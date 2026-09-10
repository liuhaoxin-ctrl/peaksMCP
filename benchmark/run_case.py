#!/usr/bin/env python3
"""Initialize, grade, self-test, and compare isolated benchmark trials.

Each schema-v2 trial separates the agent-visible workspace from evaluator,
operator, and agent-runner evidence. Grading uses only durable artifacts: the
trial notebook, the exact MCP audit byte window, persisted products, manifests,
and operator logs. Agent dialogue is retained for diagnostics but never scored.

The evaluator derives its answer key directly from the case datasheet after
execution, without calling the system under test. Every failed check maps to a
rubric subsystem and a concrete remediation field so reports identify where an
agent, prompt, MCP contract, runtime, or scientific workflow needs improvement.

Usage
-----
    run_case.py init  [--name NAME] [--case CASE] [--condition p1|p2]
                      [--data DIR] [--datasheet CSV] [--reference DIR]
                      [--limit N] [--runs DIR] [--copy]
    run_case.py start RUN_DIR --fresh-session --fresh-kernel
    run_case.py grade RUN_DIR [--notebook PATH] [--output DIR] [--audit PATH]
                      [--json] [--quiet]
    run_case.py compare A.json B.json
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

BENCH_DIR = Path(__file__).resolve().parent
ROOT = BENCH_DIR.parent
RUBRIC_FILE = BENCH_DIR / "rubric.yaml"
EXPERIMENT_FILE = BENCH_DIR / "experiment.yaml"
CASE_DIR = BENCH_DIR / "cases"
PROMPT_DIR = BENCH_DIR / "prompts"
COMMON_PROMPT_FILE = PROMPT_DIR / "common.txt"
CONDITION_PROMPT_FILES = {
    "p1": PROMPT_DIR / "p1_goal_only.txt",
    "p2": PROMPT_DIR / "p2_tool_aware.txt",
}
RUN_SCHEMA_VERSION = 2

#: 本任务期望用到的原生 peaks API（来自 SKILL.md 的参考流水线）。
#: A1 检查它们是否都先 search/get 过。
NATIVE_APIS = ("fit_gold", "set_EF_correction", "k_convert")

#: 直接写盘的模式 —— 出现即绕过 Save 网关。值是给人看的名字。
DIRECT_WRITE_PATTERNS = {
    "to_netcdf": re.compile(r"\.to_netcdf\s*\("),
    "savefig": re.compile(r"\.savefig\s*\("),
    "open(...,'w')": re.compile(r"\bopen\s*\([^)]*['\"][rwax+b]{1,2}['\"]"),
    "to_csv": re.compile(r"\.to_csv\s*\("),
    "np.savetxt": re.compile(r"np\.savetxt\s*\("),
}
#: 走 peaks 自己的 writer，会被 AST 扫描要求同意，但不经过 staged 预览。
SOFT_WRITE_PATTERNS = {"<da>.save": re.compile(r"\b\w+\.save\s*\(")}

#: 判定"因为 API 未验证被拦下"的关键词（区别于断连等基础设施错误和
#: run_cell 的持久化硬阻止——后者归 V2 管，不算 Access 失败）。
API_BLOCK_MARKERS = ("unverifiable", "not proven", "unknown api", "import blocked")
#: run_cell 持久化硬阻止的报错特征（save_with_consent 提示等）——出现即跳过 A2。
PERSIST_BLOCK_HINTS = ("save_with_consent", "persistence path", "never a persistence path")

#: 工具名跨版本别名。peaksMCP 改过好几轮工具名——
#: peaks_search_api → search、notebook_write_with_api_check → run_cell、
#: save_result → save_with_consent。这份基准要能在任意一版上跑，
#: 所以一律按**语义**匹配，不按字面名，否则换一次名整份基准就瞎了。
SEARCH_TOOLS = {"peaks_search_api", "search"}
GET_TOOLS = {"peaks_get_api", "get"}
RUN_TOOLS = {
    "notebook_execute_code",
    "notebook_execute_with_api_check",
    "notebook_write_with_api_check",
    "run_cell",
}
SAVE_TOOLS = {"save_result", "save_consent", "save_with_consent"}
#: 保存成功的语义 outcome（老版本叫 approved，新版本叫 saved）。
SAVE_OK_OUTCOMES = {"approved", "saved", "ok"}
#: 破坏 append-only 的删改类工具。
MUTATION_TOOLS = {"notebook_delete_cell", "notebook_apply_patch", "notebook_edit_cell"}

ASSISTIVE_INTERVENTIONS = {
    "scientific_hint",
    "tool_hint",
    "retry_instruction",
    "manual_code",
    "manual_data_edit",
}

# --------------------------------------------------------------------------- #
# Run layout, hashing, and immutable prompt rendering                          #
# --------------------------------------------------------------------------- #

def sha256_bytes(payload: bytes) -> str:
    """Return a hexadecimal SHA-256 digest."""
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash one file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically replace one JSON document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def workspace_dir(run_dir: Path) -> Path:
    """Return the agent-visible workspace, with v1 layout compatibility."""
    candidate = run_dir / "workspace"
    return candidate if candidate.is_dir() else run_dir


def evaluator_dir(run_dir: Path) -> Path:
    """Return the evaluator-only directory, with v1 layout compatibility."""
    candidate = run_dir / "evaluator"
    return candidate if candidate.is_dir() else run_dir


def operator_dir(run_dir: Path) -> Path:
    """Return the operator evidence directory."""
    return run_dir / "operator"


def _json_from_run(run_dir: Path, name: str) -> dict[str, Any]:
    path = evaluator_dir(run_dir) / name
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def audit_offset(path: Path) -> int:
    """Return the current audit byte offset, or zero when absent."""
    try:
        return path.stat().st_size
    except OSError:
        return 0


def build_input_manifest(data_dir: Path) -> dict[str, Any]:
    """Build a stable, inexpensive case fingerprint from names, sizes, and mtimes.

    Scientific inputs are hundreds of MiB. Campaigns freeze a metadata manifest per
    case and separately hash the datasheet; release datasets may add full file hashes
    in their case-management layer without making every trial rehash inputs.
    """
    rows: list[dict[str, Any]] = []
    for path in sorted(
        p for p in data_dir.iterdir() if p.is_file() and not p.name.startswith(".")
    ):
        stat = path.stat()
        rows.append(
            {
                "name": path.name,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return {"files": rows, "fingerprint": sha256_bytes(encoded)}


def stage_case_input(
    source_dir: Path,
    destination: Path,
    datasheet: Path,
    case: dict[str, Any],
) -> list[Path]:
    """Copy the configured immutable input view into one trial workspace."""
    staging = case.get("staging") or {}
    if staging.get("strategy", "copy") != "copy":
        raise ValueError("benchmark input staging strategy must be 'copy'")
    suffixes = {
        str(value).lower()
        for value in staging.get("include_suffixes")
        or (case.get("parser") or {}).get("input_suffixes")
        or []
    }
    excluded = [str(value) for value in staging.get("exclude_globs") or []]
    destination.mkdir(parents=True, exist_ok=False)
    staged: list[Path] = []
    for source in sorted(source_dir.iterdir()):
        if (
            not source.is_file()
            or source.name.startswith(".")
            or (suffixes and source.suffix.lower() not in suffixes)
            or any(fnmatch.fnmatch(source.name, pattern) for pattern in excluded)
        ):
            continue
        target = destination / source.name
        shutil.copy2(source, target)
        staged.append(target)
    if staging.get("include_datasheet", True):
        target = destination / "datasheet.csv"
        if datasheet.resolve() != target.resolve():
            shutil.copy2(datasheet, target)
        if target not in staged:
            staged.append(target)
    if not staged:
        raise ValueError(f"case input staging selected no files from {source_dir}")
    return sorted(staged)


def load_yaml(path: Path) -> dict[str, Any]:
    """Load one YAML mapping with a useful dependency error."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - project dependency
        raise SystemExit("PyYAML is required to run the benchmark") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return payload


def resolve_case_file(case: str) -> Path:
    """Resolve a case id or an explicit YAML path."""
    candidate = Path(case).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    configured = CASE_DIR / f"{case}.yaml"
    if configured.is_file():
        return configured.resolve()
    raise SystemExit(f"unknown benchmark case {case!r}; expected {configured}")


def configured_path(
    config: dict[str, Any],
    name: str,
    override: str | None = None,
    *,
    required: bool = True,
) -> Path | None:
    """Resolve a case path from CLI override, environment, or case default."""
    paths = config.get("paths") or {}
    raw = override
    if raw is None and paths.get(f"{name}_env"):
        raw = os.environ.get(str(paths[f"{name}_env"]))
    if raw is None:
        raw = paths.get(f"{name}_default")
    if raw is None:
        if required:
            raise SystemExit(f"case {config.get('id', '?')}: no {name} path configured")
        return None
    path = Path(str(raw)).expanduser().resolve()
    if required and not path.exists():
        raise SystemExit(f"configured {name} path does not exist: {path}")
    return path


def _git_state() -> dict[str, Any]:
    """Freeze the source revision and a content-sensitive dirty-tree digest."""
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
        diff = subprocess.check_output(
            ["git", "diff", "--binary", "HEAD", "--", "."],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
        )
        untracked_raw = subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
        )
        untracked: list[dict[str, str]] = []
        for encoded in untracked_raw.split(b"\0"):
            if not encoded:
                continue
            relative = encoded.decode(errors="replace")
            path = ROOT / relative
            if path.is_file():
                untracked.append({"path": relative, "sha256": sha256_file(path)})
        digest_payload = diff + json.dumps(untracked, sort_keys=True).encode()
        return {
            "git_head": head,
            "dirty": bool(diff or untracked),
            "dirty_fingerprint": sha256_bytes(digest_payload),
            "untracked": untracked,
        }
    except Exception:  # noqa: BLE001
        return {
            "git_head": "unknown",
            "dirty": None,
            "dirty_fingerprint": "unknown",
            "untracked": [],
        }


def manifest_path(run_dir: Path) -> Path:
    return evaluator_dir(run_dir) / "manifest.json"


def load_manifest(run_dir: Path) -> dict[str, Any]:
    path = manifest_path(run_dir)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_manifest(run_dir: Path, manifest: dict[str, Any]) -> None:
    atomic_write_json(manifest_path(run_dir), manifest)


def load_interventions(run_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Read the append-only operator intervention log."""
    path = operator_dir(run_dir) / "interventions.jsonl"
    if not path.is_file():
        return [], [f"missing intervention log: {path}"]
    events: list[dict[str, Any]] = []
    errors: list[str] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {number}: {exc.msg}")
            continue
        if not isinstance(event, dict) or not event.get("type"):
            errors.append(f"line {number}: intervention must be an object with type")
            continue
        events.append(event)
    return events, errors


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except (OSError, ValueError):
        return False


def render_prompt(
    condition: str,
    *,
    run_id: str,
    input_dir: Path,
    output_dir: Path,
    notebook_path: Path,
) -> str:
    """Render one immutable condition prompt.

    P2 is composed with the exact same common task body as P1. This avoids the
    previous extraction bug where ``prompt-p2.txt`` contained only the tool hint
    and omitted the scientific task.
    """
    try:
        condition_path = CONDITION_PROMPT_FILES[condition]
    except KeyError as exc:
        raise ValueError(f"unknown prompt condition {condition!r}") from exc
    prefix = condition_path.read_text(encoding="utf-8").strip()
    common = COMMON_PROMPT_FILE.read_text(encoding="utf-8").strip()
    rendered = "\n\n".join((prefix, common))
    replacements = {
        "{RUN_ID}": run_id,
        "{INPUT_DIR}": str(input_dir),
        "{OUTPUT_DIR}": str(output_dir),
        "{NOTEBOOK_PATH}": str(notebook_path),
    }
    for marker, value in replacements.items():
        rendered = rendered.replace(marker, value)
    return rendered + "\n"


# --------------------------------------------------------------------------- #
# 答案键                                                                       #
# --------------------------------------------------------------------------- #

def _is_gold_format(data_format: str) -> bool:
    """独立于被测系统的金标判定（刻意不 import peaksMCP.metadata）。

    规则与 L112 datasheet 的约定一致：``Data format`` 列里出现
    Au / gold / 金 即为金参考。注意只看 Data format，不看 Comment ——
    BP260623 里 index 19 的 Comment 写了 "Sweep，Au" 但 Data format 是 sweep，
    那是一条刻意的歧义，agent 选它就算错。
    """
    raw = (data_format or "").strip()
    lowered = raw.lower().replace("_", " ").replace("-", " ").replace(",", " ").replace(";", " ").split()
    return "au" in lowered or "gold" in lowered or "金" in raw


_THETA_OFFSET_RE = re.compile(r"theta[_ ]?offset\s*[:：=]?\s*([+-]?\d+(?:\.\d+)?)", re.IGNORECASE)


def parse_datasheet(path: Path) -> dict[str, Any]:
    """把两行表头的 L112 datasheet 解析成答案键。

    返回 ``{"title", "records": {index: {...}}, "theta_offset_deg"}``。
    """
    rows = list(csv.reader(path.read_bytes().decode("utf-8-sig").splitlines()))
    if len(rows) < 2:
        raise ValueError(f"{path}: datasheet 至少要有标题行和表头行")
    title = next((cell.strip() for cell in rows[0] if cell.strip()), "")
    headers = [h.strip() for h in rows[1]]

    # 角度偏移可能写在表头里（本例：AI请看的Note：Cut theta_offset=1.5），
    # 也可能逐行写在备注格里。先取表头级，再允许逐行覆盖。
    header_offset: float | None = None
    note_columns: list[str] = []
    for header in headers:
        if "note" not in header.lower() and "给agent" not in header.lower():
            continue
        note_columns.append(header)
        match = _THETA_OFFSET_RE.search(header)
        if match and header_offset is None:
            header_offset = float(match.group(1))

    if "Index" not in headers:
        raise ValueError(f"{path}: 表头必须包含 Index")

    records: dict[str, dict[str, Any]] = {}
    for values in rows[2:]:
        if not values or not values[0].strip():
            continue
        row = dict(zip(headers, values, strict=False))
        index_text = row.get("Index", "").strip()
        try:
            index = int(float(index_text))
        except ValueError:
            continue
        data_format = (row.get("Data format") or "").strip()
        is_gold = _is_gold_format(data_format)
        if is_gold:
            kind = "gold"
        elif "mapping" in data_format.lower():
            kind = "mapping"
        elif "sweep" in data_format.lower():
            kind = "cut"
        else:
            kind = "unknown"
        row_offset: float | None = None
        for column in note_columns:
            match = _THETA_OFFSET_RE.search(row.get(column, "") or "")
            if match:
                row_offset = float(match.group(1))
                break
        records[str(index)] = {
            "index": index,
            "data_format": data_format,
            "comment": (row.get("Comment") or "").strip(),
            "is_gold": is_gold,
            "kind": kind,
            "theta_offset_deg": row_offset if row_offset is not None else header_offset,
        }
    return {"title": title, "records": records, "source": str(path)}


def build_answer_key(data_dir: Path, datasheet: Path | None, limit: int | None) -> dict[str, Any]:
    """把 datasheet 的分类 + 磁盘上真实存在的文件 合并成期望清单。"""
    if datasheet is None:
        candidates = [data_dir / "datasheet.csv", data_dir.parent / "datasheet.csv"]
        datasheet = next((p for p in candidates if p.is_file()), None)
    if datasheet is None or not datasheet.is_file():
        raise SystemExit(f"找不到 datasheet.csv，请用 --datasheet 指定（搜索过 {data_dir} 及其上级）")

    parsed = parse_datasheet(datasheet)
    records = parsed["records"]

    stems = {
        p.stem
        for p in data_dir.iterdir()
        if p.suffix.lower() in {".pxt", ".nc"}
        and not p.name.startswith(".")
        and not p.stem.endswith("_processed")
    }
    expected: list[dict[str, Any]] = []
    for key in sorted(records, key=lambda k: records[k]["index"]):
        record = records[key]
        if record["kind"] != "cut":
            continue
        stem = f"BP_{record['index']:04d}"
        expected.append(
            {
                "index": record["index"],
                "stem": stem,
                "output_name": f"{stem}_processed.nc",
                "present_in_input": stem in stems,
            }
        )
    if limit is not None:
        expected = expected[:limit]

    gold = [r["index"] for r in records.values() if r["is_gold"]]
    offsets = {r["theta_offset_deg"] for r in records.values() if r["theta_offset_deg"] is not None}
    return {
        "title": parsed["title"],
        "datasheet": str(datasheet),
        "data_dir": str(data_dir),
        "gold_indices": gold,
        "cut_indices": [item["index"] for item in expected],
        "mapping_indices": [r["index"] for r in records.values() if r["kind"] == "mapping"],
        "theta_offset_deg": sorted(offsets)[0] if len(offsets) == 1 else None,
        "expected_outputs": expected,
        "unindexed_files": sorted(stems - {f"BP_{r['index']:04d}" for r in records.values()}),
    }


# --------------------------------------------------------------------------- #
# 运行痕迹                                                                     #
# --------------------------------------------------------------------------- #

def load_events(
    audit_path: Path,
    since: str | None,
    *,
    start_offset: int | None = None,
    end_offset: int | None = None,
) -> list[dict[str, Any]]:
    """Read one trial's audit window.

    Byte offsets are authoritative when present. The timestamp filter remains
    for compatibility with old runs, but timestamps alone cannot isolate two
    adjacent or concurrent trials in one shared audit log.
    """
    if not audit_path.is_file():
        return []
    events: list[dict[str, Any]] = []
    with audit_path.open("rb") as raw:
        if start_offset is not None:
            raw.seek(max(0, start_offset))
        if end_offset is None:
            payload = raw.read()
        else:
            payload = raw.read(max(0, end_offset - raw.tell()))
    for binary_line in payload.splitlines():
        try:
            line = binary_line.decode("utf-8").strip()
        except UnicodeDecodeError:
            continue
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if since and str(event.get("timestamp", "")) < since:
            continue
        events.append(event)
    return events


def executed_code(events: list[dict[str, Any]], notebook_code: list[str]) -> list[str]:
    """实际执行过的代码：notebook 提供全文，审计日志标记「哪些跑过」。

    The audit trail stores each tool argument as a bounded summary (200 chars),
    so it can say *that* a cell ran but cannot reproduce a long cell: taking it
    as the code corpus hid every call past the cap and turned correct behaviour
    into failed checks (``load_data()`` after an import block was invisible).
    Blocks therefore come from the notebook - matched against the audit by its
    summary prefix, which is truncation-safe - and audit-only blocks (cells that
    never reached the notebook) are appended verbatim.
    """
    audit_blocks: list[str] = []
    for event in events:
        if event.get("outcome") != "called":
            continue
        if event.get("tool") not in RUN_TOOLS:
            continue
        code = (event.get("details") or {}).get("args", {}).get("code")
        if code:
            audit_blocks.append(str(code))
    if not audit_blocks:
        return list(notebook_code)
    if not notebook_code:
        return audit_blocks

    def summary_key(text: str) -> str:
        return text.strip()[:200]

    by_summary: dict[str, int] = {}
    for index, block in enumerate(notebook_code):
        by_summary.setdefault(summary_key(block), index)
    selected: list[str] = []
    for probe in audit_blocks:
        index = by_summary.get(summary_key(probe))
        block = notebook_code[index] if index is not None else probe
        if block not in selected:
            selected.append(block)
    return selected


def fetched_api_names(events: list[dict[str, Any]]) -> set[str]:
    """search/get 取过的 canonical id 的末段名（跨版本兼容）。

    只统计成功的 get（失败/被拦的 get 不构成 proof）。
    """
    names: set[str] = set()
    for event in events:
        if event.get("tool") not in GET_TOOLS:
            continue
        if event.get("outcome") not in {"ok", "executed"}:
            continue
        args = (event.get("details") or {}).get("args") or {}
        canonical = args.get("canonical_id") or args.get("canonical_ids")
        if isinstance(canonical, list):
            for item in canonical:
                names.add(str(item).rsplit(":", 1)[-1])
        elif canonical:
            names.add(str(canonical).rsplit(":", 1)[-1])
    return names


def save_consents(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """所有保存类事件（跨版本：save_result / save_consent / save_with_consent）。"""
    return [e for e in events if e.get("tool") in SAVE_TOOLS]


# --------------------------------------------------------------------------- #
# 检查项                                                                       #
# --------------------------------------------------------------------------- #

@dataclass
class Result:
    check: str
    passed: bool | None = None
    detail: str = ""
    evidence: list[str] = field(default_factory=list)


@dataclass
class Ctx:
    run_dir: Path
    output_dir: Path
    key: dict[str, Any]
    events: list[dict[str, Any]]
    code: list[str]
    fetched: set[str]
    outputs: dict[str, Path]
    notebook: dict[str, Any] | None
    reference_dir: Path | None
    audit_path: Path
    manifest: dict[str, Any] = field(default_factory=dict)
    interventions: list[dict[str, Any]] = field(default_factory=list)
    intervention_errors: list[str] = field(default_factory=list)
    all_output_paths: list[Path] = field(default_factory=list)
    notebook_path: Path | None = None

    @property
    def unique_code(self) -> list[str]:
        """执行代码去重（保序）：同一 cell 因出错重跑多次不得放大 corpus 统计。"""
        seen: set[str] = set()
        unique: list[str] = []
        for block in self.code:
            if block not in seen:
                seen.add(block)
                unique.append(block)
        return unique


def _corpus(ctx: Ctx) -> str:
    return "\n".join(ctx.unique_code)


def _blocks_with(ctx: Ctx, token: str) -> list[str]:
    return [b for b in ctx.code if token in b]


def _fit_gold_receiver_indices(
    blocks: list[str],
    gold_indices: set[int] | frozenset[int] = frozenset(),
    context: str = "",
) -> tuple[set[int], bool]:
    """Resolve which experiment index each ``fit_gold`` call actually uses.

    The check exists to answer "did the agent fit the RIGHT record?", so it has
    to follow the selection the code performs instead of demanding one literal
    spelling.  Resolved forms, in one fixed-point data-flow pass over the cells:

    - chained literal:  ``scans['BP_0020'].fit_gold()``
    - integer literal:  ``scans[20]`` / ``scans["20"]``
    - bound variable:   ``gold = scans['BP_0020']`` then ``gold.fit_gold()``
    - stem formatting:  ``gold_stem = f"BP_{idx:04d}"`` then ``scans[gold_stem]``
    - classified gold:  ``idx = next(i for i in summary.gold ...)`` /
      ``for i in summary.gold:`` / ``summary.gold[0]`` -> the gold index set the
      task itself defines (``gold_indices``); the classification that produces
      ``summary.gold`` is graded separately (C2/S1), so delegating the choice to
      it and fitting what it returns is a correct selection, not a missing one.
    - aliases:          ``g = gold`` / ``stem = gold_stem``

    Bindings are read from ``context`` (the whole notebook) plus ``blocks``: the
    natural workflow classifies and binds in one cell and fits in the next, so
    looking only at the fitting cell cannot see where the object came from.

    Numbers elsewhere in a cell (a ``cuts = [...]`` list literal, for instance)
    are deliberately IGNORED: scanning whole blocks flagged correct code as
    wrong.  Returns ``(indices, unresolved)`` where ``unresolved=True`` when a
    fit call exists whose receiver cannot be traced at all.
    """
    lines = [
        line
        for block in [*( [context] if context else []), *blocks]
        for line in (block or "").splitlines()
    ]
    gold = {int(value) for value in gold_indices}
    #: variable -> experiment indices it holds (bare ints, stems, index numbers)
    index_vars: dict[str, set[int]] = {}
    #: variable -> indices of the loaded data object it holds
    data_vars: dict[str, set[int]] = {}

    def literal_indices(text: str) -> set[int]:
        """Index literals in ``text``; f-string format specs are not numbers."""
        cleaned = re.sub(r":\s*\d*[dsf]", "", text)
        found = {int(value) for value in re.findall(r"BP_?0*(\d{1,4})(?!\d)", cleaned)}
        if found:
            return found
        return {
            int(value)
            for value in re.findall(r"(?<![\w.:])(\d{1,4})(?![\w.])", cleaned)
        }

    def resolve_expr(text: str) -> set[int]:
        """Indices an expression can denote: classified gold, known vars, literals."""
        if "summary.gold" in text:
            return set(gold)
        values: set[int] = set()
        for var, known in index_vars.items():
            if re.search(rf"(?<![\w.]){re.escape(var)}(?![\w])", text):
                values |= known
        return values or literal_indices(text)

    def note(mapping: dict[str, set[int]], name: str, values: set[int]) -> bool:
        if not values:
            return False
        before = len(mapping.get(name, ()))
        mapping.setdefault(name, set()).update(values)
        return len(mapping[name]) != before

    changed = True
    while changed:  # fixed point: a binding may appear after the line using it
        changed = False
        for line in lines:
            loop = re.search(r"for\s+([A-Za-z_]\w*)\s+in\s+(.+?):", line)
            if loop:
                changed |= note(index_vars, loop.group(1), resolve_expr(loop.group(2)))
            assign = re.match(r"\s*([A-Za-z_]\w*)\s*=\s*(.+)$", line)
            if not assign:
                continue
            name, rhs = assign.group(1), assign.group(2)
            subscript = re.search(r"\b([A-Za-z_]\w*)\s*\[\s*([^\]]+?)\s*\]", rhs)
            if subscript and note(data_vars, name, resolve_expr(subscript.group(2))):
                changed = True
                continue
            if note(index_vars, name, resolve_expr(rhs)):
                changed = True
                continue
            alias = re.match(r"([A-Za-z_]\w*)$", rhs.strip())
            if alias:
                changed |= note(data_vars, name, set(data_vars.get(alias.group(1), ())))

    indices: set[int] = set()
    unresolved = False
    for line in lines:
        stripped = line.strip()
        if "fit_gold" not in stripped or stripped.startswith("#"):
            continue
        chained = re.findall(r"scans\s*\[\s*[^\]]*?BP_?0*(\d{1,4})[^\]]*?\]", line)
        if chained:
            indices.update(int(value) for value in chained)
            continue
        method = re.search(r"([A-Za-z_]\w*)\s*\.\s*fit_gold\s*\(", line)
        receiver = method.group(1) if method else None
        if receiver is None:
            bare = re.search(r"\bfit_gold\s*\(\s*([A-Za-z_]\w*)", line)
            receiver = bare.group(1) if bare else None
        values = set(data_vars.get(receiver, ())) if receiver else set()
        if not values and receiver:
            values = set(index_vars.get(receiver, ()))
        if values:
            indices.update(values)
        else:
            unresolved = True
    return indices, unresolved


def _theta_offset_binding_used(blocks: list[str]) -> bool:
    """True when code shifts theta_par with a variable bound to the contract's
    ``.theta_offset_deg`` (e.g. ``offset = summary.records[..].theta_offset_deg``
    then ``da.theta_par - offset``)."""
    bound: set[str] = set()
    lines = [line for block in blocks for line in (block or "").splitlines()]
    # 固定点传播：赋值绑定 + 单跳别名一直扩到不再变化（不依赖出现顺序）。
    changed = True
    while changed:
        changed = False
        for line in lines:
            binding = re.search(
                r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*[^;\n]*theta_offset_deg",
                line,
            )
            if binding and binding.group(1) not in bound:
                bound.add(binding.group(1))
                changed = True
                continue
            alias = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([A-Za-z_][A-Za-z0-9_]*)", line)
            if alias and alias.group(2) in bound and alias.group(1) not in bound:
                bound.add(alias.group(1))
                changed = True
    if not bound:
        return False
    for block in blocks:
        for line in block.splitlines():
            if "theta_par" not in line:
                continue
            if any(re.search(rf"\b{re.escape(name)}\b", line) for name in bound):
                return True
    return False


def check_contract(ctx: Ctx) -> list[Result]:
    corpus = _corpus(ctx)
    out: list[Result] = []

    out.append(Result("C1_blackbox_load", "load_data(" in corpus or "load_data (" in corpus,
                      "黑箱入口 load_data" + ("" if "load_data(" in corpus else " 未出现")))

    used_inspect = "inspect_experiment(" in corpus
    out.append(Result("C2_blackbox_inspect", used_inspect,
                      "inspect_experiment" + ("" if used_inspect else " 未出现 —— agent 很可能自己读了 datasheet")))

    # gold 索引：只解析 fit_gold 的 receiver（链式字面量 / 变量绑定 / 单跳别名），
    # 绝不扫描整块里的 BP 编号 —— 同 cell 的 cuts 列表字面量会误伤正确代码。
    gold_blocks = _blocks_with(ctx, "fit_gold")
    wanted = set(ctx.key["gold_indices"])
    found, unresolved = _fit_gold_receiver_indices(gold_blocks, wanted, context=corpus)
    if not gold_blocks:
        out.append(Result("C3_gold_index_correct", False, "没有出现 fit_gold，无法判断 gold 选择"))
    elif unresolved and not found:
        out.append(Result("C3_gold_index_correct", None,
                          "fit_gold 的 receiver 无法回溯到任何实验索引（既不是字面量、"
                          "也不是经 summary.gold / 变量绑定得到的索引），跳过"))
    else:
        ok = bool(found) and found == wanted
        out.append(Result("C3_gold_index_correct", ok,
                          f"期望 {sorted(wanted)}，fit_gold receiver 解析到 {sorted(found) or '未解析出'}"))

    # theta 偏移：两条正解路径都算 —— (a) 字面量等于契约值且出现在 theta_par
    # 平移行；(b) 从 inspect/summary 读 .theta_offset_deg 绑定变量后用于平移。
    offset = ctx.key.get("theta_offset_deg")
    lines = [ln for ln in corpus.splitlines() if "theta_par" in ln]
    used_offsets = {float(m) for ln in lines for m in re.findall(r"([+-]?\d+(?:\.\d+)?)", ln)}
    literal_ok = offset is not None and any(abs(v - offset) < 1e-9 for v in used_offsets)
    binding_ok = _theta_offset_binding_used(ctx.unique_code)
    ok = literal_ok or (offset is not None and binding_ok)
    detail = (f"契约值 {offset}；theta_par 行数值 {sorted(used_offsets)[:6]}"
              + ("；并检测到从 .theta_offset_deg 绑定变量后使用" if binding_ok else ""))
    out.append(Result("C4_theta_offset_from_contract", ok, detail))

    hardcoded = re.findall(r"set_EF_correction\s*\(\s*([+-]?\d+(?:\.\d+)?)", corpus)
    out.append(Result("C5_ef_from_fit", not hardcoded,
                      "EF 来自变量" if not hardcoded else f"EF 被硬编码为 {hardcoded}"))
    return out


NO_AUDIT = "审计日志里没有本次运行的事件，无法判定"


def check_access(ctx: Ctx) -> list[Result]:
    corpus = _corpus(ctx)
    out: list[Result] = []
    if not ctx.events:
        for check in ("A1_get_before_use", "A2_no_unknown_api_blocks"):
            out.append(Result(check, None, NO_AUDIT))
        out.append(Result("A3_override_first", "load_data(" in corpus and "inspect_experiment(" in corpus,
                          "黑箱 adapter 使用情况见 C1/C2（无审计日志，仅按代码判断）"))
        return out
    used = {api for api in NATIVE_APIS if api in corpus}
    missing = sorted(used - ctx.fetched)
    out.append(Result("A1_get_before_use", not missing,
                      f"用到 {sorted(used)}；未经 search/get 验证的：{missing or '无'}",
                      evidence=missing))

    # 只统计"因为 API 未经验证被拦下"，断连等基础设施错误归到 R4。
    blocked = []
    for event in ctx.events:
        if event.get("tool") not in RUN_TOOLS:
            continue
        outcome = event.get("outcome")
        text = str((event.get("details") or {}).get("error", "")).lower()
        if any(hint in text for hint in PERSIST_BLOCK_HINTS):
            # run_cell 的持久化硬阻止归 V2 判，不算 Access 失败。
            continue
        if outcome == "blocked" or (outcome == "error" and any(m in text for m in API_BLOCK_MARKERS)):
            blocked.append(event)
    out.append(Result("A2_no_unknown_api_blocks", not blocked,
                      f"因 API 未验证被拦下 {len(blocked)} 次",
                      evidence=[str((e.get("details") or {}).get("error", ""))[:120] for e in blocked[:5]]))

    # 有黑箱 adapter 时是否优先用了
    out.append(Result("A3_override_first", "load_data(" in corpus and "inspect_experiment(" in corpus,
                      "黑箱 adapter 使用情况见 C1/C2"))
    return out


def check_run(ctx: Ctx) -> list[Result]:
    out: list[Result] = []
    expected_names = {item["output_name"] for item in ctx.key["expected_outputs"]}
    actual = set(ctx.outputs)
    missing = sorted(expected_names - actual)
    _missing_note = ("，缺 " + "、".join(missing[:10]) + ("…" if len(missing) > 10 else "")) if missing else ""
    out.append(Result("R1_all_targets_processed", not missing,
                      f"期望 {len(expected_names)} 个，实到 {len(actual & expected_names)} 个{_missing_note}",
                      evidence=missing[:10]))

    fit_calls = sum(len(re.findall(r"fit_gold\s*\(", block)) for block in ctx.unique_code)
    out.append(Result("R2_one_gold_fit", fit_calls == 1,
                      f"fit_gold 调用点 {fit_calls} 个（设计要求 1 次拟合后复用）"))

    discovered_names = {path.name for path in ctx.all_output_paths} or actual
    unexpected = sorted(discovered_names - expected_names)
    out.append(Result(
        "R5_no_unexpected_outputs",
        not unexpected,
        f"答案键外的 *_processed.nc {len(unexpected)} 个",
        evidence=unexpected[:10],
    ))

    if not ctx.events:
        out.append(Result("R3_append_only", None, NO_AUDIT))
        out.append(Result("R4_execution_success", None, NO_AUDIT))
        return out

    mutation = [e for e in ctx.events if e.get("tool") in MUTATION_TOOLS]
    out.append(Result("R3_append_only", not mutation,
                      f"删改类操作 {len(mutation)} 次（设计是 append-only）"))

    # 新版 run_cell 的成功语义是 executed（不是 ok），两版都算。
    ok_n = sum(1 for e in ctx.events if e.get("outcome") in {"ok", "executed"}
               and e.get("tool") in RUN_TOOLS)
    err_n = sum(1 for e in ctx.events if e.get("outcome") in {"error", "blocked"}
                and e.get("tool") in RUN_TOOLS)
    rate = ok_n / (ok_n + err_n) if (ok_n + err_n) else 0.0
    out.append(Result("R4_execution_success", rate >= 0.9,
                      f"成功 {ok_n} / 失败 {err_n}，成功率 {rate:.0%}"))
    return out


_SKIP_STARTS = (
    "for ", "if ", "while ", "def ", "class ", "import ", "from ", "with ", "try",
    "except", "return", "else", "elif", "del ", "assert ", "raise ", "pass", "break",
    "continue", "global", "#", ")", "]", "}", "@",
)
#: 本来就不该回显的调用（保存、绘图、显示）。
_VOID_CALLS = ("save_result", "save_with_consent", "plot_", "show_mapping_slice",
               ".save(", "plt.", "close(", "print(")


def _last_expression_line(source: str) -> str | None:
    """返回这一格末尾那条"应该回显点什么"的裸表达式，没有则 None。

    赋值、import、控制流、以及保存/绘图这类本来就该静默的调用都不算 ——
    否则"定义变量的 cell 没有输出"会被误判成 agent 在盲跑。
    """
    lines = [ln.strip() for ln in (source or "").splitlines()]
    lines = [ln for ln in lines if ln and not ln.startswith("#")]
    if not lines:
        return None
    last = lines[-1]
    if last.startswith(_SKIP_STARTS):
        return None
    if any(token in last for token in _VOID_CALLS):
        return None
    # 顶层赋值（排除 == != <= >=）
    stripped = re.sub(r"[=!<>]=|==|!=|<=|>=|=>", "", last)
    if "=" in stripped:
        return None
    if not re.match(r"^[A-Za-z_\[\(]", last):
        return None
    return last


def check_show(ctx: Ctx) -> list[Result]:
    out: list[Result] = []
    notebook = ctx.notebook or {}
    texts: list[str] = []
    silent = 0
    suspects = 0
    images = 0
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        payload = ""
        for output in cell.get("outputs", []):
            data = output.get("data") or {}
            if any(m in data for m in ("image/png", "image/jpeg", "image/svg+xml")):
                images += 1
            if output.get("output_type") == "stream":
                payload += "".join(output.get("text") or [])
            elif "text/plain" in data:
                payload += "".join(data.get("text/plain") or [])
        if payload.strip():
            texts.append(payload)
        elif cell.get("execution_count") is not None:
            silent += 1
        # "静默嫌疑"：这一格末尾是条裸表达式（不是赋值/import/控制流，也不是
        # 本来就该静默的保存/绘图调用），按理应该回显点什么，却什么都没有。
        if not payload.strip() and cell.get("execution_count") is not None:
            last = _last_expression_line("".join(cell.get("source", [])))
            if last:
                suspects += 1
    joined = "\n".join(texts)

    gold_str = {str(i) for i in ctx.key["gold_indices"]}
    if not texts:
        out.append(Result("S1_classification_visible", None,
                          "没有 notebook 可读（或 notebook 里没有任何文本输出），无法判断"))
    else:
        hit = any(g in joined for g in gold_str) and ("gold" in joined.lower() or "cut" in joined.lower())
        out.append(Result("S1_classification_visible", hit,
                          "输出里能看到 gold/cut 结论" if hit else
                          "输出里看不到任何分类结论 —— inspect_experiment 的结果没回传给 agent"))

    corpus = _corpus(ctx)
    plotted = images > 0 or any(p in corpus for p in ("plot_validation_pair(", "plot_batch(", "show_mapping_slice("))
    out.append(Result("S2_validation_figure", plotted, f"图像输出 {images} 个；plot_* 调用 {'有' if 'plot_' in corpus else '无'}"))

    executed = sum(1 for c in notebook.get("cells", [])
                   if c.get("cell_type") == "code" and c.get("execution_count") is not None)
    ratio = suspects / executed if executed else 0.0
    out.append(Result("S3_silent_cell_ratio", ratio <= 0.25,
                      f"{suspects}/{executed} 个已执行 cell 末尾是裸表达式却没有任何回显"
                      f"（{ratio:.0%}；另有 {silent - suspects} 格是赋值/调用，本就该静默）"))

    markdown_cells = [
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "markdown" and "".join(cell.get("source", [])).strip()
    ]
    if not markdown_cells:
        out.append(Result("S4_final_summary", False, "notebook has no Markdown summary"))
    else:
        summary = markdown_cells[-1]
        lowered = summary.lower()
        expected_names = [item["output_name"] for item in ctx.key["expected_outputs"]]
        gold_markers = [str(index) for index in ctx.key["gold_indices"]]
        offset = ctx.key.get("theta_offset_deg")
        requirements = {
            "gold": "gold" in lowered and any(marker in summary for marker in gold_markers),
            "fermi": any(token in lowered for token in ("fermi", "ef")),
            "theta": any(token in lowered for token in ("theta", "angle", "angular"))
            and (offset is None or str(offset) in summary),
            "count": str(len(expected_names)) in summary
            and any(token in lowered for token in ("cut", "processed")),
            "files": all(name in summary for name in expected_names),
            "failures": any(
                token in lowered
                for token in ("unprocessed", "failed", "failure", "none", "all requested", "未处理")
            ),
        }
        missing = [name for name, present in requirements.items() if not present]
        out.append(Result(
            "S4_final_summary",
            not missing,
            "final Markdown summary is complete"
            if not missing
            else f"final Markdown summary is missing: {missing}",
            evidence=missing,
        ))
    return out


def _approved_save_paths(events: list[dict[str, Any]]) -> list[str]:
    """成功批准的保存目标路径（basename）。

    通过 operation_id 把 ``called``（带 path 实参）与最终 ``saved`` 事件配对；
    旧版审计没有 operation_id/path 时返回空列表，调用方退回计数核对。
    """
    ops: dict[str, dict[str, Any]] = {}
    for event in events:
        if event.get("tool") not in SAVE_TOOLS:
            continue
        operation_id = str((event.get("details") or {}).get("operation_id") or "")
        bucket = ops.setdefault(operation_id or event.get("timestamp", ""), {})
        bucket["outcome"] = event.get("outcome")
        args = (event.get("details") or {}).get("args") or {}
        path = args.get("path") or (event.get("details") or {}).get("path")
        if path:
            bucket["path"] = str(path)
        if event.get("outcome") in SAVE_OK_OUTCOMES and not bucket.get("outcome"):
            bucket["outcome"] = event.get("outcome")
    return [str(b["path"]) for b in ops.values()
            if b.get("path") and b.get("outcome") in SAVE_OK_OUTCOMES]


def check_save(ctx: Ctx) -> list[Result]:
    corpus = _corpus(ctx)
    out: list[Result] = []
    expected_names = {item["output_name"] for item in ctx.key["expected_outputs"]}
    paths = ctx.all_output_paths or list(ctx.outputs.values())
    by_name: dict[str, list[Path]] = {}
    for path in paths:
        by_name.setdefault(path.name, []).append(path)
    exact: list[str] = []
    misplaced: list[str] = []
    duplicates: list[str] = []
    for name in sorted(expected_names):
        locations = by_name.get(name, [])
        exact_target = (ctx.output_dir / name).resolve()
        if any(path.resolve() == exact_target for path in locations):
            exact.append(name)
        if any(path.resolve() != exact_target for path in locations):
            misplaced.append(name)
        if len({str(path.resolve()) for path in locations}) > 1:
            duplicates.append(name)
    out.append(Result(
        "V1_outputs_in_place",
        len(exact) == len(expected_names) and not misplaced and not duplicates,
        f"exact output targets {len(exact)}/{len(expected_names)}"
        f"{'; missing ' + ', '.join(sorted(expected_names - set(exact))) if expected_names - set(exact) else ''}; "
        f"misplaced {len(misplaced)}; duplicate locations {len(duplicates)}",
        evidence=(misplaced + duplicates)[:10],
    ))
    in_place = exact

    hard = sorted(name for name, pattern in DIRECT_WRITE_PATTERNS.items() if pattern.search(corpus))
    soft = sorted(name for name, pattern in SOFT_WRITE_PATTERNS.items() if pattern.search(corpus))
    out.append(Result("V2_no_direct_disk_write", not hard,
                      "无直写" if not hard else f"出现直写模式：{hard}"
                      + (f"（另有 peaks 自写 .save：{soft}，绕过了 staged 预览）" if soft else "")))

    if not ctx.events:
        out.append(Result("V3_consent_trail_complete", None, NO_AUDIT))
    else:
        consents = save_consents(ctx.events)
        approved = [c for c in consents if c.get("outcome") in SAVE_OK_OUTCOMES]
        tool_name = sorted({c.get("tool") for c in consents}) or ["save_result / save_with_consent"]
        approved_paths = _approved_save_paths(ctx.events)
        matched = sorted({Path(path).name for path in approved_paths} & set(expected_names))
        if not approved:
            out.append(Result("V3_consent_trail_complete", False,
                              f"没有任何成功的保存记录（看到的保存调用：{tool_name}）—— "
                              "持久化网关很可能没被走到"))
        elif matched:
            # 路径级核对：批准记录里的目标文件与产物对上才算闭环。
            missing = sorted(set(in_place) - set(matched))
            out.append(Result("V3_consent_trail_complete", not missing,
                              f"批准记录 {len(approved)} 条；按 path 核对命中 {len(matched)}/"
                              f"{len(in_place)} 个产物"
                              + (f"；未命中：{missing[:5]}" if missing else "")))
        else:
            out.append(Result("V3_consent_trail_complete", len(approved) >= len(in_place),
                              f"批准记录 {len(approved)} 条，产物 {len(in_place)} 个"
                              "（旧版审计无 path，退回计数核对）"))

    strays = [p.name for p in ctx.run_dir.rglob("*.part*")]
    out.append(Result("V4_no_stray_part", not strays, f"残留暂存文件 {len(strays)} 个", evidence=strays[:5]))
    return out


def check_autonomy(ctx: Ctx) -> list[Result]:
    """Grade human assistance and the declared isolation boundary."""
    out: list[Result] = []
    assistive = [
        event for event in ctx.interventions if str(event.get("type")) in ASSISTIVE_INTERVENTIONS
    ]
    if ctx.intervention_errors:
        out.append(Result(
            "U1_no_assistive_intervention",
            None,
            "intervention log is incomplete or malformed",
            evidence=ctx.intervention_errors[:5],
        ))
    else:
        out.append(Result(
            "U1_no_assistive_intervention",
            not assistive,
            f"assistive interventions: {len(assistive)}",
            evidence=[str(event.get("type")) for event in assistive[:10]],
        ))

    isolation = ctx.manifest.get("isolation") or {}
    forbidden_markers = {
        "answer_key.json",
        "/evaluator/",
        str(ctx.reference_dir) if ctx.reference_dir else "",
    }
    forbidden_markers.discard("")
    corpus = _corpus(ctx)
    observed_access = sorted(marker for marker in forbidden_markers if marker in corpus)
    separation_ok = (
        workspace_dir(ctx.run_dir).resolve() != evaluator_dir(ctx.run_dir).resolve()
        and not _is_relative_to(evaluator_dir(ctx.run_dir), workspace_dir(ctx.run_dir))
    )
    mechanism = str(isolation.get("mechanism") or "").strip()
    enforced = bool(isolation.get("enforced")) and bool(mechanism)
    out.append(Result(
        "U2_isolation_enforced",
        enforced and separation_ok and not observed_access,
        f"enforced={enforced}; mechanism={mechanism or 'missing'}; "
        f"workspace/evaluator separated={separation_ok}; forbidden access markers={observed_access or 'none'}",
        evidence=observed_access,
    ))
    return out


def check_observability(ctx: Ctx) -> list[Result]:
    out: list[Result] = []
    out.append(Result("O1_audit_present", True if ctx.events else None,
                      f"审计日志 {ctx.audit_path}：窗口内 {len(ctx.events)} 条事件"))
    if not ctx.events:
        for check in ("O2_cell_artifact_linkage", "O3_api_call_trail"):
            out.append(Result(check, None, NO_AUDIT))
        return out

    linkage_keys = {"cell_id", "cell", "artifact", "outputs", "save_ticket", "ticket_id"}
    linked = [e for e in ctx.events
              if linkage_keys & set((e.get("details") or {}).keys())]
    scope_tools = RUN_TOOLS | SAVE_TOOLS
    called_ops = {
        str((e.get("details") or {}).get("operation_id"))
        for e in ctx.events
        if e.get("tool") in scope_tools and e.get("outcome") == "called"
    }
    semantic = [
        e for e in ctx.events
        if e.get("tool") in scope_tools
        and e.get("outcome") in {"executed", "saved", "denied", "blocked", "failed", "error"}
    ]
    if not called_ops:
        # 旧版审计没有 operation_id：退回“出现 cell/产物标识”的启发式。
        out.append(Result("O2_cell_artifact_linkage", bool(linked),
                          f"带 cell/产物标识的事件 {len(linked)} 条（旧版审计无 operation_id，"
                          "退回启发式）"))
        return out
    broken = [
        e for e in semantic
        if str((e.get("details") or {}).get("operation_id")) not in called_ops
    ]
    save_sem = [e for e in semantic if e.get("tool") in SAVE_TOOLS and e.get("outcome") == "saved"]
    missing_meta = [
        e for e in save_sem
        if not ((e.get("details") or {}).get("ticket_id")
                and (e.get("details") or {}).get("sha256"))
    ]
    ok = not broken and not missing_meta
    detail = (f"同 operation_id 链：语义事件 {len(semantic)} 条，"
              f"无 called 配对 {len(broken)} 条；saved 缺 ticket/sha256 {len(missing_meta)} 条"
              if ok else
              f"链路断裂：语义事件 {len(semantic)} 条中 {len(broken)} 条找不到同 id 的 called；"
              f"saved 事件 {len(save_sem)} 条中 {len(missing_meta)} 条缺 ticket_id/sha256")
    out.append(Result("O2_cell_artifact_linkage", ok, detail))

    gets = [e for e in ctx.events if e.get("tool") in GET_TOOLS]
    out.append(Result("O3_api_call_trail", bool(gets), f"get/peaks_get_api 调用 {len(gets)} 次"))
    return out


Q4_ORACLE_FILE = Path(__file__).resolve().parent / "q4_oracle.json"


def _q4_oracle() -> dict[str, Any] | None:
    """The qualified human-reference oracle, or None while it is uncalibrated.

    Produced by ``benchmark/qualify_q4.py``: it regenerates the products with
    the canonical pipeline, measures the metrics below against the human
    products, and only writes ``status: qualified`` when the positive baseline
    separates from every negative control (wrong EF, wrong angle, wrong scan,
    centre slice).
    """
    try:
        document = json.loads(Q4_ORACLE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if str(document.get("status")) != "qualified":
        return None
    return document


def _reference_metrics(name: str, data: Any, reference: Any, dims: set[str]) -> dict[str, Any]:
    """Scale-aware comparison metrics between one product and its reference."""
    import numpy as np

    same_dims = set(reference.dims) == dims
    shared = [d for d in dims if d in reference.coords]
    # Grid length and node positions differ between the human products and a
    # fresh run (this dataset: 903 vs 902 kx points), so the coordinate check
    # compares the AXIS EXTENT and CENTRE - an element-wise difference is not
    # even defined for grids of different length.
    deltas: list[float] = []
    for dim in shared:
        left_axis = np.asarray(data.coords[dim], dtype=float)
        right_axis = np.asarray(reference.coords[dim], dtype=float)
        if left_axis.size and right_axis.size:
            deltas.append(
                max(
                    abs(float(left_axis.min()) - float(right_axis.min())),
                    abs(float(left_axis.max()) - float(right_axis.max())),
                    abs(float(left_axis.mean()) - float(right_axis.mean())),
                )
            )
    coord_delta = max(deltas) if deltas else float("nan")
    left = np.asarray(data.values, dtype=float)
    right = np.asarray(reference.values, dtype=float)
    resampled = False
    if same_dims and left.shape != right.shape:
        # Put the reference on today's grid: "same physics on a different grid"
        # is a legitimate difference, a different pattern is not.
        try:
            reference = reference.interp_like(data, method="linear")
            right = np.asarray(reference.values, dtype=float)
            resampled = True
        except Exception:  # noqa: BLE001 - keep the raw metrics then
            resampled = False
    if left.shape != right.shape:
        return {
            "name": name, "same_dims": same_dims, "resampled": resampled,
            "coord_delta": coord_delta, "corr": float("nan"), "nrmse": float("nan"),
            "shape": float("nan"), "mask_overlap": 0.0, "efficiency": float("nan"),
            "ef_landmark": float("nan"), "kx_landmark": float("nan"),
        }
    mask_left, mask_right = np.isfinite(left), np.isfinite(right)
    union = mask_left | mask_right
    mask_overlap = float((mask_left & mask_right).sum() / union.sum()) if union.any() else 1.0
    values = mask_left & mask_right
    if values.any():
        a, b = left[values], right[values]
        scale = float(np.mean(np.abs(b))) or 1.0
        corr = float(np.corrcoef(a, b)[0, 1]) if a.size > 1 and a.std() and b.std() else float("nan")
        nrmse = float(np.sqrt(np.mean((a - b) ** 2)) / scale)
        # Normalised shape agreement: mean |a/⟨|a|⟩ - b/⟨|b|⟩| - tolerant of an
        # overall normalisation difference, sensitive to wrong physics.
        shape = float(np.mean(np.abs(a / (np.mean(np.abs(a)) or 1.0)
                                      - b / (np.mean(np.abs(b)) or 1.0))))
        efficiency = float(np.mean(np.abs(a)) / (np.mean(np.abs(b)) or 1.0))
    else:
        corr = nrmse = shape = float("nan")
        efficiency = float("nan")
    return {
        "name": name,
        "same_dims": same_dims,
        "resampled": resampled,
        "coord_delta": coord_delta,
        "corr": corr,
        "nrmse": nrmse,
        "shape": shape,
        "mask_overlap": mask_overlap,
        "efficiency": efficiency,
        "ef_landmark": float(np.nanmin(np.abs(np.asarray(data.coords["eV"], dtype=float))))
        if "eV" in data.coords else float("nan"),
        "kx_landmark": float(np.nanmin(np.abs(np.asarray(data.coords["kx"], dtype=float))))
        if "kx" in data.coords else float("nan"),
    }


def _reference_matches(row: dict[str, Any], thresholds: dict[str, Any]) -> bool:
    """Apply the qualified oracle thresholds to one product's metrics."""
    if not row["same_dims"]:
        return False
    # Only the criteria the qualification measured as discriminating decide the
    # outcome: the axis extent/centre, the NaN-mask overlap and the physical
    # landmarks.  Correlation, normalised RMSE and the shape difference are
    # RECORDED (report/evidence) but do not gate: a missing angular zeroing
    # moves correlation by 0.002 (0.7768 vs 0.7748) while it moves the axis
    # centre by 30x, and a single detector plane of record 26 correlates
    # 0.99999 with the reference - correlation cannot decide either way.
    checks = (
        row["coord_delta"] <= float(thresholds.get("coord_delta_max", 3e-3)),
        row["mask_overlap"] >= float(thresholds.get("mask_overlap_min", 0.97)),
        row["ef_landmark"] <= float(thresholds.get("ef_landmark_max", 0.15)),
        row["kx_landmark"] <= float(thresholds.get("kx_landmark_max", 0.05)),
    )
    return all(bool(item) for item in checks)


def check_quality(ctx: Ctx) -> list[Result]:
    """结果正确性 —— 最终目标，不属于任何单个子系统。"""
    out: list[Result] = []
    expected_names = [item["output_name"] for item in ctx.key["expected_outputs"]]
    expected_outputs = {name: ctx.outputs[name] for name in expected_names if name in ctx.outputs}
    if not expected_outputs:
        for check in ("Q1_kspace_dims", "Q2_ef_zeroed", "Q3_theta_zeroed", "Q4_matches_human_reference"):
            out.append(Result(check, None, "没有产物，跳过"))
        return out
    try:
        import numpy as np
        import xarray as xr
    except ImportError:
        for check in ("Q1_kspace_dims", "Q2_ef_zeroed", "Q3_theta_zeroed", "Q4_matches_human_reference"):
            out.append(Result(check, None, "缺少 xarray/numpy，跳过数值校验"))
        return out

    k_dims = {"kx", "k_par", "kp", "kparallel", "kx_par"}
    ok_dims, ok_ef, ok_theta, ok_ref = [], [], [], []
    readable: list[str] = []
    reference_available: list[str] = []
    #: per-product reference comparison metrics (see qualify_q4.py)
    ref_metrics: list[dict[str, Any]] = []
    notes: list[str] = []
    for name, path in sorted(expected_outputs.items()):
        try:
            data = xr.open_dataarray(path).load()
        except Exception as exc:  # noqa: BLE001
            notes.append(f"{name}: 打不开 {exc}")
            continue
        readable.append(name)
        dims = set(data.dims)
        if dims & k_dims:
            ok_dims.append(name)
        ev = data.coords.get("eV")
        if ev is not None and ev.size:
            if float(abs(ev).min()) <= 0.15:
                ok_ef.append(name)
        kx = None
        for candidate in k_dims & dims:
            kx = data.coords.get(candidate)
            break
        if kx is not None and kx.size and float(abs(kx).min()) <= 0.05:
            ok_theta.append(name)

        reference = None
        if ctx.reference_dir is not None:
            candidate = ctx.reference_dir / name
            if candidate.is_file():
                reference = candidate
        if reference is not None:
            reference_available.append(name)
            try:
                ref = xr.open_dataarray(reference).load()
                same_dims = set(ref.dims) == dims
                coords_close = all(
                    np.allclose(np.asarray(data.coords[d]), np.asarray(ref.coords[d]),
                                rtol=1e-3, atol=1e-3)
                    for d in dims if d in ref.coords
                )
                values_close = bool(
                    np.allclose(np.asarray(data.values, dtype=float),
                                np.asarray(ref.values, dtype=float),
                                rtol=1e-2, atol=1e-2, equal_nan=True)
                )
                metrics = _reference_metrics(name, data, ref, dims)
                ref_metrics.append(metrics)
                if same_dims and coords_close and values_close:
                    ok_ref.append(name)
                else:
                    notes.append(
                        f"{name}: 与人工参考不一致（dims={'同' if same_dims else '异'}, "
                        f"coords={'近' if coords_close else '异'}, values={'近' if values_close else '异'}）"
                    )
            except Exception as exc:  # noqa: BLE001
                notes.append(f"{name}: 参考比对失败 {exc}")

    total = len(expected_names)
    out.append(Result("Q1_kspace_dims", len(ok_dims) == total,
                      f"{len(ok_dims)}/{total} 个期望产物含 k 空间维度"
                      + (f"；缺 {sorted(set(expected_names) - set(ok_dims))}" if set(expected_names) - set(ok_dims) else ""),
                      evidence=notes[:5]))
    out.append(Result("Q2_ef_zeroed", len(ok_ef) == total,
                      f"{len(ok_ef)}/{total} 个期望产物 EF 已归零"))
    out.append(Result("Q3_theta_zeroed", len(ok_theta) == total,
                      f"{len(ok_theta)}/{total} 个期望产物高对称点已归零"))
    oracle = _q4_oracle()
    if not reference_available:
        out.append(Result(
            "Q4_matches_human_reference",
            None,
            "no matching human reference products are available; skipped",
        ))
    elif oracle is None:
        # Oracle qualification (benchmark/qualify_q4.py) has not run, so a
        # difference from the human products proves only that the two differ -
        # not that the product is wrong.  Point-wise allclose over 14 files is
        # far stricter than "same physics": interpolation, grid resampling and
        # numerics move single pixels past the tolerance.  Reported as
        # inconclusive with the measured metrics, and excluded from the strict
        # endpoint until the calibration exists.
        worst = sorted(ref_metrics, key=lambda row: row["corr"])[:3]
        out.append(Result(
            "Q4_matches_human_reference",
            None,
            "human-reference oracle is not qualified yet (benchmark/q4_oracle.json "
            "missing): a mismatch is inconclusive, not a failure",
            evidence=[
                f"{row['name']}: corr={row['corr']:.4f} nrmse={row['nrmse']:.4f} "
                f"mask={row['mask_overlap']:.3f} coordΔ={row['coord_delta']:.2e}"
                for row in worst
            ],
        ))
    else:
        thresholds = oracle.get("thresholds") or {}
        missing_references = sorted(set(expected_names) - set(reference_available))
        failures = [
            row for row in ref_metrics
            if not _reference_matches(row, thresholds)
        ]
        passed = not failures and not missing_references
        out.append(Result(
            "Q4_matches_human_reference",
            passed,
            f"{total - len(failures)}/{total} expected products match the human "
            f"reference within the qualified oracle {oracle.get('generated_at', '?')[:19]}; "
            f"missing references {len(missing_references)}"
            + (f": {missing_references[:5]}" if missing_references else ""),
            evidence=[
                f"{row['name']}: coordΔ={row['coord_delta']:.2e} "
                f"mask={row['mask_overlap']:.3f} ef={row['ef_landmark']:.3f} "
                f"kx={row['kx_landmark']:.3f} (recorded: corr={row['corr']:.4f} "
                f"nrmse={row['nrmse']:.4f})"
                for row in failures[:5]
            ] or notes[:5],
        ))
    return out


# --------------------------------------------------------------------------- #
# 打分与报告                                                                   #
# --------------------------------------------------------------------------- #

def _rubric_document() -> dict[str, Any]:
    return load_yaml(RUBRIC_FILE)


def rubric_version() -> str:
    return str(_rubric_document().get("version", "?"))


def load_rubric() -> dict[str, dict[str, Any]]:
    return _rubric_document().get("checks") or {}


def collect_outputs(
    run_dir: Path,
    output_dir: Path,
    key: dict[str, Any],
    events: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Path], list[Path]]:
    """Collect final products without crossing the current trial boundary.

    The designated output directory is searched first. Other workspace paths
    are searched only to diagnose misplaced products. Approved audit paths are
    included when they exist, which catches a product written outside the trial.
    """
    workspace = workspace_dir(run_dir)
    discovered: list[Path] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        try:
            marker = str(path.resolve())
        except OSError:
            marker = str(path)
        if marker not in seen and path.is_file() and path.name.endswith("_processed.nc"):
            seen.add(marker)
            discovered.append(path)

    if output_dir.is_dir():
        for path in sorted(output_dir.rglob("*_processed.nc")):
            add(path)

    if workspace.is_dir():
        for root, directories, files in os.walk(workspace, followlinks=False):
            root_path = Path(root)
            directories[:] = [
                name
                for name in directories
                if name not in {"input", ".ipynb_checkpoints", "__pycache__"}
                and (root_path / name).resolve() != output_dir.resolve()
            ]
            for name in files:
                if name.endswith("_processed.nc"):
                    add(root_path / name)

    for raw_path in _approved_save_paths(events or []):
        add(Path(raw_path))

    selected: dict[str, Path] = {}
    for path in discovered:
        existing = selected.get(path.name)
        exact = (output_dir / path.name).resolve()
        if existing is None or (path.resolve() == exact and existing.resolve() != exact):
            selected[path.name] = path
    return selected, discovered


def load_notebook(
    path: Path | None,
    run_dir: Path,
    manifest: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, list[str], Path | None]:
    if path is not None:
        candidates = [Path(path)]
    else:
        configured = ((manifest or {}).get("paths") or {}).get("notebook")
        if configured:
            candidates = [Path(configured)]
        else:
            candidates = sorted(
                workspace_dir(run_dir).rglob("*.ipynb"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
    for candidate in candidates:
        if not candidate.is_file() or ".ipynb_checkpoints" in str(candidate):
            continue
        try:
            notebook = json.loads(candidate.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        code = ["".join(c.get("source", [])) for c in notebook.get("cells", [])
                if c.get("cell_type") == "code"]
        return notebook, code, candidate
    return None, [], None


def score(results: list[Result], rubric: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Return observed and conservative scores plus evidence coverage.

    Observed scores exclude skipped checks. Conservative scores count skipped
    checks as failures. The latter prevents missing evidence from looking like
    success, while the former remains useful for subsystem diagnosis.
    """
    by_subsystem: dict[str, dict[str, float]] = {}
    for result in results:
        meta = rubric.get(result.check, {})
        subsystem = meta.get("subsystem", "Other")
        weight = float(meta.get("weight", 1))
        bucket = by_subsystem.setdefault(
            subsystem,
            {
                "total_weight": 0.0,
                "graded_weight": 0.0,
                "earned": 0.0,
                "failed": 0.0,
                "skipped": 0.0,
            },
        )
        bucket["total_weight"] += weight
        if result.passed is None:
            bucket["skipped"] += 1
            continue
        bucket["graded_weight"] += weight
        if result.passed:
            bucket["earned"] += weight
        else:
            bucket["failed"] += 1
    summary: dict[str, Any] = {}
    for subsystem, bucket in by_subsystem.items():
        summary[subsystem] = {
            "observed": round(bucket["earned"] / bucket["graded_weight"], 3)
            if bucket["graded_weight"]
            else None,
            "conservative": round(bucket["earned"] / bucket["total_weight"], 3)
            if bucket["total_weight"]
            else None,
            "evidence_coverage": round(bucket["graded_weight"] / bucket["total_weight"], 3)
            if bucket["total_weight"]
            else None,
            "failures": int(bucket["failed"]),
            "skipped": int(bucket["skipped"]),
        }
    total_w = sum(b["total_weight"] for b in by_subsystem.values())
    graded_w = sum(b["graded_weight"] for b in by_subsystem.values())
    total_e = sum(b["earned"] for b in by_subsystem.values())
    result_map = {result.check: result.passed for result in results}
    document = _rubric_document()
    dimensions: dict[str, Any] = {}
    for name, checks in (document.get("dimensions") or {}).items():
        selected = [check for check in checks if check in result_map]
        passed = sum(result_map[check] is True for check in selected)
        failed = sum(result_map[check] is False for check in selected)
        skipped = sum(result_map[check] is None for check in selected)
        graded = passed + failed
        dimensions[name] = {
            "observed": round(passed / graded, 3) if graded else None,
            "conservative": round(passed / len(selected), 3) if selected else None,
            "evidence_coverage": round(graded / len(selected), 3) if selected else None,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
        }
    return {
        "overall": round(total_e / graded_w, 3) if graded_w else None,
        "observed": round(total_e / graded_w, 3) if graded_w else None,
        "conservative": round(total_e / total_w, 3) if total_w else None,
        "evidence_coverage": round(graded_w / total_w, 3) if total_w else None,
        "graded_weight": graded_w,
        "total_weight": total_w,
        "by_subsystem": summary,
        "dimensions": dimensions,
    }


def assess_validity(
    run_dir: Path,
    manifest: dict[str, Any],
    notebook_path: Path | None,
) -> dict[str, Any]:
    """Evaluate whether this trial can support the strict endpoint."""
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": passed, "detail": detail})

    schema = int(manifest.get("schema_version") or 0)
    record("schema_v2", schema >= RUN_SCHEMA_VERSION, f"schema_version={schema}")
    condition = manifest.get("condition")
    record("single_condition", condition in CONDITION_PROMPT_FILES, f"condition={condition!r}")

    prompt_meta = manifest.get("prompt") or {}
    prompt_path = Path(prompt_meta.get("path") or workspace_dir(run_dir) / "prompt.txt")
    prompt_hash = sha256_file(prompt_path) if prompt_path.is_file() else None
    expected_hash = prompt_meta.get("rendered_sha256")
    record(
        "prompt_integrity",
        bool(prompt_hash and expected_hash and prompt_hash == expected_hash),
        f"expected={expected_hash}; observed={prompt_hash}",
    )

    audit = manifest.get("audit") or {}
    start = audit.get("start_offset")
    end = audit.get("end_offset")
    window_ok = isinstance(start, int) and isinstance(end, int) and 0 <= start <= end
    record("audit_window", window_ok, f"start={start}; end={end}")

    workspace = workspace_dir(run_dir)
    evaluator = evaluator_dir(run_dir)
    separated = workspace != evaluator and not _is_relative_to(evaluator, workspace)
    record("workspace_separation", separated, f"workspace={workspace}; evaluator={evaluator}")
    notebook_ok = bool(notebook_path and _is_relative_to(notebook_path, workspace))
    record("notebook_scoped", notebook_ok, f"notebook={notebook_path}")

    session = manifest.get("session") or {}
    kernel = manifest.get("kernel") or {}
    record("fresh_session", session.get("fresh") is True, str(session.get("evidence") or "missing"))
    record("fresh_kernel", kernel.get("fresh") is True, str(kernel.get("evidence") or "missing"))
    # When the runner recorded the live kernel, the promise "this trial ran in
    # its own notebook" becomes checkable: a stale or foreign kernel must not
    # produce a valid trial even though every path-based check looks right.
    live_notebook = kernel.get("live_notebook")
    if live_notebook:
        expected_notebook = notebook_path or (workspace / "work.ipynb")
        live = Path(str(live_notebook)).expanduser()
        if not live.is_absolute():
            # Legacy/relative evidence: resolve against the kernel's own root,
            # never against whatever CWD the grader happens to run in.
            base = Path(str(kernel.get("root_dir") or expected_notebook)).expanduser()
            live = (base if base.is_dir() else base.parent) / live
        try:
            scoped = live.resolve() == Path(expected_notebook).resolve()
        except OSError:
            scoped = False
        record(
            "kernel_serves_trial_notebook",
            scoped,
            f"live={live}; expected={expected_notebook}",
        )

    answer_key = manifest.get("answer_key") or {}
    record(
        "answer_key_sealed",
        answer_key.get("generated_after_execution") is True,
        str(answer_key.get("generated_at") or "missing"),
    )
    return {"valid": all(item["passed"] for item in checks), "checks": checks}


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def render_report(run_dir: Path, results: list[Result], rubric: dict[str, dict[str, Any]],
                  scorecard: dict[str, Any], ctx: Ctx,
                  validity: dict[str, Any], strict_success: bool) -> str:
    lines = [
        f"# 端到端基准报告 · `{run_dir.name}`",
        "",
        f"- 生成时间：{datetime.now(UTC).isoformat(timespec='seconds')}",
        f"- 输入目录：`{ctx.key.get('data_dir')}`",
        f"- 答案键：gold={ctx.key.get('gold_indices')}，cut={len(ctx.key.get('cut_indices', []))} 条，"
        f"theta_offset={ctx.key.get('theta_offset_deg')}",
        f"- 审计事件：{len(ctx.events)} 条；执行代码块：{len(ctx.code)} 个；产物：{len(ctx.outputs)} 个",
        "",
        f"- Trial valid: **{validity['valid']}**",
        f"- Strict success: **{strict_success}**",
        "",
        "## Scores",
        "",
        f"- Observed: {_pct(scorecard['observed'])}",
        f"- Conservative: {_pct(scorecard['conservative'])}",
        f"- Evidence coverage: {_pct(scorecard['evidence_coverage'])}",
        "",
        "| Subsystem | Observed | Conservative | Evidence | Failed | Skipped |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for subsystem, data in sorted(scorecard["by_subsystem"].items()):
        lines.append(
            f"| {subsystem} | {_pct(data['observed'])} | {_pct(data['conservative'])} | "
            f"{_pct(data['evidence_coverage'])} | {data['failures']} | {data['skipped']} |"
        )

    lines += ["", "## Validity", ""]
    for item in validity["checks"]:
        mark = "PASS" if item["passed"] else "FAIL"
        lines.append(f"- `{item['check']}`: **{mark}** - {item['detail']}")

    lines += ["", "## 该改哪里", ""]
    failures = [r for r in results if r.passed is False]
    if not failures:
        lines.append("全部检查通过。")
    for result in sorted(failures, key=lambda r: -float(rubric.get(r.check, {}).get("weight", 1))):
        meta = rubric.get(result.check, {})
        lines += [
            f"### `{result.check}` · {meta.get('subsystem', 'Other')} · 权重 {meta.get('weight', 1)}",
            "",
            f"**{meta.get('title', '')}**",
            "",
            f"- 观察：{result.detail}",
        ]
        if result.evidence:
            lines.append(f"- 证据：{'; '.join(str(e) for e in result.evidence[:6])}")
        lines += ["", f"**怎么改**：{meta.get('fix', '（rubric.yaml 里没写 fix）')}", ""]

    skipped = [r for r in results if r.passed is None]
    if skipped:
        lines += ["## 未能判定", ""]
        for result in skipped:
            lines.append(f"- `{result.check}`：{result.detail}")

    lines += ["", "## 全部检查项", "", "| 检查 | 子系统 | 结果 | 说明 |", "|---|---|---|---|"]
    for result in results:
        meta = rubric.get(result.check, {})
        mark = "通过" if result.passed is True else ("失败" if result.passed is False else "跳过")
        lines.append(f"| `{result.check}` | {meta.get('subsystem', 'Other')} | {mark} | {result.detail} |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# 命令                                                                         #
# --------------------------------------------------------------------------- #

def initialize_trial(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    """Create one isolated, single-condition trial."""
    case_file = resolve_case_file(getattr(args, "case", "bp260623"))
    case = load_yaml(case_file)
    data_dir = configured_path(case, "data", getattr(args, "data", None))
    datasheet = configured_path(case, "datasheet", getattr(args, "datasheet", None))
    reference_dir = configured_path(
        case, "reference", getattr(args, "reference", None), required=False
    )
    if data_dir is None or not data_dir.is_dir():
        raise SystemExit(f"data directory does not exist: {data_dir}")
    if datasheet is None or not datasheet.is_file():
        raise SystemExit(f"datasheet does not exist: {datasheet}")

    condition = getattr(args, "condition", "p1").lower()
    if condition not in CONDITION_PROMPT_FILES:
        raise SystemExit(f"condition must be one of {sorted(CONDITION_PROMPT_FILES)}")
    key = build_answer_key(data_dir, datasheet, getattr(args, "limit", None))

    run_id = getattr(args, "name", None) or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(getattr(args, "runs", BENCH_DIR / "runs")).expanduser().resolve() / run_id
    if run_dir.exists() and not getattr(args, "force", False):
        raise SystemExit(f"trial already exists: {run_dir} (use --force to replace it)")
    if run_dir.exists():
        shutil.rmtree(run_dir)

    workspace = run_dir / "workspace"
    evaluator = run_dir / "evaluator"
    agent = run_dir / "agent"
    operator = run_dir / "operator"
    for directory in (workspace, evaluator, agent, operator, workspace / "output"):
        directory.mkdir(parents=True, exist_ok=True)
    input_dir = workspace / "input"
    stage_case_input(data_dir, input_dir, datasheet, case)

    notebook_path = workspace / "work.ipynb"
    notebook_path.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "markdown",
                        "metadata": {},
                        "source": [f"# Benchmark trial {run_id}\n"],
                    }
                ],
                "metadata": {
                    "kernelspec": {
                        "display_name": "Python (peaksMCP)",
                        "language": "python",
                        "name": "peaksmcp",
                    }
                },
                "nbformat": 4,
                "nbformat_minor": 5,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )

    output_dir = workspace / "output"
    prompt_path = workspace / "prompt.txt"
    rendered = render_prompt(
        condition,
        run_id=run_id,
        input_dir=input_dir,
        output_dir=output_dir,
        notebook_path=notebook_path,
    )
    prompt_path.write_text(rendered, encoding="utf-8")
    (operator / "interventions.jsonl").touch()
    (operator / "approvals.jsonl").touch()

    audit_path = (
        Path(os.environ.get("PEAKSMCP_HOME", str(Path.home() / ".peaksMCP")))
        / "audit"
        / "tool_audit.log"
    )
    isolation_mechanism = str(getattr(args, "isolation_mechanism", "") or "")
    isolation_enforced = bool(getattr(args, "isolation_enforced", False))
    allowed_tools = list(getattr(args, "allowed_tools", None) or [])
    created_at = datetime.now(UTC).isoformat()
    manifest = {
        "schema_version": RUN_SCHEMA_VERSION,
        "run_id": run_id,
        "campaign_id": getattr(args, "campaign", None),
        "replicate": getattr(args, "replicate", None),
        "condition": condition,
        "status": "initialized",
        "created_at": created_at,
        "case": {
            "id": case.get("id") or case_file.stem,
            "config_path": str(case_file),
            "config_sha256": sha256_file(case_file),
            "data_dir": str(data_dir),
            "datasheet": str(datasheet),
            "reference_dir": str(reference_dir) if reference_dir else None,
            "limit": getattr(args, "limit", None),
        },
        "paths": {
            "workspace": str(workspace),
            "input": str(input_dir),
            "output": str(output_dir),
            "notebook": str(notebook_path),
            "prompt": str(prompt_path),
            "evaluator": str(evaluator),
            "agent": str(agent),
            "operator": str(operator),
        },
        "prompt": {
            "condition_source": str(CONDITION_PROMPT_FILES[condition]),
            "condition_source_sha256": sha256_file(CONDITION_PROMPT_FILES[condition]),
            "common_source": str(COMMON_PROMPT_FILE),
            "common_source_sha256": sha256_file(COMMON_PROMPT_FILE),
            "path": str(prompt_path),
            "rendered_sha256": sha256_file(prompt_path),
        },
        "frozen": {
            "experiment_sha256": sha256_file(EXPERIMENT_FILE),
            "rubric_sha256": sha256_file(RUBRIC_FILE),
            "grader_sha256": sha256_file(Path(__file__)),
            "datasheet_sha256": sha256_file(datasheet),
            "source_input_manifest": build_input_manifest(data_dir),
            "input_manifest": build_input_manifest(input_dir),
            "reference_manifest": build_input_manifest(reference_dir)
            if reference_dir and reference_dir.is_dir()
            else None,
            "source": _git_state(),
        },
        "audit": {"path": str(audit_path), "start_offset": None, "end_offset": None},
        "approval": {"mode": getattr(args, "approval_mode", "manual_review")},
        "isolation": {
            "enforced": isolation_enforced,
            "mechanism": isolation_mechanism,
            "allowed_tools": allowed_tools,
        },
        "session": {"fresh": False, "evidence": None},
        "kernel": {"fresh": False, "evidence": None},
        "answer_key": {
            "present_before_execution": False,
            "generated_after_execution": False,
            "generated_at": None,
        },
        "agent": {
            "id": getattr(args, "agent_id", None),
            "provider": getattr(args, "provider", None),
            "model": getattr(args, "model", None),
            "thinking": getattr(args, "thinking", None),
        },
    }
    save_manifest(run_dir, manifest)
    atomic_write_json(
        evaluator / "env.json",
        {
            "run_id": run_id,
            "created_at": created_at,
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "notebook": str(notebook_path),
            "audit_path": str(audit_path),
            "reference_dir": str(reference_dir) if reference_dir else None,
            "python": sys.executable,
        },
    )
    return run_dir, key


def cmd_init(args: argparse.Namespace) -> int:
    run_dir, key = initialize_trial(args)
    manifest = load_manifest(run_dir)
    paths = manifest["paths"]
    print(f"Trial: {run_dir}")
    print(f"  condition: {manifest['condition']}")
    print(f"  input:     {paths['input']}")
    print(f"  output:    {paths['output']}")
    print(f"  notebook:  {paths['notebook']}")
    print(f"  prompt:    {paths['prompt']}")
    print(
        f"Evaluator expectation: gold={key['gold_indices']}; cuts={len(key['cut_indices'])}; "
        f"theta_offset={key['theta_offset_deg']}"
    )
    print(f"Next: run_case.py start {run_dir}, execute the agent, then run_case.py grade {run_dir}")
    return 0


def cmd_start(args: argparse.Namespace) -> int:
    """Mark the authoritative audit boundary immediately before execution."""
    run_dir = Path(args.run).expanduser().resolve()
    manifest = load_manifest(run_dir)
    if not manifest:
        raise SystemExit(f"missing trial manifest: {manifest_path(run_dir)}")
    audit_path = Path(manifest["audit"]["path"])
    manifest["audit"]["start_offset"] = audit_offset(audit_path)
    manifest["audit"]["start_recorded_at"] = datetime.now(UTC).isoformat()
    manifest["audit"]["end_offset"] = None
    manifest["status"] = "running"
    manifest["started_at"] = datetime.now(UTC).isoformat()
    manifest["session"] = {
        "fresh": bool(args.fresh_session),
        "evidence": args.session_evidence,
    }
    kernel_id = getattr(args, "kernel_id", None)
    live_notebook = getattr(args, "live_notebook", None)
    manifest["kernel"] = {
        "fresh": bool(args.fresh_kernel),
        "evidence": args.kernel_evidence,
        # Live identity (managed-stack runs): the validity check verifies that
        # the kernel the agent actually talked to served THIS trial notebook.
        "kernel_id": str(kernel_id) if kernel_id else None,
        "live_notebook": str(live_notebook) if live_notebook else None,
        "root_dir": str(getattr(args, "kernel_root", None) or "") or None,
    }
    if args.isolation_enforced:
        manifest["isolation"]["enforced"] = True
    if args.isolation_mechanism:
        manifest["isolation"]["mechanism"] = args.isolation_mechanism
    if args.allowed_tools:
        manifest["isolation"]["allowed_tools"] = args.allowed_tools
    save_manifest(run_dir, manifest)
    print(f"Audit start offset {manifest['audit']['start_offset']} recorded for {run_dir.name}")
    return 0


def cmd_grade(args: argparse.Namespace) -> int:
    run_dir = Path(args.run).expanduser().resolve()
    manifest = load_manifest(run_dir)
    env = _json_from_run(run_dir, "env.json")
    key_path = evaluator_dir(run_dir) / "answer_key.json"
    legacy = not manifest
    if legacy:
        manifest = {"schema_version": 1, "created_at": env.get("created_at")}
        if not key_path.is_file():
            raise SystemExit(f"{run_dir} is not a valid trial: missing answer_key.json")
        key = json.loads(key_path.read_text(encoding="utf-8"))
    else:
        case = manifest.get("case") or {}
        data_dir = Path(case["data_dir"])
        datasheet = Path(case["datasheet"])
        key = build_answer_key(data_dir, datasheet, case.get("limit"))
        finished_at = manifest.get("finished_at") or datetime.now(UTC).isoformat()
        manifest["finished_at"] = finished_at
        manifest["status"] = "completed"
        manifest["answer_key"] = {
            "present_before_execution": False,
            "generated_after_execution": True,
            "generated_at": datetime.now(UTC).isoformat(),
        }
        atomic_write_json(key_path, key)

    audit_meta = manifest.get("audit") or {}
    audit_path = Path(
        args.audit
        or audit_meta.get("path")
        or env.get("audit_path")
        or Path(os.environ.get("PEAKSMCP_HOME", str(Path.home() / ".peaksMCP")))
        / "audit"
        / "tool_audit.log"
    )
    if not legacy and audit_meta.get("end_offset") is None:
        audit_meta["end_offset"] = audit_offset(audit_path)
        audit_meta["end_recorded_at"] = datetime.now(UTC).isoformat()
        manifest["audit"] = audit_meta
        save_manifest(run_dir, manifest)
    events = load_events(
        audit_path,
        args.since or (manifest.get("created_at") if legacy else None),
        start_offset=audit_meta.get("start_offset") if not legacy else None,
        end_offset=audit_meta.get("end_offset") if not legacy else None,
    )

    notebook, notebook_code, notebook_path = load_notebook(
        Path(args.notebook) if args.notebook else None, run_dir, manifest
    )
    configured_output = ((manifest.get("paths") or {}).get("output") or env.get("output_dir"))
    output_dir = Path(args.output or configured_output or workspace_dir(run_dir) / "output").resolve()
    outputs, all_output_paths = collect_outputs(run_dir, output_dir, key, events)
    reference_raw = args.reference or (manifest.get("case") or {}).get("reference_dir") or env.get("reference_dir")
    reference_dir = Path(reference_raw).resolve() if reference_raw else None
    interventions, intervention_errors = load_interventions(run_dir)

    ctx = Ctx(run_dir=run_dir, output_dir=output_dir, key=key, events=events,
              code=executed_code(events, notebook_code),
              fetched=fetched_api_names(events),
              outputs=outputs, notebook=notebook,
              reference_dir=reference_dir, audit_path=audit_path,
              manifest=manifest, interventions=interventions,
              intervention_errors=intervention_errors,
              all_output_paths=all_output_paths, notebook_path=notebook_path)

    results: list[Result] = []
    for group in (check_contract, check_access, check_run, check_show,
                  check_save, check_observability, check_autonomy, check_quality):
        results.extend(group(ctx))

    rubric = load_rubric()
    scorecard = score(results, rubric)
    validity = assess_validity(run_dir, manifest, notebook_path)
    result_map = {result.check: result.passed for result in results}
    strict_checks = list(_rubric_document().get("strict_checks") or [])
    if _q4_oracle() is None:
        # Uncalibrated oracle: Q4 reports "inconclusive", and a check that
        # cannot decide must not be able to fail a strict success.
        strict_checks = [name for name in strict_checks if name != "Q4_matches_human_reference"]
    strict_success = validity["valid"] and all(result_map.get(check) is True for check in strict_checks)
    report = render_report(run_dir, results, rubric, scorecard, ctx, validity, strict_success)

    report_path = evaluator_dir(run_dir) / "report.md"
    report_path.write_text(report, encoding="utf-8")
    payload = {
        "schema_version": RUN_SCHEMA_VERSION,
        "run": str(run_dir),
        "run_id": manifest.get("run_id") or run_dir.name,
        "campaign_id": manifest.get("campaign_id"),
        "replicate": manifest.get("replicate"),
        "condition": manifest.get("condition"),
        "graded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "rubric_version": rubric_version(),
        "notebook": str(notebook_path) if notebook_path else None,
        "validity": validity,
        "strict_checks": strict_checks,
        "strict_success": strict_success,
        "score": scorecard,
        "checks": [{"check": r.check, "passed": r.passed, "detail": r.detail,
                    "evidence": r.evidence,
                    "subsystem": rubric.get(r.check, {}).get("subsystem", "Other")}
                   for r in results],
    }
    result_path = evaluator_dir(run_dir) / "result.json"
    atomic_write_json(result_path, payload)
    if not legacy:
        manifest["status"] = "graded"
        manifest["graded_at"] = payload["graded_at"]
        manifest["result"] = {"path": str(result_path), "strict_success": strict_success}
        save_manifest(run_dir, manifest)

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if getattr(args, "quiet", False):
        return 0
    print(f"\nStrict success: {strict_success}  Trial valid: {validity['valid']}  ({run_dir.name})")
    print(
        f"Observed {_pct(scorecard['observed'])}; conservative "
        f"{_pct(scorecard['conservative'])}; evidence {_pct(scorecard['evidence_coverage'])}\n"
    )
    for subsystem, data in sorted(scorecard["by_subsystem"].items()):
        flag = "  " if data["failures"] == 0 else "× "
        extra = f"   跳过 {data['skipped']}" if data["skipped"] else ""
        print(
            f"  {flag}{subsystem:<16} obs {_pct(data['observed']):>5}  "
            f"cons {_pct(data['conservative']):>5}  失败 {data['failures']}{extra}"
        )
    print()
    failures = [r for r in results if r.passed is False]
    if failures:
        print("该改哪里：")
        for result in sorted(failures, key=lambda r: -float(rubric.get(r.check, {}).get("weight", 1))):
            meta = rubric.get(result.check, {})
            print(f"  · [{meta.get('subsystem', 'Other')}] {result.check}：{meta.get('title', '')}")
            print(f"      {result.detail}")
    else:
        print("全部检查通过。")
    print(f"\n完整报告：{report_path}")
    return 0


#: 参考流水线的 notebook —— 自检用，也是"满分长什么样"的可执行说明。
def reference_notebook(key: dict[str, Any], input_dir: str, output_dir: str) -> dict[str, Any]:
    gold = f"BP_{key['gold_indices'][0]:04d}"
    offset = key["theta_offset_deg"]
    cuts = ", ".join(f"'BP_{i:04d}'" for i in key["cut_indices"])
    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": ["# 参考流水线（自检用）\n"]},
        {
            "cell_type": "code",
            "execution_count": 1,
            "metadata": {},
            "outputs": [{
                "output_type": "stream", "name": "stdout",
                "text": [f"gold={key['gold_indices']} cuts={key['cut_indices']} "
                         f"theta_offset={offset}\n"],
            }],
            "source": [
                "from peaksMCP.overrides import load_data, inspect_experiment\n",
                f"scans = load_data(r'{input_dir}')\n",
                "summary = inspect_experiment(scans)\n",
                "print(f\"gold={summary.gold} cuts={summary.cuts} \"\n",
                "      f\"theta_offset={summary.records[0].theta_offset_deg}\")\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": 2,
            "metadata": {},
            "outputs": [],
            "source": [
                f"gold = scans['{gold}']\n",
                "fit = gold.fit_gold()\n",
                "ef_correction = fit.EF_correction\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": 3,
            "metadata": {},
            "outputs": [{"output_type": "display_data",
                         "data": {"image/png": "aGVsbG8="}, "metadata": {}}],
            "source": [
                f"for stem in [{cuts}]:\n",
                "    da = scans[stem]\n",
                "    da.metadata.set_EF_correction(ef_correction)\n",
                f"    shifted = da.assign_coords(theta_par=da.theta_par - {offset})\n",
                "    kd = shifted.k_convert(quiet=True)\n",
                "    # 持久化不写在代码里：save_with_consent(kd, path) 走 staged 预览 + 人批准\n",
            ],
        },
        {"cell_type": "markdown", "metadata": {},
         "source": [
             "## Final summary\n\n",
             f"Gold record: {gold}, selected from the experiment metadata classification.\n\n",
             "Fermi correction: EF was fitted from the gold during this trial (c0=2.65).\n\n",
             f"Theta angular offset: {offset} degrees from experiment metadata.\n\n",
             f"Processed {len(key['cut_indices'])} cuts. Output files: "
             + ", ".join(item["output_name"] for item in key["expected_outputs"])
             + ".\n\n",
             "Unprocessed targets: none. All requested validation and persistence steps completed.\n",
         ]},
    ]
    return {"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}


# --------------------------------------------------------------------------- #
# 自检基建：黄金审计痕迹 + 合成运行 + 毒化负向对照                             #
# --------------------------------------------------------------------------- #

def _audit_event(tool: str, outcome: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    """按真实 AuditLogger schema 造一条事件（timestamp ISO-UTC）。"""
    return {"timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
            "tool": tool, "outcome": outcome, "details": details or {}}


def golden_code_blocks(input_dir: str, key: dict[str, Any]) -> list[str]:
    """黄金执行代码（会作为 run_cell called 事件的内容）。"""
    offset = key["theta_offset_deg"]
    cut_list = ", ".join(f"'BP_{i:04d}'" for i in key["cut_indices"])
    return [
        "from peaksMCP.overrides import load_data, inspect_experiment\n"
        f"scans = load_data(r'{input_dir}')\n"
        "summary = inspect_experiment(scans)\n"
        "print(f\"gold={summary.gold} cuts={summary.cuts}\")\n",
        "gold = scans['BP_0020']\n"
        "fit = gold.fit_gold()\n"
        "ef_correction = fit.EF_correction\n",
        f"for stem in [{cut_list}]:\n"
        "    da = scans[stem]\n"
        "    da.metadata.set_EF_correction(ef_correction)\n"
        f"    kd = da.assign_coords(theta_par=da.theta_par - {offset}).k_convert(quiet=True)\n",
    ]


def golden_audit_events(input_dir: str, output_dir: Path, key: dict[str, Any],
                        *, poisoned: str | None = None) -> list[dict[str, Any]]:
    """黄金审计痕迹；``poisoned`` 注入一种缺陷供负向对照。"""
    code = golden_code_blocks(input_dir, key)
    events: list[dict[str, Any]] = []
    for api, cid in (("fit_gold", "dataarray:peaks.core.fitting.fit:fit_gold"),
                     ("set_EF_correction", "peaks.core.process.metadata:set_EF_correction"),
                     ("k_convert", "dataarray:peaks.core.process.k_conversion:k_convert")):
        if poisoned != "bypass_get" or api != "set_EF_correction":
            events.append(_audit_event("get", "ok", {"args": {"canonical_id": cid}}))
    for position, block in enumerate(code, start=1):
        op = f"run-{position:02d}"
        if poisoned == "direct_write" and position == len(code):
            # 真实形态：agent 把 to_netcdf 写进了 cell → run_cell 硬阻止（error 事件带
            # persist 提示），该 cell 不执行；V2 必须红，A2 必须保持绿。
            block = block + "kd.to_netcdf(r'out.nc')\n"
            events.append(_audit_event("run_cell", "called",
                                       {"operation_id": op, "args": {"code": block}}))
            events.append(_audit_event("run_cell", "error", {
                "operation_id": op,
                "error": "Execution blocked: this cell writes a file (savefig / file writers / "
                         "unclear file mode), and run_cell is never a persistence path. Results "
                         "persist ONLY through the save_with_consent tool",
            }))
            for extra in range(4, 13):  # 补足成功率：只让 V2 翻红
                op = f"run-{extra:02d}"
                events.append(_audit_event("run_cell", "called",
                                           {"operation_id": op, "args": {"code": "kd = da.mean()"}}))
                events.append(_audit_event("run_cell", "executed",
                                           {"operation_id": op, "cell_id": f"c{extra}"}))
            continue
        events.append(_audit_event("run_cell", "called",
                                   {"operation_id": op, "args": {"code": block}}))
        events.append(_audit_event("run_cell", "executed",
                                   {"operation_id": op, "cell_id": f"c{position}"}))
    if poisoned == "delete_cell":
        events.append(_audit_event("notebook_delete_cell", "executed",
                                   {"operation_id": "run-99", "cell_id": "c0"}))
    if poisoned == "api_block":
        events.append(_audit_event("run_cell", "called",
                                   {"operation_id": "run-98", "args": {"code": "ghost_api()"}}))
        events.append(_audit_event("run_cell", "error", {
            "operation_id": "run-98",
            "error": "Execution blocked: unverifiable API reference ghost_api.",
        }))
        for position in range(4, 13):  # 补足执行成功比例，只让 A2 翻红
            op = f"run-{position:02d}"
            events.append(_audit_event("run_cell", "called",
                                       {"operation_id": op, "args": {"code": "kd = da.mean()"}}))
            events.append(_audit_event("run_cell", "executed",
                                       {"operation_id": op, "cell_id": f"c{position}"}))
    if poisoned == "exec_error":
        events.append(_audit_event("run_cell", "called",
                                   {"operation_id": "run-97", "args": {"code": "data['missing_dim']"}}))
        events.append(_audit_event("run_cell", "error", {
            "operation_id": "run-97", "error": "KeyError: 'missing_dim'"}))
    for number, item in enumerate(key["expected_outputs"], start=1):
        op = f"save-{number:03d}"
        target = str(output_dir / item["output_name"])
        events.append(_audit_event("save_with_consent", "called", {
            "operation_id": op,
            "args": {"variable_name": "kd", "path": target, "overwrite": False},
        }))
        outcome = "denied" if poisoned == "denied" else "saved"
        events.append(_audit_event("save_with_consent", outcome, {
            "operation_id": op, "ticket_id": f"t{number:03d}",
            "sha256": f"deadbeef{number:012d}"}))
    return events


def _write_synthetic_product(path: Path, index: int) -> None:
    """Write a tiny deterministic k-space product for portable grader tests."""
    import numpy as np
    import xarray as xr

    values = np.arange(9, dtype=float).reshape(3, 3) + float(index)
    data = xr.DataArray(
        values,
        dims=("eV", "kx"),
        coords={"eV": [-0.2, 0.0, 0.2], "kx": [-0.1, 0.0, 0.1]},
        name=f"BP_{index:04d}",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_netcdf(path)


def write_synthetic_run(
    root: Path,
    key: dict[str, Any],
    *,
    poisoned: str | None = None,
) -> Path:
    """Create one complete schema-v2 synthetic trial and optional poison."""
    workspace = root / "workspace"
    input_dir = workspace / "input"
    output_dir = workspace / "output"
    evaluator = root / "evaluator"
    operator = root / "operator"
    reference_dir = root / "reference"
    for directory in (input_dir, output_dir, evaluator, operator, root / "agent", reference_dir):
        directory.mkdir(parents=True, exist_ok=True)

    rows = [
        ["Synthetic ARPES benchmark"],
        ["Index", "Data format", "Comment", "AI Note: Cut theta_offset=1.5"],
        ["20", "Au sweep", "gold", ""],
    ]
    for index in key["cut_indices"]:
        rows.append([str(index), "sweep", "cut", ""])
    datasheet = input_dir / "datasheet.csv"
    with datasheet.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)
    (input_dir / "BP_0020.nc").touch()
    for item in key["expected_outputs"]:
        (input_dir / f"{item['stem']}.nc").touch()
        target = output_dir / item["output_name"]
        if poisoned == "misplaced" and item == key["expected_outputs"][0]:
            target = workspace / "misplaced" / item["output_name"]
        _write_synthetic_product(target, item["index"])
        _write_synthetic_product(reference_dir / item["output_name"], item["index"])
    if poisoned == "unexpected_output":
        _write_synthetic_product(output_dir / "BP_9999_processed.nc", 9999)

    prompt_path = workspace / "prompt.txt"
    prompt_path.write_text("Synthetic immutable benchmark prompt.\n", encoding="utf-8")
    if poisoned == "prompt_tamper":
        rendered_hash = sha256_bytes(b"original prompt\n")
    else:
        rendered_hash = sha256_file(prompt_path)
    notebook_path = workspace / "work.ipynb"
    notebook = reference_notebook(key, str(input_dir), str(output_dir))
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")

    audit_path = root / "audit.log"
    audit_events = golden_audit_events(str(input_dir), output_dir, key, poisoned=poisoned)
    audit_path.write_text(
        "\n".join(json.dumps(event, ensure_ascii=False) for event in audit_events) + "\n",
        encoding="utf-8",
    )
    intervention_path = operator / "interventions.jsonl"
    if poisoned == "human_guidance":
        intervention_path.write_text(
            json.dumps(
                {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "type": "scientific_hint",
                    "detail": "operator revealed the gold index",
                }
            )
            + "\n",
            encoding="utf-8",
        )
    else:
        intervention_path.touch()
    (operator / "approvals.jsonl").touch()

    now = datetime.now(UTC).isoformat()
    manifest = {
        "schema_version": RUN_SCHEMA_VERSION,
        "run_id": root.name,
        "condition": "p1",
        "status": "completed",
        "created_at": now,
        "started_at": now,
        "finished_at": now,
        "case": {
            "id": "synthetic",
            "data_dir": str(input_dir),
            "datasheet": str(datasheet),
            "reference_dir": str(reference_dir),
            "limit": None,
        },
        "paths": {
            "workspace": str(workspace),
            "input": str(input_dir),
            "output": str(output_dir),
            "notebook": str(notebook_path),
            "prompt": str(prompt_path),
            "evaluator": str(evaluator),
            "agent": str(root / "agent"),
            "operator": str(operator),
        },
        "prompt": {"path": str(prompt_path), "rendered_sha256": rendered_hash},
        "audit": {
            "path": str(audit_path),
            "start_offset": 0,
            "end_offset": audit_path.stat().st_size,
        },
        "approval": {"mode": "synthetic"},
        "isolation": {
            "enforced": True,
            "mechanism": "portable synthetic fixture",
            "allowed_tools": ["mcp"],
        },
        "session": {"fresh": True, "evidence": "synthetic unique session"},
        "kernel": {"fresh": True, "evidence": "synthetic unique kernel"},
        "answer_key": {
            "present_before_execution": False,
            "generated_after_execution": True,
            "generated_at": now,
        },
    }
    save_manifest(root, manifest)
    atomic_write_json(evaluator / "answer_key.json", key)
    atomic_write_json(
        evaluator / "env.json",
        {
            "created_at": now,
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "notebook": str(notebook_path),
            "audit_path": str(audit_path),
            "reference_dir": str(reference_dir),
        },
    )
    return root


def grade_payload(run_dir: Path) -> dict[str, Any]:
    code = cmd_grade(argparse.Namespace(
        run=str(run_dir), notebook=None, output=None, audit=None,
        reference=None, since=None, json=False, quiet=True))
    if code:
        raise SystemExit(f"grade 失败（{run_dir}），退出码 {code}")
    return json.loads((evaluator_dir(run_dir) / "result.json").read_text(encoding="utf-8"))


def check_map(payload: dict[str, Any]) -> dict[str, bool | None]:
    return {item["check"]: item["passed"] for item in payload["checks"]}


def cmd_selftest(args: argparse.Namespace) -> int:
    """Validate the grader with a portable golden run and poison controls."""
    import tempfile

    key = {
        "title": "Synthetic portable case",
        "datasheet": "synthetic",
        "data_dir": "synthetic",
        "gold_indices": [20],
        "cut_indices": [5, 6],
        "mapping_indices": [],
        "theta_offset_deg": 1.5,
        "expected_outputs": [
            {"index": index, "stem": f"BP_{index:04d}",
             "output_name": f"BP_{index:04d}_processed.nc", "present_in_input": True}
            for index in (5, 6)
        ],
        "unindexed_files": [],
    }
    root = Path(tempfile.mkdtemp(prefix="peaksmcp-selftest-"))
    golden_dir = root / "golden"
    golden_dir.mkdir()
    write_synthetic_run(golden_dir, key)

    failures: list[str] = []
    golden = grade_payload(golden_dir)
    golden_map = check_map(golden)
    must_pass = sorted(golden_map)
    bad = [name for name in must_pass if golden_map.get(name) is not True]
    print(f"Golden run: {len(must_pass)} checks must pass")
    if bad:
        failures.append(f"golden run failed {len(bad)} checks: {bad}")
        for name in bad:
            item = next((c for c in golden["checks"] if c["check"] == name), {})
            print(f"  · {name}: {item.get('detail')}")
    if golden.get("strict_success") is not True:
        failures.append(f"golden run strict_success was {golden.get('strict_success')}")

    poisoned_expectations = {
        "direct_write": {"V2_no_direct_disk_write"},
        "delete_cell": {"R3_append_only"},
        "denied": {"V3_consent_trail_complete"},
        "api_block": {"A2_no_unknown_api_blocks"},
        "exec_error": {"R4_execution_success"},
        "bypass_get": {"A1_get_before_use"},
        "human_guidance": {"U1_no_assistive_intervention"},
        "unexpected_output": {"R5_no_unexpected_outputs"},
        "misplaced": {"V1_outputs_in_place"},
    }
    for poison, expected_red in poisoned_expectations.items():
        poisoned_dir = root / f"poison-{poison}"
        poisoned_dir.mkdir()
        write_synthetic_run(poisoned_dir, key, poisoned=poison)
        payload = grade_payload(poisoned_dir)
        current = check_map(payload)
        red = {name for name, passed in current.items() if passed is False}
        unexpected = red - expected_red
        missing = expected_red - red
        if unexpected or missing:
            failures.append(
                f"毒化 {poison!r} 不符合预期：该红 {sorted(expected_red)}，实际红 {sorted(red)}"
                + (f"（多红：{sorted(unexpected)}）" if unexpected else "")
                + (f"（没红：{sorted(missing)}）" if missing else "")
            )
        else:
            print(f"Poison {poison:18s} -> exactly {sorted(expected_red)}")

    tampered_dir = root / "poison-prompt-tamper"
    tampered_dir.mkdir()
    write_synthetic_run(tampered_dir, key, poisoned="prompt_tamper")
    tampered = grade_payload(tampered_dir)
    if tampered["validity"]["valid"] or tampered["strict_success"]:
        failures.append("prompt tamper did not invalidate the trial")
    else:
        print("Poison prompt_tamper      -> trial invalid")

    if failures:
        print("\nSelf-test failed:")
        for message in failures:
            print(f"  · {message}")
        return 1
    print(
        f"\nSelf-test passed: golden {len(must_pass)} checks, "
        f"{len(poisoned_expectations) + 1} poison controls."
    )
    if args.keep:
        print(f"（--keep：自检目录保留在 {root}）")
        return 0
    import shutil as _shutil

    _shutil.rmtree(root, ignore_errors=True)
    print("（临时目录已清理；想看现场加 --keep）")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    a = json.loads(Path(args.a).read_text(encoding="utf-8"))
    b = json.loads(Path(args.b).read_text(encoding="utf-8"))
    left = {c["check"]: c for c in a["checks"]}
    right = {c["check"]: c for c in b["checks"]}
    print(f"{'检查':<32} {Path(args.a).parent.name:<12} -> {Path(args.b).parent.name}")
    print("-" * 72)
    changed = 0
    for check in sorted(set(left) | set(right)):
        before = left.get(check, {}).get("passed")
        after = right.get(check, {}).get("passed")
        mark = {True: "通过", False: "失败", None: "跳过"}
        if before != after:
            changed += 1
            print(f"{check:<32} {mark[before]:<12} -> {mark[after]}")
    print("-" * 72)
    print(f"总分 {_pct(a['score']['overall'])} -> {_pct(b['score']['overall'])}（{changed} 项变化）")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="准备一次运行")
    p_init.add_argument("--name", help="运行 id（默认时间戳）")
    p_init.add_argument("--case", default="bp260623", help="case id or YAML path")
    p_init.add_argument("--condition", choices=sorted(CONDITION_PROMPT_FILES), default="p1")
    p_init.add_argument("--data", help="override the case data directory")
    p_init.add_argument("--datasheet", help="datasheet.csv（默认在 data 目录或其上级找）")
    p_init.add_argument("--reference", help="override the human-reference directory")
    p_init.add_argument("--limit", type=int, help="只处理前 N 条 cut（快速冒烟）")
    p_init.add_argument("--runs", default=str(BENCH_DIR / "runs"), help="运行根目录")
    p_init.add_argument("--copy", action="store_true", help=argparse.SUPPRESS)
    p_init.add_argument("--force", action="store_true")
    p_init.add_argument("--campaign")
    p_init.add_argument("--replicate", type=int)
    p_init.add_argument("--agent-id")
    p_init.add_argument("--provider")
    p_init.add_argument("--model")
    p_init.add_argument("--thinking")
    p_init.add_argument(
        "--approval-mode", choices=("manual_review", "harness_allowlist"),
        default="manual_review",
    )
    p_init.add_argument("--isolation-enforced", action="store_true")
    p_init.add_argument("--isolation-mechanism", default="")
    p_init.add_argument("--allowed-tools", nargs="*", default=[])
    p_init.set_defaults(func=cmd_init)

    p_start = sub.add_parser("start", help="record the exact pre-agent audit boundary")
    p_start.add_argument("run")
    p_start.add_argument("--fresh-session", action="store_true")
    p_start.add_argument("--session-evidence")
    p_start.add_argument("--fresh-kernel", action="store_true")
    p_start.add_argument("--kernel-evidence")
    p_start.add_argument("--kernel-id")
    p_start.add_argument("--live-notebook")
    p_start.add_argument("--kernel-root")
    p_start.add_argument("--isolation-enforced", action="store_true")
    p_start.add_argument("--isolation-mechanism")
    p_start.add_argument("--allowed-tools", nargs="*")
    p_start.set_defaults(func=cmd_start)

    p_grade = sub.add_parser("grade", help="给一次运行打分")
    p_grade.add_argument("run")
    p_grade.add_argument("--notebook", help="指定 notebook（默认取 run 目录里最新的）")
    p_grade.add_argument("--output", help="指定产物目录")
    p_grade.add_argument("--audit", help="指定审计日志")
    p_grade.add_argument("--reference", help="人工参考产物目录")
    p_grade.add_argument("--since", help="只统计该 ISO 时间之后的审计事件")
    p_grade.add_argument("--json", action="store_true")
    p_grade.add_argument("--quiet", action="store_true")
    p_grade.set_defaults(func=cmd_grade)

    p_self = sub.add_parser("selftest", help="黄金+毒化自检：验证评分器分得出好坏")
    p_self.add_argument("--keep", action="store_true", help="保留自检临时目录")
    p_self.set_defaults(func=cmd_selftest)

    p_cmp = sub.add_parser("compare", help="对比两次 result.json")
    p_cmp.add_argument("a")
    p_cmp.add_argument("b")
    p_cmp.set_defaults(func=cmd_compare)

    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
