#!/usr/bin/env python3
"""Initialize, grade, self-test, and compare isolated benchmark trials.

Each schema-v2 trial separates the agent-visible workspace from evaluator,
operator, and agent-runner evidence. Grading uses the trial notebook, the exact
MCP audit byte window, a test-privileged snapshot of the live kernel namespace,
manifests, and operator logs. Scientific arrays remain in memory and the
notebook; agent dialogue is retained for diagnostics but never scored.

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
import ast
import csv
import fnmatch
import hashlib
import importlib
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
    "u1": PROMPT_DIR / "u1_natural_2d.txt",
}
STANDALONE_PROMPT_CONDITIONS = frozenset({"u1"})
RUN_SCHEMA_VERSION = 2

#: 本任务期望用到的原生 peaks API（来自 SKILL.md 的参考流水线）。
#: A1 检查它们是否都先 search/get 过。
NATIVE_APIS = ("pxt2nc", "load_experiment", "fit_gold", "assign_normal_emission", "k_convert")

#: 非方法型直接写盘模式 —— 出现即绕过 Save 网关。
DIRECT_WRITE_PATTERNS = {
    "savefig": re.compile(r"\.savefig\s*\("),
    "open(...,'w')": re.compile(r"\bopen\s*\([^)]*['\"][rwax+b]{1,2}['\"]"),
    "np.savetxt": re.compile(r"np\.savetxt\s*\("),
}
#: pandas/xarray serializers write only when a target argument is supplied.
#: Each value lists target positional indices followed by accepted target
#: keyword names. Literal ``None`` preserves the APIs' in-memory return mode.
DIRECT_WRITE_METHOD_TARGETS = {
    "to_csv": ((0,), frozenset({"path_or_buf"})),
    "to_excel": ((0,), frozenset({"excel_writer"})),
    "to_feather": ((0,), frozenset({"path"})),
    "to_hdf": ((0,), frozenset({"path_or_buf"})),
    "to_json": ((0,), frozenset({"path_or_buf"})),
    "to_netcdf": ((0,), frozenset({"path"})),
    "to_parquet": ((0,), frozenset({"path"})),
    "to_pickle": ((0,), frozenset({"path"})),
    "to_zarr": ((0, 1), frozenset({"store", "chunk_store"})),
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
INSPECT_TOOLS = {"inspect_notebook"}
PI_CLIENT_ALLOWED_TOOLS = frozenset({"mcp", "mcpScript"})
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


def build_raw_hashes(data_dir: Path) -> dict[str, str]:
    """Hash immutable PXT inputs for conversion-cache provenance checks."""
    return {
        path.name: sha256_file(path)
        for path in sorted(data_dir.glob("*.pxt"))
        if path.is_file()
    }


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


def _git_state(root: Path = ROOT) -> dict[str, Any]:
    """Freeze the source revision and a content-sensitive dirty-tree digest."""
    root = root.resolve()
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL
        ).decode().strip()
        diff = subprocess.check_output(
            ["git", "diff", "--binary", "HEAD", "--", "."],
            cwd=root,
            stderr=subprocess.DEVNULL,
        )
        untracked_raw = subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=root,
            stderr=subprocess.DEVNULL,
        )
        untracked: list[dict[str, str]] = []
        for encoded in untracked_raw.split(b"\0"):
            if not encoded:
                continue
            relative = encoded.decode(errors="replace")
            path = root / relative
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


def _imported_peaks_git_state() -> dict[str, Any]:
    """Identify and freeze the repository backing the imported ``peaks`` package."""
    try:
        module = importlib.import_module("peaks")
        module_path = Path(module.__file__).resolve()
        repository_root = Path(
            subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=module_path.parent,
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        ).resolve()
    except Exception:  # noqa: BLE001
        return {
            "module_path": "unknown",
            "repository_root": "unknown",
            "git_head": "unknown",
            "dirty": None,
            "dirty_fingerprint": "unknown",
            "untracked": [],
        }
    return {
        "module_path": str(module_path),
        "repository_root": str(repository_root),
        **_git_state(repository_root),
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

    P1/P2 share the common scientific task. U1 is deliberately standalone: it
    reproduces a natural user's short request so API and tool guidance, rather
    than a benchmark-provided recipe, must carry the workflow.
    """
    try:
        condition_path = CONDITION_PROMPT_FILES[condition]
    except KeyError as exc:
        raise ValueError(f"unknown prompt condition {condition!r}") from exc
    prefix = condition_path.read_text(encoding="utf-8").strip()
    if condition in STANDALONE_PROMPT_CONDITIONS:
        rendered = prefix
    else:
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
    into failed checks (a loader call after an import block was invisible).
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
        args = (event.get("details") or {}).get("args", {})
        if args.get("cell_type", "code") != "code":
            continue
        code = args.get("code")
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
    # The audit writes a tool call as TWO events sharing one operation_id: the
    # "called" event carries the arguments, the outcome event carries the
    # result.  Reading `args` off the success event therefore found nothing and
    # every native API looked unproven (A1 was red on runs whose calls provably
    # went through `get`).  Pair them by operation_id, and keep the rule that
    # only a get which actually SUCCEEDED is a proof.  Older/legacy shapes that
    # put the outcome and the arguments on one event keep working.
    called: dict[str, set[str]] = {}
    succeeded: set[str] = set()
    standalone: set[str] = set()

    def leaves_of(details: dict[str, Any]) -> set[str]:
        args = details.get("args") or {}
        canonical = args.get("canonical_id") or args.get("canonical_ids")
        if canonical is None:
            return set()
        items = canonical if isinstance(canonical, list) else [canonical]
        return {str(item).rsplit(":", 1)[-1] for item in items}

    for event in events:
        if event.get("tool") not in GET_TOOLS:
            continue
        details = event.get("details") or {}
        operation = str(details.get("operation_id") or "")
        outcome = str(event.get("outcome") or "")
        if operation:
            if outcome in {"ok", "executed"}:
                succeeded.add(operation)
            called.setdefault(operation, set()).update(leaves_of(details))
        elif outcome in {"ok", "executed"}:
            standalone.update(leaves_of(details))
    for operation, leaves in called.items():
        if operation in succeeded:
            names.update(leaves)
    names.update(standalone)
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
    live_evidence: dict[str, Any] = field(default_factory=dict)
    client_tool_evidence: dict[str, Any] = field(default_factory=dict)

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


_PI_WRAPPER_ERROR_LINE = re.compile(
    r'^\s*(?:Error:\s*)?(?:Tool\s+"[^"\r\n]+"\s+not found(?:[.!:]|$)'
    r"|Input validation error\s*:)",
    re.IGNORECASE,
)


def _pi_wrapper_reports_failure(message: dict[str, Any], text: str) -> bool:
    """Recognize MCP client wrapper failures hidden behind ``isError=false``."""
    details = message.get("details") or {}
    if isinstance(details, dict):
        if details.get("error"):
            return True
        mcp_result = details.get("mcpResult") or {}
        if isinstance(mcp_result, dict) and mcp_result.get("isError") is True:
            return True
        calls = details.get("calls") or []
        if isinstance(calls, list) and any(
            isinstance(call, dict) and call.get("ok") is False for call in calls
        ):
            return True
    first_line = next((part.strip() for part in text.splitlines() if part.strip()), "")
    return bool(_PI_WRAPPER_ERROR_LINE.match(first_line))


def collect_pi_client_tool_evidence(run_dir: Path) -> dict[str, Any]:
    """Extract structured Pi tool failures without grading conversation text."""
    session_dir = run_dir / "agent" / "session"
    paths = sorted(session_dir.glob("*.jsonl")) if session_dir.is_dir() else []
    if not paths:
        return {
            "status": "unavailable",
            "session_files": [],
            "error_count": 0,
            "schema_validation_errors": 0,
            "errors": [],
            "successful_tool_names": [],
            "disallowed_successful_tools": [],
        }

    calls: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []
    successful_tool_names: list[str] = []
    disallowed_successful_tools: list[dict[str, Any]] = []
    malformed = 0
    for path in paths:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
        ):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if event.get("type") != "message":
                continue
            message = event.get("message") or {}
            content = message.get("content") or []
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict) or item.get("type") != "toolCall":
                    continue
                call_id = str(item.get("id") or "")
                arguments = item.get("arguments") or {}
                target = (
                    arguments.get("tool")
                    if isinstance(arguments, dict) and arguments.get("tool")
                    else item.get("name")
                )
                if call_id:
                    calls[call_id] = {
                        "top_level_tool": str(item.get("name") or "unknown"),
                        "target": str(target or "unknown"),
                        "session_file": path.name,
                        "line": line_number,
                    }
            if message.get("role") != "toolResult":
                continue
            call_id = str(message.get("toolCallId") or "")
            call = calls.get(call_id) or {}
            top_level_tool = str(
                message.get("toolName") or call.get("top_level_tool") or "unknown"
            )
            text = "\n".join(
                str(item.get("text") or "")
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            )
            failed = bool(message.get("isError")) or any(
                isinstance(item, dict) and bool(item.get("isError")) for item in content
            )
            if top_level_tool in PI_CLIENT_ALLOWED_TOOLS:
                failed = failed or _pi_wrapper_reports_failure(message, text)
            if not failed:
                successful_tool_names.append(top_level_tool)
                if top_level_tool not in PI_CLIENT_ALLOWED_TOOLS:
                    disallowed_successful_tools.append({
                        "tool_call_id": call_id,
                        "tool": top_level_tool,
                        "session_file": path.name,
                        "line": line_number,
                    })
                continue
            first_line = next((part.strip() for part in text.splitlines() if part.strip()), "Error")
            kind = "schema_validation" if "input validation error" in text.lower() else "tool_error"
            errors.append(
                {
                    "tool_call_id": call_id,
                    "tool": call.get("target", "unknown"),
                    "kind": kind,
                    "message": first_line[:240],
                    "session_file": path.name,
                    "line": line_number,
                }
            )
    return {
        "status": "ok",
        "session_files": [path.name for path in paths],
        "malformed_lines": malformed,
        "error_count": len(errors),
        "schema_validation_errors": sum(
            error["kind"] == "schema_validation" for error in errors
        ),
        "errors": errors,
        "successful_tool_names": sorted(set(successful_tool_names)),
        "disallowed_successful_tools": disallowed_successful_tools,
    }


def _parse_notebook_block(block: str) -> ast.Module | None:
    """Parse a notebook block, ignoring standalone IPython command lines."""
    try:
        return ast.parse(block)
    except SyntaxError:
        filtered = "\n".join(
            "" if line.lstrip().startswith(("%", "!", "?")) else line
            for line in block.splitlines()
        )
        try:
            return ast.parse(filtered)
        except SyntaxError:
            return None


_PATH_CONSTRUCTORS = frozenset({
    "Path", "PosixPath", "WindowsPath",
    "pathlib.Path", "pathlib.PosixPath", "pathlib.WindowsPath",
})
_PATH_RECEIVER_METHODS = frozenset({
    "open", "read_bytes", "read_text", "iterdir", "glob", "rglob",
    "exists", "is_dir", "is_file", "stat",
    "write_bytes", "write_text", "unlink", "rmdir", "rename", "replace",
})
_NUMPY_PATH_READERS = frozenset({
    "load", "loadtxt", "genfromtxt", "fromfile", "memmap", "open_memmap",
    "recfromcsv", "recfromtxt",
})
_IO_PATH_SPECS: dict[str, tuple[tuple[int, ...], frozenset[str]]] = {
    "open": ((0,), frozenset({"file"})),
    "builtins.open": ((0,), frozenset({"file"})),
    "io.open": ((0,), frozenset({"file"})),
    "os.walk": ((0,), frozenset({"top"})),
    "os.listdir": ((0,), frozenset({"path"})),
    "os.scandir": ((0,), frozenset({"path"})),
    "os.stat": ((0,), frozenset({"path"})),
    "os.lstat": ((0,), frozenset({"path"})),
    "os.access": ((0,), frozenset({"path"})),
    "os.readlink": ((0,), frozenset({"path"})),
    "os.remove": ((0,), frozenset({"path"})),
    "os.unlink": ((0,), frozenset({"path"})),
    "os.rmdir": ((0,), frozenset({"path"})),
    "os.rename": ((0, 1), frozenset({"src", "dst"})),
    "os.replace": ((0, 1), frozenset({"src", "dst"})),
    "peaks.load_experiment": ((0,), frozenset({"source", "metadata"})),
    "load_experiment": ((0,), frozenset({"source", "metadata"})),
    "peaks.pxt2nc": ((0,), frozenset({"source", "metadata"})),
    "pxt2nc": ((0,), frozenset({"source", "metadata"})),
}


def _qualified_ast_name(node: ast.AST, aliases: dict[str, str]) -> str:
    """Resolve a simple imported name or attribute chain."""
    if isinstance(node, ast.Name):
        return aliases.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        parent = _qualified_ast_name(node.value, aliases)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _static_path_values(
    node: ast.AST,
    bindings: dict[str, set[str]],
    aliases: dict[str, str],
) -> set[str]:
    """Resolve the small, literal-only path expressions used in trials."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value}
    if isinstance(node, ast.Name):
        return set(bindings.get(node.id, set()))
    if isinstance(node, ast.JoinedStr):
        values = {""}
        for part in node.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                additions = {part.value}
            elif isinstance(part, ast.FormattedValue):
                additions = _static_path_values(part.value, bindings, aliases)
            else:
                return set()
            values = {left + right for left in values for right in additions}
        return values
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Div)):
        left = _static_path_values(node.left, bindings, aliases)
        right = _static_path_values(node.right, bindings, aliases)
        if isinstance(node.op, ast.Add):
            return {prefix + suffix for prefix in left for suffix in right}
        return {str(Path(prefix) / suffix) for prefix in left for suffix in right}
    if not isinstance(node, ast.Call):
        return set()
    name = _qualified_ast_name(node.func, aliases)
    if name in _PATH_CONSTRUCTORS or name == "str":
        return _static_path_values(node.args[0], bindings, aliases) if node.args else set()
    if isinstance(node.func, ast.Attribute):
        base = _static_path_values(node.func.value, bindings, aliases)
        if node.func.attr in {"absolute", "expanduser", "resolve", "with_name", "with_suffix"}:
            return base
        if node.func.attr == "joinpath":
            for argument in node.args:
                additions = _static_path_values(argument, bindings, aliases)
                base = {str(Path(prefix) / suffix) for prefix in base for suffix in additions}
            return base
    return set()


def _bind_static_paths(
    target: ast.AST,
    values: set[str],
    bindings: dict[str, set[str]],
    aliases: dict[str, str],
) -> None:
    """Update one simple Notebook name binding without retaining stale values."""
    if not isinstance(target, ast.Name):
        return
    aliases.pop(target.id, None)
    if values:
        bindings[target.id] = set(values)
    else:
        bindings.pop(target.id, None)


def _call_path_values(
    call: ast.Call,
    bindings: dict[str, set[str]],
    aliases: dict[str, str],
) -> set[str]:
    """Return statically known paths actually consumed by one I/O call."""
    name = _qualified_ast_name(call.func, aliases)
    root, _, method = name.partition(".")
    method = name.rsplit(".", 1)[-1]
    spec = _IO_PATH_SPECS.get(name)
    if spec is None and root == "pandas" and method.startswith("read_"):
        spec = ((0,), frozenset({"filepath_or_buffer", "path_or_buf", "io"}))
    elif spec is None and root == "xarray" and method.startswith("open_"):
        spec = ((0,), frozenset({"filename_or_obj", "filename_or_obj_or_dict"}))
    elif spec is None and root == "numpy" and method in _NUMPY_PATH_READERS:
        spec = ((0,), frozenset({"file", "fname", "filename"}))
    elif spec is None and method in DIRECT_WRITE_METHOD_TARGETS:
        spec = DIRECT_WRITE_METHOD_TARGETS[method]

    values: set[str] = set()
    if spec is not None:
        positions, keywords = spec
        for position in positions:
            if position < len(call.args):
                values |= _static_path_values(call.args[position], bindings, aliases)
        for keyword in call.keywords:
            if keyword.arg in keywords:
                values |= _static_path_values(keyword.value, bindings, aliases)

    if isinstance(call.func, ast.Attribute) and call.func.attr in _PATH_RECEIVER_METHODS:
        receiver = _static_path_values(call.func.value, bindings, aliases)
        if receiver:
            values |= receiver
            if call.func.attr in {"rename", "replace"} and call.args:
                values |= _static_path_values(call.args[0], bindings, aliases)
    return values


def _outside_notebook_io_paths(blocks: list[str], workspace: Path) -> list[str]:
    """Find static paths outside ``workspace`` that are passed to Notebook I/O."""
    aliases: dict[str, str] = {}
    bindings: dict[str, set[str]] = {}
    observed: set[str] = set()
    for block in blocks:
        tree = _parse_notebook_block(block)
        if tree is None:
            continue
        nodes = [
            node for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign, ast.Call))
        ]
        nodes.sort(key=lambda node: (getattr(node, "lineno", 0), getattr(node, "col_offset", 0)))
        for node in nodes:
            if isinstance(node, ast.Import):
                for item in node.names:
                    bound = item.asname or item.name.split(".")[0]
                    bindings.pop(bound, None)
                    aliases[bound] = (
                        item.name if item.asname else item.name.split(".")[0]
                    )
                continue
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for item in node.names:
                    bound = item.asname or item.name
                    bindings.pop(bound, None)
                    aliases[bound] = f"{module}.{item.name}"
                continue
            if isinstance(node, ast.Assign):
                values = _static_path_values(node.value, bindings, aliases)
                for target in node.targets:
                    _bind_static_paths(target, values, bindings, aliases)
                continue
            if isinstance(node, ast.AnnAssign):
                values = (
                    _static_path_values(node.value, bindings, aliases)
                    if node.value is not None else set()
                )
                _bind_static_paths(node.target, values, bindings, aliases)
                continue
            for raw in _call_path_values(node, bindings, aliases):
                try:
                    candidate = Path(raw).expanduser()
                    if not candidate.is_absolute():
                        candidate = workspace / candidate
                    candidate = candidate.resolve()
                except (OSError, ValueError):
                    continue
                if not _is_relative_to(candidate, workspace):
                    observed.add(str(candidate))
    return sorted(observed)


def _direct_write_method_names(blocks: list[str]) -> set[str]:
    """Return target-bearing pandas/xarray writer methods in notebook code."""
    found: set[str] = set()
    for block in blocks:
        tree = _parse_notebook_block(block)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            method = node.func.attr
            target_spec = DIRECT_WRITE_METHOD_TARGETS.get(method)
            if target_spec is None:
                continue
            positions, keywords = target_spec
            positional_target = any(
                position < len(node.args)
                and not (
                    isinstance(node.args[position], ast.Constant)
                    and node.args[position].value is None
                )
                for position in positions
            )
            keyword_target = any(
                (keyword.arg is None or keyword.arg in keywords)
                and not (
                    isinstance(keyword.value, ast.Constant)
                    and keyword.value.value is None
                )
                for keyword in node.keywords
            )
            if positional_target or keyword_target:
                found.add(method)
    return found


def _call_count(blocks: list[str], name: str) -> int:
    """Count real Python calls to ``name`` across already-deduplicated cells."""
    count = 0
    for block in blocks:
        tree = _parse_notebook_block(block)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            if (
                isinstance(function, ast.Name) and function.id == name
                or isinstance(function, ast.Attribute) and function.attr == name
            ):
                count += 1
    return count


def _scientific_contract(ctx: Ctx) -> dict[str, Any]:
    """Return the frozen, case-specific workflow expectations."""
    case = ctx.manifest.get("case") or {}
    return dict(case.get("scientific_contract") or {})


def _expected_call_counts(ctx: Ctx) -> dict[str, int]:
    raw = _scientific_contract(ctx).get("expected_call_counts") or {}
    return {str(name): int(count) for name, count in raw.items()}


def _executed_notebook_code(ctx: Ctx) -> list[str]:
    """Executed code cells in notebook order, preserving repeated sources."""
    cells = [
        "".join(cell.get("source") or [])
        for cell in (ctx.notebook or {}).get("cells", [])
        if cell.get("cell_type") == "code" and cell.get("execution_count") is not None
    ]
    return cells or list(ctx.code)


def _successful_get_ids(events: list[dict[str, Any]]) -> list[str]:
    """Canonical ids from successful get operations, retaining duplicates."""
    called: dict[str, list[str]] = {}
    succeeded: set[str] = set()
    standalone: list[str] = []

    def ids(details: dict[str, Any]) -> list[str]:
        args = details.get("args") or {}
        value = args.get("canonical_id") or args.get("canonical_ids")
        if value is None:
            return []
        return [str(item) for item in (value if isinstance(value, list) else [value])]

    for position, event in enumerate(events):
        if event.get("tool") not in GET_TOOLS:
            continue
        details = event.get("details") or {}
        operation = str(details.get("operation_id") or "")
        outcome = str(event.get("outcome") or "")
        if operation:
            if outcome in {"ok", "executed"}:
                succeeded.add(operation)
            called.setdefault(operation, []).extend(ids(details))
        elif outcome in {"ok", "executed"}:
            standalone.extend(ids(details))
        elif outcome == "called":
            called.setdefault(f"legacy-{position}", []).extend(ids(details))
    return [item for operation, values in called.items() if operation in succeeded for item in values] + standalone


_RUN_CELL_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_RUN_CELL_MEDIA_MIMES = frozenset({
    "image/png", "image/jpeg", "image/svg+xml",
    "application/vnd.peaksmcp.image-omitted+json",
    "application/vnd.jupyter.widget-view+json",
    "application/vnd.holoviews_load.v0+json",
    "application/vnd.plotly.v1+json",
    "application/vnd.bokehjs_exec.v0+json",
})
_RUN_CELL_STDOUT_HEAD_MAX = 80
_RUN_CELL_SUMMARY_MAX_LINES = 3
_RUN_CELL_SUMMARY_MAX_LINE_LENGTH = 200

# A completed benchmark notebook has exactly three durable review figures:
# the gold-fit diagnostic, one compact grid containing every processed cut,
# and one representative before/after comparison.  Progress widgets are
# transient execution UI, not scientific results, and make a reopened
# notebook unnecessarily noisy.
_NOTEBOOK_STATIC_IMAGE_MIMES = frozenset({"image/png", "image/jpeg", "image/svg+xml"})
_NOTEBOOK_WIDGET_MIME = "application/vnd.jupyter.widget-view+json"
_NOTEBOOK_EXPECTED_FIGURES = 3
_NOTEBOOK_MAX_FINAL_MARKDOWN_CHARS = 2_000


def _notebook_output_text(value: Any) -> str:
    """Join Jupyter text fragments and match run_cell's ANSI cleanup."""
    text = "".join(str(part) for part in value) if isinstance(value, list) else str(value)
    return _RUN_CELL_ANSI_ESCAPE.sub("", text)


def _cell_has_archived_text(cell: dict[str, Any]) -> bool:
    """Whether a cell reread exposes meaningful text absent from its run reply.

    Besides the bounded stdout summary, ``run_cell`` always returns the first
    80 raw stdout characters. A reread is justified only when neither channel
    covers all stdout, another stream was suppressed, or a standalone
    ``text/plain`` result was archived. Rich-display ``text/plain`` fallbacks
    (especially ``<Figure ...>``) do not make an otherwise redundant reread
    useful.
    """
    stdout: list[tuple[int, str]] = []
    stdout_chunks: list[str] = []
    first_media: int | None = None
    has_error = False
    has_suppressed_stream = False
    has_standalone_plain = False
    for order, output in enumerate(cell.get("outputs") or []):
        if not isinstance(output, dict):
            continue
        if output.get("output_type") == "error":
            has_error = True
            continue
        if output.get("output_type") == "stream":
            raw = output.get("text") or ""
            raw_text = "".join(str(part) for part in raw) if isinstance(raw, list) else str(raw)
            text = _RUN_CELL_ANSI_ESCAPE.sub("", raw_text)
            if output.get("name") == "stdout":
                stdout_chunks.append(raw_text)
                stdout.extend(
                    (order, line.rstrip()) for line in text.splitlines() if line.strip()
                )
            elif text.strip():
                has_suppressed_stream = True
            continue
        data = output.get("data") or {}
        if not isinstance(data, dict):
            continue
        data_mimes = {str(mime) for mime, value in data.items() if value}
        if first_media is None and data_mimes & _RUN_CELL_MEDIA_MIMES:
            first_media = order
        plain = data.get("text/plain")
        if plain and data_mimes == {"text/plain"}:
            plain_text = _notebook_output_text(plain).strip()
            if plain_text and not plain_text.startswith("<Figure"):
                has_standalone_plain = True

    before_media = [
        line for order, line in stdout if first_media is None or order < first_media
    ]
    summary_visible = (
        not has_error
        and bool(before_media)
        and len(before_media) <= _RUN_CELL_SUMMARY_MAX_LINES
        and all(len(line) <= _RUN_CELL_SUMMARY_MAX_LINE_LENGTH for line in before_media)
    )
    all_stdout_in_summary = summary_visible and len(before_media) == len(stdout)
    raw_stdout = "".join(stdout_chunks)
    all_stdout_in_head = not raw_stdout[_RUN_CELL_STDOUT_HEAD_MAX:].strip()
    stdout_omitted = bool(stdout) and not (all_stdout_in_summary or all_stdout_in_head)
    return stdout_omitted or has_suppressed_stream or has_standalone_plain


def _redundancy_metrics(ctx: Ctx) -> dict[str, Any]:
    """Evidence-only efficiency signals; none are inferred from model prose."""
    successful_gets = _successful_get_ids(ctx.events)
    get_counts = {item: successful_gets.count(item) for item in dict.fromkeys(successful_gets)}
    duplicate_gets = {item: count - 1 for item, count in get_counts.items() if count > 1}
    executed_blocks = _executed_notebook_code(ctx)
    unused_gets = sorted(
        item
        for item in get_counts
        if _call_count(executed_blocks, item.rsplit(":", 1)[-1]) == 0
    )

    queries = [
        str(((event.get("details") or {}).get("args") or {}).get("query") or "").strip()
        for event in ctx.events
        if event.get("tool") in SEARCH_TOOLS and event.get("outcome") == "called"
    ]
    query_counts = {query: queries.count(query) for query in dict.fromkeys(queries) if query}
    repeated_searches = {query: count - 1 for query, count in query_counts.items() if count > 1}

    outcomes: dict[str, set[str]] = {}
    cell_ids: dict[str, str] = {}
    for event in ctx.events:
        details = event.get("details") or {}
        operation = str(details.get("operation_id") or "")
        if not operation:
            continue
        outcomes.setdefault(operation, set()).add(str(event.get("outcome") or ""))
        if details.get("cell_id") is not None:
            cell_ids[operation] = str(details["cell_id"])
    calls = [
        event for event in ctx.events
        if event.get("outcome") == "called"
        and event.get("tool") in (RUN_TOOLS | INSPECT_TOOLS)
    ]
    notebook_cells = (ctx.notebook or {}).get("cells", [])
    cells_by_id = {
        str(cell.get("id")): cell
        for cell in notebook_cells
        if cell.get("id") is not None
    }
    cells_by_index = {str(index): cell for index, cell in enumerate(notebook_cells)}
    cell_positions = {
        cell_id: str(index)
        for index, cell in enumerate(notebook_cells)
        if (cell_id := str(cell.get("id") or ""))
    }
    immediate_rereads: list[str] = []
    for current, following in zip(calls, calls[1:], strict=False):
        if current.get("tool") not in RUN_TOOLS or following.get("tool") not in INSPECT_TOOLS:
            continue
        operation = str((current.get("details") or {}).get("operation_id") or "")
        terminal = outcomes.get(operation, set())
        if not terminal.intersection({"ok", "executed"}) or terminal.intersection(
            {"error", "blocked", "failed"}
        ):
            continue
        args = (following.get("details") or {}).get("args") or {}
        if args.get("target") != "cell" or not args.get("with_text_outputs"):
            continue
        requested = str(args.get("cell") or "")
        produced = cell_ids.get(operation, "")
        produced_cell = cells_by_id.get(produced) or cells_by_index.get(requested)
        if produced_cell is not None and _cell_has_archived_text(produced_cell):
            continue
        if (
            not requested
            or not produced
            or requested == produced
            or requested == cell_positions.get(produced)
        ):
            immediate_rereads.append(produced or requested or operation)
    return {
        "duplicate_gets": duplicate_gets,
        "unused_gets": unused_gets,
        "repeated_searches": repeated_searches,
        "immediate_cell_rereads": immediate_rereads,
    }


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
        try:
            tree = ast.parse(cleaned, mode="eval")
        except SyntaxError:
            return set()
        values: set[int] = set()
        for node in ast.walk(tree):
            value = getattr(node, "value", None)
            if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 9999:
                values.add(value)
            elif isinstance(value, str) and re.fullmatch(r"\d{1,4}", value):
                values.add(int(value))
        return values

    def resolve_expr(text: str) -> set[int]:
        """Indices an expression can denote: classified gold, known vars, literals."""
        if re.search(r"\b[A-Za-z_]\w*\.gold\b|\.is_gold\b", text):
            return set(gold)
        values: set[int] = set()
        for mapping in (index_vars, data_vars):
            for var, known in mapping.items():
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
            # ``experiment.gold[0]`` indexes the classified-gold list; the
            # literal 0 is a list position, not experiment scan 0.
            classified_gold_item = re.fullmatch(
                r"\s*[A-Za-z_]\w*\.gold\s*\[\s*[^\]]+\s*\]\s*", rhs
            )
            if classified_gold_item:
                changed |= note(index_vars, name, set(gold))
                continue
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
        direct = re.search(
            r"\b[A-Za-z_]\w*\s*\[\s*([^\]]+?)\s*\]\s*\.\s*fit_gold\s*\(",
            line,
        )
        if direct:
            values = resolve_expr(direct.group(1))
            if values:
                indices.update(values)
            else:
                unresolved = True
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


def _fit_gold_uses_classified_receiver(blocks: list[str], context: str = "") -> bool:
    """Whether every gold fit is selected through ExperimentIndex classification.

    Resolving a receiver to the expected numeric index is insufficient: a literal
    such as ``exp[20]`` can guess the answer without using ``exp.gold`` or the
    ``is_gold`` classification carried by ExperimentIndex records. This small
    taint analysis follows classification-derived indices, stems, and data aliases
    into each ``fit_gold`` receiver.
    """
    trees = [
        tree
        for block in [context, *blocks]
        if block and (tree := _parse_notebook_block(block)) is not None
    ]
    classified: set[str] = set()

    def uses_classification(node: ast.AST | None) -> bool:
        if node is None:
            return False
        return any(
            isinstance(part, ast.Attribute) and part.attr in {"gold", "is_gold"}
            or isinstance(part, ast.Name) and part.id in classified
            for part in ast.walk(node)
        )

    def bind(target: ast.AST) -> bool:
        names = {
            part.id
            for part in ast.walk(target)
            if isinstance(part, ast.Name) and isinstance(part.ctx, ast.Store)
        }
        before = len(classified)
        classified.update(names)
        return len(classified) != before

    changed = True
    while changed:
        changed = False
        for tree in trees:
            for node in ast.walk(tree):
                if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
                    value = node.value
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    if uses_classification(value):
                        changed |= any(bind(target) for target in targets)
                elif isinstance(node, (ast.For, ast.AsyncFor)) and uses_classification(node.iter):
                    changed |= bind(node.target)
                elif isinstance(node, ast.comprehension) and uses_classification(node.iter):
                    changed |= bind(node.target)

    provenance: list[bool] = []
    for tree in trees:
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Attribute) and node.func.attr == "fit_gold":
                provenance.append(uses_classification(node.func.value))
            elif isinstance(node.func, ast.Name) and node.func.id == "fit_gold":
                provenance.append(bool(node.args) and uses_classification(node.args[0]))
    return bool(provenance) and all(provenance)


def _theta_offset_binding_used(blocks: list[str]) -> bool:
    """Whether every normal-emission theta value derives from record metadata."""
    trees = [
        tree
        for block in blocks
        if block and (tree := _parse_notebook_block(block)) is not None
    ]
    bound: set[str] = set()

    def uses_record_offset(node: ast.AST | None) -> bool:
        if node is None:
            return False
        return any(
            isinstance(part, ast.Attribute) and part.attr == "theta_offset_deg"
            or isinstance(part, ast.Name) and part.id in bound
            for part in ast.walk(node)
        )

    def bind(target: ast.AST) -> bool:
        names = {
            part.id
            for part in ast.walk(target)
            if isinstance(part, ast.Name) and isinstance(part.ctx, ast.Store)
        }
        before = len(bound)
        bound.update(names)
        return len(bound) != before

    changed = True
    while changed:
        changed = False
        for tree in trees:
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
                    continue
                value = node.value
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if uses_record_offset(value):
                    changed |= any(bind(target) for target in targets)

    theta_arguments: list[ast.AST] = []
    for tree in trees:
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            if not (
                isinstance(function, ast.Name) and function.id == "assign_normal_emission"
                or isinstance(function, ast.Attribute) and function.attr == "assign_normal_emission"
            ):
                continue
            theta_arguments.extend(
                keyword.value for keyword in node.keywords if keyword.arg == "theta_par"
            )
    return bool(theta_arguments) and all(uses_record_offset(value) for value in theta_arguments)


def check_contract(ctx: Ctx) -> list[Result]:
    corpus = _corpus(ctx)
    out: list[Result] = []

    converted = "pxt2nc(" in corpus or "pxt2nc (" in corpus
    expected_pxt = _expected_call_counts(ctx).get("pxt2nc")
    conversion_ok = not converted if expected_pxt == 0 else converted
    out.append(Result(
        "C1_blackbox_load",
        conversion_ok,
        (
            "输入已预转换，正确跳过 peaks.pxt2nc"
            if expected_pxt == 0 and not converted
            else "预转换输入不应再次调用 peaks.pxt2nc"
            if expected_pxt == 0
            else "公开入口 peaks.pxt2nc" + ("" if converted else " 未出现")
        ),
    ))

    loaded = "load_experiment(" in corpus or "load_experiment (" in corpus
    manual_metadata = bool(re.search(
        r"datasheet\.csv[^\n]*(?:read_text|open)|csv\.reader\s*\(|read_csv\s*\(",
        corpus,
        flags=re.IGNORECASE,
    ))
    out.append(Result(
        "C2_blackbox_inspect",
        loaded and not manual_metadata,
        (
            "公开入口 peaks.load_experiment（含 metadata 与分类）"
            if loaded and not manual_metadata
            else "已调用 load_experiment，但又手工读取 datasheet；应直接使用 ExperimentIndex"
            if loaded
            else "公开入口 peaks.load_experiment 未出现 —— agent 很可能手工读取或分类"
        ),
    ))

    # gold 索引：只解析 fit_gold 的 receiver（链式字面量 / 变量绑定 / 单跳别名），
    # 绝不扫描整块里的 BP 编号 —— 同 cell 的 cuts 列表字面量会误伤正确代码。
    gold_blocks = _blocks_with(ctx, "fit_gold")
    wanted = set(ctx.key["gold_indices"])
    found, unresolved = _fit_gold_receiver_indices(gold_blocks, wanted, context=corpus)
    classified_receiver = _fit_gold_uses_classified_receiver(gold_blocks, context=corpus)
    if not gold_blocks:
        out.append(Result("C3_gold_index_correct", False, "没有出现 fit_gold，无法判断 gold 选择"))
    elif unresolved and not found:
        out.append(Result("C3_gold_index_correct", None,
                          "fit_gold 的 receiver 无法回溯到任何实验索引（既不是字面量、"
                          "也不是经 summary.gold / 变量绑定得到的索引），跳过"))
    else:
        ok = bool(found) and found == wanted and classified_receiver
        out.append(Result("C3_gold_index_correct", ok,
                          f"期望 {sorted(wanted)}，fit_gold receiver 解析到 "
                          f"{sorted(found) or '未解析出'}；ExperimentIndex 分类来源 "
                          f"{'已证明' if classified_receiver else '缺失'}"))

    # 数值碰巧等于答案不构成 metadata provenance；只认 record.theta_offset_deg
    # 的直接使用或经变量/容器传播后传给 assign_normal_emission(theta_par=...).
    offset = ctx.key.get("theta_offset_deg")
    binding_ok = _theta_offset_binding_used(ctx.unique_code)
    ok = offset is not None and binding_ok
    detail = (
        f"契约值 {offset}；record.theta_offset_deg -> theta_par 来源"
        f"{'已证明' if binding_ok else '缺失（纯字面量不被接受）'}"
    )
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
        for check in ("A1_get_before_use", "A2_no_unknown_api_blocks", "A4_no_redundant_tool_calls"):
            out.append(Result(check, None, NO_AUDIT))
        expected_pxt = _expected_call_counts(ctx).get("pxt2nc")
        pxt_ok = "pxt2nc(" not in corpus if expected_pxt == 0 else "pxt2nc(" in corpus
        out.append(Result("A3_override_first", pxt_ok and "load_experiment(" in corpus,
                          "公开 peaks 入口使用情况见 C1/C2（无审计日志，仅按代码判断）"))
        return out
    used = {api for api in NATIVE_APIS if api in corpus}
    missing = sorted(used - ctx.fetched)
    out.append(Result("A1_get_before_use", not missing,
                      f"用到 {sorted(used)}；未经 search/get 验证的：{missing or '无'}",
                      evidence=missing))

    # 只统计"因为 API 未经验证被拦下"，断连、安全扫描和持久化错误不算
    # Access 失败。一个 run_cell 操作可能同时写 backend + wrapper 两条
    # blocked 事件，所以必须按 operation_id 去重。
    blocked_by_operation: dict[str, dict[str, Any]] = {}
    for position, event in enumerate(ctx.events):
        if event.get("tool") not in RUN_TOOLS:
            continue
        outcome = event.get("outcome")
        details = event.get("details") or {}
        text = str(details.get("error", "")).lower()
        if any(hint in text for hint in PERSIST_BLOCK_HINTS):
            # run_cell 的持久化硬阻止归 V2 判，不算 Access 失败。
            continue
        unknown_refs = details.get("unknown_refs") or []
        is_api_block = bool(unknown_refs) or (
            outcome == "error" and any(marker in text for marker in API_BLOCK_MARKERS)
        )
        if not is_api_block:
            continue
        operation = str(details.get("operation_id") or f"legacy-{position}")
        blocked_by_operation.setdefault(operation, event)
    blocked = list(blocked_by_operation.values())
    out.append(Result("A2_no_unknown_api_blocks", not blocked,
                      f"因 API 未验证被拦下 {len(blocked)} 次",
                      evidence=[
                          str(
                              (e.get("details") or {}).get("unknown_refs")
                              or (e.get("details") or {}).get("error", "")
                          )[:120]
                          for e in blocked[:5]
                      ]))

    expected_pxt = _expected_call_counts(ctx).get("pxt2nc")
    pxt_ok = "pxt2nc(" not in corpus if expected_pxt == 0 else "pxt2nc(" in corpus
    out.append(Result("A3_override_first", pxt_ok and "load_experiment(" in corpus,
                      "公开 peaks 入口使用情况见 C1/C2"))
    redundant = _redundancy_metrics(ctx)
    redundant_count = (
        sum(redundant["duplicate_gets"].values())
        + len(redundant["unused_gets"])
        + sum(redundant["repeated_searches"].values())
        + len(redundant["immediate_cell_rereads"])
    )
    out.append(Result(
        "A4_no_redundant_tool_calls",
        redundant_count == 0,
        "；".join(
            (
                f"重复 get {sum(redundant['duplicate_gets'].values())}",
                f"未使用 get {len(redundant['unused_gets'])}",
                f"重复 search {sum(redundant['repeated_searches'].values())}",
                f"成功 cell 后立即回读 {len(redundant['immediate_cell_rereads'])}",
            )
        ),
        evidence=[
            *(f"duplicate get: {name}" for name in redundant["duplicate_gets"]),
            *(f"unused get: {name}" for name in redundant["unused_gets"]),
            *(f"repeated search: {query}" for query in redundant["repeated_searches"]),
            *(f"cell reread: {cell}" for cell in redundant["immediate_cell_rereads"]),
        ][:10],
    ))
    return out


def check_client_tools(ctx: Ctx) -> list[Result]:
    """Require a clean Pi-visible tool trajectory, including schema failures."""
    agent = ctx.manifest.get("agent") or {}
    execution = agent.get("execution") or {}
    runner = str(agent.get("runner") or execution.get("runner") or "")
    if runner not in {"pi", "pi-tui"}:
        return [Result(
            "A5_no_client_tool_errors",
            True,
            f"runner={runner or 'unknown'}; Pi client-tool check not applicable",
        )]
    evidence = ctx.client_tool_evidence
    if evidence.get("status") != "ok":
        return [Result(
            "A5_no_client_tool_errors",
            None,
            "Pi session has no structured client tool evidence",
        )]
    errors = list(evidence.get("errors") or [])
    disallowed = list(evidence.get("disallowed_successful_tools") or [])
    schema_errors = int(evidence.get("schema_validation_errors") or 0)
    return [Result(
        "A5_no_client_tool_errors",
        not errors and not disallowed,
        f"Pi-visible tool errors {len(errors)}; schema validation {schema_errors}; "
        f"disallowed successful tools {len(disallowed)}",
        evidence=[
            f"{item.get('tool', 'unknown')}: {item.get('kind', 'tool_error')}: "
            f"{item.get('message', '')}"
            for item in errors[:10]
        ] + [
            f"disallowed successful tool: {item.get('tool', 'unknown')}"
            for item in disallowed[:10]
        ],
    )]


def check_run(ctx: Ctx) -> list[Result]:
    out: list[Result] = []
    # The task is DONE when the notebook contains the required output - a
    # persisted file is a policy step, not the definition of completion.  A
    # target therefore counts when the privileged live-kernel snapshot found a
    # momentum-space DataArray for it. The code heuristic remains only as an
    # offline compatibility fallback for portable grader tests.
    items = list(ctx.key["expected_outputs"])
    actual = set(ctx.outputs)
    processed = set(ctx.live_evidence.get("processed_stems") or [])
    for block in ctx.unique_code:
        if "k_convert" not in block:
            continue
        for item in items:
            if item["stem"] in block:
                processed.add(item["stem"])
    missing = sorted(
        item["stem"]
        for item in items
        if item["stem"] not in processed
    )
    expected_stems = {item["stem"] for item in items}
    unexpected = sorted(
        set(ctx.live_evidence.get("unexpected_processed_stems") or [])
        | (set(ctx.live_evidence.get("processed_stems") or []) - expected_stems)
    )
    _missing_note = ("，缺 " + "、".join(missing[:10]) + ("…" if len(missing) > 10 else "")) if missing else ""
    _unexpected_note = (
        "，额外处理 " + "、".join(unexpected[:10]) + ("…" if len(unexpected) > 10 else "")
        if unexpected
        else ""
    )
    covered = len({i["stem"] for i in items} - set(missing))
    out.append(Result(
        "R1_all_targets_processed",
        not missing and not unexpected,
        f"期望 {len(items)} 个，live namespace / notebook 已处理 {covered} 个"
        f"{_missing_note}{_unexpected_note}",
        evidence=(
            [*(f"missing: {stem}" for stem in missing),
             *(f"unexpected: {stem}" for stem in unexpected)]
        )[:10],
    ))

    executed_blocks = _executed_notebook_code(ctx)
    fit_calls = _call_count(executed_blocks, "fit_gold")
    out.append(Result("R2_one_gold_fit", fit_calls == 1,
                      f"fit_gold 调用点 {fit_calls} 个（设计要求 1 次拟合后复用）"))

    expected_calls = _expected_call_counts(ctx)
    if expected_calls:
        actual_calls = {
            name: _call_count(executed_blocks, name) for name in expected_calls
        }
        mismatches = {
            name: {"expected": expected_calls[name], "actual": actual_calls[name]}
            for name in expected_calls
            if actual_calls[name] != expected_calls[name]
        }
        out.append(Result(
            "R6_no_redundant_scientific_execution",
            not mismatches,
            "；".join(
                f"{name}={actual_calls[name]}（期望 {expected_calls[name]}）"
                for name in expected_calls
            ),
            evidence=[
                f"{name}: expected {counts['expected']}, actual {counts['actual']}"
                for name, counts in mismatches.items()
            ],
        ))
    else:
        out.append(Result(
            "R6_no_redundant_scientific_execution",
            None,
            "case 未声明 expected_call_counts",
        ))

    discovered_names = {path.name for path in ctx.all_output_paths} or actual
    persisted_images = []
    workspace = workspace_dir(ctx.run_dir)
    if workspace.is_dir():
        persisted_images = [
            str(path.relative_to(workspace))
            for path in workspace.rglob("*")
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg", ".pdf"}
        ]
    unexpected = sorted(discovered_names)
    out.append(Result(
        "R5_no_unexpected_outputs",
        not unexpected and not persisted_images,
        f"落盘分析 NetCDF {len(unexpected)} 个；落盘图片 {len(persisted_images)} 个",
        evidence=(unexpected + persisted_images)[:10],
    ))

    if not ctx.events:
        out.append(Result("R3_append_only", None, NO_AUDIT))
        out.append(Result("R4_execution_success", None, NO_AUDIT))
        return out

    mutation = [e for e in ctx.events if e.get("tool") in MUTATION_TOOLS]
    out.append(Result("R3_append_only", not mutation,
                      f"删改类操作 {len(mutation)} 次（设计是 append-only）"))

    # Count one terminal result per run_cell operation. Backend + wrapper may
    # log the same refusal twice, while an "executed" audit event can still
    # point to a notebook cell whose output is a Python error.
    error_cell_ids = {
        str(cell.get("id"))
        for cell in (ctx.notebook or {}).get("cells", [])
        if cell.get("cell_type") == "code"
        and any(output.get("output_type") == "error" for output in cell.get("outputs", []))
    }
    terminal_by_operation: dict[str, bool] = {}
    code_by_operation: dict[str, str] = {}
    cell_source = {
        str(cell.get("id")): "".join(cell.get("source") or [])
        for cell in (ctx.notebook or {}).get("cells", [])
        if cell.get("cell_type") == "code"
    }
    for position, event in enumerate(ctx.events):
        if event.get("tool") not in RUN_TOOLS:
            continue
        outcome = event.get("outcome")
        details = event.get("details") or {}
        operation = str(details.get("operation_id") or f"legacy-{position}")
        if outcome == "called":
            code_by_operation[operation] = str((details.get("args") or {}).get("code") or "")
            continue
        if outcome not in {"ok", "executed", "error", "blocked", "failed"}:
            continue
        failed = outcome in {"error", "blocked", "failed"}
        if outcome in {"ok", "executed"} and str(details.get("cell_id")) in error_cell_ids:
            failed = True
        if details.get("cell_id") is not None and str(details["cell_id"]) in cell_source:
            code_by_operation[operation] = cell_source[str(details["cell_id"])]
        terminal_by_operation[operation] = terminal_by_operation.get(operation, False) or failed
    err_n = sum(terminal_by_operation.values())
    successful_code = {
        code_by_operation.get(operation) or operation
        for operation, failed in terminal_by_operation.items()
        if not failed
    }
    ok_n = len(successful_code)
    rate = ok_n / (ok_n + err_n) if (ok_n + err_n) else 0.0
    out.append(Result("R4_execution_success", ok_n > 0 and err_n == 0,
                      f"成功 {ok_n} / 失败 {err_n}，成功率 {rate:.0%}"))
    return out


_SKIP_STARTS = (
    "for ", "if ", "while ", "def ", "class ", "import ", "from ", "with ", "try",
    "except", "return", "else", "elif", "del ", "assert ", "raise ", "pass", "break",
    "continue", "global", "#", ")", "]", "}", "@",
)
#: 本来就不该回显的调用（保存、绘图、显示）。
_VOID_CALLS = (
    "save_result",
    "save_with_consent",
    "plot_",
    ".save(",
    "plt.",
    "close(",
    "print(",
)


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


_SUMMARY_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_.])[+-]?\d+(?:\.\d+)?")
_THETA_SUMMARY_LABEL_RE = re.compile(
    r"(?:theta[\s_-]*(?:angular[\s_-]*)?offset|angular[\s_-]*offset|angle[\s_-]*source)",
    re.IGNORECASE,
)
_PROCESSED_STEMS_RECEIPT_RE = re.compile(
    r"(?:^|[;\s])processed_stems=([A-Za-z0-9_.-]+(?:,[A-Za-z0-9_.-]+)*)"
)


def _processed_stems_container_names(source: str) -> set[str]:
    """Return mapping names used to build a processed-stems receipt."""
    tree = _parse_notebook_block(source)
    if tree is None or not any(
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and "processed_stems=" in node.value
        for node in ast.walk(tree)
    ):
        return set()

    names: set[str] = set()
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "sorted"
            and node.args
        ):
            continue
        argument = node.args[0]
        if isinstance(argument, ast.Name):
            names.add(argument.id)
        elif (
            isinstance(argument, ast.Call)
            and isinstance(argument.func, ast.Attribute)
            and argument.func.attr == "keys"
            and isinstance(argument.func.value, ast.Name)
        ):
            names.add(argument.func.value.id)
    return names


def _theta_summary_complete(summary: str, expected_offset: float | None) -> bool:
    """Whether the summary states a theta value and its exact metadata field."""
    lowered = summary.lower()
    if "record.theta_offset_deg" not in lowered:
        return False
    labels = list(_THETA_SUMMARY_LABEL_RE.finditer(summary))
    if not labels:
        return False
    if expected_offset is None:
        return True
    expected = float(expected_offset)
    for label in labels:
        # Keep the number tied to the theta statement so an EF or count elsewhere
        # cannot accidentally satisfy this provenance requirement.
        statement = summary[label.start() : label.end() + 96]
        if any(float(token) == expected for token in _SUMMARY_NUMBER_RE.findall(statement)):
            return True
    return False


def check_show(ctx: Ctx) -> list[Result]:
    out: list[Result] = []
    notebook = ctx.notebook or {}
    texts: list[str] = []
    silent = 0
    suspects = 0
    images = 0
    duplicate_images: list[str] = []
    widget_outputs: list[str] = []
    seen_image_payloads: set[str] = set()
    stdout_line_counts: list[int] = []
    stdout_max_line_lengths: list[int] = []
    processed_stem_receipt_lines: list[str] = []
    processed_stem_container_names: set[str] = set()
    for cell_index, cell in enumerate(notebook.get("cells", []), start=1):
        if cell.get("cell_type") != "code":
            continue
        payload = ""
        stdout = ""
        previous_image_label: str | None = None
        previous_image_order: int | None = None
        for output_index, output in enumerate(cell.get("outputs", []), start=1):
            if not isinstance(output, dict):
                continue
            data = output.get("data") or {}
            if not isinstance(data, dict):
                data = {}
            location = f"cell {cell_index} output {output_index}"
            if _NOTEBOOK_WIDGET_MIME in data:
                widget_outputs.append(location)
            image_data = {
                mime: data[mime]
                for mime in sorted(_NOTEBOOK_STATIC_IMAGE_MIMES)
                if data.get(mime)
            }
            if image_data:
                images += 1
                image_payload = json.dumps(
                    image_data,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                fingerprint = sha256_bytes(image_payload.encode("utf-8"))
                plain_label = _notebook_output_text(data.get("text/plain") or "").strip()
                repeated_payload = fingerprint in seen_image_payloads
                adjacent_figure_label = (
                    bool(plain_label)
                    and plain_label.startswith("<Figure")
                    and plain_label == previous_image_label
                    and previous_image_order == output_index - 1
                )
                if repeated_payload or adjacent_figure_label:
                    duplicate_images.append(location)
                seen_image_payloads.add(fingerprint)
                previous_image_label = plain_label or None
                previous_image_order = output_index
            if output.get("output_type") == "stream":
                stream_text = "".join(output.get("text") or [])
                payload += stream_text
                stdout += stream_text
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
        if cell.get("execution_count") is not None:
            stdout_lines = [line.rstrip() for line in stdout.splitlines() if line.strip()]
            stdout_line_counts.append(len(stdout_lines))
            stdout_max_line_lengths.append(max((len(line) for line in stdout_lines), default=0))
            processed_stem_receipt_lines.extend(
                line for line in stdout_lines if "processed_stems=" in line
            )
            if any("processed_stems=" in line for line in stdout_lines):
                processed_stem_container_names.update(
                    _processed_stems_container_names("".join(cell.get("source", [])))
                )
    joined = "\n".join(texts)
    text_lengths = [len(text) for text in texts]

    gold_str = {str(i) for i in ctx.key["gold_indices"]}
    if not texts:
        out.append(Result("S1_classification_visible", None,
                          "没有 notebook 可读（或 notebook 里没有任何文本输出），无法判断"))
    else:
        hit = any(g in joined for g in gold_str) and ("gold" in joined.lower() or "cut" in joined.lower())
        out.append(Result("S1_classification_visible", hit,
                          "输出里能看到 gold/cut 结论" if hit else
                          "输出里看不到任何分类结论 —— load_experiment 的结果没回传给 agent"))

    plotted = images > 0
    out.append(Result("S2_validation_figure", plotted, f"Notebook 图像输出 {images} 个"))

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
        expected_stems = sorted(item["stem"] for item in ctx.key["expected_outputs"])
        gold_markers = [str(index) for index in ctx.key["gold_indices"]]
        offset = ctx.key.get("theta_offset_deg")
        expected_receipt = "processed_stems=" + ",".join(expected_stems)
        receipt_values = [
            match.group(1)
            for line in processed_stem_receipt_lines
            for match in _PROCESSED_STEMS_RECEIPT_RE.finditer(line)
        ]
        missing_summary_stems = [stem for stem in expected_stems if stem not in summary]
        variable_named = any(
            token in lowered
            for token in ("variable", "notebook", "cell", "result dictionary")
        ) or any(
            re.search(rf"(?<!\w){re.escape(name)}(?!\w)", summary)
            for name in processed_stem_container_names
        )
        receipt_exact = (
            len(processed_stem_receipt_lines) == 1
            and receipt_values == [",".join(expected_stems)]
        )
        receipt_reused = expected_receipt in summary
        variable_issues: list[str] = []
        if missing_summary_stems:
            variable_issues.append(f"missing stems={missing_summary_stems}")
        if not variable_named:
            variable_issues.append("result variable not named")
        if len(processed_stem_receipt_lines) != 1:
            variable_issues.append(
                f"processed_stems receipt count={len(processed_stem_receipt_lines)} (expected 1)"
            )
        elif not receipt_exact:
            variable_issues.append(
                f"processed_stems receipt={receipt_values or ['malformed']}; "
                f"expected={','.join(expected_stems)}"
            )
        if not receipt_reused:
            variable_issues.append("final Markdown did not reuse the exact processed_stems token")
        requirements = {
            "gold": "gold" in lowered and any(marker in summary for marker in gold_markers),
            "fermi": any(token in lowered for token in ("fermi", "ef")),
            "theta": _theta_summary_complete(summary, offset),
            "count": str(len(expected_stems)) in summary
            and any(token in lowered for token in ("cut", "processed")),
            "variables": not missing_summary_stems
            and variable_named
            and receipt_exact
            and receipt_reused,
            "cache": (
                any(
                    token in lowered
                    for token in (
                        "preconverted", "already converted", "netcdf input",
                        "netcdf representation", "needs_conversion",
                        "无需转换", "已转换",
                    )
                )
                if _expected_call_counts(ctx).get("pxt2nc") == 0
                else "cache" in lowered
                and any(
                    token in lowered
                    for token in (
                        "created", "generated", "converted", "completed",
                        "创建", "生成", "转换", "完成",
                    )
                )
            ),
            "validation": any(token in lowered for token in ("validation", "figure", "plot", "验证", "图")),
            "failures": any(
                token in lowered
                for token in ("unprocessed", "failed", "failure", "none", "all requested", "未处理")
            ),
        }
        missing = [name for name, present in requirements.items() if not present]
        detail = (
            "final Markdown summary is complete"
            if not missing
            else f"final Markdown summary is missing: {missing}"
        )
        evidence = list(missing)
        if "variables" in missing:
            detail += "; " + "; ".join(variable_issues)
            evidence.extend(f"missing_stem:{stem}" for stem in missing_summary_stems)
            evidence.extend(variable_issues)
        out.append(Result(
            "S4_final_summary",
            not missing,
            detail,
            evidence=evidence,
        ))
    total_text = sum(text_lengths)
    largest_text = max(text_lengths, default=0)
    final_markdown_chars = len(markdown_cells[-1]) if markdown_cells else 0
    max_stdout_lines = max(stdout_line_counts, default=0)
    max_stdout_line_length = max(stdout_max_line_lengths, default=0)
    noisy_cells = sum(lines > 3 for lines in stdout_line_counts)
    long_line_cells = sum(length > 200 for length in stdout_max_line_lengths)
    semantic_images = images - len(duplicate_images)
    readable = (
        total_text <= 20_000
        and largest_text <= 4_000
        and final_markdown_chars <= _NOTEBOOK_MAX_FINAL_MARKDOWN_CHARS
        and noisy_cells == 0
        and long_line_cells == 0
        and images == _NOTEBOOK_EXPECTED_FIGURES
        and semantic_images == _NOTEBOOK_EXPECTED_FIGURES
        and not duplicate_images
        and not widget_outputs
    )
    out.append(Result(
        "S5_notebook_readable",
        readable,
        f"Notebook code output text {total_text} chars total; largest cell {largest_text} chars; "
        f"final Markdown {final_markdown_chars} chars "
        f"(limit {_NOTEBOOK_MAX_FINAL_MARKDOWN_CHARS}); "
        f"max stdout {max_stdout_lines} non-empty lines; cells above 3 lines {noisy_cells}; "
        f"longest stdout line {max_stdout_line_length} chars; cells above 200 chars {long_line_cells}; "
        f"static figures {images}/{_NOTEBOOK_EXPECTED_FIGURES}; "
        f"semantic figures {semantic_images}/{_NOTEBOOK_EXPECTED_FIGURES}; "
        f"duplicate figure outputs {len(duplicate_images)}; widget outputs {len(widget_outputs)}",
        evidence=([
            f"code cell {index}: {lines} stdout lines; longest {length} chars"
            for index, (lines, length) in enumerate(
                zip(stdout_line_counts, stdout_max_line_lengths, strict=True)
            )
            if lines > 3 or length > 200
        ] + ([
            f"final Markdown: {final_markdown_chars} chars "
            f"(limit {_NOTEBOOK_MAX_FINAL_MARKDOWN_CHARS})"
        ] if final_markdown_chars > _NOTEBOOK_MAX_FINAL_MARKDOWN_CHARS else [])
          + [f"duplicate figure: {location}" for location in duplicate_images]
          + [f"widget: {location}" for location in widget_outputs])[:10],
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
        not paths,
        f"analysis-file persistence is disabled for this experiment; found {len(paths)} processed files; "
        f"misplaced {len(misplaced)}; duplicate locations {len(duplicates)}",
        evidence=(misplaced + duplicates)[:10],
    ))
    in_place = exact

    hard = sorted(
        {name for name, pattern in DIRECT_WRITE_PATTERNS.items() if pattern.search(corpus)}
        | _direct_write_method_names(ctx.unique_code)
    )
    soft = sorted(name for name, pattern in SOFT_WRITE_PATTERNS.items() if pattern.search(corpus))
    out.append(Result("V2_no_direct_disk_write", not hard,
                      "无直写" if not hard else f"出现直写模式：{hard}"
                      + (f"（另有 peaks 自写 .save：{soft}，绕过了 staged 预览）" if soft else "")))

    consents = save_consents(ctx.events)
    if not consents and not paths:
        out.append(Result("V3_consent_trail_complete", True,
                          "no analysis persistence requested, as required"))
    elif not ctx.events:
        out.append(Result("V3_consent_trail_complete", None, NO_AUDIT))
    else:
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


def check_conversion_cache(ctx: Ctx) -> list[Result]:
    """Verify the sole automatic persistence exception: the PXT conversion cache."""
    out: list[Result] = []
    input_dir = Path((ctx.manifest.get("paths") or {}).get("input") or workspace_dir(ctx.run_dir) / "input")
    before = (ctx.manifest.get("frozen") or {}).get("raw_sha256") or {}
    after = build_raw_hashes(input_dir) if input_dir.is_dir() else {}
    if before:
        immutable = after == before
        immutable_detail = (
            f"raw PXT SHA-256 unchanged: {len(after)}/{len(before)}"
            if immutable
            else "raw PXT hash set changed"
        )
        immutable_evidence = sorted(set(before) ^ set(after))[:10]
    else:
        frozen_input = (ctx.manifest.get("frozen") or {}).get("input_manifest") or {}
        current_input = build_input_manifest(input_dir) if input_dir.is_dir() else {}
        immutable = bool(frozen_input) and current_input == frozen_input
        immutable_detail = (
            f"preconverted input manifest unchanged: {len(current_input.get('files') or [])} files"
            if immutable
            else "preconverted input manifest changed"
        )
        immutable_evidence = []
    out.append(Result(
        "V5_raw_inputs_immutable",
        immutable,
        immutable_detail,
        evidence=immutable_evidence,
    ))

    calls = _call_count(_executed_notebook_code(ctx), "pxt2nc")
    reports = list(ctx.live_evidence.get("conversion_reports") or [])
    converted = sum(int(row.get("converted") or 0) for row in reports)
    failed = sum(int(row.get("failed") or 0) for row in reports)
    cache_files = [
        path for path in workspace_dir(ctx.run_dir).rglob("*.nc")
        if path.is_file() and not path.name.endswith("_processed.nc")
    ]
    expected_pxt = _expected_call_counts(ctx).get("pxt2nc")
    if expected_pxt == 0:
        reused = calls == 0 and failed == 0 and bool(cache_files)
        detail = (
            f"preconverted input: pxt2nc call sites {calls}; "
            f"conversion reports {len(reports)}; NetCDF inputs {len(cache_files)}"
        )
    else:
        call_count_ok = calls == expected_pxt if expected_pxt is not None else calls == 1
        reused = call_count_ok and converted > 0 and failed == 0 and bool(cache_files)
        detail = (
            f"pxt2nc call sites {calls}; converted items {converted}; failures {failed}; "
            f"cache files {len(cache_files)}"
        )
    out.append(Result(
        "V6_pxt_cache_reused",
        reused,
        detail,
        evidence=[str(path.relative_to(workspace_dir(ctx.run_dir))) for path in cache_files[:10]],
    ))
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
    workspace_path = workspace_dir(ctx.run_dir).resolve()
    observed_access.extend(
        _outside_notebook_io_paths(_executed_notebook_code(ctx), workspace_path)
    )
    observed_access = sorted(set(observed_access))
    separation_ok = (
        workspace_dir(ctx.run_dir).resolve() != evaluator_dir(ctx.run_dir).resolve()
        and not _is_relative_to(evaluator_dir(ctx.run_dir), workspace_dir(ctx.run_dir))
    )
    mechanism = str(isolation.get("mechanism") or "").strip()
    declared = bool(isolation.get("enforced")) and bool(mechanism)
    # Fallback for trials recorded before the field existed: the same promises
    # are already checkable from the kernel evidence - the managed host is
    # rooted at the trial workspace and serves the trial's own notebook.
    derived: list[str] = []
    kernel_meta = ctx.manifest.get("kernel") or {}
    host_root = str(kernel_meta.get("root_dir") or "")
    live_notebook = str(kernel_meta.get("live_notebook") or "")
    workspace = workspace_path
    if host_root:
        try:
            root_path = Path(host_root).expanduser().resolve()
        except OSError:
            root_path = None
        if root_path is not None and (root_path == workspace or _is_relative_to(workspace, root_path)):
            derived.append("managed host rooted at the trial workspace")
    if live_notebook:
        try:
            live_path = Path(live_notebook).expanduser().resolve()
        except OSError:
            live_path = None
        if live_path is not None and _is_relative_to(live_path, workspace):
            derived.append("live kernel serves the trial notebook")
    if separation_ok:
        derived.append("evaluator outside the workspace")
    enforced = declared or len(derived) >= 2
    if not mechanism and derived:
        mechanism = "; ".join(derived)
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
    if same_dims:
        # Axis order is storage layout, not scientific content.  Align by the
        # named dimensions before shape checks/interpolation so (kx, eV) and
        # (eV, kx) compare as the same product.
        reference = reference.transpose(*data.dims)
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
    required = ("coord_delta", "mask_overlap", "ef_landmark", "kx_landmark", "efficiency", "corr")
    if any(row.get(name) is None for name in required):
        return False
    # Only the criteria the qualification measured as discriminating decide the
    # outcome: the axis extent/centre, the NaN-mask overlap and the physical
    # landmarks. Correlation is also required at a deliberately broad 0.98
    # threshold; it complements rather than replaces coordinate/mask checks,
    # because correlation alone cannot detect every wrong-angle or centre-slice
    # control. Normalised RMSE and shape remain diagnostic only.
    checks = (
        row["coord_delta"] <= float(thresholds.get("coord_delta_max", 3e-3)),
        row["mask_overlap"] >= float(thresholds.get("mask_overlap_min", 0.97)),
        row["ef_landmark"] <= float(thresholds.get("ef_landmark_max", 0.15)),
        row["kx_landmark"] <= float(thresholds.get("kx_landmark_max", 0.05)),
        # Intensity scale, not shape: selecting the centre plane of the scanned
        # deflector axis reproduces the human product's scale (1.00), while
        # integrating that axis is 43.8x brighter.  Correlation cannot see the
        # difference (0.7768 vs 1.0000 only because one file differs), the mask
        # and the grids cannot either.
        float(thresholds.get("efficiency_min", 0.5))
        <= row["efficiency"]
        <= float(thresholds.get("efficiency_max", 2.0)),
        row.get("corr") is not None
        and row["corr"] >= float(thresholds.get("corr_min", 0.98)),
    )
    return all(bool(item) for item in checks)


def _gold_fit_quality(live: dict[str, Any]) -> Result:
    """Validate the balanced Fermi-edge window from the live fit object."""
    if live.get("status") != "ok":
        return Result("Q5_gold_fit_balanced", None, "live gold-fit evidence is unavailable")
    gold_fits = list(live.get("gold_fits") or [])
    balanced: list[dict[str, Any]] = []
    for row in gold_fits:
        window = row.get("fit_window") or {}
        lower = int(window.get("lower_points") or 0)
        upper = int(window.get("upper_points") or 0)
        start = window.get("start_eV")
        center = window.get("center_eV")
        stop = window.get("stop_eV")
        symmetric_energy = all(value is not None for value in (start, center, stop)) and abs(
            (float(center) - float(start)) - (float(stop) - float(center))
        ) <= 1e-6
        if (
            lower == upper
            and lower >= 8
            and symmetric_energy
            and float(row.get("outlier_fraction", 1.0)) <= 0.05
            and row.get("uniform") is True
        ):
            balanced.append(row)
    return Result(
        "Q5_gold_fit_balanced",
        bool(balanced),
        (
            f"{len(balanced)}/{len(gold_fits)} live gold fits have equal plateaus, "
            "uniform EF and <=5% outliers"
        ),
        evidence=[str(row)[:300] for row in gold_fits if row not in balanced][:3],
    )


def check_quality(ctx: Ctx) -> list[Result]:
    """结果正确性 —— 最终目标，不属于任何单个子系统。"""
    out: list[Result] = [_gold_fit_quality(ctx.live_evidence)]
    expected_names = [item["output_name"] for item in ctx.key["expected_outputs"]]
    expected_stems = [item["stem"] for item in ctx.key["expected_outputs"]]
    live = ctx.live_evidence
    if live.get("status") == "ok":
        products = live.get("products") or {}
        found = [stem for stem in expected_stems if stem in products]
        missing = sorted(set(expected_stems) - set(found))
        k_dims = {"kx", "k_par", "kp", "kparallel", "kx_par"}
        dims_ok = [stem for stem in found if k_dims & set(products[stem].get("dims") or [])]
        ef_ok = [
            stem for stem in found
            if products[stem].get("ef_landmark") is not None
            and float(products[stem]["ef_landmark"]) <= 0.15
        ]
        theta_ok = [
            stem for stem in found
            if products[stem].get("kx_landmark") is not None
            and float(products[stem]["kx_landmark"]) <= 0.05
        ]
        total = len(expected_stems)
        out.append(Result("Q1_kspace_dims", len(dims_ok) == total,
                          f"{len(dims_ok)}/{total} live products contain a k-space dimension",
                          evidence=missing[:10]))
        out.append(Result("Q2_ef_zeroed", len(ef_ok) == total,
                          f"{len(ef_ok)}/{total} live products cross EF=0"))
        out.append(Result("Q3_theta_zeroed", len(theta_ok) == total,
                          f"{len(theta_ok)}/{total} live products cross kx=0"))
        oracle = _q4_oracle()
        if oracle is None:
            out.append(Result("Q4_matches_human_reference", None,
                              "qualified human-reference oracle is unavailable"))
        else:
            thresholds = {**(oracle.get("thresholds") or {}), "corr_min": 0.98}
            rows = [products[stem] for stem in found]
            failures = [row for row in rows if row.get("reference_missing") or not _reference_matches(row, thresholds)]
            passed = len(found) == total and not failures
            failed_stems = [stem for stem in found if products[stem] in failures]
            q4_evidence = [f"{stem}: missing live product" for stem in missing[:5]]
            q4_evidence.extend(
                f"{stem}: corr={products[stem].get('corr')} coordΔ={products[stem].get('coord_delta')} "
                f"mask={products[stem].get('mask_overlap')}"
                for stem in failed_stems[:5]
            )
            out.append(Result(
                "Q4_matches_human_reference",
                passed,
                f"{total - len(missing) - len(failures)}/{total} live products satisfy coordinates, mask, landmarks and corr>=0.98",
                evidence=q4_evidence,
            ))
        return out
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
                aligned_ref = ref.transpose(*data.dims) if same_dims else ref
                coords_close = all(
                    np.allclose(np.asarray(data.coords[d]), np.asarray(aligned_ref.coords[d]),
                                rtol=1e-3, atol=1e-3)
                    for d in dims if d in aligned_ref.coords
                )
                values_close = bool(
                    data.shape == aligned_ref.shape
                    and np.allclose(
                        np.asarray(data.values, dtype=float),
                        np.asarray(aligned_ref.values, dtype=float),
                        rtol=1e-2,
                        atol=1e-2,
                        equal_nan=True,
                    )
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
        measured_names = {str(row.get("name")) for row in ref_metrics}
        missing_metrics = sorted(set(reference_available) - measured_names)
        failures = [
            row for row in ref_metrics
            if not _reference_matches(row, thresholds)
        ]
        passed = not failures and not missing_references and not missing_metrics
        matched = len(ref_metrics) - len(failures)
        out.append(Result(
            "Q4_matches_human_reference",
            passed,
            f"{matched}/{total} expected products match the human "
            f"reference within the qualified oracle {oracle.get('generated_at', '?')[:19]}; "
            f"missing references {len(missing_references)}"
            + (f": {missing_references[:5]}" if missing_references else "")
            + f"; metric errors {len(missing_metrics)}"
            + (f": {missing_metrics[:5]}" if missing_metrics else ""),
            evidence=([
                f"{row['name']}: coordΔ={row['coord_delta']:.2e} "
                f"mask={row['mask_overlap']:.3f} ef={row['ef_landmark']:.3f} "
                f"kx={row['kx_landmark']:.3f} scale={row['efficiency']:.3f} "
                f"(recorded: corr={row['corr']:.4f} nrmse={row['nrmse']:.4f})"
                for row in failures[:5]
            ] + [f"missing metrics: {name}" for name in missing_metrics[:5]]) or notes[:5],
        ))
    return out


# --------------------------------------------------------------------------- #
# 打分与报告                                                                   #
# --------------------------------------------------------------------------- #

def _rubric_document() -> dict[str, Any]:
    return load_yaml(RUBRIC_FILE)


def rubric_version() -> str:
    return str(_rubric_document().get("version", "?"))


def _normalized_error(text: str) -> str:
    """Collapse run-specific paths and identifiers into comparable failure text."""
    value = re.sub(r"(?:/[^\s,;:]+)+", "<PATH>", str(text))
    value = re.sub(r"\b[0-9a-f]{12,}\b", "<ID>", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value).strip()
    return value[:500]


def _canonical_ids_from_events(events: list[dict[str, Any]]) -> list[str]:
    ids: set[str] = set()
    for event in events:
        args = (event.get("details") or {}).get("args") or {}
        values = args.get("canonical_ids") or args.get("api_ids") or args.get("canonical_id")
        if isinstance(values, str):
            ids.add(values)
        elif isinstance(values, list):
            ids.update(str(value) for value in values)
    return sorted(ids)


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


#: A trial that executed fewer code cells than this never engaged with the task
#: (observed: two campaign trials ran a single cell each and were still counted
#: as valid samples, which flatters nothing but corrupts the statistics).
MIN_EXECUTED_CELLS = 3


def assess_validity(
    run_dir: Path,
    manifest: dict[str, Any],
    notebook_path: Path | None,
    *,
    events: list[dict[str, Any]] | None = None,
    executed_cells: int | None = None,
) -> dict[str, Any]:
    """Evaluate whether this trial can support the strict endpoint.

    ``events`` and ``executed_cells`` turn the two "did anything happen at all"
    promises into checks: a trial with an empty audit window or a single
    executed cell cannot support any endpoint, however clean its paths are.
    """
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

    if events is not None:
        record(
            "agent_activity",
            len(events) > 0,
            f"audit events inside the trial window: {len(events)}",
        )
    if executed_cells is not None:
        record(
            "executed_cells",
            executed_cells >= MIN_EXECUTED_CELLS,
            f"executed code cells: {executed_cells} (minimum {MIN_EXECUTED_CELLS})",
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
            "scientific_contract": dict(case.get("scientific_contract") or {}),
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
            "common_source": (
                None if condition in STANDALONE_PROMPT_CONDITIONS else str(COMMON_PROMPT_FILE)
            ),
            "common_source_sha256": (
                None
                if condition in STANDALONE_PROMPT_CONDITIONS
                else sha256_file(COMMON_PROMPT_FILE)
            ),
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
            "raw_sha256": build_raw_hashes(input_dir),
            "reference_manifest": build_input_manifest(reference_dir)
            if reference_dir and reference_dir.is_dir()
            else None,
            "source": _git_state(),
            "peaks": _imported_peaks_git_state(),
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
              all_output_paths=all_output_paths, notebook_path=notebook_path,
              live_evidence=_json_from_run(run_dir, "live_evidence.json"),
              client_tool_evidence=collect_pi_client_tool_evidence(run_dir))

    results: list[Result] = []
    for group in (check_contract, check_access, check_client_tools, check_run, check_show,
                  check_save, check_conversion_cache, check_observability,
                  check_autonomy, check_quality):
        results.extend(group(ctx))

    rubric = load_rubric()
    scorecard = score(results, rubric)
    executed_cells = sum(
        1
        for cell in (notebook or {}).get("cells", [])
        if cell.get("cell_type") == "code" and cell.get("execution_count") is not None
    )
    validity = assess_validity(
        run_dir,
        manifest,
        notebook_path,
        events=events,
        executed_cells=executed_cells,
    )
    result_map = {result.check: result.passed for result in results}
    strict_checks = list(_rubric_document().get("strict_checks") or [])
    if _q4_oracle() is None:
        # Uncalibrated oracle: Q4 reports "inconclusive", and a check that
        # cannot decide must not be able to fail a strict success.
        strict_checks = [name for name in strict_checks if name != "Q4_matches_human_reference"]
    strict_success = validity["valid"] and all(result_map.get(check) is True for check in strict_checks)
    report = render_report(run_dir, results, rubric, scorecard, ctx, validity, strict_success)

    primary = next(
        (result for check in strict_checks for result in results
         if result.check == check and result.passed is not True),
        None,
    )
    primary_failure = None
    if primary is not None:
        primary_failure = {
            "check_id": primary.check,
            "stage": rubric.get(primary.check, {}).get("subsystem", "Other"),
            "canonical_api_ids": _canonical_ids_from_events(events),
            "normalized_error": _normalized_error(primary.detail),
        }

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
        "check_map": result_map,
        "primary_failure": primary_failure,
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
    processed_stems = ",".join(sorted(item["stem"] for item in key["expected_outputs"]))
    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": ["# 参考流水线（自检用）\n"]},
        {
            "cell_type": "code",
            "execution_count": 1,
            "metadata": {},
            "outputs": [{
                "output_type": "stream", "name": "stdout",
                "text": ["load_experiment: records=3; needs_conversion=3\n"],
            }],
            "source": [
                "import peaks\n",
                f"initial_experiment = peaks.load_experiment(r'{input_dir}')\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": 2,
            "metadata": {},
            "outputs": [{"output_type": "stream", "name": "stdout",
                         "text": ["pxt2nc: converted=3; cached=0; failed=0\n"]}],
            "source": [f"conversion = peaks.pxt2nc(r'{input_dir}')\n"],
        },
        {
            "cell_type": "code",
            "execution_count": 3,
            "metadata": {},
            "outputs": [{"output_type": "stream", "name": "stdout",
                         "text": [f"gold={key['gold_indices']} cuts={key['cut_indices']} "
                                  f"theta_offset={offset}\n"]}],
            "source": [
                "experiment = peaks.load_experiment(conversion.destination)\n",
                "theta_offsets = {int(r.index): r.theta_offset_deg for r in experiment.records}\n",
                "print(f\"gold={experiment.gold} cuts={experiment.cuts} \"\n",
                "      f\"theta_offset={theta_offsets[experiment.cuts[0]]}\")\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": 4,
            "metadata": {},
            "outputs": [{
                "output_type": "display_data",
                "data": {
                    "image/png": "Z29sZC1kaWFnbm9zdGlj",
                    "text/plain": "<Figure size 1200x900 with 7 Axes>",
                },
                "metadata": {},
            }],
            "source": [
                "gold_index = experiment.gold[0]\n",
                "gold = experiment[gold_index]\n",
                "fit = gold.fit_gold(plot=True, show=False)\n",
                "# Gold-fit diagnostic rendered once.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": 5,
            "metadata": {},
            "outputs": [
                {
                    "output_type": "stream",
                    "name": "stdout",
                    "text": [f"processed_stems={processed_stems}\n"],
                },
                {
                    "output_type": "display_data",
                    "data": {
                        "image/png": "Y3V0LWdyaWQ=",
                        "text/plain": "<Figure size 1500x720 with 15 Axes>",
                    },
                    "metadata": {},
                },
                {
                    "output_type": "display_data",
                    "data": {
                        "image/png": "YmVmb3JlLWFmdGVy",
                        "text/plain": "<Figure size 1100x400 with 2 Axes>",
                    },
                    "metadata": {},
                },
            ],
            "source": [
                "cut_results = {}\n",
                f"for stem in [{cuts}]:\n",
                "    da = experiment[stem]\n",
                "    offset = theta_offsets[int(stem[-4:])]\n",
                "    shifted = da.metadata.assign_normal_emission(theta_par=offset)\n",
                "    cut_results[stem] = shifted.k_convert(EF_correction=fit, quiet=True)\n",
                "print('processed_stems=' + ','.join(sorted(cut_results)))\n",
                "# One all-cut grid and one before/after figure rendered inline.\n",
            ],
        },
        {"cell_type": "markdown", "metadata": {},
         "source": [
             "## Final summary\n\n",
             f"Gold record: {gold}, selected from the experiment metadata classification.\n\n",
             "Fermi correction: EF was fitted from the gold during this trial (c0=2.65).\n\n",
             f"Theta angular offset: {offset} degrees from record.theta_offset_deg.\n\n",
             f"Processed {len(key['cut_indices'])} cuts in result dictionary cut_results. "
             f"processed_stems={processed_stems}.\n\n",
             "Cache: conversion completed and generated validated NetCDF entries.\n\n",
             "Unprocessed targets: none. Inline figure validation completed; no analysis results were persisted.\n",
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
    cut_list = ", ".join(f"'BP_{i:04d}'" for i in key["cut_indices"])
    return [
        "import peaks\n"
        f"initial_experiment = peaks.load_experiment(r'{input_dir}')\n",
        f"conversion = peaks.pxt2nc(r'{input_dir}')\n",
        "experiment = peaks.load_experiment(conversion.destination)\n"
        "theta_offsets = {int(r.index): r.theta_offset_deg for r in experiment.records}\n"
        "print(f\"gold={experiment.gold} cuts={experiment.cuts}\")\n",
        "gold_index = experiment.gold[0]\n"
        "gold = experiment[gold_index]\n"
        "fit = gold.fit_gold(plot=True, show=False)\n",
        f"for stem in [{cut_list}]:\n"
        "    da = experiment[stem]\n"
        "    offset = theta_offsets[int(stem[-4:])]\n"
        "    shifted = da.metadata.assign_normal_emission(theta_par=offset)\n"
        "    globals()[f'{stem}_kspace'] = shifted.k_convert(EF_correction=fit, quiet=True)\n",
    ]


def golden_audit_events(input_dir: str, output_dir: Path, key: dict[str, Any],
                        *, poisoned: str | None = None) -> list[dict[str, Any]]:
    """黄金审计痕迹；``poisoned`` 注入一种缺陷供负向对照。"""
    code = golden_code_blocks(input_dir, key)
    if poisoned == "hardcoded_gold":
        code[-2] = (
            "gold = experiment[20]\n"
            "fit = gold.fit_gold(plot=True, show=False)\n"
        )
    if poisoned == "hardcoded_theta":
        code[-1] = code[-1].replace(
            "offset = theta_offsets[int(stem[-4:])]",
            f"offset = {key['theta_offset_deg']}",
        )
    if poisoned == "duplicate_science":
        code.append(
            "pilot_record = next(r for r in experiment.records "
            "if r.index == experiment.cuts[0])\n"
            "pilot = experiment[pilot_record.index].metadata.assign_normal_emission("
            "theta_par=pilot_record.theta_offset_deg)\n"
            "pilot_k = pilot.k_convert(EF_correction=fit, quiet=True)\n",
        )
    events: list[dict[str, Any]] = []
    for api, cid in (
        ("pxt2nc", "top_level:peaks.core.fileIO.experiment:pxt2nc"),
        ("load_experiment", "top_level:peaks.core.fileIO.experiment:load_experiment"),
        ("fit_gold", "dataarray:peaks.core.fitting.fit:fit_gold"),
        ("assign_normal_emission", "metadata:peaks.core.metadata.metadata_methods:assign_normal_emission"),
        ("k_convert", "dataarray:peaks.core.process.k_conversion:k_convert"),
    ):
        if poisoned != "bypass_get" or api != "assign_normal_emission":
            events.append(_audit_event("get", "ok", {"args": {"canonical_id": cid}}))
    for position, block in enumerate(code, start=1):
        op = f"run-{position:02d}"
        if poisoned == "direct_write" and position == len(code):
            # 真实形态：agent 把 to_netcdf 写进了 cell → run_cell 硬阻止（error 事件带
            # persist 提示），该 cell 不执行；V2 必须红，A2 必须保持绿。
            block = "kd.to_netcdf(r'out.nc')\n" + block
            events.append(_audit_event("run_cell", "called",
                                       {"operation_id": op, "args": {"code": block}}))
            events.append(_audit_event("run_cell", "error", {
                "operation_id": op,
                "error": "Execution blocked: this cell writes a file (savefig / file writers / "
                         "unclear file mode), and run_cell is never a persistence path. Results "
                         "must remain in notebook variables and inline outputs",
            }))
            for extra in range(4, 13):  # 补足独立成功操作：只让 V2 翻红
                op = f"run-{extra:02d}"
                events.append(_audit_event("run_cell", "called",
                                           {"operation_id": op, "args": {"code": f"kd_{extra} = da.mean()"}}))
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
        for position in range(4, 13):  # 补足独立成功操作，只让 A2 翻红
            op = f"run-{position:02d}"
            events.append(_audit_event("run_cell", "called",
                                       {"operation_id": op, "args": {"code": f"kd_{position} = da.mean()"}}))
            events.append(_audit_event("run_cell", "executed",
                                       {"operation_id": op, "cell_id": f"c{position}"}))
    if poisoned == "exec_error":
        events.append(_audit_event("run_cell", "called",
                                   {"operation_id": "run-97", "args": {"code": "data['missing_dim']"}}))
        events.append(_audit_event("run_cell", "error", {
            "operation_id": "run-97", "error": "KeyError: 'missing_dim'"}))
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
    raw_paths = [input_dir / "BP_0020.pxt"]
    for item in key["expected_outputs"]:
        raw_paths.append(input_dir / f"{item['stem']}.pxt")
        _write_synthetic_product(reference_dir / item["output_name"], item["index"])
    for position, path in enumerate(raw_paths):
        path.write_bytes(f"synthetic-pxt-{position}\n".encode())
    raw_hashes = build_raw_hashes(input_dir)

    cache_dir = workspace / "input_netcdf"
    cache_dir.mkdir()
    for position, path in enumerate(raw_paths):
        _write_synthetic_product(cache_dir / f"{path.stem}.nc", position)
    if poisoned == "cache_miss":
        for path in cache_dir.glob("*.nc"):
            path.unlink()
    if poisoned == "raw_tamper":
        raw_paths[0].write_bytes(b"tampered synthetic PXT\n")
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
    if poisoned == "hardcoded_gold":
        notebook["cells"][4]["source"] = [
            "gold = experiment[20]\n",
            "fit = gold.fit_gold(plot=True, show=False)\n",
            "# Gold-fit diagnostic rendered once.\n",
        ]
    if poisoned == "hardcoded_theta":
        notebook["cells"][5]["source"] = [
            line.replace(
                "offset = theta_offsets[int(stem[-4:])]",
                f"offset = {key['theta_offset_deg']}",
            )
            for line in notebook["cells"][5]["source"]
        ]
    if poisoned == "duplicate_science":
        notebook["cells"].insert(-1, {
            "cell_type": "code",
            "execution_count": 6,
            "metadata": {},
            "outputs": [],
            "source": [
                "pilot_record = next(r for r in experiment.records "
                "if r.index == experiment.cuts[0])\n",
                "pilot = experiment[pilot_record.index].metadata.assign_normal_emission("
                "theta_par=pilot_record.theta_offset_deg)\n",
                "pilot_k = pilot.k_convert(EF_correction=fit, quiet=True)\n",
            ],
        })
    figure_outputs = notebook["cells"][5]["outputs"]
    if poisoned == "widget_noise":
        figure_outputs.append({
            "output_type": "display_data",
            "data": {
                _NOTEBOOK_WIDGET_MIME: {
                    "version_major": 2,
                    "version_minor": 0,
                    "model_id": "progress-widget",
                },
                "text/plain": "Converting data to k-space: 0%",
            },
            "metadata": {},
        })
    if poisoned == "duplicate_figure":
        image_output = next(
            output
            for output in figure_outputs
            if any((output.get("data") or {}).get(mime) for mime in _NOTEBOOK_STATIC_IMAGE_MIMES)
        )
        figure_outputs.append(json.loads(json.dumps(image_output)))
    if poisoned == "excess_figure":
        figure_outputs.append({
            "output_type": "display_data",
            "data": {
                "image/png": "ZXh0cmEtZmlndXJl",
                "text/plain": "<Figure size 640x480 with 1 Axes>",
            },
            "metadata": {},
        })
    if poisoned == "oversized_final_markdown":
        summary_cell = notebook["cells"][-1]
        summary = "".join(summary_cell["source"])
        summary_cell["source"].append(
            "x" * (_NOTEBOOK_MAX_FINAL_MARKDOWN_CHARS + 1 - len(summary))
        )
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

    session_dir = root / "agent" / "session"
    session_dir.mkdir()
    pi_call_id = "synthetic-health-check"
    pi_tool = "peaks_status" if poisoned == "wrapper_tool_error" else "inspect_notebook"
    pi_args = {} if poisoned == "wrapper_tool_error" else {"target": "kernel"}
    pi_call = {
        "type": "message",
        "message": {
            "role": "assistant",
            "content": [{
                "type": "toolCall",
                "id": pi_call_id,
                "name": "mcp",
                "arguments": {"tool": pi_tool, "args": pi_args},
            }],
        },
    }
    if poisoned == "wrapper_tool_error":
        pi_result = {
            "type": "message",
            "message": {
                "role": "toolResult",
                "toolCallId": pi_call_id,
                "toolName": "mcp",
                "content": [{
                    "type": "text",
                    "text": (
                        'Tool "peaks_status" not found. Use mcp({ search: "..." }) '
                        "to search."
                    ),
                }],
                "details": {
                    "mode": "call",
                    "error": "tool_not_found",
                    "requestedTool": "peaks_status",
                },
                "isError": False,
            },
        }
    else:
        pi_result = {
            "type": "message",
            "message": {
                "role": "toolResult",
                "toolCallId": pi_call_id,
                "toolName": "mcp",
                "content": [{"type": "text", "text": '{"kernel":"ready"}'}],
                "details": {"mode": "call"},
                "isError": False,
            },
        }
    (session_dir / "synthetic.jsonl").write_text(
        "\n".join(json.dumps(row) for row in (pi_call, pi_result)) + "\n",
        encoding="utf-8",
    )

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
            "scientific_contract": {
                "expected_call_counts": {
                    "pxt2nc": 1,
                    "load_experiment": 2,
                    "fit_gold": 1,
                    "assign_normal_emission": 1,
                    "k_convert": 1,
                }
            },
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
        "frozen": {"raw_sha256": raw_hashes},
        "audit": {
            "path": str(audit_path),
            "start_offset": 0,
            "end_offset": audit_path.stat().st_size,
        },
        "approval": {"mode": "synthetic"},
        "agent": {"runner": "pi"},
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
    product_metrics = {
        item["stem"]: {
            "variable": f"{item['stem']}_kspace",
            "dims": ["eV", "kx"],
            "shape": [3, 3],
            "ef_landmark": 0.0,
            "kx_landmark": 0.0,
            "same_dims": True,
            "coord_delta": 0.0,
            "mask_overlap": 1.0,
            "efficiency": 1.0,
            "corr": 1.0,
        }
        for item in key["expected_outputs"]
    }
    atomic_write_json(
        evaluator / "live_evidence.json",
        {
            "status": "ok",
            "processed_stems": sorted(product_metrics),
            "products": product_metrics,
            "conversion_reports": [
                {"converted": len(raw_paths), "cached": 0, "failed": 0},
            ],
            "gold_fits": [{
                "variable": "gold_fit",
                "fit_window": {
                    "start_eV": -0.1,
                    "center_eV": 0.0,
                    "stop_eV": 0.1,
                    "lower_points": 16,
                    "upper_points": 16,
                    "total_points": 33,
                },
                "outlier_fraction": 0.01,
                "uniform": True,
            }],
        },
    )
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
        "direct_write": {"V2_no_direct_disk_write", "R4_execution_success"},
        "delete_cell": {"R3_append_only"},
        "api_block": {"A2_no_unknown_api_blocks", "R4_execution_success"},
        "exec_error": {"R4_execution_success"},
        "bypass_get": {"A1_get_before_use"},
        "human_guidance": {"U1_no_assistive_intervention"},
        "unexpected_output": {
            "R5_no_unexpected_outputs",
            "V1_outputs_in_place",
            "V3_consent_trail_complete",
        },
        "raw_tamper": {"V5_raw_inputs_immutable"},
        "cache_miss": {"V6_pxt_cache_reused"},
        "hardcoded_gold": {"C3_gold_index_correct"},
        "hardcoded_theta": {"C4_theta_offset_from_contract"},
        "duplicate_science": {"R6_no_redundant_scientific_execution"},
        "widget_noise": {"S5_notebook_readable"},
        "duplicate_figure": {"S5_notebook_readable"},
        "excess_figure": {"S5_notebook_readable"},
        "oversized_final_markdown": {"S5_notebook_readable"},
        "wrapper_tool_error": {"A5_no_client_tool_errors"},
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
