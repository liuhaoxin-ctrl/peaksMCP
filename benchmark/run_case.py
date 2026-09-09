#!/usr/bin/env python3
"""End-to-end agent benchmark: 让一个 agent 真的把一批 ARPES cut 预处理完，然后打分。

三次命令完成一轮实验::

    # 1) 准备一次运行（生成输入、答案键、空 notebook、渲染好的提示词）
    python benchmark/run_case.py init --name bp260623-p1

    # 2) 把 prompt.txt 粘给任何配好 MCP 的 agent，让它跑完（人只负责点同意）

    # 3) 打分
    python benchmark/run_case.py grade runs/<run-id>

设计要点（决定了这份基准能不能真的用起来）：

* **评分只看痕迹**。审计日志 + 产物 + notebook，不读任何 agent 的对话记录。
  所以 pi-agent / Claude Desktop / WorkBuddy / 任何 agent 都能用同一套。
* **答案键独立于被测系统**。自己解析 datasheet，绝不调用 ``inspect_experiment``
  来生成期望值——否则分类有 bug 时基准会自己给自己打满分。
* **每条失败都指向一个子系统**，并带上 rubric.yaml 里的 fix 文案：
  跑完就知道该改哪里，而不是只知道"分数低"。

Usage
-----
    run_case.py init  [--name NAME] [--data DIR] [--datasheet CSV] [--reference DIR]
                      [--limit N] [--runs DIR] [--copy]
    run_case.py grade RUN_DIR [--notebook PATH] [--output DIR] [--audit PATH]
                      [--json] [--quiet]
    run_case.py compare A.json B.json
"""

from __future__ import annotations

import argparse
import csv
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
TASK_FILE = BENCH_DIR / "task.md"

DEFAULT_DATA = Path("/Users/haoxin/Documents/实验数据/BP260623/data")
DEFAULT_REFERENCE = Path("/Users/haoxin/Documents/实验数据/BP260623/data_netcdf")

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

#: 不应被当成"peaks API"的通用方法名。
COMMON_METHODS = frozenset(
    """assign_coords sel isel mean sum max min plot imshow pcolormesh contourf
    transpose squeeze dropna isnull fillna astype copy values item reshape
    expand_dims rename stack unstack to_numpy compute load close append format
    join split strip replace startswith endswith get keys items values""".split()
)


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

    stems = {p.stem for p in data_dir.iterdir() if p.suffix.lower() in {".pxt", ".nc"}}
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

def load_events(audit_path: Path, since: str | None) -> list[dict[str, Any]]:
    """读审计日志，只保留 since 之后的事件。"""
    if not audit_path.is_file():
        return []
    events: list[dict[str, Any]] = []
    with audit_path.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
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
    """实际执行过的代码：优先取审计日志，缺失时回落到 notebook。"""
    blocks: list[str] = []
    for event in events:
        if event.get("outcome") != "called":
            continue
        if event.get("tool") not in RUN_TOOLS:
            continue
        code = (event.get("details") or {}).get("args", {}).get("code")
        if code:
            blocks.append(str(code))
    if not blocks:
        blocks = list(notebook_code)
    return blocks


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
    key: dict[str, Any]
    events: list[dict[str, Any]]
    code: list[str]
    fetched: set[str]
    outputs: dict[str, Path]
    notebook: dict[str, Any] | None
    reference_dir: Path | None
    audit_path: Path

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


def _fit_gold_receiver_indices(blocks: list[str]) -> tuple[set[int], bool]:
    """Resolve which experiment index each fit_gold call actually uses.

    Looks ONLY at the receiver of ``*.fit_gold(...)``:

    - chained literal:  ``scans['BP_0020'].fit_gold()`` / ``scans["BP_0020"]...``
    - bound variable:   ``gold = scans['BP_0020']`` followed by ``gold.fit_gold()``
      (binding may live anywhere earlier in the same block, incl. an alias var)
    - alias rename:     ``g = scans[...]; gold = g`` etc. (single hop)

    Numbers elsewhere in the block (e.g. a ``cuts = [...]`` list literal in the
    same cell) are deliberately IGNORED - scanning the whole block would flag
    correct code as wrong.  Returns ``(indices, unresolved)`` where
    ``unresolved=True`` when a fit call exists whose receiver cannot be
    traced to a literal index (e.g. a loop over ``summary.gold``).
    """
    indices: set[int] = set()
    unresolved = False
    for block in blocks:
        lines = (block or "").splitlines()
        for line in lines:
            if "fit_gold" not in line:
                continue
            direct = re.findall(r"BP_0*(\d{1,4})", line)
            if direct:
                indices.update(int(value) for value in direct)
                continue
            match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*fit_gold\s*\(", line)
            if not match:
                continue
            receiver = match.group(1)
            found_here = False
            for other in lines:
                binding = re.search(
                    rf"\b{re.escape(receiver)}\s*=\s*(?:[^;]*?)scans\s*\[\s*['\"]?BP_?0*(\d+)['\"]?\s*\]",
                    other,
                )
                if binding:
                    indices.add(int(binding.group(1)))
                    found_here = True
                    break
            if not found_here:
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
    found, unresolved = _fit_gold_receiver_indices(gold_blocks)
    wanted = set(ctx.key["gold_indices"])
    if not gold_blocks:
        out.append(Result("C3_gold_index_correct", False, "没有出现 fit_gold，无法判断 gold 选择"))
    elif unresolved and not found:
        out.append(Result("C3_gold_index_correct", None,
                          "fit_gold 的 receiver 无法回溯到字面索引（如遍历 summary.gold），跳过"))
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
    out.append(Result("R1_all_targets_processed", not missing,
                      f"期望 {len(expected_names)} 个，实到 {len(actual & expected_names)} 个",
                      evidence=missing[:10]))

    fit_calls = sum(len(re.findall(r"fit_gold\s*\(", block)) for block in ctx.unique_code)
    out.append(Result("R2_one_gold_fit", fit_calls <= 1,
                      f"fit_gold 调用点 {fit_calls} 个（设计要求 1 次拟合后复用）"))

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
    in_place = sorted(expected_names & set(ctx.outputs))
    out.append(Result("V1_outputs_in_place", bool(in_place),
                      f"输出目录里命中的期望产物 {len(in_place)}/{len(expected_names)}"))

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
    out.append(Result("O2_cell_artifact_linkage", bool(linked),
                      f"带 cell/产物标识的事件 {len(linked)} 条 —— "
                      + ("可重建链路" if linked else "事后无法把产物溯源到某次执行")))

    gets = [e for e in ctx.events if e.get("tool") in GET_TOOLS]
    out.append(Result("O3_api_call_trail", bool(gets), f"get/peaks_get_api 调用 {len(gets)} 次"))
    return out


def check_quality(ctx: Ctx) -> list[Result]:
    """结果正确性 —— 最终目标，不属于任何单个子系统。"""
    out: list[Result] = []
    if not ctx.outputs:
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
    notes: list[str] = []
    for name, path in sorted(ctx.outputs.items()):
        try:
            data = xr.open_dataarray(path)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"{name}: 打不开 {exc}")
            continue
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
            try:
                ref = xr.open_dataarray(reference)
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
                if same_dims and coords_close and values_close:
                    ok_ref.append(name)
                else:
                    notes.append(
                        f"{name}: 与人工参考不一致（dims={'同' if same_dims else '异'}, "
                        f"coords={'近' if coords_close else '异'}, values={'近' if values_close else '异'}）"
                    )
            except Exception as exc:  # noqa: BLE001
                notes.append(f"{name}: 参考比对失败 {exc}")

    total = len(ctx.outputs)
    out.append(Result("Q1_kspace_dims", len(ok_dims) == total,
                      f"{len(ok_dims)}/{total} 个产物含 k 空间维度", evidence=notes[:5]))
    out.append(Result("Q2_ef_zeroed", len(ok_ef) == total,
                      f"{len(ok_ef)}/{total} 个产物 EF 已归零"))
    out.append(Result("Q3_theta_zeroed", len(ok_theta) == total,
                      f"{len(ok_theta)}/{total} 个产物高对称点已归零"))
    out.append(Result("Q4_matches_human_reference", bool(ok_ref) and len(ok_ref) == total,
                      f"{len(ok_ref)}/{total} 个产物与人工参考一致", evidence=notes[:8]))
    return out


# --------------------------------------------------------------------------- #
# 打分与报告                                                                   #
# --------------------------------------------------------------------------- #

def _rubric_document() -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        return {}
    return yaml.safe_load(RUBRIC_FILE.read_text(encoding="utf-8")) or {}


def rubric_version() -> str:
    return str(_rubric_document().get("version", "?"))


def load_rubric() -> dict[str, dict[str, Any]]:
    return _rubric_document().get("checks") or {}


def collect_outputs(run_dir: Path, output_dir: Path | None, key: dict[str, Any]) -> dict[str, Path]:
    """产物可以来自约定输出目录，也可以散在 run 目录里。"""
    found: dict[str, Path] = {}
    search_roots = [p for p in (output_dir, run_dir / "output", run_dir) if p and Path(p).is_dir()]
    for root in search_roots:
        for path in Path(root).rglob("*_processed.nc"):
            found.setdefault(path.name, path)
    return found


def load_notebook(path: Path | None, run_dir: Path) -> tuple[dict[str, Any] | None, list[str], Path | None]:
    if path is not None:
        candidates = [Path(path)]
    else:
        candidates = sorted(run_dir.rglob("*.ipynb"), key=lambda p: p.stat().st_mtime, reverse=True)
        candidates += sorted((run_dir.parent).rglob("*.ipynb"),
                             key=lambda p: p.stat().st_mtime, reverse=True)[:5]
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
    """按子系统汇总加权得分。

    ``passed is None``（无法判定，例如没有 notebook 或缺 xarray）既不加权也不扣分，
    否则"缺证据"会被算成"做得差"，把分数拖到没有意义。
    """
    by_subsystem: dict[str, dict[str, float]] = {}
    for result in results:
        meta = rubric.get(result.check, {})
        subsystem = meta.get("subsystem", "Other")
        weight = float(meta.get("weight", 1))
        bucket = by_subsystem.setdefault(
            subsystem, {"weight": 0.0, "earned": 0.0, "failed": 0.0, "skipped": 0.0}
        )
        if result.passed is None:
            bucket["skipped"] += 1
            continue
        bucket["weight"] += weight
        if result.passed:
            bucket["earned"] += weight
        else:
            bucket["failed"] += 1
    summary: dict[str, Any] = {}
    for subsystem, bucket in by_subsystem.items():
        summary[subsystem] = {
            "score": round(bucket["earned"] / bucket["weight"], 3) if bucket["weight"] else None,
            "failures": int(bucket["failed"]),
            "skipped": int(bucket["skipped"]),
        }
    total_w = sum(b["weight"] for b in by_subsystem.values())
    total_e = sum(b["earned"] for b in by_subsystem.values())
    return {
        "overall": round(total_e / total_w, 3) if total_w else None,
        "graded_weight": total_w,
        "by_subsystem": summary,
    }


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def render_report(run_dir: Path, results: list[Result], rubric: dict[str, dict[str, Any]],
                  scorecard: dict[str, Any], ctx: Ctx) -> str:
    lines = [
        f"# 端到端基准报告 · `{run_dir.name}`",
        "",
        f"- 生成时间：{datetime.now(UTC).isoformat(timespec='seconds')}",
        f"- 输入目录：`{ctx.key.get('data_dir')}`",
        f"- 答案键：gold={ctx.key.get('gold_indices')}，cut={len(ctx.key.get('cut_indices', []))} 条，"
        f"theta_offset={ctx.key.get('theta_offset_deg')}",
        f"- 审计事件：{len(ctx.events)} 条；执行代码块：{len(ctx.code)} 个；产物：{len(ctx.outputs)} 个",
        "",
        f"## 总分：{_pct(scorecard['overall'])}",
        "",
        "| 子系统 | 得分 | 失败项 | 无法判定 |",
        "|---|---|---|---|",
    ]
    for subsystem, data in sorted(scorecard["by_subsystem"].items()):
        lines.append(f"| {subsystem} | {_pct(data['score'])} | {data['failures']} | {data['skipped']} |")

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

def render_prompt_sections(task_text: str, input_dir: str, output_dir: str) -> dict[str, str]:
    """从 task.md 抽出 P1/P2 两段并填入真实路径（纯文本，可直接粘给 agent）。"""
    import re as _re

    sections: dict[str, str] = {}
    for name in ("P1", "P2"):
        marker = f"## {name}"
        if marker not in task_text:
            continue
        rest = task_text.split(marker, 1)[1].split("\n", 1)[1]
        next_heading = [match.start() for match in _re.finditer(r"^## ", rest, _re.M)]
        body = rest[: next_heading[0]] if next_heading else rest
        fenced = _re.findall(r"```text\n(.*?)```", body, _re.S)
        text = fenced[0].strip() if fenced else body.strip()
        sections[name.lower()] = (text.replace("{INPUT_DIR}", input_dir)
                                       .replace("{OUTPUT_DIR}", output_dir))
    return sections


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def cmd_init(args: argparse.Namespace) -> int:
    data_dir = Path(args.data).expanduser().resolve()
    if not data_dir.is_dir():
        raise SystemExit(f"数据目录不存在：{data_dir}")
    key = build_answer_key(data_dir, Path(args.datasheet).resolve() if args.datasheet else None, args.limit)

    run_id = args.name or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.runs).expanduser().resolve() / run_id
    if run_dir.exists() and not args.force:
        raise SystemExit(f"运行目录已存在：{run_dir}（加 --force 覆盖）")
    if run_dir.exists():
        shutil.rmtree(run_dir)
    input_dir = run_dir / "input"
    output_dir = run_dir / "output"
    output_dir.mkdir(parents=True)

    if args.copy:
        shutil.copytree(data_dir, input_dir)
    else:
        input_dir.symlink_to(data_dir, target_is_directory=True)

    notebooks = list(data_dir.glob("*.ipynb"))
    work = run_dir / "work.ipynb"
    work.write_text(json.dumps({
        "cells": [{"cell_type": "markdown", "metadata": {}, "source": [f"# {run_id}\n"]}],
        "metadata": {"kernelspec": {"display_name": "Python (peaksMCP)", "language": "python",
                                    "name": "peaksmcp"}},
        "nbformat": 4, "nbformat_minor": 5,
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    try:
        from jupyter_client.kernelspec import KernelSpecManager

        KernelSpecManager().get_kernel_spec("peaksmcp")
    except Exception:  # noqa: BLE001
        print("\n注意：找不到 peaksmcp kernelspec —— 请先 `peaksMCP dash` 拉一次内核，")
        print("否则在 Jupyter 里打开 work.ipynb 会没有可用的 MCP 内核。")

    audit_path = Path(os.environ.get("PEAKSMCP_HOME", str(Path.home() / ".peaksMCP"))) / "audit" / "tool_audit.log"
    env = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "notebook": str(work),
        "audit_path": str(audit_path),
        "reference_dir": str(Path(args.reference).resolve()) if args.reference else None,
        "peaksMCP_head": _git_head(),
        "python": sys.executable,
    }
    (run_dir / "env.json").write_text(json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "answer_key.json").write_text(json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")

    prompt = TASK_FILE.read_text(encoding="utf-8")
    rendered = (prompt.replace("{INPUT_DIR}", str(input_dir)).replace("{OUTPUT_DIR}", str(output_dir)))
    (run_dir / "prompt.md").write_text(rendered, encoding="utf-8")
    for name, text in render_prompt_sections(rendered, str(input_dir), str(output_dir)).items():
        (run_dir / f"prompt-{name}.txt").write_text(text + "\n", encoding="utf-8")

    print(f"运行目录：{run_dir}")
    print(f"  输入  {input_dir}")
    print(f"  输出  {output_dir}")
    print(f"  notebook  {work}")
    print(f"  prompt     {run_dir / 'prompt.md'}（纯文本版：prompt-p1.txt / prompt-p2.txt）")
    print()
    print(f"答案键：gold={key['gold_indices']}  cut={len(key['cut_indices'])} 条  "
          f"theta_offset={key['theta_offset_deg']}")
    if key["unindexed_files"]:
        print(f"  干扰项（datasheet 里没有记录的文件）：{key['unindexed_files']}")
    print()
    print("下一步：")
    print(f"  1. 在 Jupyter 里打开 {work}（保存同意卡渲染在这里）")
    print("  2. 把 prompt-p1.txt（裸提示）整段粘给 agent；P2 变体用 prompt-p2.txt")
    print(f"  3. agent 跑完后：python benchmark/run_case.py grade {run_dir}")
    if notebooks:
        print(f"（注意：数据目录里已有 {len(notebooks)} 个 notebook，评分时会优先取 run 目录下最新的）")
    return 0


def cmd_grade(args: argparse.Namespace) -> int:
    run_dir = Path(args.run).expanduser().resolve()
    if not (run_dir / "answer_key.json").is_file():
        raise SystemExit(
            f"{run_dir} 不是一次有效的运行：缺少 answer_key.json。\n"
            "先跑 init 生成运行目录：python benchmark/run_case.py init --name <id>"
        )
    env = json.loads((run_dir / "env.json").read_text(encoding="utf-8")) if (run_dir / "env.json").is_file() else {}
    key = json.loads((run_dir / "answer_key.json").read_text(encoding="utf-8"))

    audit_path = Path(args.audit or env.get("audit_path")
                      or Path(os.environ.get("PEAKSMCP_HOME", str(Path.home() / ".peaksMCP")))
                      / "audit" / "tool_audit.log")
    events = load_events(audit_path, args.since or env.get("created_at"))

    notebook, notebook_code, notebook_path = load_notebook(
        Path(args.notebook) if args.notebook else None, run_dir)
    output_dir = Path(args.output) if args.output else Path(env.get("output_dir") or run_dir / "output")
    outputs = collect_outputs(run_dir, output_dir, key)
    reference_dir = Path(args.reference).resolve() if args.reference else (
        Path(env["reference_dir"]) if env.get("reference_dir") else None)

    ctx = Ctx(run_dir=run_dir, key=key, events=events,
              code=executed_code(events, notebook_code),
              fetched=fetched_api_names(events),
              outputs=outputs, notebook=notebook,
              reference_dir=reference_dir, audit_path=audit_path)

    results: list[Result] = []
    for group in (check_contract, check_access, check_run, check_show,
                  check_save, check_observability, check_quality):
        results.extend(group(ctx))

    rubric = load_rubric()
    scorecard = score(results, rubric)
    report = render_report(run_dir, results, rubric, scorecard, ctx)

    report_path = run_dir / "report.md"
    report_path.write_text(report, encoding="utf-8")
    payload = {
        "run": str(run_dir),
        "graded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "rubric_version": rubric_version(),
        "notebook": str(notebook_path) if notebook_path else None,
        "score": scorecard,
        "checks": [{"check": r.check, "passed": r.passed, "detail": r.detail,
                    "evidence": r.evidence,
                    "subsystem": rubric.get(r.check, {}).get("subsystem", "Other")}
                   for r in results],
    }
    (run_dir / "result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print(f"\n总分 {_pct(scorecard['overall'])}  （{run_dir.name}）\n")
    for subsystem, data in sorted(scorecard["by_subsystem"].items()):
        flag = "  " if data["failures"] == 0 else "× "
        extra = f"   跳过 {data['skipped']}" if data["skipped"] else ""
        print(f"  {flag}{subsystem:<16} {_pct(data['score']):>5}   失败 {data['failures']}{extra}")
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
         "source": [f"处理了 {len(key['cut_indices'])} 条 cut，gold 用 {gold}，theta 偏移 {offset}。\n"]},
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
                events.append(_audit_event("run_cell", "executed",
                                           {"operation_id": f"run-{extra:02d}", "cell_id": f"c{extra}"}))
            continue
        events.append(_audit_event("run_cell", "called",
                                   {"operation_id": op, "args": {"code": block}}))
        events.append(_audit_event("run_cell", "executed",
                                   {"operation_id": op, "cell_id": f"c{position}"}))
    if poisoned == "delete_cell":
        events.append(_audit_event("notebook_delete_cell", "executed",
                                   {"operation_id": "run-99", "cell_id": "c0"}))
    if poisoned == "api_block":
        events.append(_audit_event("run_cell", "error", {
            "operation_id": "run-98",
            "error": "Execution blocked: unverifiable API reference ghost_api.",
        }))
        for position in range(4, 13):  # 补足执行成功比例，只让 A2 翻红
            events.append(_audit_event("run_cell", "executed",
                                       {"operation_id": f"run-{position:02d}", "cell_id": f"c{position}"}))
    if poisoned == "exec_error":
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


def write_synthetic_run(root: Path, key: dict[str, Any], *, output_symlink: Path | None = None,
                        audit_events: list[dict[str, Any]] | None = None,
                        notebook: dict[str, Any] | None = None) -> Path:
    """把一次合成运行的痕迹（产物/审计/notebook）落到 root。"""
    input_dir = root / "input"
    output_dir = root / "output"
    if output_symlink is not None:
        output_dir.symlink_to(output_symlink, target_is_directory=True)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    (root / "answer_key.json").write_text(json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")
    now = datetime.now(UTC).isoformat(timespec="seconds")
    env = {"run_id": root.name, "created_at": now,
           "input_dir": str(input_dir), "output_dir": str(output_dir),
           "audit_path": str(root / "audit.log"),
           "reference_dir": str(DEFAULT_REFERENCE)}
    (root / "env.json").write_text(json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8")
    if audit_events is None:
        audit_events = golden_audit_events(str(input_dir), output_dir, key)
    (root / "audit.log").write_text(
        "\n".join(json.dumps(event, ensure_ascii=False) for event in audit_events) + "\n",
        encoding="utf-8")
    if notebook is None:
        notebook = reference_notebook(key, str(input_dir), str(output_dir))
    (root / "work.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    return root


def grade_payload(run_dir: Path) -> dict[str, Any]:
    code = cmd_grade(argparse.Namespace(
        run=str(run_dir), notebook=None, output=None, audit=None,
        reference=str(DEFAULT_REFERENCE), since=None, json=False))
    if code:
        raise SystemExit(f"grade 失败（{run_dir}），退出码 {code}")
    return json.loads((run_dir / "result.json").read_text(encoding="utf-8"))


def check_map(payload: dict[str, Any]) -> dict[str, bool | None]:
    return {item["check"]: item["passed"] for item in payload["checks"]}


def cmd_selftest(args: argparse.Namespace) -> int:
    """验证评分器本身分得出好坏（黄金 + 毒化负向对照），临时目录默认清理。

    1. 黄金运行：人工参考产物 + 与真实 AuditLogger schema 一致的审计痕迹
       （get × 3、run_cell called/executed、save_with_consent saved × N），
       24 条检查必须全过（含审计相关的 A1/A2/R3/R4/V3/O1/O2/O3/S3）；
    2. 毒化变体：每次只注入一种缺陷，断言恰好对应的一条/一组检查翻红：
       直写→V2、delete_cell→R3、denied→V3、API-block→A2、执行错误→R4、
       绕过 get→A1。这样"自检过=评分器没坏"才真正成立。
    """
    import tempfile

    data_dir = Path(args.data).expanduser().resolve()
    reference_dir = Path(args.reference).expanduser().resolve() if args.reference else DEFAULT_REFERENCE
    if not reference_dir.is_dir():
        raise SystemExit(f"自检需要人工参考产物目录：{reference_dir}")

    key = build_answer_key(data_dir, Path(args.datasheet).resolve() if args.datasheet else None, args.limit)
    root = Path(tempfile.mkdtemp(prefix="peaksmcp-selftest-"))
    input_dir = root / "input"
    input_dir.symlink_to(data_dir, target_is_directory=True)
    golden_dir = root / "golden"
    golden_dir.mkdir()
    output_dir = golden_dir / "output"
    output_dir.mkdir()
    copied = 0
    for item in key["expected_outputs"]:
        source = reference_dir / item["output_name"]
        if source.is_file():
            shutil.copy2(source, output_dir / item["output_name"])
            copied += 1
    write_synthetic_run(golden_dir, key, audit_events=golden_audit_events(str(input_dir), output_dir, key))

    failures: list[str] = []
    golden = grade_payload(golden_dir)
    golden_map = check_map(golden)
    must_pass = sorted(golden_map)
    bad = [name for name in must_pass if golden_map.get(name) is not True]
    print(f"黄金运行（24 项全过才算通过；审计相关检查已纳入）：复制了 {copied} 个人工参考产物")
    if bad:
        failures.append(f"黄金运行失败 {len(bad)} 项：{bad}")
        for name in bad:
            item = next((c for c in golden["checks"] if c["check"] == name), {})
            print(f"  · {name}: {item.get('detail')}")

    # 毒化负向对照：每注入一种缺陷，恰好该红的一组检查变红。
    poisoned_expectations = {
        "direct_write": {"V2_no_direct_disk_write"},
        "delete_cell": {"R3_append_only"},
        "denied": {"V3_consent_trail_complete"},
        "api_block": {"A2_no_unknown_api_blocks"},
        "exec_error": {"R4_execution_success"},
        "bypass_get": {"A1_get_before_use"},
    }
    for poison, expected_red in poisoned_expectations.items():
        poisoned_dir = root / f"poison-{poison}"
        poisoned_dir.mkdir()
        events = golden_audit_events(str(input_dir), output_dir, key, poisoned=poison)
        write_synthetic_run(poisoned_dir, key, output_symlink=output_dir,
                            audit_events=events)
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
            print(f"毒化 {poison:12s} → 恰好红 {sorted(expected_red)} ✓")

    if failures:
        print("\n自检失败：")
        for message in failures:
            print(f"  · {message}")
        return 1
    print(f"\n自检通过：黄金 24 项全过，毒化对照 {len(poisoned_expectations)} 组全部符合预期，评分器可用。")
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
    p_init.add_argument("--data", default=str(DEFAULT_DATA), help="原始数据目录")
    p_init.add_argument("--datasheet", help="datasheet.csv（默认在 data 目录或其上级找）")
    p_init.add_argument("--reference", default=str(DEFAULT_REFERENCE), help="人工参考产物目录")
    p_init.add_argument("--limit", type=int, help="只处理前 N 条 cut（快速冒烟）")
    p_init.add_argument("--runs", default=str(BENCH_DIR / "runs"), help="运行根目录")
    p_init.add_argument("--copy", action="store_true", help="复制数据而不是建软链")
    p_init.add_argument("--force", action="store_true")
    p_init.set_defaults(func=cmd_init)

    p_grade = sub.add_parser("grade", help="给一次运行打分")
    p_grade.add_argument("run")
    p_grade.add_argument("--notebook", help="指定 notebook（默认取 run 目录里最新的）")
    p_grade.add_argument("--output", help="指定产物目录")
    p_grade.add_argument("--audit", help="指定审计日志")
    p_grade.add_argument("--reference", help="人工参考产物目录")
    p_grade.add_argument("--since", help="只统计该 ISO 时间之后的审计事件")
    p_grade.add_argument("--json", action="store_true")
    p_grade.set_defaults(func=cmd_grade)

    p_self = sub.add_parser("selftest", help="黄金+毒化自检：验证评分器分得出好坏")
    p_self.add_argument("--data", default=str(DEFAULT_DATA))
    p_self.add_argument("--datasheet")
    p_self.add_argument("--reference", default=str(DEFAULT_REFERENCE))
    p_self.add_argument("--limit", type=int)
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
