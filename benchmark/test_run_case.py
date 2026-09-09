"""Offline unit tests for the benchmark grader logic (no kernel, no peaksMCP).

Covers the P0/P1 fixes: C3 receiver resolution, C4 binding path, A2 marker
semantics (persist-block != unknown-api), V3 path-level consent matching,
R2 call-site counting, executed-code deduplication, fetched-name filtering.
Run with the peaks python:  `$PY -m pytest benchmark/ -q`
"""

from __future__ import annotations

from benchmark import run_case as rc
from benchmark.run_case import (
    Ctx,
    _approved_save_paths,
    _fit_gold_receiver_indices,
    _theta_offset_binding_used,
    check_access,
    check_contract,
    check_observability,
    check_run,
    check_save,
)

GOLD_KEY = {
    "gold_indices": [20],
    "cut_indices": [5, 6, 9],
    "theta_offset_deg": 1.5,
    "expected_outputs": [
        {"index": 5, "stem": "BP_0005", "output_name": "BP_0005_processed.nc"},
        {"index": 6, "stem": "BP_0006", "output_name": "BP_0006_processed.nc"},
        {"index": 9, "stem": "BP_0009", "output_name": "BP_0009_processed.nc"},
    ],
}


def _ctx(code, events=None, key=None, outputs=None) -> Ctx:
    return Ctx(
        run_dir=rc.Path("."),
        key=key or GOLD_KEY,
        events=events or [],
        code=code,
        fetched=set(),
        outputs=outputs or {},
        notebook=None,
        reference_dir=None,
        audit_path=rc.Path("audit.log"),
    )


def _by_name(results):
    return {result.check: result for result in results}


# --------------------------------------------------------------------------- #
# C3: fit_gold receiver resolution                                            #
# --------------------------------------------------------------------------- #

def test_c3_bound_variable_with_same_cell_cut_list_passes():
    code = [
        "cuts = ['BP_0005','BP_0009','BP_0020']\n"
        "gold = scans['BP_0020']\nfit = gold.fit_gold()\n"
        "ef = fit.EF_correction",
    ]
    result = _by_name(check_contract(_ctx(code)))["C3_gold_index_correct"]
    assert result.passed is True, result.detail


def test_c3_chained_literal_passes():
    code = ["kd = scans['BP_0020'].fit_gold()\nef = kd.EF_correction\n"]
    assert _fit_gold_receiver_indices(code) == ({20}, False)


def test_c3_wrong_gold_fails():
    code = ["gold = scans['BP_0019']\nfit = gold.fit_gold()\n"]
    result = _by_name(check_contract(_ctx(code)))["C3_gold_index_correct"]
    assert result.passed is False


def test_c3_unresolved_loop_is_skipped_not_failed():
    code = ["for i in summary.gold:\n    g = scans[f'BP_{i:04d}']\n    g.fit_gold()\n"]
    result = _by_name(check_contract(_ctx(code)))["C3_gold_index_correct"]
    assert result.passed is None


# --------------------------------------------------------------------------- #
# C4: theta offset literal vs contract-binding                                #
# --------------------------------------------------------------------------- #

def test_c4_numeric_literal_passes():
    code = ["shifted = da.assign_coords(theta_par=da.theta_par - 1.5)\n"]
    result = _by_name(check_contract(_ctx(code)))["C4_theta_offset_from_contract"]
    assert result.passed is True


def test_c4_binding_from_inspect_passes():
    code = [
        "summary = inspect_experiment(scans)\n"
        "offset = summary.records[0].theta_offset_deg\n",
        "shifted = da.assign_coords(theta_par=da.theta_par - offset)\n",
    ]
    assert _theta_offset_binding_used(code) is True
    result = _by_name(check_contract(_ctx(code)))["C4_theta_offset_from_contract"]
    assert result.passed is True


def test_c4_no_offset_at_all_fails():
    code = ["shifted = da.assign_coords(theta_par=da.theta_par - 0.0)\n"]
    result = _by_name(check_contract(_ctx(code)))["C4_theta_offset_from_contract"]
    assert result.passed is False


# --------------------------------------------------------------------------- #
# A2: only genuine unknown-API blocks count; persist-block does not           #
# --------------------------------------------------------------------------- #

def _run_error(text):
    return [{"timestamp": "2026-01-01T00:00:00", "tool": "run_cell",
             "outcome": "error", "details": {"error": text}}]


def test_a2_persist_block_does_not_count_as_unknown_api():
    message = ("Execution blocked: this cell writes a file (savefig / file writers), "
               "and run_cell is never a persistence path. Results persist ONLY "
               "through the save_with_consent tool")
    result = _by_name(check_access(_ctx([], events=_run_error(message))))[
        "A2_no_unknown_api_blocks"]
    assert result.passed is True, result.detail


def test_a2_unknown_api_error_counts():
    message = "Execution blocked: unverifiable API reference ghost_api."
    result = _by_name(check_access(_ctx([], events=_run_error(message))))[
        "A2_no_unknown_api_blocks"]
    assert result.passed is False


# --------------------------------------------------------------------------- #
# V3: path-level consent matching + fallback                                  #
# --------------------------------------------------------------------------- #

def _save_trail(approved=True):
    events = []
    for number in (5, 9):
        operation_id = f"save-{number}"
        events.append({"timestamp": "2026-01-01T00:00:00", "tool": "save_with_consent",
                       "outcome": "called",
                       "details": {"operation_id": operation_id,
                                   "args": {"path": f"/out/BP_000{number}_processed.nc"}}})
        events.append({"timestamp": "2026-01-01T00:00:01", "tool": "save_with_consent",
                       "outcome": "saved" if approved else "denied",
                       "details": {"operation_id": operation_id, "ticket_id": "t",
                                   "sha256": "a" * 16}})
    return events


def test_v3_path_level_matching():
    outputs = {f"BP_000{i}_processed.nc": rc.Path(f"/out/BP_000{i}_processed.nc")
               for i in (5, 9)}
    result = _by_name(check_save(_ctx([], events=_save_trail(True), outputs=outputs)))[
        "V3_consent_trail_complete"]
    assert result.passed is True, result.detail
    assert "按 path 核对命中 2/2" in result.detail


def test_v3_approved_without_corresponding_product_fails():
    outputs = {f"BP_000{i}_processed.nc": rc.Path(f"/out/BP_000{i}_processed.nc")
               for i in (5, 9)}
    outputs["BP_0006_processed.nc"] = rc.Path("/out/BP_0006_processed.nc")
    result = _by_name(check_save(_ctx([], events=_save_trail(True), outputs=outputs)))[
        "V3_consent_trail_complete"]
    assert result.passed is False, "save 了 2 个但产物 3 个且有一条没被批准过 → 应失败"


def test_v3_denied_fails():
    outputs = {f"BP_000{i}_processed.nc": rc.Path(f"/out/BP_000{i}_processed.nc")
               for i in (5, 9)}
    result = _by_name(check_save(_ctx([], events=_save_trail(False), outputs=outputs)))[
        "V3_consent_trail_complete"]
    assert result.passed is False


# --------------------------------------------------------------------------- #
# R2 / dedupe / fetched                                                        #
# --------------------------------------------------------------------------- #

def test_r2_counts_call_sites_not_blocks():
    # 同块重跑（错误后重试的典型形态）去重后只算 1 个调用点 → 通过。
    rerun = [
        "gold = scans['BP_0020']\nfit = gold.fit_gold()\n",
        "gold = scans['BP_0020']\nfit = gold.fit_gold()\n",
        "kd = scans[s].k_convert(quiet=True)\n",
    ]
    result = _by_name(check_run(_ctx(rerun, events=[])))["R2_one_gold_fit"]
    assert result.passed is True and "1 个" in result.detail
    # 两段不同的 fit 代码（两次独立拟合）→ 失败。
    twice = [
        "gold = scans['BP_0020']\nfit = gold.fit_gold()\nef = fit.EF_correction\n",
        "gold2 = scans['BP_0020']\nfit2 = gold2.fit_gold()\nef2 = fit2.EF_correction\n",
    ]
    result = _by_name(check_run(_ctx(twice, events=[])))["R2_one_gold_fit"]
    assert result.passed is False and "2 个" in result.detail


def test_deduplicated_corpus():
    code = ["a = 1\nload_data('x')\n", "a = 1\nload_data('x')\n"]
    ctx = _ctx(code)
    assert ctx.unique_code == code[:1]


def test_fetched_names_ignore_failed_gets():
    events = [
        {"timestamp": "t1", "tool": "get", "outcome": "ok",
         "details": {"args": {"canonical_id": "dataarray:peaks.fitting:fit_gold"}}},
        {"timestamp": "t2", "tool": "get", "outcome": "error",
         "details": {"args": {"canonical_id": "x:set_EF_correction"}, "error": "boom"}},
    ]
    assert rc.fetched_api_names(events) == {"fit_gold"}


def test_approved_save_paths_pair_by_operation_id():
    paths = _approved_save_paths(_save_trail(True))
    assert sorted(paths) == ["/out/BP_0005_processed.nc", "/out/BP_0009_processed.nc"]


# --------------------------------------------------------------------------- #
# O2: operation_id chain completeness                                          #
# --------------------------------------------------------------------------- #

def _op_event(tool, outcome, operation_id, **details):
    return {"timestamp": "2026-01-01T00:00:00", "tool": tool, "outcome": outcome,
            "details": {"operation_id": operation_id, **details}}


def test_o2_chain_complete_passes():
    events = [
        _op_event("run_cell", "called", "op-1", args={"code": "a=1"}),
        _op_event("run_cell", "executed", "op-1", cell_id="c1"),
        _op_event("save_with_consent", "called", "op-2",
                  args={"path": "/out/x.nc"}),
        _op_event("save_with_consent", "saved", "op-2",
                  ticket_id="t1", sha256="a" * 16),
    ]
    result = _by_name(check_observability(_ctx([], events=events)))["O2_cell_artifact_linkage"]
    assert result.passed is True, result.detail


def test_o2_missing_called_pair_fails():
    events = [
        _op_event("run_cell", "called", "op-1", args={"code": "a=1"}),
        _op_event("run_cell", "executed", "op-1", cell_id="c1"),
        _op_event("save_with_consent", "saved", "op-ghost", ticket_id="t1"),
    ]
    result = _by_name(check_observability(_ctx([], events=events)))["O2_cell_artifact_linkage"]
    assert result.passed is False
    assert "找不到同 id 的 called" in result.detail


def test_o2_saved_without_ticket_or_sha_fails():
    events = [
        _op_event("save_with_consent", "called", "op-2", args={"path": "/out/x.nc"}),
        _op_event("save_with_consent", "saved", "op-2", ticket_id="t1"),  # 缺 sha256
    ]
    result = _by_name(check_observability(_ctx([], events=events)))["O2_cell_artifact_linkage"]
    assert result.passed is False
    assert "缺 ticket_id/sha256" in result.detail


def test_o2_legacy_audit_without_op_ids_uses_fallback():
    events = [
        {"timestamp": "2026-01-01T00:00:00", "tool": "run_cell", "outcome": "executed",
         "details": {"cell_id": "c1"}},
    ]
    result = _by_name(check_observability(_ctx([], events=events)))["O2_cell_artifact_linkage"]
    assert result.passed is True
    assert "退回启发式" in result.detail
