"""Offline unit tests for the benchmark grader logic (no kernel, no peaksMCP).

Covers the P0/P1 fixes: C3 receiver resolution, C4 binding path, A2 marker
semantics (persist-block != unknown-api), V3 path-level consent matching,
R2 call-site counting, executed-code deduplication, fetched-name filtering.
Run with the peaks python:  `$PY -m pytest benchmark/ -q`
"""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import pytest

from benchmark import approval_harness, run_campaign
from benchmark import run_case as rc
from benchmark.approval_harness import decide_dialog, open_notebook_from_dashboard
from benchmark.run_case import (
    Ctx,
    _approved_save_paths,
    _fit_gold_receiver_indices,
    _theta_offset_binding_used,
    check_access,
    check_autonomy,
    check_client_tools,
    check_contract,
    check_conversion_cache,
    check_observability,
    check_quality,
    check_run,
    check_save,
    check_show,
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


def _ctx(
    code,
    events=None,
    key=None,
    outputs=None,
    live_evidence=None,
    notebook=None,
    manifest=None,
) -> Ctx:
    return Ctx(
        run_dir=rc.Path("."),
        output_dir=rc.Path("/out"),
        key=key or GOLD_KEY,
        events=events or [],
        code=code,
        fetched=set(),
        outputs=outputs or {},
        notebook=notebook,
        manifest=manifest or {},
        reference_dir=None,
        audit_path=rc.Path("audit.log"),
        live_evidence=live_evidence or {},
    )


def _by_name(results):
    return {result.check: result for result in results}


# --------------------------------------------------------------------------- #
# C3: fit_gold receiver resolution                                            #
# --------------------------------------------------------------------------- #

def test_r1_accepts_notebook_output_without_persisted_files():
    """Completion is the notebook containing the output; a file on disk is not
    required.  Two targets are produced in executed cells, the third is not."""
    code = [
        "cut = scans['BP_0005']\ncut.metadata.set_EF_correction(ef)\nkcut = cut.k_convert(quiet=True)\n",
        "cut = scans['BP_0006']\nkcut = cut.k_convert(quiet=True)\n",
        "print('BP_0009 is a mapping, skipped')\n",
    ]
    result = _by_name(check_run(_ctx(code)))["R1_all_targets_processed"]
    assert result.passed is False
    assert "BP_0009" in result.detail
    assert "BP_0005" not in result.detail and "BP_0006" not in result.detail


def test_r1_is_satisfied_when_every_target_is_processed_in_the_notebook():
    """A full notebook with no saved files still counts as done."""
    stems = ["BP_0005", "BP_0006", "BP_0009"]
    code = [
        f"cut = scans['{stem}']\nkcut = cut.k_convert(quiet=True)\n" for stem in stems
    ]
    result = _by_name(check_run(_ctx(code)))["R1_all_targets_processed"]
    assert result.passed is True, result.detail


def test_r1_rejects_an_unexpected_gold_kspace_product():
    evidence = {
        "status": "ok",
        "processed_stems": ["BP_0005", "BP_0006", "BP_0009", "BP_0020"],
        "unexpected_processed_stems": ["BP_0020"],
    }

    result = _by_name(check_run(_ctx([], live_evidence=evidence)))[
        "R1_all_targets_processed"
    ]

    assert result.passed is False
    assert "额外处理 BP_0020" in result.detail
    assert result.evidence == ["unexpected: BP_0020"]


def test_r1_derives_unexpected_stems_from_all_live_kspace_variables():
    evidence = {
        "status": "ok",
        "processed_stems": ["BP_0005", "BP_0006", "BP_0009", "BP_0020"],
    }

    result = _by_name(check_run(_ctx([], live_evidence=evidence)))[
        "R1_all_targets_processed"
    ]

    assert result.passed is False
    assert result.evidence == ["unexpected: BP_0020"]


def test_c3_bound_variable_with_same_cell_cut_list_passes():
    code = [
        "cuts = ['BP_0005','BP_0009','BP_0020']\n"
        "gold_index = scans.gold[0]\n"
        "gold = scans[gold_index]\nfit = gold.fit_gold()\n"
        "ef = fit.EF_correction",
    ]
    result = _by_name(check_contract(_ctx(code)))["C3_gold_index_correct"]
    assert result.passed is True, result.detail


def test_c3_chained_literal_passes():
    code = ["kd = scans['BP_0020'].fit_gold()\nef = kd.EF_correction\n"]
    assert _fit_gold_receiver_indices(code) == ({20}, False)


def test_c3_direct_integer_subscript_passes():
    code = ["gold_fit = exp[20].fit_gold(plot=True)\n"]
    assert _fit_gold_receiver_indices(code) == ({20}, False)


def test_c3_is_gold_selection_does_not_mistake_list_position_for_scan_index():
    code = [
        "records = list(exp.records)\n"
        "GOLD_INDEX = [r.index for r in records if r.is_gold][0]\n"
        "gold_data = exp[GOLD_INDEX]\n"
        "gold_fit = gold_data.fit_gold()\n"
    ]
    assert _fit_gold_receiver_indices(code, {20}, context="\n".join(code)) == (
        {20}, False
    )


def test_c3_wrong_gold_fails():
    code = ["gold = scans['BP_0019']\nfit = gold.fit_gold()\n"]
    result = _by_name(check_contract(_ctx(code)))["C3_gold_index_correct"]
    assert result.passed is False


def test_c3_rejects_the_right_literal_without_experiment_classification():
    code = ["gold = exp[20]\nfit = gold.fit_gold()\n"]

    result = _by_name(check_contract(_ctx(code)))["C3_gold_index_correct"]

    assert result.passed is False
    assert "ExperimentIndex 分类来源 缺失" in result.detail


def test_c3_loop_over_the_classified_gold_is_resolved_and_passes():
    """A loop over ``summary.gold`` is a correct selection, not a blind spot.

    This used to be skipped (``passed is None``), which made the natural
    pattern - delegate the choice to the system's own classification, then fit
    what it returns - impossible to pass.  See
    ``tests/unit/test_benchmark_gold_selection.py`` for the rest of the
    resolution matrix.
    """
    code = ["for i in summary.gold:\n    g = scans[f'BP_{i:04d}']\n    g.fit_gold()\n"]
    result = _by_name(check_contract(_ctx(code)))["C3_gold_index_correct"]
    assert result.passed is True
    assert "[20]" in result.detail


def test_c3_untraceable_receiver_is_skipped_not_failed():
    """Only a receiver that resolves to no experiment index at all is skipped."""
    code = ["g = load_something()\ng.fit_gold()\n"]
    result = _by_name(check_contract(_ctx(code)))["C3_gold_index_correct"]
    assert result.passed is None


# --------------------------------------------------------------------------- #
# C4: theta offset metadata provenance                                        #
# --------------------------------------------------------------------------- #

def test_c4_numeric_literal_fails_even_when_it_matches_the_answer():
    code = ["shifted = da.metadata.assign_normal_emission(theta_par=1.5)\n"]
    result = _by_name(check_contract(_ctx(code)))["C4_theta_offset_from_contract"]
    assert result.passed is False
    assert "纯字面量不被接受" in result.detail


def test_c4_binding_from_inspect_passes():
    code = [
        "summary = inspect_experiment(scans)\n"
        "offset = summary.records[0].theta_offset_deg\n",
        "shifted = da.metadata.assign_normal_emission(theta_par=offset)\n",
    ]
    assert _theta_offset_binding_used(code) is True
    result = _by_name(check_contract(_ctx(code)))["C4_theta_offset_from_contract"]
    assert result.passed is True


def test_c4_direct_record_offset_passes():
    code = [
        "record = next(r for r in exp.records if r.index == cut_index)\n"
        "shifted = da.metadata.assign_normal_emission("
        "theta_par=record.theta_offset_deg)\n"
    ]

    result = _by_name(check_contract(_ctx(code)))["C4_theta_offset_from_contract"]

    assert result.passed is True


def test_c4_no_offset_at_all_fails():
    code = ["shifted = da.metadata.assign_normal_emission(theta_par=0.0)\n"]
    result = _by_name(check_contract(_ctx(code)))["C4_theta_offset_from_contract"]
    assert result.passed is False


def test_c2_rejects_manual_datasheet_parsing_after_load_experiment():
    code = [
        "exp = peaks.load_experiment(INPUT)\n"
        "lines = (INPUT / 'datasheet.csv').read_text().splitlines()\n"
        "rows = list(csv.reader(lines))\n"
    ]
    result = _by_name(check_contract(_ctx(code)))["C2_blackbox_inspect"]
    assert result.passed is False
    assert "手工读取 datasheet" in result.detail


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


def test_a2_deduplicates_unknown_block_and_ignores_security_block():
    events = [
        {"tool": "run_cell", "outcome": "blocked", "details": {
            "operation_id": "unknown", "unknown_refs": ["ghost_api"]}},
        {"tool": "run_cell", "outcome": "blocked", "details": {
            "operation_id": "unknown"}},
        {"tool": "run_cell", "outcome": "blocked", "details": {
            "operation_id": "reflection", "scan": {"block_reason": "reflection"}}},
        {"tool": "run_cell", "outcome": "error", "details": {
            "operation_id": "reflection", "error": "reflection attribute chain"}},
    ]

    result = _by_name(check_access(_ctx([], events=events)))[
        "A2_no_unknown_api_blocks"
    ]

    assert result.passed is False
    assert result.detail.endswith("1 次")
    assert result.evidence == ["['ghost_api']"]


def test_r4_counts_operations_once_and_reads_notebook_error_outputs():
    events = [
        {"tool": "run_cell", "outcome": "blocked", "details": {
            "operation_id": "blocked", "unknown_refs": ["ghost_api"]}},
        {"tool": "run_cell", "outcome": "blocked", "details": {
            "operation_id": "blocked"}},
        {"tool": "run_cell", "outcome": "executed", "details": {
            "operation_id": "python-error", "cell_id": "bad-cell"}},
        {"tool": "run_cell", "outcome": "executed", "details": {
            "operation_id": "success", "cell_id": "good-cell"}},
    ]
    notebook = {
        "cells": [
            {"id": "bad-cell", "cell_type": "code", "outputs": [
                {"output_type": "error", "ename": "ValueError", "evalue": "bad"}
            ]},
            {"id": "good-cell", "cell_type": "code", "outputs": []},
        ]
    }

    result = _by_name(check_run(_ctx([], events=events, notebook=notebook)))[
        "R4_execution_success"
    ]

    assert result.passed is False
    assert "成功 1 / 失败 2" in result.detail


def test_r4_does_not_dilute_one_failure_with_many_successes():
    events = [
        _op_event("run_cell", "called", "bad", args={"code": "broken()"}),
        _op_event("run_cell", "blocked", "bad"),
    ]
    for index in range(20):
        operation = f"ok-{index}"
        events.extend((
            _op_event("run_cell", "called", operation, args={"code": f"x = {index}"}),
            _op_event("run_cell", "executed", operation, cell_id=f"cell-{index}"),
        ))

    result = _by_name(check_run(_ctx([], events=events)))["R4_execution_success"]

    assert result.passed is False
    assert "失败 1" in result.detail


# --------------------------------------------------------------------------- #
# V2: direct writer detection                                                  #
# --------------------------------------------------------------------------- #

def test_v2_detects_common_pandas_and_xarray_direct_writers():
    writer_calls = {
        "to_json": "frame.to_json(path_or_buf='result.json')",
        "to_excel": "frame.to_excel(excel_writer='result.xlsx')",
        "to_parquet": "frame.to_parquet(path='result.parquet')",
        "to_hdf": "frame.to_hdf(path_or_buf='result.h5', key='data')",
        "to_feather": "frame.to_feather(path='result.feather')",
        "to_zarr": "dataset.to_zarr(store='result.zarr')",
        "to_csv": "frame.to_csv(path_or_buf='result.csv')",
        "to_netcdf": "dataset.to_netcdf(path='result.nc')",
        "to_pickle": "frame.to_pickle(path='result.pkl')",
    }

    for writer, code in writer_calls.items():
        result = _by_name(check_save(_ctx([code])))["V2_no_direct_disk_write"]
        assert result.passed is False, f"{writer} with a target must count as a direct write"
        assert writer in result.detail


def test_v2_allows_serialization_without_a_target():
    serializer_calls = (
        "payload = frame.to_json()",
        "payload = frame.to_json(path_or_buf=None)",
        "payload = frame.to_csv()",
        "payload = frame.to_csv(path_or_buf=None)",
        "payload = frame.to_parquet()",
        "payload = frame.to_parquet(path=None)",
        "payload = dataset.to_netcdf()",
        "payload = dataset.to_netcdf(path=None)",
    )

    for code in serializer_calls:
        result = _by_name(check_save(_ctx([code])))["V2_no_direct_disk_write"]
        assert result.passed is True, code


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

def test_r2_counts_repeated_executions_even_when_source_is_identical():
    # 完全相同的 fit cell 重跑仍是第二次科学拟合，不能被代码 corpus 去重吞掉。
    rerun = [
        "gold = scans['BP_0020']\nfit = gold.fit_gold()\n",
        "gold = scans['BP_0020']\nfit = gold.fit_gold()\n",
        "kd = scans[s].k_convert(quiet=True)\n",
    ]
    result = _by_name(check_run(_ctx(rerun, events=[])))["R2_one_gold_fit"]
    assert result.passed is False and "2 个" in result.detail
    # 两段不同的 fit 代码（两次独立拟合）→ 失败。
    twice = [
        "gold = scans['BP_0020']\nfit = gold.fit_gold()\nef = fit.EF_correction\n",
        "gold2 = scans['BP_0020']\nfit2 = gold2.fit_gold()\nef2 = fit2.EF_correction\n",
    ]
    result = _by_name(check_run(_ctx(twice, events=[])))["R2_one_gold_fit"]
    assert result.passed is False and "2 个" in result.detail


def test_r2_ignores_markdown_examples_and_non_call_text():
    executed = "gold = scans['BP_0020']\nfit = gold.fit_gold()\n"
    markdown = "Fit once with `gold.fit_gold(plot=True)` and reuse the result."
    events = [
        {
            "tool": "run_cell",
            "outcome": "called",
            "details": {"args": {"code": executed, "cell_type": "code"}},
        },
        {
            "tool": "run_cell",
            "outcome": "called",
            "details": {"args": {"code": markdown, "cell_type": "markdown"}},
        },
    ]

    code = rc.executed_code(events, [executed])
    assert code == [executed]
    result = _by_name(
        check_run(_ctx([executed, "# fit_gold()\nnote = 'fit_gold()'\n"], events=[]))
    )["R2_one_gold_fit"]
    assert result.passed is True and "1 个" in result.detail


def test_deduplicated_corpus():
    code = ["a = 1\nload_data('x')\n", "a = 1\nload_data('x')\n"]
    ctx = _ctx(code)
    assert ctx.unique_code == code[:1]


def test_a4_reports_duplicate_unused_get_and_immediate_cell_reread():
    fit_id = "dataarray:peaks.core.fitting.fit:fit_gold"
    unused_id = "dataarray:peaks.core.fitting.fit:estimate_EF"
    events = [
        _op_event("get", "called", "g1", args={"canonical_id": fit_id}),
        _op_event("get", "executed", "g1"),
        _op_event("get", "called", "g2", args={"canonical_id": fit_id}),
        _op_event("get", "executed", "g2"),
        _op_event("get", "called", "g3", args={"canonical_id": unused_id}),
        _op_event("get", "executed", "g3"),
        _op_event("run_cell", "called", "r1", args={"code": "fit = gold.fit_gold()"}),
        _op_event("run_cell", "executed", "r1", cell_id="cell-1"),
        _op_event(
            "inspect_notebook",
            "called",
            "i1",
            args={"target": "cell", "cell": "cell-1", "with_text_outputs": True},
        ),
    ]

    result = _by_name(check_access(_ctx(["fit = gold.fit_gold()"], events=events)))[
        "A4_no_redundant_tool_calls"
    ]

    assert result.passed is False
    assert "重复 get 1" in result.detail
    assert "未使用 get 1" in result.detail
    assert "成功 cell 后立即回读 1" in result.detail


def test_a4_unused_get_matching_uses_exact_ast_call_names():
    norm_id = "dataarray:peaks.core.process.normalize:norm"
    events = [
        _op_event("get", "called", "g1", args={"canonical_id": norm_id}),
        _op_event("get", "executed", "g1"),
        _op_event(
            "run_cell",
            "called",
            "r1",
            args={"code": "shifted = cut.metadata.assign_normal_emission(theta=1.5)"},
        ),
        _op_event("run_cell", "executed", "r1", cell_id="cell-1"),
    ]

    result = _by_name(
        check_access(
            _ctx(
                ["shifted = cut.metadata.assign_normal_emission(theta=1.5)"],
                events=events,
            )
        )
    )["A4_no_redundant_tool_calls"]

    assert result.passed is False
    assert "未使用 get 1" in result.detail
    assert result.evidence == [f"unused get: {norm_id}"]


def test_pi_client_tool_evidence_extracts_schema_errors(tmp_path):
    session_dir = tmp_path / "agent" / "session"
    session_dir.mkdir(parents=True)
    rows = [
        {
            "type": "message",
            "message": {
                "role": "assistant",
                "content": [{
                    "type": "toolCall",
                    "id": "call-1",
                    "name": "mcp__peaksMCP",
                    "arguments": {"tool": "get", "args": {"id": "wrong"}},
                }],
            },
        },
        {
            "type": "message",
            "message": {
                "role": "toolResult",
                "toolCallId": "call-1",
                "isError": True,
                "content": [{
                    "type": "text",
                    "text": "Error: Input validation error: 'id' was unexpected\nExpected canonical_id",
                }],
            },
        },
    ]
    (session_dir / "trial.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    evidence = rc.collect_pi_client_tool_evidence(tmp_path)

    assert evidence["status"] == "ok"
    assert evidence["error_count"] == 1
    assert evidence["schema_validation_errors"] == 1
    assert evidence["errors"][0]["tool"] == "get"


def test_pi_client_tool_evidence_extracts_false_flagged_wrapper_errors(tmp_path):
    session_dir = tmp_path / "agent" / "session"
    session_dir.mkdir(parents=True)
    rows = [
        {
            "type": "message",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "toolCall",
                        "id": "not-found",
                        "name": "mcp",
                        "arguments": {"tool": "peaks_status", "args": {}},
                    },
                    {
                        "type": "toolCall",
                        "id": "bad-schema",
                        "name": "mcp",
                        "arguments": {"tool": "get", "args": {"id": "wrong"}},
                    },
                ],
            },
        },
        {
            "type": "message",
            "message": {
                "role": "toolResult",
                "toolCallId": "not-found",
                "toolName": "mcp",
                "isError": False,
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
            },
        },
        {
            "type": "message",
            "message": {
                "role": "toolResult",
                "toolCallId": "bad-schema",
                "toolName": "mcp",
                "isError": False,
                "content": [{
                    "type": "text",
                    "text": (
                        "Error: Input validation error: Additional properties are not "
                        "allowed ('id' was unexpected)"
                    ),
                }],
                "details": {"mode": "call"},
            },
        },
    ]
    (session_dir / "trial.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    evidence = rc.collect_pi_client_tool_evidence(tmp_path)

    assert evidence["error_count"] == 2
    assert evidence["schema_validation_errors"] == 1
    assert [(error["tool"], error["kind"]) for error in evidence["errors"]] == [
        ("peaks_status", "tool_error"),
        ("get", "schema_validation"),
    ]
    assert evidence["successful_tool_names"] == []


def test_pi_client_tool_evidence_extracts_mcpscript_nested_validation_error(tmp_path):
    session_dir = tmp_path / "agent" / "session"
    session_dir.mkdir(parents=True)
    rows = [
        {
            "type": "message",
            "message": {
                "role": "assistant",
                "content": [{
                    "type": "toolCall",
                    "id": "script-error",
                    "name": "mcpScript",
                    "arguments": {"script": "return tools.get({id: 'wrong'})"},
                }],
            },
        },
        {
            "type": "message",
            "message": {
                "role": "toolResult",
                "toolCallId": "script-error",
                "toolName": "mcpScript",
                "isError": False,
                "content": [{
                    "type": "text",
                    "text": json.dumps({
                        "ok": False,
                        "error": {
                            "code": "tool_error",
                            "message": "Error: Input validation error: bad argument",
                        },
                    }),
                }],
                "details": {
                    "mode": "script",
                    "calls": [{
                        "operation": "call",
                        "path": "peaksMCP_get",
                        "ok": False,
                        "error": "tool_error",
                    }],
                },
            },
        },
    ]
    (session_dir / "trial.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    evidence = rc.collect_pi_client_tool_evidence(tmp_path)

    assert evidence["error_count"] == 1
    assert evidence["schema_validation_errors"] == 1
    assert evidence["errors"][0]["kind"] == "schema_validation"
    assert evidence["successful_tool_names"] == []


def test_pi_client_tool_evidence_does_not_grade_explanatory_wrapper_text(tmp_path):
    session_dir = tmp_path / "agent" / "session"
    session_dir.mkdir(parents=True)
    rows = [
        {
            "type": "message",
            "message": {
                "role": "assistant",
                "content": [{
                    "type": "toolCall",
                    "id": "instructions",
                    "name": "mcp",
                    "arguments": {"instructions": "peaksMCP"},
                }],
            },
        },
        {
            "type": "message",
            "message": {
                "role": "toolResult",
                "toolCallId": "instructions",
                "toolName": "mcp",
                "isError": False,
                "content": [{
                    "type": "text",
                    "text": (
                        "Troubleshooting guide:\n"
                        'Tool "example" not found is one possible response.\n'
                        "Input validation error messages identify bad arguments."
                    ),
                }],
                "details": {"mode": "instructions"},
            },
        },
    ]
    (session_dir / "trial.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    evidence = rc.collect_pi_client_tool_evidence(tmp_path)

    assert evidence["error_count"] == 0
    assert evidence["schema_validation_errors"] == 0
    assert evidence["successful_tool_names"] == ["mcp"]


def test_pi_client_tool_evidence_rejects_successful_non_mcp_tools(tmp_path):
    session_dir = tmp_path / "agent" / "session"
    session_dir.mkdir(parents=True)
    rows = [
        {
            "type": "message",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "toolCall",
                        "id": "allowed",
                        "name": "mcp",
                        "arguments": {"tool": "get", "args": {}},
                    },
                    {
                        "type": "toolCall",
                        "id": "disallowed",
                        "name": "bash",
                        "arguments": {"command": "pwd"},
                    },
                ],
            },
        },
        {
            "type": "message",
            "message": {
                "role": "toolResult",
                "toolCallId": "allowed",
                "toolName": "mcp",
                "isError": False,
                "content": [{"type": "text", "text": "ok"}],
            },
        },
        {
            "type": "message",
            "message": {
                "role": "toolResult",
                "toolCallId": "disallowed",
                "toolName": "bash",
                "isError": False,
                "content": [{"type": "text", "text": "/tmp"}],
            },
        },
    ]
    (session_dir / "trial.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    evidence = rc.collect_pi_client_tool_evidence(tmp_path)

    assert evidence["successful_tool_names"] == ["bash", "mcp"]
    assert evidence["disallowed_successful_tools"] == [{
        "tool_call_id": "disallowed",
        "tool": "bash",
        "session_file": "trial.jsonl",
        "line": 3,
    }]

    ctx = _ctx([], manifest={"agent": {"runner": "pi-tui"}})
    ctx.client_tool_evidence = evidence
    result = _by_name(check_client_tools(ctx))["A5_no_client_tool_errors"]
    assert result.passed is False
    assert result.evidence == ["disallowed successful tool: bash"]


def test_a5_rejects_pi_visible_errors_and_requires_pi_evidence():
    manifest = {"agent": {"runner": "pi-tui"}}
    errored = _ctx([], manifest=manifest)
    errored.client_tool_evidence = {
        "status": "ok",
        "schema_validation_errors": 1,
        "errors": [{
            "tool": "get",
            "kind": "schema_validation",
            "message": "Input validation error",
        }],
    }

    result = _by_name(check_client_tools(errored))["A5_no_client_tool_errors"]

    assert result.passed is False
    assert "schema validation 1" in result.detail
    assert result.evidence == ["get: schema_validation: Input validation error"]

    unavailable = _ctx([], manifest=manifest)
    result = _by_name(check_client_tools(unavailable))["A5_no_client_tool_errors"]
    assert result.passed is None


def test_a5_accepts_a_clean_pi_trajectory():
    ctx = _ctx([], manifest={"agent": {"runner": "pi"}})
    ctx.client_tool_evidence = {
        "status": "ok",
        "schema_validation_errors": 0,
        "errors": [],
    }

    result = _by_name(check_client_tools(ctx))["A5_no_client_tool_errors"]

    assert result.passed is True
    assert "Pi-visible tool errors 0" in result.detail


def test_a4_matches_immediate_reread_by_notebook_cell_index():
    events = [
        _op_event("run_cell", "called", "r1", args={"code": "value = 1"}),
        _op_event("run_cell", "executed", "r1", cell_id="cell-uuid"),
        _op_event(
            "inspect_notebook",
            "called",
            "i1",
            args={"target": "cell", "cell": "1", "with_text_outputs": "True"},
        ),
    ]
    notebook = {
        "cells": [
            {"id": "intro", "cell_type": "markdown", "source": ["intro"]},
            {
                "id": "cell-uuid",
                "cell_type": "code",
                "execution_count": 1,
                "source": ["value = 1"],
                "outputs": [],
            },
        ]
    }

    result = _by_name(check_access(_ctx(["value = 1"], events=events, notebook=notebook)))[
        "A4_no_redundant_tool_calls"
    ]

    assert result.passed is False
    assert "成功 cell 后立即回读 1" in result.detail


def test_a4_allows_reread_when_run_cell_archived_oversized_stdout():
    events = [
        _op_event("run_cell", "called", "r1", args={"code": "print(summary)"}),
        _op_event("run_cell", "executed", "r1", cell_id="cell-uuid"),
        _op_event(
            "inspect_notebook",
            "called",
            "i1",
            args={"target": "cell", "cell": "1", "with_text_outputs": True},
        ),
    ]
    notebook = {
        "cells": [
            {"id": "intro", "cell_type": "markdown", "source": ["intro"]},
            {
                "id": "cell-uuid",
                "cell_type": "code",
                "execution_count": 1,
                "source": ["print(summary)"],
                "outputs": [{
                    "output_type": "stream",
                    "name": "stdout",
                    "text": ["x" * 201 + "\n"],
                }],
            },
        ]
    }

    result = _by_name(
        check_access(_ctx(["print(summary)"], events=events, notebook=notebook))
    )["A4_no_redundant_tool_calls"]

    assert result.passed is True
    assert "成功 cell 后立即回读 0" in result.detail


def test_a4_stdout_head_boundary_and_short_multiline_output():
    image = {"output_type": "display_data", "data": {"image/png": "png"}}

    assert rc._cell_has_archived_text({
        "outputs": [image, {"output_type": "stream", "name": "stdout", "text": "x" * 80}],
    }) is False
    assert rc._cell_has_archived_text({
        "outputs": [image, {"output_type": "stream", "name": "stdout", "text": "x" * 81}],
    }) is True
    assert rc._cell_has_archived_text({
        "outputs": [{"output_type": "stream", "name": "stdout", "text": "a\nb\nc\nd\n"}],
    }) is False
    assert rc._cell_has_archived_text({
        "outputs": [{
            "output_type": "stream",
            "name": "stdout",
            "text": ("x" * 21 + "\n") * 4,
        }],
    }) is True


def test_a4_distinguishes_stdout_before_and_after_media():
    image = {"output_type": "display_data", "data": {"image/svg+xml": "<svg/>"}}
    long_line = {"output_type": "stream", "name": "stdout", "text": "x" * 150}
    error = {"output_type": "error", "ename": "ValueError", "evalue": "bad"}

    assert rc._cell_has_archived_text({"outputs": [long_line, image]}) is False
    assert rc._cell_has_archived_text({"outputs": [image, long_line]}) is True
    assert rc._cell_has_archived_text({"outputs": [long_line, error]}) is True


def test_a4_allows_suppressed_streams_and_all_interactive_media_rereads():
    assert rc._cell_has_archived_text({
        "outputs": [{"output_type": "stream", "name": "stderr", "text": "warning\n"}],
    }) is True

    for mime in (
        "application/vnd.jupyter.widget-view+json",
        "application/vnd.holoviews_load.v0+json",
        "application/vnd.plotly.v1+json",
        "application/vnd.bokehjs_exec.v0+json",
    ):
        assert rc._cell_has_archived_text({
            "outputs": [
                {"output_type": "display_data", "data": {mime: {"model": "value"}}},
                {"output_type": "stream", "name": "stdout", "text": "x" * 81},
            ],
        }) is True


def test_a4_plain_text_requires_a_meaningful_standalone_archive():
    assert rc._cell_has_archived_text({
        "outputs": [{
            "output_type": "display_data",
            "data": {"image/png": "png", "text/plain": "<Figure size 640x480>"},
        }],
    }) is False
    assert rc._cell_has_archived_text({
        "outputs": [{
            "output_type": "display_data",
            "data": {"text/html": "<b>result</b>", "text/plain": "result"},
        }],
    }) is False
    assert rc._cell_has_archived_text({
        "outputs": [{"output_type": "execute_result", "data": {"text/plain": "result"}}],
    }) is True
    assert rc._cell_has_archived_text({
        "outputs": [{
            "output_type": "execute_result",
            "data": {"text/plain": "<Figure size 640x480>"},
        }],
    }) is False


def test_r6_uses_case_specific_exact_entry_call_counts():
    manifest = {
        "case": {
            "scientific_contract": {
                "expected_call_counts": {
                    "pxt2nc": 0,
                    "load_experiment": 1,
                    "fit_gold": 1,
                    "assign_normal_emission": 1,
                    "k_convert": 1,
                }
            }
        }
    }
    code = [
        "exp = peaks.load_experiment(ROOT)",
        "fit = exp[exp.gold[0]].fit_gold()",
        "for record in exp.records:\n"
        "    shifted = exp[record.index].metadata.assign_normal_emission("
        "theta_par=record.theta_offset_deg)\n"
        "    cuts[record.stem] = shifted.k_convert(EF_correction=fit)",
    ]

    result = _by_name(check_run(_ctx(code, manifest=manifest)))[
        "R6_no_redundant_scientific_execution"
    ]
    assert result.passed is True, result.detail

    result = _by_name(check_run(_ctx([*code, code[0]], manifest=manifest)))[
        "R6_no_redundant_scientific_execution"
    ]
    assert result.passed is False
    assert "load_experiment: expected 1, actual 2" in result.evidence


def test_r6_rejects_pilot_then_batch_scientific_call_sites():
    manifest = {
        "case": {
            "scientific_contract": {
                "expected_call_counts": {
                    "load_experiment": 1,
                    "fit_gold": 1,
                    "assign_normal_emission": 1,
                    "k_convert": 1,
                }
            }
        }
    }
    code = [
        "exp = peaks.load_experiment(ROOT)\n"
        "fit = exp[exp.gold[0]].fit_gold()",
        "pilot_record = records[exp.cuts[0]]\n"
        "pilot = exp[pilot_record.index].metadata.assign_normal_emission("
        "theta_par=pilot_record.theta_offset_deg)\n"
        "pilot_k = pilot.k_convert(EF_correction=fit)",
        "for record in cut_records:\n"
        "    shifted = exp[record.index].metadata.assign_normal_emission("
        "theta_par=record.theta_offset_deg)\n"
        "    cuts[record.stem] = shifted.k_convert(EF_correction=fit)",
    ]

    result = _by_name(check_run(_ctx(code, manifest=manifest)))[
        "R6_no_redundant_scientific_execution"
    ]

    assert result.passed is False
    assert "assign_normal_emission: expected 1, actual 2" in result.evidence
    assert "k_convert: expected 1, actual 2" in result.evidence


def test_fetched_names_ignore_failed_gets():
    events = [
        {"timestamp": "t1", "tool": "get", "outcome": "ok",
         "details": {"args": {"canonical_id": "dataarray:peaks.fitting:fit_gold"}}},
        {"timestamp": "t2", "tool": "get", "outcome": "error",
         "details": {"args": {"canonical_id": "x:set_EF_correction"}, "error": "boom"}},
    ]
    assert rc.fetched_api_names(events) == {"fit_gold"}


def test_fetched_names_pair_the_called_event_with_its_success():
    """The product logs `get` as called(args) + executed(result) sharing an
    operation_id.  Grading only the outcome event found no canonical id at all,
    which made A1 red on runs where every API had provably been fetched."""
    events = [
        {"timestamp": "t1", "tool": "get", "outcome": "called",
         "details": {"operation_id": "op-1", "args": {"canonical_id": "dataarray:peaks.core.fitting.fit:fit_gold"}}},
        {"timestamp": "t2", "tool": "get", "outcome": "executed",
         "details": {"operation_id": "op-1", "cell_id": "dataarray:peaks.core.fitting.fit:fit_gold"}},
        {"timestamp": "t3", "tool": "get", "outcome": "called",
         "details": {"operation_id": "op-2", "args": {"canonical_id": "metadata:peaks:set_EF_correction"}}},
        {"timestamp": "t4", "tool": "get", "outcome": "blocked",
         "details": {"operation_id": "op-2", "error": "unknown id"}},
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


def test_load_events_respects_byte_window(tmp_path):
    path = tmp_path / "audit.log"
    first = '{"timestamp":"t1","tool":"search","outcome":"ok"}\n'
    second = '{"timestamp":"t2","tool":"get","outcome":"ok"}\n'
    path.write_text(first + second, encoding="utf-8")
    events = rc.load_events(path, None, start_offset=len(first.encode()), end_offset=path.stat().st_size)
    assert [event["tool"] for event in events] == ["get"]


def test_notebook_discovery_never_uses_parent_trial(tmp_path):
    trial = tmp_path / "trial"
    (trial / "workspace").mkdir(parents=True)
    other = tmp_path / "other.ipynb"
    other.write_text('{"cells": []}', encoding="utf-8")
    notebook, code, path = rc.load_notebook(None, trial, {})
    assert notebook is None and code == [] and path is None


def test_v1_requires_exact_output_directory(tmp_path):
    output = tmp_path / "output"
    misplaced = tmp_path / "elsewhere" / "BP_0005_processed.nc"
    misplaced.parent.mkdir()
    misplaced.touch()
    ctx = _ctx([], outputs={misplaced.name: misplaced})
    ctx.output_dir = output
    ctx.all_output_paths = [misplaced]
    result = _by_name(check_save(ctx))["V1_outputs_in_place"]
    assert result.passed is False


def test_q4_skips_when_reference_is_unavailable(tmp_path):
    import numpy as np
    import xarray as xr

    path = tmp_path / "BP_0005_processed.nc"
    xr.DataArray(
        np.ones((3, 3)),
        dims=("eV", "kx"),
        coords={"eV": [-0.1, 0.0, 0.1], "kx": [-0.1, 0.0, 0.1]},
    ).to_netcdf(path)
    key = {**GOLD_KEY, "expected_outputs": GOLD_KEY["expected_outputs"][:1], "cut_indices": [5]}
    ctx = _ctx([], key=key, outputs={path.name: path})
    result = _by_name(check_quality(ctx))["Q4_matches_human_reference"]
    assert result.passed is None


def _batch_stem_receipt_cell(
    stems: tuple[str, ...] = ("BP_0005", "BP_0006", "BP_0009"),
) -> dict:
    value = ",".join(sorted(stems))
    return {
        "cell_type": "code",
        "execution_count": 1,
        "source": ["print('processed_stems=' + ','.join(sorted(cut_results)))\n"],
        "outputs": [{
            "output_type": "stream",
            "name": "stdout",
            "text": [f"processed_stems={value}\n"],
        }],
    }


def test_s4_requires_variables_cache_and_validation_in_final_markdown():
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "source": [
                    "Gold BP_0020. EF fitted. Theta offset 1.5. Processed 3 cuts. "
                    "Unprocessed: none. BP_0005_processed.nc"
                ],
            }
        ]
    }
    ctx = _ctx([])
    ctx.notebook = notebook
    result = _by_name(check_show(ctx))["S4_final_summary"]
    assert result.passed is False
    assert {"variables", "cache", "validation"} <= set(result.evidence)


def test_s4_accepts_notebook_variable_and_conversion_cache_summary():
    notebook = {
        "cells": [_batch_stem_receipt_cell(), {
            "cell_type": "markdown",
            "source": [
                "Gold BP_0020; EF fitted; theta_offset=1.5 deg from record.theta_offset_deg; "
                "processed 3 cuts. "
                "Notebook result dictionary cuts_processed; "
                "processed_stems=BP_0005,BP_0006,BP_0009. "
                "Cache created: conversion completed. Validation figures complete. Unprocessed: none."
            ],
        }]
    }
    ctx = _ctx([])
    ctx.notebook = notebook
    assert _by_name(check_show(ctx))["S4_final_summary"].passed is True


def test_s4_accepts_a_named_result_dictionary_as_live_variable_evidence():
    notebook = {
        "cells": [_batch_stem_receipt_cell(), {
            "cell_type": "markdown",
            "source": [
                "Gold BP_0020; EF fitted; theta_offset=1.5 deg from record.theta_offset_deg; "
                "processed 3 cuts. "
                "Result dictionary cuts_processed; "
                "processed_stems=BP_0005,BP_0006,BP_0009. "
                "Preconverted cache reused. Validation figures complete. Unprocessed: none."
            ],
        }]
    }
    ctx = _ctx([])
    ctx.notebook = notebook
    assert _by_name(check_show(ctx))["S4_final_summary"].passed is True


def test_s4_accepts_the_receipt_container_name_without_magic_phrase():
    notebook = {
        "cells": [_batch_stem_receipt_cell(), {
            "cell_type": "markdown",
            "source": [
                "Gold BP_0020; EF fitted; theta_offset=1.5 deg from record.theta_offset_deg; "
                "processed 3 cuts. All final cuts live in the stem-keyed dictionary "
                "`cut_results`; processed_stems=BP_0005,BP_0006,BP_0009. "
                "Preconverted cache reused. Validation figures complete. Unprocessed: none."
            ],
        }]
    }
    ctx = _ctx([])
    ctx.notebook = notebook

    assert _by_name(check_show(ctx))["S4_final_summary"].passed is True


def test_s4_rejects_an_unrelated_container_name():
    notebook = {
        "cells": [_batch_stem_receipt_cell(), {
            "cell_type": "markdown",
            "source": [
                "Gold BP_0020; EF fitted; theta_offset=1.5 deg from record.theta_offset_deg; "
                "processed 3 cuts. Final cuts are in the stem-keyed dictionary `other`; "
                "processed_stems=BP_0005,BP_0006,BP_0009. Preconverted cache reused. "
                "Validation figures complete. Unprocessed: none."
            ],
        }]
    }

    result = _by_name(check_show(_ctx([], notebook=notebook)))["S4_final_summary"]

    assert result.passed is False
    assert "result variable not named" in result.evidence


def test_s4_accepts_semantic_preconverted_cache_wording():
    notebook = {
        "cells": [_batch_stem_receipt_cell(), {
            "cell_type": "markdown",
            "source": [
                "Gold BP_0020; EF fitted; theta_offset=1.5 deg from record.theta_offset_deg; "
                "processed 3 cuts. "
                "Notebook result dictionary cuts_processed; "
                "processed_stems=BP_0005,BP_0006,BP_0009. "
                "All inputs had needs_conversion=False and used their NetCDF representation. "
                "Validation figures complete. Unprocessed: none."
            ],
        }]
    }
    manifest = {
        "case": {
            "scientific_contract": {
                "expected_call_counts": {"pxt2nc": 0},
            }
        }
    }

    result = _by_name(check_show(_ctx([], notebook=notebook, manifest=manifest)))[
        "S4_final_summary"
    ]

    assert result.passed is True


def test_s4_rejects_theta_source_without_observed_numeric_value():
    notebook = {
        "cells": [_batch_stem_receipt_cell(), {
            "cell_type": "markdown",
            "source": [
                "Gold BP_0020; EF fitted; processed 3 cuts. "
                "Angle source: record.theta_offset_deg. "
                "Result dictionary cuts_processed; "
                "processed_stems=BP_0005,BP_0006,BP_0009. "
                "Preconverted cache reused. Validation figures complete. Unprocessed: none."
            ],
        }]
    }

    result = _by_name(check_show(_ctx([], notebook=notebook)))["S4_final_summary"]

    assert result.passed is False
    assert result.evidence == ["theta"]


def test_s4_rejects_numeric_theta_without_exact_source_field():
    notebook = {
        "cells": [_batch_stem_receipt_cell(), {
            "cell_type": "markdown",
            "source": [
                "Gold BP_0020; EF fitted; theta offset 1.5 deg from metadata; processed 3 cuts. "
                "Result dictionary cuts_processed; "
                "processed_stems=BP_0005,BP_0006,BP_0009. "
                "Preconverted cache reused. Validation figures complete. Unprocessed: none."
            ],
        }]
    }

    result = _by_name(check_show(_ctx([], notebook=notebook)))["S4_final_summary"]

    assert result.passed is False
    assert result.evidence == ["theta"]


def test_s4_rejects_generic_gold_claim_without_selected_index_or_stem():
    notebook = {
        "cells": [_batch_stem_receipt_cell(), {
            "cell_type": "markdown",
            "source": [
                "Gold scan fitted once; EF fitted; "
                "theta_offset=1.5 deg from record.theta_offset_deg; processed 3 cuts. "
                "Result dictionary cuts_processed; "
                "processed_stems=BP_0005,BP_0006,BP_0009. "
                "Preconverted cache reused. Validation figures complete. Unprocessed: none."
            ],
        }]
    }

    result = _by_name(check_show(_ctx([], notebook=notebook)))["S4_final_summary"]

    assert result.passed is False
    assert result.evidence == ["gold"]


def test_s4_requires_one_live_processed_stems_receipt():
    summary = {
        "cell_type": "markdown",
        "source": [
            "Gold BP_0020; EF fitted; theta_offset=1.5 deg from record.theta_offset_deg; "
            "processed 3 cuts in result dictionary cuts_processed; "
            "processed_stems=BP_0005,BP_0006,BP_0009. "
            "Preconverted cache reused. Validation figures complete. Unprocessed: none."
        ],
    }

    result = _by_name(check_show(_ctx([], notebook={"cells": [summary]})))["S4_final_summary"]

    assert result.passed is False
    assert "processed_stems receipt count=0 (expected 1)" in result.evidence


def test_s4_lists_missing_stems_when_summary_infers_consecutive_numbers():
    notebook = {
        "cells": [
            _batch_stem_receipt_cell(),
            {
                "cell_type": "markdown",
                "source": [
                    "Gold BP_0020; EF fitted; "
                    "theta_offset=1.5 deg from record.theta_offset_deg; processed 3 cuts. "
                    "Result dictionary cuts_processed; processed_stems=BP_0001,BP_0002,BP_0003. "
                    "Preconverted cache reused. Validation figures complete. Unprocessed: none."
                ],
            },
        ]
    }

    result = _by_name(check_show(_ctx([], notebook=notebook)))["S4_final_summary"]

    assert result.passed is False
    assert {"missing_stem:BP_0005", "missing_stem:BP_0006", "missing_stem:BP_0009"} <= set(
        result.evidence
    )
    assert "final Markdown did not reuse the exact processed_stems token" in result.evidence


def _figure_output(payload: str, label: str) -> dict:
    return {
        "output_type": "display_data",
        "data": {"image/png": payload, "text/plain": label},
        "metadata": {},
    }


def _three_review_figures() -> list[dict]:
    return [
        _figure_output("gold", "<Figure size 1200x900 with 7 Axes>"),
        _figure_output("grid", "<Figure size 1500x720 with 15 Axes>"),
        _figure_output("validation", "<Figure size 1100x400 with 2 Axes>"),
    ]


def test_s5_rejects_large_notebook_text_dumps():
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": 1,
                "source": ["print(report)"],
                "outputs": [
                    {"output_type": "stream", "name": "stdout", "text": ["x" * 4_001]}
                ],
            }
        ]
    }
    ctx = _ctx([])
    ctx.notebook = notebook

    result = _by_name(check_show(ctx))["S5_notebook_readable"]

    assert result.passed is False
    assert "largest cell 4001 chars" in result.detail


def test_s5_rejects_more_than_three_stdout_lines_even_when_short():
    notebook = {
        "cells": [{
            "cell_type": "code",
            "execution_count": 1,
            "source": ["print(summary)"],
            "outputs": [{
                "output_type": "stream",
                "name": "stdout",
                "text": ["one\ntwo\nthree\nfour\n"],
            }],
        }]
    }

    result = _by_name(check_show(_ctx([], notebook=notebook)))["S5_notebook_readable"]

    assert result.passed is False
    assert "max stdout 4" in result.detail


def test_s5_rejects_a_201_character_stdout_line():
    notebook = {
        "cells": [{
            "cell_type": "code",
            "execution_count": 1,
            "source": ["print(summary)"],
            "outputs": [{
                "output_type": "stream",
                "name": "stdout",
                "text": ["x" * 201 + "\n"],
            }],
        }]
    }

    result = _by_name(check_show(_ctx([], notebook=notebook)))["S5_notebook_readable"]

    assert result.passed is False
    assert "longest stdout line 201 chars" in result.detail


def test_s5_accepts_a_200_character_stdout_line():
    notebook = {
        "cells": [{
            "cell_type": "code",
            "execution_count": 1,
            "source": ["print(summary)"],
            "outputs": [
                {
                    "output_type": "stream",
                    "name": "stdout",
                    "text": ["x" * 200 + "\n"],
                },
                *_three_review_figures(),
            ],
        }]
    }

    result = _by_name(check_show(_ctx([], notebook=notebook)))["S5_notebook_readable"]

    assert result.passed is True


@pytest.mark.parametrize(
    ("markdown_chars", "expected"),
    [(2_000, True), (2_001, False)],
)
def test_s5_enforces_final_markdown_character_limit(markdown_chars, expected):
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": 1,
                "source": ["render_review_figures()"],
                "outputs": _three_review_figures(),
            },
            {
                "cell_type": "markdown",
                "source": ["x" * markdown_chars],
            },
        ]
    }

    result = _by_name(check_show(_ctx([], notebook=notebook)))["S5_notebook_readable"]

    assert result.passed is expected
    assert f"final Markdown {markdown_chars} chars" in result.detail


def test_s5_accepts_exactly_three_unique_static_review_figures():
    notebook = {
        "cells": [{
            "cell_type": "code",
            "execution_count": 1,
            "source": ["render_review_figures()"],
            "outputs": _three_review_figures(),
        }]
    }

    result = _by_name(check_show(_ctx([], notebook=notebook)))["S5_notebook_readable"]

    assert result.passed is True
    assert "static figures 3/3" in result.detail
    assert "semantic figures 3/3" in result.detail
    assert "widget outputs 0" in result.detail


@pytest.mark.parametrize("count", [2, 4])
def test_s5_rejects_missing_or_excess_static_figure_outputs(count):
    figures = _three_review_figures()
    if count == 2:
        figures.pop()
    else:
        figures.append(_figure_output("extra", "<Figure size 640x480 with 1 Axes>"))
    notebook = {
        "cells": [{
            "cell_type": "code",
            "execution_count": 1,
            "source": ["render_review_figures()"],
            "outputs": figures,
        }]
    }

    result = _by_name(check_show(_ctx([], notebook=notebook)))["S5_notebook_readable"]

    assert result.passed is False
    assert f"static figures {count}/3" in result.detail


def test_s5_rejects_adjacent_semantic_duplicate_figures_with_different_payloads():
    notebook = {
        "cells": [{
            "cell_type": "code",
            "execution_count": 1,
            "source": ["figure"],
            "outputs": [
                _figure_output("gold", "<Figure size 1200x900 with 7 Axes>"),
                _figure_output("first-render", "<Figure size 1500x720 with 15 Axes>"),
                _figure_output("second-render", "<Figure size 1500x720 with 15 Axes>"),
            ],
        }]
    }

    result = _by_name(check_show(_ctx([], notebook=notebook)))["S5_notebook_readable"]

    assert result.passed is False
    assert "static figures 3/3" in result.detail
    assert "semantic figures 2/3" in result.detail
    assert "duplicate figure outputs 1" in result.detail
    assert result.evidence == ["duplicate figure: cell 1 output 3"]


def test_s5_rejects_per_cut_jupyter_widget_outputs():
    widgets = [
        {
            "output_type": "display_data",
            "data": {
                "application/vnd.jupyter.widget-view+json": (
                    {} if index == 0 else {"model_id": f"cut-{index}"}
                ),
                "text/plain": "Converting data to k-space: 0%",
            },
        }
        for index in range(14)
    ]
    notebook = {
        "cells": [{
            "cell_type": "code",
            "execution_count": 1,
            "source": ["convert_all_cuts()"],
            "outputs": [*widgets, *_three_review_figures()],
        }]
    }

    result = _by_name(check_show(_ctx([], notebook=notebook)))["S5_notebook_readable"]

    assert result.passed is False
    assert "widget outputs 14" in result.detail
    assert result.evidence[0] == "widget: cell 1 output 1"


def test_conversion_cache_requires_one_successful_conversion_and_cache_files(tmp_path):
    trial = tmp_path / "trial"
    input_dir = trial / "workspace" / "input"
    cache_dir = trial / "workspace" / "input_netcdf"
    input_dir.mkdir(parents=True)
    cache_dir.mkdir()
    raw = input_dir / "BP_0005.pxt"
    raw.write_bytes(b"raw")
    (cache_dir / "BP_0005.nc").write_bytes(b"cache")
    ctx = _ctx(
        ["report = peaks.pxt2nc('input')"],
        live_evidence={
            "status": "ok",
            "conversion_reports": [{"converted": 1, "cached": 0, "failed": 0}],
        },
    )
    ctx.run_dir = trial
    ctx.manifest = {
        "paths": {"input": str(input_dir)},
        "frozen": {"raw_sha256": {raw.name: rc.sha256_file(raw)}},
        "case": {"scientific_contract": {"expected_call_counts": {"pxt2nc": 1}}},
    }
    checks = _by_name(check_conversion_cache(ctx))
    assert checks["V5_raw_inputs_immutable"].passed is True
    assert checks["V6_pxt_cache_reused"].passed is True


def test_preconverted_case_requires_zero_pxt_calls_and_immutable_netcdf(tmp_path):
    trial = tmp_path / "trial"
    input_dir = trial / "workspace" / "input"
    input_dir.mkdir(parents=True)
    (input_dir / "BP_0005.nc").write_bytes(b"converted")
    manifest = {
        "paths": {"input": str(input_dir)},
        "case": {
            "scientific_contract": {
                "expected_call_counts": {"pxt2nc": 0},
            }
        },
        "frozen": {
            "raw_sha256": {},
            "input_manifest": rc.build_input_manifest(input_dir),
        },
    }
    ctx = _ctx(["exp = peaks.load_experiment(INPUT)"], manifest=manifest)
    ctx.run_dir = trial

    checks = _by_name(check_conversion_cache(ctx))

    assert checks["V5_raw_inputs_immutable"].passed is True
    assert checks["V6_pxt_cache_reused"].passed is True
    assert _by_name(check_contract(ctx))["C1_blackbox_load"].passed is True


def test_quality_uses_live_metrics_without_processed_files(monkeypatch):
    key = {**GOLD_KEY, "expected_outputs": GOLD_KEY["expected_outputs"][:1], "cut_indices": [5]}
    evidence = {
        "status": "ok",
        "processed_stems": ["BP_0005"],
        "products": {
            "BP_0005": {
                "dims": ["eV", "kx"], "ef_landmark": 0.0, "kx_landmark": 0.0,
                "same_dims": True, "coord_delta": 0.0, "mask_overlap": 1.0,
                "efficiency": 1.0, "corr": 0.99,
            }
        },
    }
    monkeypatch.setattr(rc, "_q4_oracle", lambda: {"thresholds": {}})
    checks = _by_name(check_quality(_ctx([], key=key, live_evidence=evidence)))
    assert all(checks[name].passed is True for name in (
        "Q1_kspace_dims", "Q2_ef_zeroed", "Q3_theta_zeroed", "Q4_matches_human_reference"
    ))


def test_live_evidence_prefers_stem_keyed_result_over_transient_plot_view(
    monkeypatch, tmp_path
):
    import contextlib
    import io

    import jupyter_client
    import numpy as np
    import xarray as xr

    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    eV = np.linspace(-0.3, 0.3, 7)
    kx = np.linspace(-1.0, 1.0, 5)
    full = xr.DataArray(
        np.arange(eV.size * kx.size, dtype=float).reshape(eV.size, kx.size),
        dims=("eV", "kx"),
        coords={"eV": eV, "kx": kx},
        attrs={"source_path": "/input/BP_0010.nc"},
    )
    full.to_netcdf(reference_dir / "BP_0010_processed.nc")
    user_ns = {
        "result_dict": {"BP_0010": full},
        "plot_view": full.sel(eV=slice(-0.1, 0.1)),
    }

    class FakeKernelClient:
        def __init__(self, **_kwargs):
            self.messages = []

        def load_connection_file(self):
            return None

        def start_channels(self):
            return None

        def wait_for_ready(self, timeout):
            return timeout

        def execute(self, code, **_kwargs):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exec(code, {"get_ipython": lambda: SimpleNamespace(user_ns=user_ns)})
            self.messages = [
                {
                    "parent_header": {"msg_id": "message-id"},
                    "msg_type": "stream",
                    "content": {"text": output.getvalue()},
                },
                {
                    "parent_header": {"msg_id": "message-id"},
                    "msg_type": "status",
                    "content": {"execution_state": "idle"},
                },
            ]
            return "message-id"

        def get_iopub_msg(self, timeout):
            assert timeout > 0
            return self.messages.pop(0)

        def stop_channels(self):
            return None

    monkeypatch.setattr(jupyter_client, "BlockingKernelClient", FakeKernelClient)
    monkeypatch.setattr(
        "jupyter_client.connect.find_connection_file", lambda *_args, **_kwargs: "kernel.json"
    )
    trial_dir = tmp_path / "trial"
    (trial_dir / "evaluator").mkdir(parents=True)

    evidence = run_campaign.capture_live_kernel_evidence(
        trial_dir,
        {"kernel_id": "fake"},
        {"expected_outputs": [{"stem": "BP_0010"}]},
        reference_dir,
    )

    product = evidence["products"]["BP_0010"]
    assert product["shape"] == list(full.shape)
    assert product["coord_delta"] == pytest.approx(0.0)
    assert product["mask_overlap"] == pytest.approx(1.0)
    assert product["corr"] == pytest.approx(1.0)


def test_file_fallback_cannot_pass_when_transposed_wrong_values_skip_metrics(
    monkeypatch, tmp_path
):
    import numpy as np
    import xarray as xr

    output_dir = tmp_path / "output"
    reference_dir = tmp_path / "reference"
    output_dir.mkdir()
    reference_dir.mkdir()
    name = "BP_0005_processed.nc"
    reference = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("eV", "kx"),
        coords={"eV": [-0.1, 0.1], "kx": [-1.0, 0.0, 1.0]},
    )
    wrong = xr.DataArray(
        np.flip(reference.values).T,
        dims=("kx", "eV"),
        coords={"kx": reference.kx, "eV": reference.eV},
    )
    reference.to_netcdf(reference_dir / name)
    wrong.to_netcdf(output_dir / name)
    key = {**GOLD_KEY, "expected_outputs": GOLD_KEY["expected_outputs"][:1], "cut_indices": [5]}
    ctx = _ctx([], key=key, outputs={name: output_dir / name})
    ctx.reference_dir = reference_dir
    monkeypatch.setattr(rc, "_q4_oracle", lambda: {"thresholds": {"corr_min": 0.98}})

    result = _by_name(check_quality(ctx))["Q4_matches_human_reference"]

    assert result.passed is False


def test_q5_requires_balanced_gold_window_and_low_outlier_fraction():
    evidence = {
        "status": "ok",
        "gold_fits": [{
            "variable": "gold_fit",
            "fit_window": {
                "start_eV": 2.623,
                "center_eV": 2.662,
                "stop_eV": 2.701,
                "lower_points": 13,
                "upper_points": 13,
                "total_points": 27,
            },
            "outlier_fraction": 24 / 902,
            "uniform": True,
        }],
    }
    result = _by_name(check_quality(_ctx([], live_evidence=evidence)))[
        "Q5_gold_fit_balanced"
    ]
    assert result.passed is True

    evidence["gold_fits"][0]["fit_window"]["upper_points"] = 4
    result = _by_name(check_quality(_ctx([], live_evidence=evidence)))[
        "Q5_gold_fit_balanced"
    ]
    assert result.passed is False


def test_autonomy_fails_on_scientific_hint():
    ctx = _ctx([])
    ctx.manifest = {"isolation": {"enforced": True, "mechanism": "sandbox"}}
    ctx.interventions = [{"type": "scientific_hint"}]
    result = _by_name(check_autonomy(ctx))["U1_no_assistive_intervention"]
    assert result.passed is False


def test_autonomy_rejects_absolute_paths_outside_trial_workspace(tmp_path):
    trial = tmp_path / "trial"
    (trial / "workspace").mkdir(parents=True)
    (trial / "evaluator").mkdir()
    ctx = _ctx(["import os\nlist(os.walk('/Users/example/project/source'))"])
    ctx.run_dir = trial
    ctx.manifest = {
        "isolation": {"enforced": True, "mechanism": "MCP-only agent surface"},
    }

    result = _by_name(check_autonomy(ctx))["U2_isolation_enforced"]

    assert result.passed is False
    assert "/Users/example/project/source" in result.evidence


def test_autonomy_does_not_treat_a_count_separator_as_root_access(tmp_path):
    trial = tmp_path / "trial"
    (trial / "workspace").mkdir(parents=True)
    (trial / "evaluator").mkdir()
    ctx = _ctx(['print("processed", 14, "/", 14)'])
    ctx.run_dir = trial
    ctx.manifest = {
        "isolation": {"enforced": True, "mechanism": "MCP-only agent surface"},
    }

    result = _by_name(check_autonomy(ctx))["U2_isolation_enforced"]

    assert result.passed is True
    assert result.evidence == []


def test_autonomy_rejects_root_single_component_and_propagated_io_paths(tmp_path):
    trial = tmp_path / "trial"
    (trial / "workspace").mkdir(parents=True)
    (trial / "evaluator").mkdir()
    sources = (
        "import os\nos.listdir('/')",
        "from peaks import load_experiment\nsource = '/tmp'\nload_experiment(source)",
        "from pathlib import Path\nsource = Path('/tmp')\nsource.read_text()",
    )

    for source in sources:
        ctx = _ctx([source])
        ctx.run_dir = trial
        ctx.manifest = {
            "isolation": {"enforced": True, "mechanism": "MCP-only agent surface"},
        }
        result = _by_name(check_autonomy(ctx))["U2_isolation_enforced"]
        assert result.passed is False, source
        assert result.evidence, source


def test_autonomy_ignores_path_like_text_but_allows_workspace_io(tmp_path):
    trial = tmp_path / "trial"
    workspace = trial / "workspace"
    workspace.mkdir(parents=True)
    (trial / "evaluator").mkdir()
    source = (
        "from peaks import load_experiment\n"
        f"source = {str(workspace / 'input')!r}\n"
        "print('/', '/tmp/not-an-access')\n"
        "load_experiment(source)\n"
    )
    ctx = _ctx([source])
    ctx.run_dir = trial
    ctx.manifest = {
        "isolation": {"enforced": True, "mechanism": "MCP-only agent surface"},
    }

    result = _by_name(check_autonomy(ctx))["U2_isolation_enforced"]

    assert result.passed is True
    assert result.evidence == []


def test_approval_harness_is_fail_closed(tmp_path):
    output = tmp_path / "output"
    expected = {"BP_0005_processed.nc"}
    target = output / "BP_0005_processed.nc"
    approved = decide_dialog("peaksMCP — 保存确认", f"路径\n{target}\n", output, expected)
    assert approved["decision"] == "approve"
    denied = decide_dialog(
        "peaksMCP — 保存确认",
        f"路径\n{tmp_path / 'wrong' / target.name}\n",
        output,
        expected,
    )
    assert denied["decision"] == "deny"


def test_approval_harness_uses_stable_marker_and_parses_any_netcdf(tmp_path):
    output = tmp_path / "output"
    final = output / "BP_0005_processed.nc"
    conversion = output / "BP_0005.nc"
    approved = decide_dialog(
        "localized title",
        f"path\n{final}\n",
        output,
        {final.name},
        "save-consent",
    )
    assert approved["decision"] == "approve"
    denied = decide_dialog(
        "peaksMCP save confirmation",
        f"path\n{conversion}\n",
        output,
        {final.name},
    )
    assert denied["decision"] == "deny"
    assert denied["reason"].endswith("expected_name=False")


def test_approval_harness_empty_allowlist_denies_expected_product(tmp_path):
    output = tmp_path / "output"
    final = output / "BP_0005_processed.nc"

    denied = decide_dialog(
        "peaksMCP save confirmation",
        f"path\n{final}\n",
        output,
        set(),
        "save-consent",
    )

    assert denied["decision"] == "deny"
    assert denied["reason"].endswith("expected_name=False")


def test_no_save_campaign_harness_omits_expected_file_flags(tmp_path, monkeypatch):
    trial = tmp_path / "trial"
    operator = trial / "operator"
    output = trial / "workspace" / "output"
    operator.mkdir(parents=True)
    output.mkdir(parents=True)
    captured = {}
    process = SimpleNamespace()

    monkeypatch.setattr(
        run_campaign.rc,
        "load_manifest",
        lambda _trial: {"paths": {"output": str(output)}},
    )
    monkeypatch.setattr(run_campaign.rc, "operator_dir", lambda _trial: operator)
    monkeypatch.setattr(run_campaign, "_wait_for_file", lambda *_args: None)

    def fake_popen(command, **_kwargs):
        captured["command"] = command
        return process

    monkeypatch.setattr(run_campaign.subprocess, "Popen", fake_popen)

    actual, _stop = run_campaign.start_approval_harness(
        trial,
        "http://127.0.0.1:8765/?token=test",
        headed=False,
        timeout=30,
    )

    assert actual is process
    assert "--expected-file" not in captured["command"]


def test_approval_harness_cli_accepts_an_empty_allowlist(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        approval_harness,
        "run_harness",
        lambda args: captured.setdefault("expected_file", args.expected_file) or 0,
    )

    result = approval_harness.main(
        [
            "--notebook-url",
            "http://127.0.0.1:8765/",
            "--output-dir",
            str(tmp_path / "output"),
            "--log",
            str(tmp_path / "approvals.jsonl"),
            "--ready-file",
            str(tmp_path / "ready.json"),
            "--stop-file",
            str(tmp_path / "stop"),
        ]
    )

    assert result == 0
    assert captured["expected_file"] == []


def test_approval_harness_opens_ready_lab_link_without_page_eval():
    calls = []
    lab_page = object()

    class Locator:
        def __init__(self, selector):
            self.selector = selector

        def count(self):
            return 1

        def wait_for(self, **kwargs):
            calls.append(("wait_for", self.selector, kwargs))

        def click(self):
            calls.append(("click", self.selector))

    class Popup:
        value = lab_page

        def __enter__(self):
            calls.append(("popup_enter",))
            return self

        def __exit__(self, *_args):
            calls.append(("popup_exit",))

    class Page:
        def locator(self, selector):
            return Locator(selector)

        def expect_popup(self, **kwargs):
            calls.append(("expect_popup", kwargs))
            return Popup()

        def wait_for_function(self, *_args, **_kwargs):
            raise AssertionError("page JavaScript evaluation violates the dashboard CSP")

    result = open_notebook_from_dashboard(Page(), 12_000)

    assert result is lab_page
    assert calls == [
        (
            "wait_for",
            "#open-lab[href]:not([href=''])",
            {"state": "attached", "timeout": 12_000},
        ),
        ("expect_popup", {"timeout": 12_000}),
        ("popup_enter",),
        ("click", "#open-lab"),
        ("popup_exit",),
    ]


def test_trial_manifest_freezes_the_imported_peaks_repository(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "BP_0001.nc").write_bytes(b"scan")
    datasheet = tmp_path / "datasheet.csv"
    datasheet.write_text("Index\n1\n", encoding="utf-8")
    case_file = tmp_path / "case.yaml"
    case_file.write_text("id: synthetic\n", encoding="utf-8")
    case = {
        "id": "synthetic",
        "staging": {"strategy": "copy", "include_suffixes": [".nc"]},
    }
    peaks_state = {
        "module_path": "/checkout/peaks/peaks/__init__.py",
        "repository_root": "/checkout/peaks",
        "git_head": "peaks-head",
        "dirty": True,
        "dirty_fingerprint": "peaks-dirty",
        "untracked": [],
    }
    monkeypatch.setattr(rc, "load_yaml", lambda _path: case)
    monkeypatch.setattr(rc, "build_answer_key", lambda *_args: GOLD_KEY)
    monkeypatch.setattr(rc, "_git_state", lambda *_args: {"git_head": "mcp-head"})
    monkeypatch.setattr(rc, "_imported_peaks_git_state", lambda: peaks_state)
    args = SimpleNamespace(
        case=str(case_file),
        data=str(source),
        datasheet=str(datasheet),
        reference=None,
        condition="u1",
        name="trial",
        runs=str(tmp_path / "runs"),
        force=False,
    )

    run_dir, _ = rc.initialize_trial(args)
    frozen = rc.load_manifest(run_dir)["frozen"]

    assert frozen["source"] == {"git_head": "mcp-head"}
    assert frozen["peaks"] == peaks_state
    assert frozen["peaks"]["git_head"] == "peaks-head"
    assert frozen["peaks"]["dirty_fingerprint"] == "peaks-dirty"


def test_stage_case_input_copies_only_configured_files(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "BP_0001.nc").write_bytes(b"scan")
    (source / "BP_0001_processed.nc").write_bytes(b"reference")
    (source / ".DS_Store").write_bytes(b"noise")
    datasheet = tmp_path / "datasheet.csv"
    datasheet.write_text("Index\n1\n", encoding="utf-8")
    destination = tmp_path / "trial" / "input"
    case = {
        "staging": {
            "strategy": "copy",
            "include_suffixes": [".nc"],
            "exclude_globs": ["*_processed.nc"],
            "include_datasheet": True,
        }
    }

    staged = rc.stage_case_input(source, destination, datasheet, case)

    assert [path.name for path in staged] == ["BP_0001.nc", "datasheet.csv"]
    assert (destination / "BP_0001.nc").read_bytes() == b"scan"
    assert not (destination / "BP_0001_processed.nc").exists()
    assert not (destination / ".DS_Store").exists()


def test_bp260623_warm_case_stages_28_netcdf_inputs_and_datasheet(tmp_path):
    case = rc.load_yaml(rc.resolve_case_file("bp260623_warm"))
    source = tmp_path / "source"
    source.mkdir()
    for index in range(1, 29):
        (source / f"BP_{index:04d}.nc").write_bytes(b"scan")
        (source / f"BP_{index:04d}_processed.nc").write_bytes(b"reference")
    (source / "BP_0001.pxt").write_bytes(b"raw")
    datasheet = tmp_path / "datasheet.csv"
    datasheet.write_text("Index\n1\n", encoding="utf-8")
    destination = tmp_path / "trial" / "input"

    staged = rc.stage_case_input(source, destination, datasheet, case)
    staged_names = [path.name for path in staged]

    assert case["parser"]["input_suffixes"] == [".nc"]
    assert case["staging"] == {
        "strategy": "copy",
        "include_suffixes": [".nc"],
        "exclude_globs": ["*_processed.nc"],
        "include_datasheet": True,
    }
    assert len([name for name in staged_names if name.endswith(".nc")]) == 28
    assert "datasheet.csv" in staged_names
    assert not any(name.endswith("_processed.nc") for name in staged_names)
    assert case["scientific_contract"]["expected_call_counts"] == {
        "pxt2nc": 0,
        "load_experiment": 1,
        "fit_gold": 1,
        "assign_normal_emission": 1,
        "k_convert": 1,
    }


def test_p2_prompt_includes_the_same_task_body(tmp_path):
    rendered = rc.render_prompt(
        "p2",
        run_id="trial",
        input_dir=tmp_path / "input",
        output_dir=tmp_path / "output",
        notebook_path=tmp_path / "work.ipynb",
    )
    assert "Completion requirements" in rendered
    assert "Fit the gold\n   exactly once" in rendered
    assert "five operations" in rendered


def test_prompt_conditions_share_common_body_and_keep_p1_tool_agnostic(tmp_path):
    kwargs = {
        "run_id": "trial",
        "input_dir": tmp_path / "input",
        "output_dir": tmp_path / "output",
        "notebook_path": tmp_path / "work.ipynb",
    }
    rendered = {condition: rc.render_prompt(condition, **kwargs) for condition in ("p1", "p2")}
    common = rc.COMMON_PROMPT_FILE.read_text(encoding="utf-8").strip()
    for marker, value in {
        "{RUN_ID}": kwargs["run_id"],
        "{INPUT_DIR}": str(kwargs["input_dir"]),
        "{OUTPUT_DIR}": str(kwargs["output_dir"]),
        "{NOTEBOOK_PATH}": str(kwargs["notebook_path"]),
    }.items():
        common = common.replace(marker, value)
    common += "\n"

    assert rendered["p1"].endswith(common)
    assert rendered["p2"].endswith(common)
    p1_prefix = rendered["p1"][: -len(common)]
    p2_prefix = rendered["p2"][: -len(common)]
    for tool in ("search", "get", "inspect_notebook", "run_cell", "save_with_consent"):
        assert tool not in p1_prefix, tool
        assert tool in p2_prefix, tool
    assert 'cell_type="markdown"' in p2_prefix


def test_u1_prompt_is_a_standalone_natural_user_request(tmp_path):
    rendered = rc.render_prompt(
        "u1",
        run_id="trial",
        input_dir=tmp_path / "converted",
        output_dir=tmp_path / "output",
        notebook_path=tmp_path / "work.ipynb",
    )

    assert f"preprocess all 2D data in {tmp_path / 'converted'}" in rendered
    assert str(tmp_path / "work.ipynb") in rendered
    assert "runtime check and peaksmcp instructions are already complete" in rendered.lower()
    assert "do not repeat status or initial notebook checks" in rendered.lower()
    assert "do not inspect or list input files" in rendered.lower()
    assert "Completion requirements" not in rendered
    assert "fit_gold" not in rendered
    assert "pxt2nc" not in rendered


def test_score_reports_conservative_and_evidence_coverage():
    results = [rc.Result("a", True), rc.Result("b", None)]
    rubric = {
        "a": {"subsystem": "X", "weight": 1},
        "b": {"subsystem": "X", "weight": 1},
    }
    score = rc.score(results, rubric)
    assert score["observed"] == 1.0
    assert score["conservative"] == 0.5
    assert score["evidence_coverage"] == 0.5


def test_rubric_makes_redundancy_and_pi_client_errors_strict():
    document = rc._rubric_document()
    strict = set(document["strict_checks"])

    assert {"A4_no_redundant_tool_calls", "A5_no_client_tool_errors"} <= strict
    assert "A5_no_client_tool_errors" in document["checks"]


def test_pi_command_enforces_mcp_only_isolation(tmp_path, monkeypatch):
    monkeypatch.setattr(
        run_campaign.rc,
        "load_manifest",
        lambda _trial: {"run_id": "trial-1"},
    )
    args = argparse.Namespace(
        pi_executable="pi",
        provider=None,
        model=None,
        thinking=None,
    )
    command = run_campaign.pi_command(args, tmp_path / "trial", "perform the task")

    assert "--no-builtin-tools" in command
    assert command[command.index("--tools") + 1] == "mcp,mcpScript"
    assert "--no-context-files" in command
    assert "--no-skills" in command
    assert "--no-prompt-templates" in command


def test_pi_tui_bootstrap_names_one_unambiguous_health_probe(tmp_path, monkeypatch):
    monkeypatch.setattr(
        run_campaign.rc,
        "load_manifest",
        lambda _trial: {"run_id": "trial-1"},
    )
    args = argparse.Namespace(
        pi_executable="pi",
        provider=None,
        model=None,
        thinking=None,
        runner="pi-tui",
    )

    command = run_campaign.pi_command(args, tmp_path / "trial", "perform the task")
    bootstrap = command[-1]

    assert "peaksMCP_inspect_notebook" in bootstrap
    assert 'target="kernel" exactly once' in bootstrap
    assert 'instructions="peaksMCP" exactly once' in bootstrap
    assert "Do not list or search tools" in bootstrap
    assert "guess another status-tool name" in bootstrap


def test_tui_prompt_uses_bracketed_paste_and_a_separate_submit_key():
    class FakeProcess:
        def __init__(self):
            self.sent = []

        def send(self, value):
            self.sent.append(value)

    process = FakeProcess()
    run_campaign._submit_tui_prompt(process, "abcdef", chunk_size=2)

    assert process.sent == ["\x1b[200~", "ab", "cd", "ef", "\x1b[201~", "\r"]


def test_tui_slash_command_uses_carriage_return_instead_of_sendline():
    class FakeProcess:
        def __init__(self):
            self.sent = []

        def send(self, value):
            self.sent.append(value)

        def sendline(self, _value):
            raise AssertionError("sendline emits LF, which Pi TUI does not submit")

    process = FakeProcess()
    run_campaign._submit_tui_command(process, "/quit")

    assert process.sent == ["/quit", "\r"]


def test_preflight_reports_missing_peaks_executable_without_traceback(
    monkeypatch, tmp_path, capsys
):
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    (data_dir / "BP_0001.pxt").write_bytes(b"raw")
    datasheet = tmp_path / "datasheet.csv"
    datasheet.write_text("index,type\n1,cut\n", encoding="utf-8")
    case = {
        "staging": {"strategy": "copy", "include_suffixes": [".pxt"]},
    }
    paths = {"data": data_dir, "datasheet": datasheet, "reference": None}

    monkeypatch.setattr(run_campaign.rc, "resolve_case_file", lambda _case: tmp_path / "case.yaml")
    monkeypatch.setattr(run_campaign.rc, "load_yaml", lambda _path: case)
    monkeypatch.setattr(
        run_campaign.rc,
        "configured_path",
        lambda _case, name, _override, required: paths[name],
    )
    monkeypatch.setattr(
        run_campaign.importlib.util,
        "find_spec",
        lambda name: None if name == "playwright" else SimpleNamespace(),
    )
    monkeypatch.setattr(
        run_campaign.shutil,
        "which",
        lambda name: "/usr/bin/pi" if name == "pi" else None,
    )

    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="Self-test passed\n", stderr="")

    monkeypatch.setattr(run_campaign.subprocess, "run", fake_run)
    args = argparse.Namespace(
        case="case.yaml",
        pi_executable="pi",
        peaks_executable="missing-peaksMCP",
        require_live_mcp=True,
    )

    assert run_campaign.preflight(args) == 1

    payload = json.loads(capsys.readouterr().out)
    checks = {check["check"]: check for check in payload["checks"]}
    assert payload["passed"] is False
    assert checks["peaks_executable"] == {
        "check": "peaks_executable",
        "passed": False,
        "required": True,
        "detail": "executable not found: 'missing-peaksMCP'",
    }
    assert checks["live_mcp"]["passed"] is False
    assert checks["live_mcp"]["detail"] == (
        "not attempted because the peaksMCP executable is unavailable"
    )
    assert len(calls) == 1
    assert calls[0][-1] == "selftest"


def test_restore_managed_host_stops_temporary_host(monkeypatch):
    from peaksMCP import cli

    stopped = []
    monkeypatch.setattr(cli, "_runfile", lambda _required=False: {"pid": 42})
    monkeypatch.setattr(cli, "_terminate_supervisor", stopped.append)

    result = run_campaign.restore_managed_host(None, timeout=1)

    assert result == {"status": "stopped_temporary_host"}
    assert stopped == [42]


def test_restore_managed_host_recreates_previous_workspace(monkeypatch, tmp_path):
    from peaksMCP import cli

    captured = []
    monkeypatch.setattr(cli, "_runfile", lambda _required=False: {"pid": 42})

    def ensure(args):
        captured.append(args)
        return {
            "profile": args.profile,
            "root_dir": args.root_dir,
            "notebook_path": args.notebook,
        }

    monkeypatch.setattr(cli, "_ensure_host", ensure)
    previous = {
        "profile": "default",
        "root_dir": str(tmp_path),
        "notebook": "peaksMCP-runtime.ipynb",
    }

    result = run_campaign.restore_managed_host(previous, timeout=12)

    assert result["status"] == "restored"
    assert captured[0].root_dir == str(tmp_path)
    assert captured[0].notebook == "peaksMCP-runtime.ipynb"


@pytest.mark.parametrize(
    "trial_result",
    [
        {"strict_success": False, "validity": {"valid": True}},
        {"strict_success": True, "validity": {"valid": False}},
    ],
)
def test_stop_on_non_strict_stops_before_next_trial_and_restores_host(
    monkeypatch, tmp_path, capsys, trial_result
):
    campaign_dir = tmp_path / "campaign"
    campaign = {
        "agent": {
            "provider": "deepseek",
            "model": "deepseek-v4-flash",
            "thinking": "low",
        },
        "trials": [
            {
                "run_id": "r001-u1",
                "replicate": 1,
                "condition": "u1",
                "order": 1,
                "path": str(campaign_dir / "trials" / "r001-u1"),
                "status": "initialized",
            },
            {
                "run_id": "r002-u1",
                "replicate": 2,
                "condition": "u1",
                "order": 1,
                "path": str(campaign_dir / "trials" / "r002-u1"),
                "status": "initialized",
            },
        ],
    }
    events = []
    monkeypatch.setattr(run_campaign, "load_campaign", lambda _path: (campaign_dir, campaign))
    monkeypatch.setattr(
        run_campaign,
        "capture_managed_host",
        lambda: {"profile": "default", "root_dir": "/before", "notebook": "work.ipynb"},
    )

    def fake_run_one(_args, trial):
        events.append(("run", trial["run_id"]))
        return trial_result

    monkeypatch.setattr(run_campaign, "run_one_trial", fake_run_one)
    monkeypatch.setattr(
        run_campaign,
        "save_campaign",
        lambda _dir, payload: events.append(
            ("save", payload["trials"][0]["status"], payload.get("stack_restoration"))
        ),
    )
    monkeypatch.setattr(
        run_campaign,
        "summarize_campaign_dir",
        lambda _dir, payload: events.append(
            (
                "summarize",
                payload["trials"][0]["valid"],
                payload["trials"][0]["strict_success"],
            )
        ),
    )
    monkeypatch.setattr(
        run_campaign,
        "restore_managed_host",
        lambda previous, timeout: events.append(("restore", previous, timeout))
        or {"status": "restored"},
    )
    args = argparse.Namespace(
        campaign=str(campaign_dir),
        manage_stack=True,
        provider=None,
        model=None,
        thinking=None,
        runner="pi-tui",
        conditions=None,
        keep_going=False,
        stop_on_non_strict=True,
        stack_timeout=180,
    )

    assert run_campaign.run_campaign(args) == 1

    assert [event for event in events if event[0] == "run"] == [("run", "r001-u1")]
    assert campaign["trials"][0]["status"] == "graded"
    assert campaign["trials"][1]["status"] == "initialized"
    assert next(i for i, event in enumerate(events) if event[0] == "save") < next(
        i for i, event in enumerate(events) if event[0] == "summarize"
    )
    assert next(i for i, event in enumerate(events) if event[0] == "summarize") < next(
        i for i, event in enumerate(events) if event[0] == "restore"
    )
    assert campaign["stack_restoration"] == {"status": "restored"}
    assert "Stopping before the next trial" in capsys.readouterr().out


def test_stop_on_non_strict_is_opt_in():
    parser = run_campaign.build_parser()

    assert parser.parse_args(["run", "campaign"]).stop_on_non_strict is False
    assert (
        parser.parse_args(["run", "campaign", "--stop-on-non-strict"]).stop_on_non_strict
        is True
    )


def test_stop_on_non_strict_blocks_resume_after_an_existing_failure(
    monkeypatch, tmp_path
):
    campaign_dir = tmp_path / "campaign"
    first = campaign_dir / "trials" / "r001-u1"
    second = campaign_dir / "trials" / "r002-u1"
    (first / "evaluator").mkdir(parents=True)
    (first / "evaluator" / "result.json").write_text(
        json.dumps({"strict_success": False, "validity": {"valid": True}}),
        encoding="utf-8",
    )
    campaign = {
        "agent": {},
        "trials": [
            {
                "run_id": "r001-u1",
                "replicate": 1,
                "condition": "u1",
                "order": 1,
                "path": str(first),
                "status": "graded",
            },
            {
                "run_id": "r002-u1",
                "replicate": 2,
                "condition": "u1",
                "order": 1,
                "path": str(second),
                "status": "initialized",
            },
        ],
    }
    executed = []
    monkeypatch.setattr(run_campaign, "load_campaign", lambda _path: (campaign_dir, campaign))
    monkeypatch.setattr(
        run_campaign,
        "run_one_trial",
        lambda _args, trial: executed.append(trial["run_id"]),
    )
    monkeypatch.setattr(run_campaign, "save_campaign", lambda *_args: None)
    monkeypatch.setattr(run_campaign, "summarize_campaign_dir", lambda *_args: {})
    args = argparse.Namespace(
        campaign=str(campaign_dir),
        manage_stack=False,
        provider=None,
        model=None,
        thinking=None,
        runner="pi-tui",
        conditions=None,
        keep_going=False,
        stop_on_non_strict=True,
    )

    assert run_campaign.run_campaign(args) == 1
    assert executed == []
    assert campaign["trials"][1]["status"] == "initialized"


def test_non_strict_result_does_not_stop_campaign_by_default(monkeypatch, tmp_path):
    campaign_dir = tmp_path / "campaign"
    campaign = {
        "agent": {},
        "trials": [
            {
                "run_id": f"r{replicate:03d}-u1",
                "replicate": replicate,
                "condition": "u1",
                "order": 1,
                "path": str(campaign_dir / "trials" / f"r{replicate:03d}-u1"),
            }
            for replicate in (1, 2)
        ],
    }
    executed = []
    monkeypatch.setattr(run_campaign, "load_campaign", lambda _path: (campaign_dir, campaign))
    monkeypatch.setattr(
        run_campaign,
        "run_one_trial",
        lambda _args, trial: executed.append(trial["run_id"])
        or {"strict_success": False, "validity": {"valid": True}},
    )
    monkeypatch.setattr(run_campaign, "save_campaign", lambda *_args: None)
    monkeypatch.setattr(run_campaign, "summarize_campaign_dir", lambda *_args: {})
    args = argparse.Namespace(
        campaign=str(campaign_dir),
        manage_stack=False,
        provider=None,
        model=None,
        thinking=None,
        runner="pi-tui",
        conditions=None,
        keep_going=False,
        stop_on_non_strict=False,
    )

    assert run_campaign.run_campaign(args) == 0
    assert executed == ["r001-u1", "r002-u1"]


def test_stack_readiness_requires_an_identified_kernel(monkeypatch):
    """A stack whose components are ready but which never names its kernel must
    not be accepted: the fresh-kernel evidence would be empty."""
    import httpx
    import pytest

    monkeypatch.setattr(
        run_campaign.subprocess,
        "run",
        lambda *args, **kwargs: argparse.Namespace(returncode=0, stdout="", stderr=""),
    )
    monkeypatch.setattr(
        "peaksMCP.observability.read_runfile",
        lambda: {"dashboard_url": "http://127.0.0.1:9", "dashboard_token": "t"},
    )
    ready = {
        "components": {name: {"state": "ready"} for name in ("jupyter", "kernel", "extension", "mcp")},
        "kernel_id": None,
        "notebook_path": "work.ipynb",
    }

    class _Response:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    ticks = iter(range(0, 10_000))
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Response(ready))
    monkeypatch.setattr(run_campaign.time, "sleep", lambda _s: None)
    monkeypatch.setattr(run_campaign.time, "monotonic", lambda: float(next(ticks)))

    from pathlib import Path

    with pytest.raises(RuntimeError, match="identified kernel"):
        run_campaign._run_state_for_notebook(Path("/tmp/x/work.ipynb"), "default", 0.1)

    ready["kernel_id"] = "kernel-abc"
    state = run_campaign._run_state_for_notebook(Path("/tmp/x/work.ipynb"), "default", 5.0)
    assert state["kernel_id"] == "kernel-abc"
    # The host reports notebook_path RELATIVE to its root; the runner must store
    # an absolute path or grading (run from another CWD) rejects a valid trial.
    from pathlib import Path as _Path

    assert state["live_notebook"] == str(_Path("/tmp/x/work.ipynb").resolve())
    assert state["root_dir"] == str(_Path("/tmp/x").resolve())


def test_validity_rejects_a_kernel_serving_another_notebook(tmp_path):
    """Path-based checks cannot see a foreign kernel; the recorded live
    notebook closes that hole."""
    run_dir = tmp_path / "r001-p1"
    (run_dir / "workspace").mkdir(parents=True)
    (run_dir / "evaluator").mkdir()
    notebook = run_dir / "workspace" / "work.ipynb"
    notebook.write_text("{}", encoding="utf-8")
    manifest = {
        "schema_version": 2,
        "condition": "p1",
        "prompt": {"path": str(run_dir / "workspace" / "prompt.txt")},
        "audit": {"start_offset": 0, "end_offset": 1},
        "session": {"fresh": True, "evidence": "fresh"},
        "kernel": {"fresh": True, "evidence": "fresh", "live_notebook": str(tmp_path / "other" / "work.ipynb")},
        "answer_key": {"generated_after_execution": True},
    }
    (run_dir / "workspace" / "prompt.txt").write_text("prompt", encoding="utf-8")
    manifest["prompt"]["rendered_sha256"] = rc.sha256_file(run_dir / "workspace" / "prompt.txt")

    validity = rc.assess_validity(run_dir, manifest, notebook)
    assert validity["valid"] is False
    checks = {item["check"]: item["passed"] for item in validity["checks"]}
    assert checks["kernel_serves_trial_notebook"] is False

    manifest["kernel"]["live_notebook"] = str(notebook)
    assert rc.assess_validity(run_dir, manifest, notebook)["valid"] is True


def test_validity_rejects_a_trial_that_did_nothing(tmp_path):
    """A clean-looking trial that executed one cell and logged no events cannot
    support any endpoint.  Two campaign trials looked like this and were still
    counted as valid samples until these two gates existed."""
    run_dir = tmp_path / "r002-p1"
    workspace = run_dir / "workspace"
    workspace.mkdir(parents=True)
    (run_dir / "evaluator").mkdir()
    notebook = workspace / "work.ipynb"
    notebook.write_text("{}", encoding="utf-8")
    prompt = workspace / "prompt.txt"
    prompt.write_text("prompt", encoding="utf-8")
    manifest = {
        "schema_version": 2,
        "condition": "p1",
        "prompt": {"path": str(prompt), "rendered_sha256": rc.sha256_file(prompt)},
        "audit": {"start_offset": 0, "end_offset": 0},
        "session": {"fresh": True, "evidence": "fresh"},
        "kernel": {"fresh": True, "evidence": "managed host; kernel_id=k1"},
        "answer_key": {"generated_after_execution": True},
    }

    empty = rc.assess_validity(run_dir, manifest, notebook, events=[], executed_cells=1)
    assert empty["valid"] is False
    checks = {item["check"]: item for item in empty["checks"]}
    assert checks["agent_activity"]["passed"] is False
    assert "0" in checks["agent_activity"]["detail"]
    assert checks["executed_cells"]["passed"] is False
    assert str(rc.MIN_EXECUTED_CELLS) in checks["executed_cells"]["detail"]

    # The same paths with real activity stay valid: the gate measures work, not
    # artefacts.
    busy = rc.assess_validity(
        run_dir,
        manifest,
        notebook,
        events=[{"tool": "run_cell"}],
        executed_cells=rc.MIN_EXECUTED_CELLS,
    )
    assert busy["valid"] is True


def test_validity_leaves_the_activity_gates_out_when_not_measured(tmp_path):
    """Backwards compatibility: callers that cannot supply activity evidence
    (older manifests, unit fixtures) are not silently marked invalid."""
    run_dir = tmp_path / "r003-p1"
    workspace = run_dir / "workspace"
    workspace.mkdir(parents=True)
    (run_dir / "evaluator").mkdir()
    notebook = workspace / "work.ipynb"
    notebook.write_text("{}", encoding="utf-8")
    prompt = workspace / "prompt.txt"
    prompt.write_text("prompt", encoding="utf-8")
    manifest = {
        "schema_version": 2,
        "condition": "p1",
        "prompt": {"path": str(prompt), "rendered_sha256": rc.sha256_file(prompt)},
        "audit": {"start_offset": 0, "end_offset": 1},
        "session": {"fresh": True, "evidence": "fresh"},
        "kernel": {"fresh": True, "evidence": "managed host; kernel_id=k1"},
        "answer_key": {"generated_after_execution": True},
    }
    validity = rc.assess_validity(run_dir, manifest, notebook)
    names = {item["check"] for item in validity["checks"]}
    assert "agent_activity" not in names and "executed_cells" not in names


def test_validity_accepts_a_relative_live_notebook_under_the_kernel_root(tmp_path):
    """Positive case: the managed host reports "work.ipynb" relative to the
    trial workspace, which is exactly what a correct isolated run looks like."""
    run_dir = tmp_path / "r002-p2"
    workspace = run_dir / "workspace"
    workspace.mkdir(parents=True)
    (run_dir / "evaluator").mkdir()
    notebook = workspace / "work.ipynb"
    notebook.write_text("{}", encoding="utf-8")
    prompt = workspace / "prompt.txt"
    prompt.write_text("prompt", encoding="utf-8")
    manifest = {
        "schema_version": 2,
        "condition": "p2",
        "prompt": {"path": str(prompt), "rendered_sha256": rc.sha256_file(prompt)},
        "audit": {"start_offset": 0, "end_offset": 1},
        "session": {"fresh": True, "evidence": "fresh"},
        "kernel": {
            "fresh": True,
            "evidence": "managed host; kernel_id=k1",
            "kernel_id": "k1",
            "live_notebook": "work.ipynb",          # relative, as the host reports it
            "root_dir": str(workspace),
        },
        "answer_key": {"generated_after_execution": True},
    }
    validity = rc.assess_validity(run_dir, manifest, notebook)
    checks = {item["check"]: item["passed"] for item in validity["checks"]}
    assert checks["kernel_serves_trial_notebook"] is True, validity["checks"]
    assert validity["valid"] is True


def test_relative_live_notebook_is_judged_against_the_trial_root_not_the_cwd(tmp_path, monkeypatch):
    """A relative live notebook resolves against the trial workspace (the host's
    root in managed runs) - never against the grader's CWD, where an unrelated
    file of the same name must not turn the check green."""
    other = tmp_path / "elsewhere"
    (other / "work.ipynb").parent.mkdir(parents=True)
    (other / "work.ipynb").write_text("{}", encoding="utf-8")

    run_dir = tmp_path / "r003-p1"
    workspace = run_dir / "workspace"
    workspace.mkdir(parents=True)
    (run_dir / "evaluator").mkdir()
    notebook = workspace / "work.ipynb"
    notebook.write_text("{}", encoding="utf-8")
    prompt = workspace / "prompt.txt"
    prompt.write_text("prompt", encoding="utf-8")
    manifest = {
        "schema_version": 2,
        "condition": "p1",
        "prompt": {"path": str(prompt), "rendered_sha256": rc.sha256_file(prompt)},
        "audit": {"start_offset": 0, "end_offset": 1},
        "session": {"fresh": True, "evidence": "fresh"},
        "kernel": {"fresh": True, "evidence": "fresh", "live_notebook": "work.ipynb"},
        "answer_key": {"generated_after_execution": True},
    }
    monkeypatch.chdir(other)  # a CWD that happens to contain work.ipynb
    checks = {item["check"]: item["passed"] for item in rc.assess_validity(run_dir, manifest, notebook)["checks"]}
    # relative "work.ipynb" belongs to the trial workspace -> accepted
    assert checks["kernel_serves_trial_notebook"] is True

    # a foreign live notebook name must fail even though the CWD has that file
    manifest["kernel"]["live_notebook"] = "elsewhere-work.ipynb"
    (other / "elsewhere-work.ipynb").write_text("{}", encoding="utf-8")
    checks = {item["check"]: item["passed"] for item in rc.assess_validity(run_dir, manifest, notebook)["checks"]}
    assert checks["kernel_serves_trial_notebook"] is False


def test_manual_runner_path_does_not_reference_an_unbound_run_state(tmp_path, monkeypatch):
    """Without --manage-stack the marker call must still work (regression: the
    managed-stack state was referenced unconditionally -> UnboundLocalError)."""
    trial_dir = tmp_path / "trial"
    workspace = trial_dir / "workspace"
    workspace.mkdir(parents=True)
    notebook = workspace / "work.ipynb"
    notebook.write_text("{}", encoding="utf-8")
    manifest = {
        "case": {"data_dir": str(workspace), "datasheet": str(workspace)},
        "paths": {"notebook": str(notebook)},
        "audit": {"path": str(tmp_path / "audit.log")},
        "agent": {},
    }
    monkeypatch.setattr(rc, "load_manifest", lambda _dir: json.loads(json.dumps(manifest)))
    monkeypatch.setattr(rc, "build_answer_key", lambda *a, **k: {"expected_outputs": []})
    monkeypatch.setattr(rc, "audit_offset", lambda _path: 0)
    monkeypatch.setattr(rc, "save_manifest", lambda *a, **k: None)
    monkeypatch.setattr(rc, "cmd_grade", lambda _args: 0)
    monkeypatch.setattr(run_campaign, "execute_agent", lambda *a, **k: {"run": "ok"})
    captured: dict = {}
    monkeypatch.setattr(
        run_campaign,
        "_mark_trial_started",
        lambda _dir, **kwargs: captured.update(kwargs),
    )
    monkeypatch.setattr(
        rc, "evaluator_dir", lambda _dir: tmp_path
    )
    (tmp_path / "result.json").write_text(
        json.dumps({"strict_success": False, "validity": {"valid": True}}), encoding="utf-8"
    )
    args = argparse.Namespace(
        runner="command", manage_stack=False, fresh_kernel=False, kernel_evidence="",
        generic_isolation_enforced=False, isolation_mechanism="", allowed_tools=[],
        provider=None, model=None, thinking=None, agent_timeout=1,
    )
    result = run_campaign.run_one_trial(args, {"path": str(trial_dir)})
    assert result["validity"]["valid"] is True
    assert captured["kernel_id"] is None and captured["live_notebook"] is None


def test_corpus_uses_the_full_notebook_cell_not_the_truncated_audit_summary():
    """Regression: the audit stores a 200-char argument summary, so a cell whose
    curated call comes after the cap was invisible and C1 failed on correct
    behaviour."""
    long_cell = (
        "import os\n"
        "input_dir = '/data'\n"
        + "note = '" + "x" * 300 + "'\n"          # pushes the call past char 200
        + "import peaks\n"
        "report = peaks.pxt2nc(input_dir)\n"
    )
    audit_summary = long_cell[:200]                # what the audit trail stores
    events = [
        {
            "outcome": "called",
            "tool": "run_cell",
            "details": {"args": {"code": audit_summary}},
        }
    ]
    blocks = rc.executed_code(events, [long_cell])
    assert blocks == [long_cell], blocks
    checks = {item.check: item.passed for item in check_contract(_ctx(blocks))}
    assert checks["C1_blackbox_load"] is True

    # an audit-only cell (never reached the notebook) is preserved verbatim
    assert rc.executed_code(events, []) == [audit_summary]
    # no audit trail at all -> the notebook is the record
    assert rc.executed_code([], [long_cell]) == [long_cell]
