"""Offline unit tests for the benchmark grader logic (no kernel, no peaksMCP).

Covers the P0/P1 fixes: C3 receiver resolution, C4 binding path, A2 marker
semantics (persist-block != unknown-api), V3 path-level consent matching,
R2 call-site counting, executed-code deduplication, fetched-name filtering.
Run with the peaks python:  `$PY -m pytest benchmark/ -q`
"""

from __future__ import annotations

import argparse
import json

from benchmark import run_campaign
from benchmark import run_case as rc
from benchmark.approval_harness import decide_dialog
from benchmark.run_case import (
    Ctx,
    _approved_save_paths,
    _fit_gold_receiver_indices,
    _theta_offset_binding_used,
    check_access,
    check_autonomy,
    check_contract,
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


def _ctx(code, events=None, key=None, outputs=None) -> Ctx:
    return Ctx(
        run_dir=rc.Path("."),
        output_dir=rc.Path("/out"),
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


def test_s4_requires_all_output_names_in_final_markdown():
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
    assert "files" in result.evidence


def test_autonomy_fails_on_scientific_hint():
    ctx = _ctx([])
    ctx.manifest = {"isolation": {"enforced": True, "mechanism": "sandbox"}}
    ctx.interventions = [{"type": "scientific_hint"}]
    result = _by_name(check_autonomy(ctx))["U1_no_assistive_intervention"]
    assert result.passed is False


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


def test_stack_readiness_requires_an_identified_kernel(monkeypatch):
    """A stack whose components are ready but which never names its kernel must
    not be accepted: the fresh-kernel evidence would be empty."""
    import httpx
    import pytest

    from peaksMCP import cli

    monkeypatch.setattr(
        cli,
        "_ensure_host",
        lambda _args: {"dashboard_url": "http://127.0.0.1:9", "dashboard_token": "t"},
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
        + "from peaksMCP.overrides import load_data\n"
        "scans = load_data(input_dir)\n"
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
