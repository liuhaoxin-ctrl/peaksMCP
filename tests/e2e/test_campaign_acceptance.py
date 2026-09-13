"""Campaign acceptance: the benchmark runner driven by a deterministic agent.

Finding 3 of the 2026-09-10 review: the real-data E2E exercises the user path
(Dashboard/MCP/Jupyter/Comm) but never the *benchmark runner* - prompt delivery,
trial isolation, the managed stack, the approval harness, the audit window and
the grader.  This test closes that gap without a model: `run_campaign.py` is run
with ``--runner command`` and the scripted agent in ``benchmark/scripted_agent.py``,
which performs the golden cut-preprocessing chain through the five MCP tools.

Asserted: the campaign exits 0, the trial is graded and VALID, the deterministic
checks pass (one PXT conversion, no final-product persistence, compact inline figures,
k-space/EF/theta alignment), and all scientific arrays remain in the live
namespace rather than processed files.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.realdata,
    pytest.mark.campaign,
    pytest.mark.browser,
    pytest.mark.slow,
]

ROOT = Path(__file__).resolve().parents[2]
RAW_PXT_DIR = Path(
    os.environ.get("PEAKSMCP_REALDATA_PXT")
    or "/Users/haoxin/Documents/实验数据/BP260623/data"
)
CASE = "bp260623"

#: Deterministic subset: everything the scripted agent explicitly guarantees.
EXPECTED_GREEN = {
    "C1_blackbox_load",
    "C2_blackbox_inspect",
    "C3_gold_index_correct",
    "C4_theta_offset_from_contract",
    "C5_ef_from_fit",
    "R1_all_targets_processed",
    "R2_one_gold_fit",
    "R4_execution_success",
    "R5_no_unexpected_outputs",
    "R6_no_redundant_scientific_execution",
    "S2_validation_figure",
    "S4_final_summary",
    "S5_notebook_readable",
    "V2_no_direct_disk_write",
    "V5_raw_inputs_immutable",
    "V6_pxt_cache_reused",
    "Q1_kspace_dims",
    "Q2_ef_zeroed",
    "Q3_theta_zeroed",
    "Q4_matches_human_reference",
}


def _run(args: list[str], timeout: float, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


def _default_ports_busy() -> str | None:
    """Reason string when the managed stack's fixed ports are already taken.

    The campaign uses the active profile (fixed 8888/8123/8765), so this
    acceptance run needs a machine without a live peaksMCP host - it is a
    deliberate, non-CI acceptance run rather than a parallel-safe test.
    """
    import socket

    for port in (8765, 8888, 8123):
        with socket.socket() as probe:
            probe.settimeout(0.5)
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                return (
                    f"a peaksMCP host already holds the default port {port}; "
                    "stop it (peaksMCP stop) before running the campaign acceptance"
                )
    return None


@pytest.mark.skipif(
    not (RAW_PXT_DIR / "BP_0015.pxt").is_file(),
    reason=f"raw beamtime data not found under {RAW_PXT_DIR}; set PEAKSMCP_REALDATA_PXT",
)
def test_campaign_runs_end_to_end_with_a_deterministic_agent(tmp_path):
    busy = _default_ports_busy()
    if busy is not None:
        pytest.skip(busy)
    # An isolated PEAKSMCP_HOME: the campaign must never fight over the ports of
    # whatever host the developer has running, and it cleans up after itself.
    env = {**os.environ, "PEAKSMCP_HOME": str(tmp_path / "home")}
    campaigns = tmp_path / "campaigns"
    created = _run(
        [
            "benchmark/run_campaign.py",
            "create",
            "--name",
            "scripted-acceptance",
            "--campaigns",
            str(campaigns),
            "--case",
            CASE,
            "--conditions",
            "p2",
            "--repetitions",
            "1",
            "--seed",
            "31",
        ],
        timeout=300,
        env=env,
    )
    assert created.returncode == 0, created.stdout + created.stderr
    campaign = campaigns / "scripted-acceptance"
    assert campaign.is_dir()

    template = (
        f"{sys.executable} {ROOT / 'benchmark' / 'scripted_agent.py'} "
        "--trial {trial} --output {output} --input {input}"
    )
    ran = _run(
        [
            "benchmark/run_campaign.py",
            "run",
            str(campaign),
            "--runner",
            "command",
            "--command-template",
            template,
            "--manage-stack",
            "--approval-mode",
            "harness_allowlist",
            "--agent-timeout",
            "900",
            "--stack-timeout",
            "300",
        ],
        timeout=1800,
        env=env,
    )
    assert ran.returncode == 0, ran.stdout[-4000:] + ran.stderr[-4000:]

    trial = json.loads((campaign / "campaign.json").read_text(encoding="utf-8"))["trials"][0]
    assert trial["status"] == "graded", trial
    result = json.loads(
        (Path(trial["path"]) / "evaluator" / "result.json").read_text(encoding="utf-8")
    )
    checks = {item["check"]: item["passed"] for item in result["checks"]}

    assert result["validity"]["valid"] is True, result["validity"]
    failed = sorted(name for name in EXPECTED_GREEN if checks.get(name) is not True)
    assert not failed, (failed, {name: checks.get(name) for name in failed})

    trial_path = Path(trial["path"])
    live = json.loads(
        (trial_path / "evaluator" / "live_evidence.json").read_text(encoding="utf-8")
    )
    assert live["status"] == "ok", live
    assert len(live["processed_stems"]) == 14, live["processed_stems"]
    assert len(live["conversion_reports"]) == 1
    assert sum(item["converted"] for item in live["conversion_reports"]) > 0

    # The Notebook and conversion cache are durable; final arrays and figures
    # are deliberately not separate disk products.
    notebook = json.loads(
        (trial_path / "workspace" / "work.ipynb").read_text(encoding="utf-8")
    )
    outputs = [
        output
        for cell in notebook["cells"]
        for output in cell.get("outputs", [])
    ]
    image_count = sum(
        any(mime in (output.get("data") or {}) for mime in ("image/png", "image/jpeg", "image/svg+xml"))
        for output in outputs
    )
    widget_count = sum(
        "application/vnd.jupyter.widget-view+json" in (output.get("data") or {})
        for output in outputs
    )
    assert image_count == 3
    assert widget_count == 0
    products = sorted(Path(trial["path"]).glob("workspace/output/*_processed.nc"))
    assert not products, f"unexpected processed products: {products}"
    caches = sorted(trial_path.glob("workspace/input_netcdf/*.nc"))
    assert caches, "pxt2nc did not create its conversion cache"
