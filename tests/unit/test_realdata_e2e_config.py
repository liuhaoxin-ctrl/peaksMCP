"""Configuration checks for the real-data E2E module."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
E2E_MODULE = ROOT / "tests" / "e2e" / "test_e2e_realdata_live.py"
DEFAULT_REFERENCE = Path("/Users/haoxin/Documents/实验数据/BP260623/data_netcdf")


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        (None, DEFAULT_REFERENCE),
        ("/tmp/peaksmcp-reference", Path("/tmp/peaksmcp-reference")),
    ],
)
def test_realdata_e2e_reference_directory_follows_benchmark_env(
    configured: str | None,
    expected: Path,
):
    env = dict(os.environ)
    if configured is None:
        env.pop("PEAKSMCP_BENCH_REFERENCE", None)
    else:
        env["PEAKSMCP_BENCH_REFERENCE"] = configured
    probe = (
        "import runpy\n"
        f"module = runpy.run_path({str(E2E_MODULE)!r}, run_name='realdata_e2e_config_probe')\n"
        "print(module['_CONVERTED_HINT'])\n"
        "print(module['METADATA_JSON'])\n"
    )

    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.stdout.splitlines() == [
        str(expected),
        str(expected / "experiment_metadata.json"),
    ]
