from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "tools" / "test.py"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(RUNNER), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_runner_lists_stable_test_layers():
    result = _run("--list")
    assert result.returncode == 0, result.stderr
    for suite in (
        "quick",
        "unit",
        "integration",
        "realdata",
        "benchmark",
        "grader",
        "e2e",
        "acceptance",
    ):
        assert suite in result.stdout


def test_runner_check_alias_is_dry_runnable_without_starting_external_services():
    result = _run("--dry-run", "check")
    assert result.returncode == 0, result.stderr
    assert "[quality]" in result.stdout
    assert "[quick]" in result.stdout
    assert "[grader]" in result.stdout
    assert "benchmark/run_case.py selftest" in result.stdout
    assert "tests/e2e" not in result.stdout
