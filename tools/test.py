#!/usr/bin/env python3
"""Run named peaksMCP test suites with the active Python interpreter."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Suite:
    description: str
    commands: tuple[tuple[str, ...], ...]


PYTEST = (sys.executable, "-m", "pytest")
RUFF = (sys.executable, "-m", "ruff", "check", "peaksMCP", "tests", "tools")
GRADER_TESTS = (
    "tests/unit/test_benchmark.py",
    "tests/unit/test_benchmark_experiment.py",
    "tests/unit/test_benchmark_gold_selection.py",
    "tests/unit/test_benchmark_q4_oracle.py",
    "tests/unit/test_realdata_e2e_config.py",
)

SUITES: dict[str, Suite] = {
    "quick": Suite(
        "Default offline gate: unit tests plus mocked component integration.",
        (PYTEST + ("-q", "-m", "not e2e and not slow"),),
    ),
    "unit": Suite(
        "All unit modules, including benchmark and frontend contract tests.",
        (PYTEST + ("-q", "tests/unit"),),
    ),
    "integration": Suite(
        "Component integration without local real data or a live kernel.",
        (PYTEST + ("-q", "tests/integration", "-m", "integration and not slow"),),
    ),
    "realdata": Suite(
        "Local real-data integration, including the opt-in live-kernel checks.",
        (PYTEST + ("-q", "tests/integration", "-m", "integration and realdata"),),
    ),
    "frontend": Suite(
        "JupyterLab single-cell bridge protocol exercised through Node.",
        (PYTEST + ("-q", "tests/unit/test_frontend_single_cell.py"),),
    ),
    "benchmark": Suite(
        "Benchmark/grader regressions followed by golden and poisoned self-tests.",
        (
            PYTEST + ("-q", *GRADER_TESTS),
            (sys.executable, "benchmark/run_case.py", "selftest"),
        ),
    ),
    "grader": Suite(
        "Golden and poisoned grader controls only; no duplicate pytest pass.",
        ((sys.executable, "benchmark/run_case.py", "selftest"),),
    ),
    "quality": Suite("Ruff source and test lint gate.", (RUFF,)),
    "e2e": Suite(
        "Real Dashboard-Jupyter-MCP-browser product path; no autonomous model.",
        (
            PYTEST
            + (
                "-q",
                "tests/e2e/test_e2e_realdata_live.py",
                "-m",
                "e2e and not campaign",
            ),
        ),
    ),
    "acceptance": Suite(
        "Deterministic scripted-agent campaign through the managed product stack.",
        (
            PYTEST
            + (
                "-q",
                "tests/e2e/test_campaign_acceptance.py",
                "-m",
                "e2e and campaign",
            ),
        ),
    ),
}

ALIASES: dict[str, tuple[str, ...]] = {
    "check": ("quality", "quick", "grader"),
    "all": ("quality", "unit", "integration", "realdata", "grader", "e2e", "acceptance"),
}


def _expand(names: list[str]) -> list[str]:
    expanded: list[str] = []
    for name in names:
        for item in ALIASES.get(name, (name,)):
            if item not in expanded:
                expanded.append(item)
    return expanded


def _display(command: tuple[str, ...]) -> str:
    return shlex.join(command)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run stable peaksMCP test suites. See tests/README.md for scope and prerequisites."
    )
    parser.add_argument("suites", nargs="*", choices=sorted((*SUITES, *ALIASES)))
    parser.add_argument("--list", action="store_true", help="list suites without running them")
    parser.add_argument("--dry-run", action="store_true", help="print commands without running them")
    args = parser.parse_args(argv)

    if args.list:
        for name, suite in SUITES.items():
            print(f"{name:12} {suite.description}")
        for name, members in ALIASES.items():
            print(f"{name:12} {' + '.join(members)}")
        return 0

    selected = _expand(args.suites or ["quick"])
    for name in selected:
        suite = SUITES[name]
        print(f"\n[{name}] {suite.description}", flush=True)
        for command in suite.commands:
            print(f"$ {_display(command)}", flush=True)
            if args.dry_run:
                continue
            completed = subprocess.run(command, cwd=ROOT, check=False)
            if completed.returncode != 0:
                return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
