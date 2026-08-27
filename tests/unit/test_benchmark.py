"""Search ranking benchmark acceptance gate (plan: ≥60 CN/EN tasks, all Top-3)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from benchmark_search import load_cases, report, run  # noqa: E402


def test_search_benchmark_meets_acceptance():
    from peaksMCP.discovery.index import build_index

    cases = load_cases()
    assert len(cases) >= 60, f"benchmark needs >= 60 cases, got {len(cases)}"
    metrics = run(build_index(), cases)
    assert metrics["top3_rate"] >= 1.0, report(metrics)
    assert metrics["mrr"] >= 0.9, report(metrics)
    assert metrics["top1_rate"] >= 0.8, report(metrics)