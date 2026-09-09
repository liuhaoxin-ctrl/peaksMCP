#!/usr/bin/env python3
"""Search benchmark: measure Top-1 / Top-3 / MRR of search ranking.

Loads ``tools/search_benchmark_queries.yaml`` (60+ Chinese and English
natural-language tasks), runs each against the live ``peaksMCP.discovery`` index
and reports ranking quality. Exits non-zero when a configured regression
threshold is breached so it can gate releases:

    python tools/benchmark_search.py                    # report only
    python tools/benchmark_search.py --top3-required 1.0   # strict gate
    python tools/benchmark_search.py --json             # machine-readable
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import yaml

TOOLS_DIR = Path(__file__).resolve().parent
ROOT = TOOLS_DIR.parent
CASES_FILE = TOOLS_DIR / "search_benchmark_queries.yaml"

sys.path.insert(0, str(ROOT))


def load_cases(path: Path = CASES_FILE) -> list[dict[str, str]]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    cases = document.get("cases") or []
    assert len(cases) >= 60, f"benchmark needs >= 60 cases, got {len(cases)}"
    return cases


def run(index, cases: list[dict[str, str]], limit: int = 5) -> dict:
    """Evaluate every case and aggregate ranking metrics.

    Parameters
    ----------
    index : peaksMCP.discovery.index.ApiIndex
        Live API index for the installed Peaks package.
    cases : list of dict
        ``{"query": ..., "expect": ...}`` tasks.
    limit : int
        Search depth used to decide whether a hit is within range.

    Returns
    -------
    dict
        Per-query results plus aggregated top1 / top3 / MRR metrics.
    """
    records: list[dict] = []
    for case in cases:
        query = str(case["query"])
        expected = str(case["expect"])
        names = [entry["name"] for entry in index.search(query, "all", limit)]
        rank = names.index(expected) + 1 if expected in names else None
        records.append(
            {
                "query": query,
                "expect": expected,
                "rank": rank,
                "top1": rank == 1,
                "top3": rank is not None and rank <= 3,
                "names": names,
            }
        )
    top1_hits = sum(1 for r in records if r["top1"])
    top3_hits = sum(1 for r in records if r["top3"])
    reciprocal_ranks = [1.0 / r["rank"] for r in records if r["rank"] is not None]
    misses = [r for r in records if not r["top3"]]
    return {
        "total": len(records),
        "top1": top1_hits,
        "top1_rate": top1_hits / len(records),
        "top3": top3_hits,
        "top3_rate": top3_hits / len(records),
        "mrr": statistics.fmean(reciprocal_ranks) if reciprocal_ranks else 0.0,
        "misses": misses,
    }


def report(metrics: dict) -> str:
    lines = [
        "Search benchmark",
        f"  cases : {metrics['total']}",
        f"  Top-1 : {metrics['top1']}/{metrics['total']} ({metrics['top1_rate']:.3f})",
        f"  Top-3 : {metrics['top3']}/{metrics['total']} ({metrics['top3_rate']:.3f})",
        f"  MRR   : {metrics['mrr']:.4f}",
    ]
    if metrics["misses"]:
        lines.append("  misses (expected not in Top-3):")
        for miss in metrics["misses"]:
            lines.append(f"    - {miss['query']!r} -> {miss['expect']!r} rank={miss['rank']} "
                         f"got={miss['names'][:3]}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="search ranking benchmark")
    parser.add_argument("--top3-required", type=float, default=1.0,
                        help="minimum Top-3 rate to pass (default 1.0)")
    parser.add_argument("--mrr-required", type=float, default=0.9,
                        help="minimum MRR to pass (default 0.9)")
    parser.add_argument("--json", action="store_true", help="emit JSON output")
    args = parser.parse_args(argv)

    from peaksMCP.discovery.index import build_index

    index = build_index()
    metrics = run(index, load_cases())
    if args.json:
        print(json.dumps(metrics, ensure_ascii=False, indent=2, default=str))
    else:
        print(report(metrics))

    ok = metrics["top3_rate"] >= args.top3_required and metrics["mrr"] >= args.mrr_required
    if not ok:
        print(f"\nFAILED: top3_rate {metrics['top3_rate']:.3f} < {args.top3_required} "
              f"or MRR {metrics['mrr']:.4f} < {args.mrr_required}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())