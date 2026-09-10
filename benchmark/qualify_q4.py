#!/usr/bin/env python3
"""Qualify the human-reference oracle behind ``Q4_matches_human_reference``.

``Q4`` compares each product with the human-made reference.  Point-wise
``np.allclose`` over fourteen files is not an oracle: ARPES processing involves
interpolation, grid resampling and version-dependent numerics, so a single
pixel/bin can move past the tolerance while the physics is identical.  A
mismatch therefore proves "different from the reference", not "wrong".

This script replaces the guess with a measurement:

1. regenerate every product with today's canonical pipeline (one gold fit,
   reused; extra axes reduced; EF applied; high-symmetry angle zeroed; then
   ``k_convert``);
2. measure the comparison metrics against the human products;
3. measure the same metrics for negative controls - wrong EF, no angular
   zeroing, wrong angle, wrong scan, centre slice instead of the integral;
4. write ``benchmark/q4_oracle.json`` with ``status: qualified`` **only** when
   the positive baseline separates from every control, together with the
   thresholds ``run_case.py`` then applies.

Usage::

    python benchmark/qualify_q4.py                 # measure and write the oracle
    python benchmark/qualify_q4.py --print-only    # measure, do not write
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.run_case import _reference_metrics  # noqa: E402

DATA_DIR = Path(
    os.environ.get("PEAKSMCP_BENCH_DATA")
    or "/Users/haoxin/Documents/实验数据/BP260623/data_netcdf"
)
REFERENCE_DIR = Path(
    os.environ.get("PEAKSMCP_BENCH_REFERENCE")
    or "/Users/haoxin/Documents/实验数据/BP260623/data_netcdf"
)
ORACLE_FILE = Path(__file__).resolve().parent / "q4_oracle.json"

#: Physical landmarks the processed products must show regardless of the
#: reference (already graded by Q2/Q3; repeated here so the oracle cannot
#: "qualify" a threshold set that accepts unprocessed data).
EF_LANDMARK_MAX = 0.15
KX_LANDMARK_MAX = 0.05

#: Controls the oracle MUST be able to detect before it may qualify.  The
#: centre-slice control is deliberately not in this list: for record 26 a
#: single deflector plane is nearly identical to the reduction, so it is a
#: question about what the human product actually is (see the goal's step 2),
#: not a discrimination requirement.
REQUIRED_CONTROLS = ("wrong_ef", "no_zeroing", "wrong_angle", "wrong_scan")


def _pipeline() -> tuple[dict[int, Any], dict[str, str]]:
    """Regenerate the products with the canonical pipeline, in memory."""
    from peaksMCP.overrides import inspect_experiment, load_data

    scans = load_data(str(DATA_DIR))
    summary = inspect_experiment(scans)
    gold_index = int(summary.gold[0])
    gold = scans[f"BP_{gold_index:04d}"]
    fit = gold.fit_gold(plot=False, show=False)
    ef = dict(fit.attrs["EF_correction"])
    offsets = {int(row.index): row.theta_offset_deg for row in summary.records}

    products: dict[int, Any] = {}
    provenance = {
        "gold_index": str(gold_index),
        "ef_correction": json.dumps(ef, sort_keys=True),
        "cuts": ",".join(str(int(i)) for i in sorted(summary.cuts, key=int)),
    }
    for raw_index in summary.cuts:
        index = int(raw_index)
        cut = scans[f"BP_{index:04d}"]
        extra = [d for d in cut.dims if d not in ("eV", "theta_par")]
        if extra:
            cut = cut.sum(extra)
        cut.metadata.set_EF_correction(ef)
        offset = offsets.get(index)
        if offset:
            cut = cut.assign_coords(theta_par=cut.theta_par - float(offset))
        products[index] = cut.k_convert(quiet=True)
    return products, provenance


def _variants(products: dict[int, Any]) -> dict[str, dict[int, Any]]:
    """Negative controls: each one is physically wrong in exactly one way."""
    from peaksMCP.overrides import inspect_experiment, load_data

    scans = load_data(str(DATA_DIR))
    summary = inspect_experiment(scans)
    offsets = {int(row.index): row.theta_offset_deg for row in summary.records}
    gold = scans[f"BP_{int(summary.gold[0]):04d}"]
    fit = gold.fit_gold(plot=False, show=False)
    ef = dict(fit.attrs["EF_correction"])

    controls: dict[str, dict[int, Any]] = {
        "wrong_ef": {}, "no_zeroing": {}, "wrong_angle": {}, "centre_slice": {},
    }
    for raw_index in summary.cuts:
        index = int(raw_index)
        raw = scans[f"BP_{index:04d}"]
        extra = [d for d in raw.dims if d not in ("eV", "theta_par")]
        base = raw.sum(extra) if extra else raw

        shifted = dict(ef)
        shifted["c0"] = float(ef.get("c0", 0.0)) + 0.30  # a wrong Fermi level
        wrong_ef = base.copy()
        wrong_ef.metadata.set_EF_correction(shifted)
        offset = offsets.get(index) or 0.0
        wrong_ef = wrong_ef.assign_coords(theta_par=wrong_ef.theta_par - float(offset))
        controls["wrong_ef"][index] = wrong_ef.k_convert(quiet=True)

        no_zeroing = base.copy()
        no_zeroing.metadata.set_EF_correction(ef)
        controls["no_zeroing"][index] = no_zeroing.k_convert(quiet=True)

        wrong_angle = base.copy()
        wrong_angle.metadata.set_EF_correction(ef)
        wrong_angle = wrong_angle.assign_coords(
            theta_par=wrong_angle.theta_par - float(offset) * 0.0 + 3.0
        )
        controls["wrong_angle"][index] = wrong_angle.k_convert(quiet=True)

        if extra:
            # The deflector-resolved record: one detector plane instead of the
            # integral the human product is built from.
            plane = raw.isel({extra[0]: raw.sizes[extra[0]] // 2})
            plane.metadata.set_EF_correction(ef)
            plane = plane.assign_coords(theta_par=plane.theta_par - float(offset))
            controls["centre_slice"][index] = plane.k_convert(quiet=True)
    controls.update(_structural_controls(scans, summary, products))
    return controls


def _structural_controls(scans: Any, summary: Any, products: dict[int, Any]) -> dict[str, dict[int, Any]]:
    """Wrong-scan and centre-slice controls (no extra conversion needed)."""
    cuts = [int(i) for i in summary.cuts]
    wrong_scan = {}
    for position, index in enumerate(cuts):
        neighbour = cuts[(position + 1) % len(cuts)]
        wrong_scan[index] = products[neighbour]

    return {"wrong_scan": wrong_scan}


def _measure(products: dict[int, Any], name_of: dict[int, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, product in products.items():
        reference_name = name_of.get(index)
        if not reference_name:
            continue
        path = REFERENCE_DIR / reference_name
        if not path.is_file():
            continue
        with xr.open_dataarray(path) as handle:
            reference = handle.load()
        rows.append(_reference_metrics(reference_name, product, reference, set(product.dims)))
    return rows


def _summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    def values(key: str) -> list[float]:
        return [float(row[key]) for row in rows if np.isfinite(row[key])]

    return {
        "files": len(rows),
        "corr_min": float(np.min(values("corr"))) if values("corr") else float("nan"),
        "nrmse_max": float(np.max(values("nrmse"))) if values("nrmse") else float("nan"),
        "shape_max": float(np.max(values("shape"))) if values("shape") else float("nan"),
        "coord_delta_max": float(np.max(values("coord_delta"))) if values("coord_delta") else float("nan"),
        "mask_overlap_min": float(np.min(values("mask_overlap"))) if values("mask_overlap") else float("nan"),
        "ef_landmark_max": float(np.max(values("ef_landmark"))) if values("ef_landmark") else float("nan"),
        "kx_landmark_max": float(np.max(values("kx_landmark"))) if values("kx_landmark") else float("nan"),
        "dims_ok": float(all(row["same_dims"] for row in rows)) if rows else 0.0,
    }


def _thresholds(positive: dict[str, float], controls: dict[str, dict[str, float]]) -> dict[str, float]:
    """Thresholds inside the gap between the baseline and the closest control."""
    # Thresholds sit inside the measured gap on the GATING metrics: comfortably
    # above the baseline, comfortably below every required control.  Recorded
    # metrics keep the raw measurements, not a threshold.
    control_coord = [
        controls[name]["coord_delta_max"]
        for name in REQUIRED_CONTROLS
        if name in controls and np.isfinite(controls[name]["coord_delta_max"])
    ]
    coord_high = min(control_coord) if control_coord else positive["coord_delta_max"] * 3.0
    thresholds = {
        "coord_delta_max": float(
            max(positive["coord_delta_max"] * 3.0,
                positive["coord_delta_max"] + 0.5 * max(0.0, coord_high - positive["coord_delta_max"]))
        ),
        "mask_overlap_min": 0.97,
        "ef_landmark_max": EF_LANDMARK_MAX,
        "kx_landmark_max": KX_LANDMARK_MAX,
    }
    control_mask = [
        controls[name]["mask_overlap_min"]
        for name in REQUIRED_CONTROLS
        if name in controls and np.isfinite(controls[name]["mask_overlap_min"])
    ]
    if control_mask:
        mask_low = max(control_mask)
        if mask_low < positive["mask_overlap_min"]:
            thresholds["mask_overlap_min"] = float(
                positive["mask_overlap_min"]
                - 0.5 * (positive["mask_overlap_min"] - mask_low)
            )
    return thresholds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()

    stubs = sorted(p.stem for p in REFERENCE_DIR.glob("*_processed.nc"))
    name_of = {int(stem.split("_")[1]): f"{stem}.nc" for stem in stubs}

    products, provenance = _pipeline()
    positive_rows = _measure(products, name_of)
    positive = _summary(positive_rows)

    controls: dict[str, dict[str, float]] = {}
    for control_name, variants in _variants(products).items():
        controls[control_name] = _summary(_measure(variants, name_of))

    print("positive baseline:", json.dumps(positive, ensure_ascii=False))
    for control_name, summary in controls.items():
        print(f"control {control_name}:", json.dumps(summary, ensure_ascii=False))

    # Separation: the baseline must be better than every control on the metrics
    # that carry the physics, otherwise the oracle cannot tell them apart.
    reasons: list[str] = []
    if positive["dims_ok"] != 1.0:
        reasons.append("positive baseline has dimension mismatches")
    if positive["ef_landmark_max"] > EF_LANDMARK_MAX:
        reasons.append("positive baseline misses the EF landmark")
    if positive["kx_landmark_max"] > KX_LANDMARK_MAX:
        reasons.append("positive baseline misses the high-symmetry landmark")
    undetectable: list[str] = []
    for control_name, summary in controls.items():
        coord_separates = (
            np.isfinite(summary["coord_delta_max"])
            and summary["coord_delta_max"] > positive["coord_delta_max"] * 2.0
        )
        mask_separates = (
            np.isfinite(summary["mask_overlap_min"])
            and summary["mask_overlap_min"] < positive["mask_overlap_min"] - 0.02
        )
        landmark_separates = (
            summary["ef_landmark_max"] > EF_LANDMARK_MAX
            or summary["kx_landmark_max"] > KX_LANDMARK_MAX
        )
        if coord_separates or mask_separates or landmark_separates:
            continue
        undetectable.append(control_name)
        if control_name in REQUIRED_CONTROLS:
            reasons.append(
                f"{control_name}: the gating criteria cannot detect it "
                f"(coordΔ {summary['coord_delta_max']:.2e} vs {positive['coord_delta_max']:.2e}, "
                f"mask {summary['mask_overlap_min']:.3f} vs {positive['mask_overlap_min']:.3f})"
            )

    document: dict[str, Any] = {
        "status": "qualified" if not reasons else "unqualified",
        "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "data_dir": str(DATA_DIR),
        "reference_dir": str(REFERENCE_DIR),
        "provenance": {
            **provenance,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "xarray": xr.__version__,
            "peaks": _version("peaks"),
            "peaksMCP": _version("peaksMCP"),
            "reference_recipe": (
                "not recoverable from the repository: the human products are "
                "inputs to the case, not artefacts of it.  The metrics below are "
                "therefore measured against them rather than derived from a "
                "documented recipe; reference file attributes are recorded too."
            ),
            "reference_attrs": _reference_attrs(name_of),
        },
        "positive": positive,
        "controls": controls,
        "thresholds": _thresholds(positive, controls),
        "gating_metrics": ["coord_delta", "mask_overlap", "ef_landmark", "kx_landmark"],
        "recorded_metrics": ["corr", "nrmse", "shape", "efficiency"],
        "undetectable_controls": undetectable,
        "required_controls": list(REQUIRED_CONTROLS),
        "unqualified_reasons": reasons,
    }
    print("status:", document["status"], "| reasons:", reasons or "none")
    if not args.print_only:
        ORACLE_FILE.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("wrote", ORACLE_FILE)
    return 0


def _version(package: str) -> str:
    try:
        from importlib.metadata import version

        return version(package)
    except Exception:  # noqa: BLE001
        return "unknown"


def _reference_attrs(name_of: dict[int, str]) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    for name in sorted(name_of.values()):
        path = REFERENCE_DIR / name
        if not path.is_file():
            continue
        try:
            with xr.open_dataarray(path) as handle:
                attrs[name] = {
                    key: str(value)[:200]
                    for key, value in (handle.attrs or {}).items()
                    if key.lower() in {"ef_correction", "hv", "polarization", "polarisation",
                                       "theta_offset", "theta_offset_deg", "history", "peaks_version"}
                }
        except Exception:  # noqa: BLE001
            continue
    return attrs


if __name__ == "__main__":
    raise SystemExit(main())
