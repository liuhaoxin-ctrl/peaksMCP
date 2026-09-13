"""Q4 gates on physical coordinates, finite masks, landmarks and correlation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr

from benchmark import run_case as rc


def _metrics(**overrides):
    row = {
        "name": "BP_0015_processed.nc",
        "same_dims": True,
        "resampled": False,
        "coord_delta": 9.4e-4,
        "corr": 0.99995,
        "nrmse": 0.4,
        "shape": 0.02,
        "mask_overlap": 0.993,
        "efficiency": 1.0,
        "ef_landmark": 7e-4,
        "kx_landmark": 4e-4,
    }
    row.update(overrides)
    return row


THRESHOLDS = {
    "coord_delta_max": 2.8e-3,
    "mask_overlap_min": 0.97,
    "ef_landmark_max": 0.15,
    "kx_landmark_max": 0.05,
    "efficiency_min": 0.5,
    "efficiency_max": 2.0,
    "corr_min": 0.98,
}


def test_the_baseline_passes_the_qualified_thresholds():
    assert rc._reference_matches(_metrics(), THRESHOLDS) is True


def test_a_low_correlation_fails_a_product():
    assert rc._reference_matches(_metrics(corr=0.97), THRESHOLDS) is False


def test_a_missing_angular_zeroing_fails_on_the_axis_centre():
    row = _metrics(coord_delta=3.1e-2, mask_overlap=0.926)
    assert rc._reference_matches(row, THRESHOLDS) is False


def test_a_wrong_fermi_level_fails_on_the_landmark_and_the_mask():
    assert rc._reference_matches(_metrics(ef_landmark=0.256, mask_overlap=0.787), THRESHOLDS) is False


def test_a_wrong_scan_fails_on_coordinates_and_mask():
    assert rc._reference_matches(_metrics(coord_delta=3.5, mask_overlap=0.079), THRESHOLDS) is False


def test_integrating_the_scanned_deflector_axis_fails_on_the_intensity_scale():
    """Record 26: the human product is the centre plane along the scanned
    deflector axis.  Integrating it is 43.8x brighter at the same grids, mask
    and landmarks, so only the scale separates the two."""
    integral = _metrics(name="BP_0026_processed.nc", efficiency=43.8)
    assert rc._reference_matches(integral, THRESHOLDS) is False
    plane = _metrics(name="BP_0026_processed.nc", efficiency=1.0, corr=1.0)
    assert rc._reference_matches(plane, THRESHOLDS) is True


def test_dimension_mismatch_fails_regardless_of_metrics():
    assert rc._reference_matches(_metrics(same_dims=False), THRESHOLDS) is False


def test_reference_metrics_aligns_dimension_order_before_comparing_values():
    reference = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("eV", "kx"),
        coords={"eV": [-0.1, 0.1], "kx": [-1.0, 0.0, 1.0]},
    )
    product = reference.transpose("kx", "eV")

    metrics = rc._reference_metrics("BP_0010_processed.nc", product, reference, set(product.dims))

    assert metrics["same_dims"] is True
    assert metrics["mask_overlap"] == 1.0
    assert metrics["corr"] == 1.0
    assert metrics["efficiency"] == 1.0


def test_an_unqualified_oracle_is_ignored_and_cannot_gate(monkeypatch, tmp_path):
    oracle = tmp_path / "q4_oracle.json"
    oracle.write_text(json.dumps({"status": "unqualified"}), encoding="utf-8")
    monkeypatch.setattr(rc, "Q4_ORACLE_FILE", oracle)
    assert rc._q4_oracle() is None


def test_a_qualified_oracle_is_loaded_with_its_thresholds(monkeypatch, tmp_path):
    oracle = tmp_path / "q4_oracle.json"
    oracle.write_text(
        json.dumps({"status": "qualified", "generated_at": "2026-09-11T00:00:00", "thresholds": THRESHOLDS}),
        encoding="utf-8",
    )
    monkeypatch.setattr(rc, "Q4_ORACLE_FILE", oracle)
    loaded = rc._q4_oracle()
    assert loaded is not None and loaded["thresholds"] == THRESHOLDS


def test_the_shipped_oracle_records_which_controls_it_cannot_detect():
    """The artefact must stay honest about its own blind spots."""
    path = Path(rc.Q4_ORACLE_FILE)
    if not path.is_file():
        return
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["gating_metrics"] == [
        "coord_delta", "mask_overlap", "ef_landmark", "kx_landmark", "efficiency", "corr",
    ]
    assert "corr" not in document["recorded_metrics"]
    assert set(document["required_controls"]) == {
        "wrong_ef", "no_zeroing", "wrong_angle", "wrong_scan", "deflector_integral",
    }
    assert document["undetectable_controls"] == [], (
        "every control must be detectable now that the deflector reduction is settled"
    )
