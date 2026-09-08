"""Real-data acceptance: override layer from converted NetCDF to a preprocessed cut figure.

Runs only when real L112-converted data is available — point
``PEAKSMCP_REALDATA_DIR`` at the ``data_netcdf`` folder (the default below is
the user's known location) and the suite exercises the full flow exactly as the
kernel workflow does:

    peaks.load(BP_0015.nc)              # our registered L112 loader provides geometry
    da.metadata.set_EF_correction(EF)   # EF from fit_gold (or constant)
    da.assign_coords(theta_par=... - offset)
    kd = da.k_convert(quiet=True)       # -> (eV BE, kx)
    fig = plot_validation_pair(da, kd)  # override before/after cut figure

Without the data the tests skip (CI stays clean); the machine with the data
acts as the acceptance gate for the override black-box functions.

Why not annotate manipulator yourself: the geometry lives on the array's peaks
``.metadata`` once it is read by an instrument loader (loc=L112 -> polar/tilt/
azi defaults); ``experiment_metadata.json`` carries only experiment conditions
(polarisation, temperature, analyser, theta_offset_deg, gold flag) and no
geometry fields.  The only quantity you must attach before ``k_convert`` is the
Fermi-level correction.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REAL_DATA_DIR = Path(
    os.environ.get("PEAKSMCP_REALDATA_DIR")
    or "/Users/haoxin/Documents/实验数据/BP260623/data_netcdf"
)
HAS_REAL_DATA = REAL_DATA_DIR.is_dir()
REQUIRES_REAL_DATA = pytest.mark.skipif(
    not HAS_REAL_DATA,
    reason=f"real L112 data not found under {REAL_DATA_DIR}; "
    "set PEAKSMCP_REALDATA_DIR to the data_netcdf folder to run",
)

#: Constant EF used in the reference notebook (two Au robust-mean 2.659 eV).
EF_CORRECTION = 2.6591
THETA_OFFSET_DEG = 1.5


@REQUIRES_REAL_DATA
def test_real_converted_cut_k_converts_with_loader_geometry():
    """Geometry comes from the peaks loader, then EF + theta offset give (eV, kx)."""
    import numpy as np
    import peaks

    from peaksMCP.pxt_utils.loader import register_l112_loader

    register_l112_loader()
    da = peaks.load(str(REAL_DATA_DIR / "BP_0015.nc"))
    assert da.dims == ("eV", "theta_par")
    assert da.metadata.scan.loc == "L112"
    manipulator = da.metadata.manipulator
    assert set(manipulator.model_dump().keys()) == {"polar", "tilt", "azi"}

    da.metadata.set_EF_correction(EF_CORRECTION)
    shifted = da.assign_coords(theta_par=da.theta_par - THETA_OFFSET_DEG)
    kd = shifted.k_convert(quiet=True)
    assert kd.dims == ("eV", "kx")
    # Output is binding energy with E_F at 0 (top edge ~0) in the core region.
    assert float(kd.eV.values[-1]) <= 0.1
    assert float(np.isnan(kd.values).mean()) < 0.1


@REQUIRES_REAL_DATA
def test_real_cut_preprocessing_figure_load_to_before_after():
    """load -> EF/offset -> k_convert -> plot_validation_pair end to end."""
    import peaks

    from peaksMCP.plotting import plot_validation_pair
    from peaksMCP.pxt_utils.loader import register_l112_loader

    register_l112_loader()
    da = peaks.load(str(REAL_DATA_DIR / "BP_0015.nc"))
    da.metadata.set_EF_correction(EF_CORRECTION)
    shifted = da.assign_coords(theta_par=da.theta_par - THETA_OFFSET_DEG)
    kd = shifted.k_convert(quiet=True)

    fig = plot_validation_pair(da, kd, shared_scale="auto")
    try:
        assert len(fig.axes) >= 2
        assert any(len(axis.lines) >= 1 for axis in fig.axes)  # EF guide
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


@REQUIRES_REAL_DATA
def test_real_experiment_metadata_json_is_consumable_by_override_load_metadata():
    """translate_datasheet output stays readable by our override loader/reader."""
    from peaksMCP.pxt_utils.metadata import load_metadata

    doc = load_metadata(REAL_DATA_DIR / "experiment_metadata.json")
    records = doc["records"]
    assert records  # non-empty
    cut = records["15"]
    # The datasheet agent note "高对称点 ~ +1.5°" is backfilled into the record.
    assert cut["theta_offset_deg"] == THETA_OFFSET_DEG
    assert cut["photon"]["polarisation"] in {"S", "P"}
