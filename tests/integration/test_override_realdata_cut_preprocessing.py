"""Acceptance: override black-box functions from a raw PXT cut to an inline figure.

The workflow mirrors a Jupyter session, not a file-processing pipeline:

1. raw data starts as a ``.pxt`` file (``PEAKSMCP_REALDATA_PXT`` -> the raw
   L112 data folder, e.g. ``.../BP260623/data``);
2. the override black-box functions are the API surface: ``convert_experiment``
   (PXT -> NetCDF, the *only* on-disk artifact), ``inspect_experiment`` for the
   experiment record, and ``plot_validation_pair`` / ``plot_batch`` for
   figures;
3. everything after conversion stays in memory — the loader geometry
   (scan.loc=L112, manipulator axes) is provided by our registered L112
   loader, EF + theta offset are attached in memory, ``k_convert`` runs in
   memory, and figures are rendered as inline figures (asserted as Figure
   objects; nothing is written as an image);
4. a second NetCDF is only written when a save is explicitly requested
   (``kd.save(path)``), proving no incidental disk writes happen.

Native ``peaks`` steps (``peaks.load``, ``da.k_convert``) are unavoidable glue:
raw PXT geometry comes from the instrument loader, and k-conversion is a peaks
accessor.  Without the raw folder these tests skip — run them on the machine
that holds the data (or point ``PEAKSMCP_REALDATA_PXT`` at it).

Constant ``EF = 2.6591`` reproduces the reference gold fits; in the live
workflow EF comes from ``fit_gold`` (peaks), which is outside the override
layer and covered by the peaks suite.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

RAW_PXT_DIR = Path(
    os.environ.get("PEAKSMCP_REALDATA_PXT")
    or "/Users/haoxin/Documents/实验数据/BP260623/data"
)
_CONVERTED_DIR = Path("/Users/haoxin/Documents/实验数据/BP260623/data_netcdf")
_METADATA_CANDIDATES = [
    Path(os.environ["PEAKSMCP_REALDATA_METADATA"])
    if os.environ.get("PEAKSMCP_REALDATA_METADATA")
    else RAW_PXT_DIR / "experiment_metadata.json",
    _CONVERTED_DIR / "experiment_metadata.json",
]
METADATA_JSON = next((p for p in _METADATA_CANDIDATES if p.is_file()), _METADATA_CANDIDATES[0])
CUT_STEM = "BP_0015"
EF_CORRECTION = 2.6591
THETA_OFFSET_DEG = 1.5

REQUIRES_RAW = pytest.mark.skipif(
    not (RAW_PXT_DIR.is_dir() and (RAW_PXT_DIR / f"{CUT_STEM}.pxt").is_file()),
    reason=f"raw L112 PXT not found under {RAW_PXT_DIR} ({CUT_STEM}.pxt missing); "
    "set PEAKSMCP_REALDATA_PXT to the raw data folder to run",
)
REQUIRES_METADATA = pytest.mark.skipif(
    not METADATA_JSON.is_file(),
    reason=f"experiment metadata json not found at {METADATA_JSON}",
)


def _convert_cut(tmp_path: Path):
    """Copy the raw PXT into tmp and convert it (the only on-disk artifact)."""
    from peaksMCP.pxt_utils.converter import convert_pxt

    source = tmp_path / f"{CUT_STEM}.pxt"
    shutil.copy2(RAW_PXT_DIR / f"{CUT_STEM}.pxt", source)
    item = convert_pxt(source, tmp_path / f"{CUT_STEM}.nc")
    assert item.output is not None
    return Path(item.output)


def _load_converted(nc_path: Path):
    """peaks.load a converted cut after registering the L112 geometry loader."""
    import peaks

    from peaksMCP.pxt_utils.loader import _register_l112_loader

    _register_l112_loader()
    return peaks.load(str(nc_path))


@REQUIRES_RAW
def test_raw_pxt_converts_to_nc_and_nothing_else_is_written(tmp_path):
    """convert_pxt (override) writes exactly the final .nc — no .part, no images."""
    nc = _convert_cut(tmp_path)
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == [f"{CUT_STEM}.nc", f"{CUT_STEM}.pxt"], files
    assert not list(tmp_path.glob("*.png")) and not list(tmp_path.glob("*.part"))

    da = _load_converted(nc)
    try:
        assert da.dims == ("eV", "theta_par")
        assert da.metadata.scan.loc == "L112"
        assert set(da.metadata.manipulator.model_dump().keys()) == {"polar", "tilt", "azi"}
    finally:
        import matplotlib.pyplot as plt

        plt.close("all")


@REQUIRES_RAW
def test_in_memory_preprocessing_and_inline_figure_with_no_image_files(tmp_path):
    """EF + theta offset in memory -> k_convert -> inline figure; no disk writes."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.figure import Figure

    from peaksMCP.plotting import plot_validation_pair

    # Conversion copy is the input; snapshot everything that is not the .nc
    # output so later assertions prove nothing else gets written.
    nc = _convert_cut(tmp_path)
    non_nc_before = {p for p in tmp_path.iterdir() if p.suffix != ".nc"}
    da = _load_converted(nc)
    da.metadata.set_EF_correction(EF_CORRECTION)
    shifted = da.assign_coords(theta_par=da.theta_par - THETA_OFFSET_DEG)
    kd = shifted.k_convert(quiet=True)

    fig = plot_validation_pair(da, kd, shared_scale="auto")  # inline in Jupyter
    try:
        assert isinstance(fig, Figure)
        assert len(fig.axes) >= 2
        assert any(len(axis.lines) >= 1 for axis in fig.axes)  # EF guide
        assert float(np.isnan(kd.values).mean()) < 0.25
        assert kd.dims == ("eV", "kx")
    finally:
        plt.close("all")

    # In-memory session wrote nothing besides the final conversion .nc.
    non_nc_after = {p for p in tmp_path.iterdir() if p.suffix != ".nc"}
    assert non_nc_after <= non_nc_before, f"unexpected files written: {non_nc_after - non_nc_before}"
    assert not list(tmp_path.glob("*.png"))


@REQUIRES_RAW
def test_save_only_happens_when_explicitly_requested(tmp_path):
    """kd.save(path) is the only other writer, and only when called."""
    kd = _load_converted(_convert_cut(tmp_path))
    kd.metadata.set_EF_correction(EF_CORRECTION)
    shifted = kd.assign_coords(theta_par=kd.theta_par - THETA_OFFSET_DEG)
    kd = shifted.k_convert(quiet=True)

    nc_files = sorted(tmp_path.glob("*.nc"))
    assert len(nc_files) == 1  # just the conversion output so far
    target = tmp_path / f"{CUT_STEM}_processed.nc"
    assert not target.exists()
    kd.save(target)  # explicit save -> allowed
    assert target.exists() and not list(tmp_path.glob("*.png"))


@REQUIRES_METADATA
def test_experiment_metadata_json_is_consumable_by_override_inspect():
    """translate_datasheet output stays readable through the inspect_experiment
    facade (the raw-document parser is an internal detail of it)."""
    from peaksMCP.overrides import inspect_experiment

    summary = inspect_experiment(METADATA_JSON)
    by_index = {row.index: row for row in summary.records}
    assert by_index
    cut = by_index[15]
    assert cut.theta_offset_deg == THETA_OFFSET_DEG
    assert cut.polarisation in {"S", "P"}
    assert cut.kind.value == "cut"
