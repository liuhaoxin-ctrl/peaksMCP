from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from peaksMCP.workflows.process_cut import preprocess_cut, process_cut
from peaksMCP.workflows.slice_view import show_mapping_slice


def _mapping(*, energy: int = 8, y: int = 12, x: int = 10) -> xr.DataArray:
    data = np.arange(energy * y * x, dtype=float).reshape(energy, y, x)
    return xr.DataArray(
        data,
        dims=("eV", "y", "x"),
        coords={
            "eV": np.linspace(2.0, 3.0, energy),
            "y": np.linspace(-5, 5, y),
            "x": np.linspace(-4, 4, x),
        },
        name="mapping",
    )


def _edc() -> xr.DataArray:
    """2-D data: one slice along y leaves a single eV curve."""
    data = np.arange(8 * 12, dtype=float).reshape(8, 12)
    return xr.DataArray(
        data,
        dims=("eV", "y"),
        coords={"eV": np.linspace(2.0, 3.0, 8), "y": np.linspace(-5, 5, 12)},
        name="edc",
    )


def test_show_mapping_slice_renders_central_energy_map(capsys):
    result = show_mapping_slice(_mapping(), dim="eV")
    assert result["status"] == "ok"
    assert result["verb"] == "show_mapping_slice"
    assert result["slice"]["position"] == 4  # middle of 8
    assert result["rendered"] == [10, 12]  # x then y
    assert len(result["figure"].axes) == 2  # map + colorbar
    out = capsys.readouterr().out
    assert "rendered for the user" in out
    assert "[debug]" not in out


def test_show_mapping_slice_curve_and_explicit_index(capsys):
    result = show_mapping_slice(_edc(), dim="y", index=-1)
    assert result["rendered"] == [8]  # single column -> curve over eV
    out = capsys.readouterr().out
    assert "curve" in out and "y=" in out


def test_show_mapping_slice_debug_is_user_gated(capsys):
    show_mapping_slice(_mapping(), dim="x", index=0, debug=False)
    out = capsys.readouterr().out
    assert "[debug]" not in out
    show_mapping_slice(_mapping(), dim="x", index=0, debug=True)
    out = capsys.readouterr().out
    assert out.count("[debug]") == 2


def test_show_mapping_slice_errors_are_plain_and_actionable():
    with pytest.raises(ValueError, match="dim 'kx' not in data dims"):
        show_mapping_slice(_mapping(), dim="kx")
    with pytest.raises(ValueError, match="out of range"):
        show_mapping_slice(_mapping(), dim="eV", index=99)
    with pytest.raises(TypeError, match="data must be an xarray.DataArray"):
        show_mapping_slice(np.zeros((2, 2)), dim="eV")
    # Slicing a 3-D mapping along one axis leaves two axes (fine); slicing a
    # 4-D mapping would leave three -> must refuse with a fix hint.
    four_d = xr.DataArray(np.zeros((2, 2, 3, 4)), dims=("a", "b", "c", "d"))
    with pytest.raises(ValueError, match="slice along one more axis first"):
        show_mapping_slice(four_d, dim="a")


def test_process_cut_contract_aliases_and_input_guard():
    """process_cut is the recommended workflow verb; preprocess_cut is its
    runtime alias, and the public contract is documented."""
    assert preprocess_cut is process_cut
    doc = (process_cut.__doc__ or "").lower()
    assert "preprocess one arpes cut" in doc
    assert "returns" in doc
    with pytest.raises(TypeError):
        process_cut("not-a-dataarray")  # fails fast, not deep inside peaks
