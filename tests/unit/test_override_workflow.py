"""End-to-end usability test for the override (black-box) tier.

Walks the cut-preprocessing workflow the model is told to follow — load a raw
PXT cut, run it through preprocessing, then render the before/after
preprocessed-cut figure — using the curated override APIs end to end.

The EF-leveling + k-space step itself is a native ``peaks`` accessor flow
(``fit_gold`` -> ``da.metadata.set_EF_correction`` -> ``da.k_convert()``) that
needs full experiment calibration and is covered by the peaks suite, so here
the documented ``(eV, kx)`` output is produced deterministically.  This keeps
the test fast, hardware-free and focused on what peaksMCP owns: the loading and
figure functions must stay callable exactly as their black-box documentation
promises (name, signature, module, aliases, no internal source path).
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.figure import Figure

from peaksMCP.discovery.index import build_index
from peaksMCP.discovery.signatures import describe_api
from peaksMCP.overrides import load_data
from peaksMCP.plotting import plot_batch, plot_validation_pair

#: Tiny real (synthetic binary) 2D PXT shipped with the test fixtures.
PXT_2D = Path(__file__).parents[1] / "fixtures" / "pxt" / "synthetic_2d_nested.pxt"


def _binding_eV(count: int) -> xr.DataArray:
    """Binding-energy axis with E_F = 0 (the documented k_convert output)."""
    return xr.DataArray(
        np.linspace(-1.0, 0.0, count),
        dims="eV",
        attrs={"units": "eV"},
    )


def _kx(width: int) -> xr.DataArray:
    """k-parallel axis in 1 / angstrom (the documented k_convert output)."""
    return xr.DataArray(
        np.linspace(-0.3, 0.3, width),
        dims="kx",
        attrs={"units": "1 / angstrom"},
    )


def _preprocessed_cut(raw: xr.DataArray) -> xr.DataArray:
    """Deterministic EF-flattened (eV, kx) twin of a raw (eV, theta_par) cut.

    Mirrors the shape the override docstring note promises from
    ``fit_gold -> da.metadata.set_EF_correction -> da.k_convert()`` so the
    plotting layer below is exercised with realistic inputs.
    """
    ev = _binding_eV(raw.sizes["eV"])
    kx = _kx(raw.sizes["theta_par"])
    intensity = np.abs(np.asarray(raw)) + (kx.values[None, :] ** 2 + (ev.values[:, None] + 0.2) ** 2)
    return xr.DataArray(
        intensity.astype("float32"),
        dims=("eV", "kx"),
        coords={"eV": ev, "kx": kx},
        attrs={"units": "counts"},
        name="BP cut (k-space)",
    )


def test_cut_preprocessing_workflow_load_to_before_after_figure():
    """load_data -> preprocessing -> plot_validation_pair must all glue together."""
    raw = load_data(PXT_2D)
    assert raw.dims == ("eV", "theta_par")
    assert raw.coords["eV"].attrs.get("units") == "eV"
    assert raw.coords["theta_par"].attrs.get("units") == "deg"
    assert raw.attrs.get("units") == "counts"

    processed = _preprocessed_cut(raw)
    assert processed.dims == ("eV", "kx")
    assert processed.coords["kx"].attrs["units"] == "1 / angstrom"

    # The cut-preprocessing figure: raw angle-space next to processed k-space.
    fig = plot_validation_pair(raw, processed, shared_scale="auto")
    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 2
    assert any(len(axis.lines) >= 1 for axis in fig.axes)  # EF guide present
    plt.close(fig)

    # Batch path also renders both cuts as a panel grid (plus optional colorbar).
    grid = plot_batch([raw, processed], max_cols=2)
    assert len(grid) == 1
    panels = [axis for axis in grid[0].axes if axis.get_visible() and axis.get_subplotspec() is not None]
    assert len(panels) == 2
    plt.close("all")


def test_black_box_docs_match_how_the_functions_are_used():
    """The curated interface each override advertises must match real usage."""
    index = build_index()
    used = {
        "load_data": {"source"},
        "plot_validation_pair": {"shared_scale"},
        "plot_batch": {"max_cols"},
    }
    for name, params in used.items():
        entry = index.get(name)
        assert entry is not None, f"{name} must be discoverable in the index"
        assert entry["tier"] == "override"
        assert entry["module"].startswith("peaksMCP.")
        assert entry.get("aliases"), f"{name} must stay alias-searchable"

        detail = describe_api(entry)
        assert "source_path" not in detail, f"{name} must stay a black box"
        assert detail["signature"]
        for param in params:
            # Parameters appear as `param=` (defaulted) or `param:` (typed).
            assert re.search(rf"\b{param}\s*[:=]", detail["signature"]), (
                f"{name} doc advertises usage that its real signature lacks: {param}"
            )
