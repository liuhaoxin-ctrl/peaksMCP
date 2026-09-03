from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from peaksMCP.plotting import plot_batch


def curve(index=0, unit="counts"):
    return xr.DataArray(np.arange(8) + index, dims="eV", coords={"eV": xr.DataArray(np.linspace(-1, 1, 8), dims="eV", attrs={"units": "eV"})}, attrs={"units": unit}, name=f"scan {index}")


def image(index=0, unit="counts"):
    return xr.DataArray(np.arange(24).reshape(4, 6) + index, dims=("eV", "theta_par"), coords={"eV": xr.DataArray(np.arange(4), dims="eV", attrs={"units": "eV"}), "theta_par": xr.DataArray(np.arange(6), dims="theta_par", attrs={"units": "deg"})}, attrs={"units": unit})


@pytest.mark.parametrize(("count", "pages"), [(1, 1), (5, 1), (6, 1), (20, 1), (21, 2)])
def test_panel_counts_pagination_and_five_column_cap(count, pages):
    figures = plot_batch([curve(i) for i in range(count)])
    assert len(figures) == pages
    assert sum(sum(axis.get_visible() for axis in figure.axes) for figure in figures) == count
    assert max(len({round(axis.get_position().x0, 3) for axis in figure.axes if axis.get_visible()}) for figure in figures) <= 5
    plt.close("all")


def test_shared_and_independent_colorbars():
    shared = plot_batch([image(0), image(1)], shared_colorbar="auto")[0]
    assert len(shared.axes) == 3
    plt.close(shared)
    separate = plot_batch([image(0, "counts"), image(1, "arb")], shared_colorbar="auto")[0]
    assert len(separate.axes) == 4
    plt.close(separate)


def test_incompatible_explicit_colorbar_fails_and_empty_is_safe():
    with pytest.raises(ValueError):
        plot_batch([image(0, "counts"), image(1, "arb")], shared_colorbar=True)
    assert plot_batch([]) == []


def test_shared_colorbar_falls_back_when_ranges_differ():
    low = image(0)
    high = image(0) * 1000  # same unit, ~1000x the dynamic range
    with pytest.warns(UserWarning):
        figures = plot_batch([low, high], shared_colorbar=True)
    assert len(figures[0].axes) == 4  # per-panel colorbars, not one shared
    plt.close("all")


def test_auto_colorbar_shares_only_when_ranges_comparable():
    shared = plot_batch([image(0), image(1)], shared_colorbar="auto")[0]
    assert len(shared.axes) == 3  # comparable ranges -> one shared colorbar
    plt.close("all")
    separate = plot_batch([image(0), image(0) * 1000], shared_colorbar="auto")[0]
    assert len(separate.axes) == 4  # ranges differ -> per-panel colorbars
    plt.close("all")


def test_titles_validation_and_hidden_blank_axes():
    with pytest.raises(ValueError):
        plot_batch([curve()], titles=[])
    figure = plot_batch([curve(i) for i in range(6)], max_cols=5)[0]
    assert len([axis for axis in figure.axes if not axis.get_visible()]) == 4
    plt.close(figure)
