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


def test_shared_colorbar_strict_true_refuses_and_auto_falls_back():
    low = image(0)
    high = image(0) * 1000  # same unit, ~1000x the dynamic range
    # Explicit True is strict: one shared colorbar or an error.
    with pytest.raises(ValueError, match="shared_colorbar='auto'"):
        plot_batch([low, high], shared_colorbar=True)
    plt.close("all")
    # "auto" may degrade to per-panel colorbars.
    figures = plot_batch([low, high], shared_colorbar="auto")
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


def kspace(index=0):
    return xr.DataArray(
        np.arange(24).reshape(4, 6) + index,
        dims=("eV", "kx"),
        coords={
            "eV": xr.DataArray(np.arange(4), dims="eV", attrs={"units": "eV"}),
            "kx": xr.DataArray(
                np.linspace(-0.3, 0.3, 6), dims="kx", attrs={"units": "1 / angstrom"}
            ),
        },
        attrs={"units": "counts"},
        name=f"k-cut {index}",
    )


def test_validation_pair_shared_scale_true_and_false():
    from matplotlib.figure import Figure

    from peaksMCP.plotting import plot_validation_pair

    fig = plot_validation_pair(image(0), kspace(0), shared_scale=True)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 3  # two panels + one shared colorbar
    assert any(len(axis.lines) >= 1 for axis in fig.axes)  # dashed EF guide
    plt.close(fig)
    # per-panel scale path renders one colorbar per panel (independent scales)
    fig2 = plot_validation_pair(image(0), kspace(0), shared_scale=False, ef_line=False)
    assert len(fig2.axes) == 4
    plt.close(fig2)


def test_validation_pair_auto_scale_shares_only_for_comparable_panels():
    from peaksMCP.plotting import plot_validation_pair

    # Same unit + comparable ranges -> one shared colorbar.
    shared = plot_validation_pair(image(0), kspace(0), shared_scale="auto")
    assert len(shared.axes) == 3
    plt.close(shared)
    # Range ratio far above the auto threshold -> independent colorbars.
    dims = plot_validation_pair(image(0), image(0) * 1000, shared_scale="auto")
    assert len(dims.axes) == 4
    plt.close(dims)


def _kinetic(index=0):
    """A raw kinetic-energy scan: purely positive eV axis."""
    import numpy as np
    import xarray as xr

    return xr.DataArray(
        np.arange(24).reshape(6, 4) + index,
        dims=("theta_par", "eV"),  # deliberately NOT (eV, other): transpose path
        coords={
            "eV": xr.DataArray(np.linspace(5, 40, 4), dims="eV", attrs={"units": "eV"}),
            "theta_par": xr.DataArray(np.linspace(-12, 12, 6), dims="theta_par", attrs={"units": "deg"}),
        },
        attrs={"units": "counts"},
        name=f"kinetic {index}",
    )


def _binding(index=0):
    """A processed binding-energy scan: purely non-positive eV axis."""
    import numpy as np
    import xarray as xr

    return xr.DataArray(
        np.arange(24).reshape(4, 6) + index,
        dims=("eV", "kx"),
        coords={
            "eV": xr.DataArray(np.linspace(-1.5, 0.0, 4), dims="eV", attrs={"units": "eV"}),
            "kx": xr.DataArray(np.linspace(-0.3, 0.3, 6), dims="kx", attrs={"units": "1 / angstrom"}),
        },
        attrs={"units": "counts"},
        name=f"binding {index}",
    )


def test_validation_pair_energy_axis_sharing_and_transpose():
    from peaksMCP.plotting import plot_validation_pair

    # Kinetic vs binding energy references: default keeps y axes separate.
    fig = plot_validation_pair(_kinetic(), _binding())
    assert len(fig.axes) == 3  # same counts unit + comparable ranges -> shared scale
    ax_raw, ax_proc = fig.axes[:2]
    assert not ax_raw.get_shared_y_axes().joined(ax_raw, ax_proc), (
        "kinetic and binding panels must not share a y axis"
    )
    plt.close(fig)
    # Same reference (two binding windows) -> shared y axis.
    fig2 = plot_validation_pair(_binding(0), _binding(1))
    ax_a, ax_b = fig2.axes[:2]
    assert ax_a.get_shared_y_axes().joined(ax_a, ax_b)
    plt.close(fig2)
    # Explicit override wins over the automatic reference check.
    fig3 = plot_validation_pair(_kinetic(), _binding(), share_energy_axis=True)
    ax_c, ax_d = fig3.axes[:2]
    assert ax_c.get_shared_y_axes().joined(ax_c, ax_d)
    plt.close(fig3)


def test_validation_pair_transposes_to_eV_first():
    """(theta_par, eV) input must render correctly: values are transposed to
    (eV, other) before pcolormesh, so axes and intensities stay aligned."""
    from matplotlib.figure import Figure

    from peaksMCP.plotting import plot_validation_pair

    fig = plot_validation_pair(_kinetic(), _binding(), share_energy_axis=False)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_validation_pair_rejects_3d_and_missing_eV():
    from peaksMCP.plotting import plot_validation_pair

    with pytest.raises(ValueError, match="2D"):
        plot_validation_pair(image(0).expand_dims(polar=[1]), kspace(0))
    no_eV = kspace(0).rename({"eV": "binding"})
    with pytest.raises(ValueError, match="eV"):
        plot_validation_pair(image(0), no_eV)
