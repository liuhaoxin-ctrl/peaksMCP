"""Curated before/after validation plots (one call, not hand-written mpl)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.figure import Figure

from .layout import _display_figures

_PRETTY_X_LABELS = {
    "theta_par": r"$\theta_{par}$ (deg)",
    "kx": r"$k_x$ ($\mathrm{\AA}^{-1}$)",
}


def _x_axis_label(dimension: str) -> str:
    """Return the canonical mathtext label for a known axis, else the name."""
    return _PRETTY_X_LABELS.get(dimension, dimension)


def plot_validation_pair(
    raw: xr.DataArray,
    processed: xr.DataArray,
    *,
    title_raw: str = "raw (θ)",
    title_processed: str = "k-space (E − E_F)",
    ylabel: str = "eV",
    figsize: tuple[float, float] = (9.0, 4.5),
    dpi: int = 180,
    cmap: str = "viridis",
    shared_scale: bool = False,
    ef_line: bool = True,
    colorbar_label: str = "counts",
) -> Figure:
    """Render a before/after validation figure for one processed cut.

    Draws the raw (angle-space) cut next to its k-space result (typically the
    k-space result of a conversion workflow) so the
    conversion can be checked at a glance.  Use this function instead of
    hand-writing the side-by-side matplotlib block: labels use mathtext, the
    panels share the y axis, the colorbar is added once and (by default) a
    dashed Fermi-level guide is drawn at 0 on the processed panel.  The figure
    renders inline in Jupyter automatically and is returned for saving.

    Parameters
    ----------
    raw : xarray.DataArray
        2D data with ``eV`` and one other dimension (angle space), e.g.
        ``(eV, theta_par)``.
    processed : xarray.DataArray
        2D k-space result with ``eV`` and one other dimension, e.g.
        ``(eV, kx)`` in binding energy (E_F = 0).
    title_raw, title_processed : str
        Panel titles.
    ylabel : str, default "eV"
        Shared y-axis label.  For EF-corrected data use
        ``r"$E - E_F$ (eV)"``.
    figsize : tuple of float, default (9.0, 4.5)
        Figure size in inches.
    dpi : int, default 180
        Figure resolution.
    cmap : str, default "viridis"
        Colormap for both panels.
    shared_scale : bool, default False
        When False each panel normalises to its own 99th-percentile maximum,
        which shows shapes best when the intensities differ.  When True one
        common maximum is used so the two panels can be compared on the same
        colour scale.
    ef_line : bool, default True
        Draw a dashed guide at 0 on the processed (k-space) panel.
    colorbar_label : str, default "counts"
        Colorbar label.

    Returns
    -------
    matplotlib.figure.Figure
        The rendered figure (also displayed inline in Jupyter).

    Raises
    ------
    ValueError
        If either input is not 2D with an ``eV`` dimension plus exactly one
        other dimension.

    Examples
    --------
    >>> from peaksMCP.plotting import plot_validation_pair
    >>> fig = plot_validation_pair(data_5, processed[5]["data"], shared_scale=True)
    """
    def panel_axes(data: xr.DataArray) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
        if "eV" not in data.dims:
            raise ValueError(f"expected an 'eV' dimension, got {data.dims}")
        others = [dim for dim in data.dims if dim != "eV"]
        if len(others) != 1:
            raise ValueError(
                f"expected a 2D array (eV + one axis), got dims {data.dims}"
            )
        xdim = others[0]
        values = np.asarray(data.values)
        if values.ndim != 2:
            raise ValueError(f"expected a 2D array, got {values.ndim}D data")
        return xdim, data.coords[xdim].values, data.coords["eV"].values, values

    raw_xdim, raw_x, raw_e, raw_values = panel_axes(raw)
    out_xdim, out_x, out_e, out_values = panel_axes(processed)

    raw_vmax = float(np.nanpercentile(raw_values, 99))
    out_vmax = float(np.nanpercentile(out_values, 99))
    if shared_scale:
        raw_vmax = out_vmax = max(raw_vmax, out_vmax)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize,
        dpi=dpi,
        sharey=True,
        constrained_layout=True,
    )
    ax_raw, ax_processed = axes
    panels = (
        (ax_raw, raw_x, raw_e, raw_values, raw_vmax, raw_xdim, title_raw),
        (ax_processed, out_x, out_e, out_values, out_vmax, out_xdim, title_processed),
    )
    im = None
    for ax, x, e, values, panel_vmax, xdim, title in panels:
        im = ax.pcolormesh(x, e, values, vmin=0, vmax=panel_vmax, cmap=cmap, shading="auto")
        ax.set_title(title)
        ax.set_xlabel(_x_axis_label(xdim))
    ax_raw.set_ylabel(ylabel)
    if ef_line:
        ax_processed.axhline(0, color="w", ls="--", lw=0.6)
    fig.colorbar(im, ax=axes, label=colorbar_label)

    _display_figures([fig])
    return fig
