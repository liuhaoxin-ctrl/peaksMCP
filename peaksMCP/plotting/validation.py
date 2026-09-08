"""Curated before/after validation plots (one call, not hand-written mpl)."""

from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.figure import Figure

from .layout import _display_figures

_PRETTY_X_LABELS = {
    "theta_par": r"$\theta_{par}$ (deg)",
    "kx": r"$k_x$ ($\mathrm{\AA}^{-1}$)",
}

#: Above this vmax ratio two panels are not colour-comparable for "auto".
_AUTO_SHARED_RATIO = 5.0


def _x_axis_label(dimension: str) -> str:
    """Return the canonical mathtext label for a known axis, else the name."""
    return _PRETTY_X_LABELS.get(dimension, dimension)


def _data_unit(data: xr.DataArray) -> str:
    """Intensity unit recorded on the DataArray ('' when absent)."""
    attrs = getattr(data, "attrs", {}) or {}
    return str(attrs.get("units") or attrs.get("unit") or "")


def _energy_reference(e: np.ndarray) -> str:
    """Classify an eV axis as kinetic (>0) or binding (<0) by its span.

    A purely positive span is a kinetic-energy axis; a purely non-positive
    span is a binding-energy axis (E_F = 0); anything crossing zero cannot be
    told apart and is treated as "unknown" (kept shareable).
    """
    if e.size == 0:
        return "unknown"
    return "kinetic" if float(e.min()) > 0 else "binding" if float(e.max()) <= 0 else "unknown"


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
    shared_scale: Literal["auto", True, False] = "auto",
    share_energy_axis: bool | None = None,
    ef_line: bool = True,
    colorbar_label: str = "counts",
) -> Figure:
    """Render a before/after validation figure for one processed cut.

    Draws the raw (angle-space) cut next to its k-space result (typically the
    k-space result of a conversion workflow) so the
    conversion can be checked at a glance.  Use this function instead of
    hand-writing the side-by-side matplotlib block: labels use mathtext, the
    colorbar policy is explicit, and (by default) a dashed Fermi-level guide
    is drawn at 0 on the processed panel.  The figure renders inline in
    Jupyter automatically and is returned for saving.

    Parameters
    ----------
    raw : xarray.DataArray
        2D data with ``eV`` and one other dimension (angle space), e.g.
        ``(eV, theta_par)``.  Rendered transposed to ``(eV, other)``.
    processed : xarray.DataArray
        2D k-space result with ``eV`` and one other dimension, e.g.
        ``(eV, kx)`` in binding energy (E_F = 0).
    title_raw, title_processed : str
        Panel titles.
    ylabel : str, default "eV"
        Shared y-axis label (each panel keeps its own y axis when the energy
        references differ).  For EF-corrected data use
        ``r"$E - E_F$ (eV)"``.
    figsize : tuple of float, default (9.0, 4.5)
        Figure size in inches.
    dpi : int, default 180
        Figure resolution.
    cmap : str, default "viridis"
        Colormap for both panels.
    shared_scale : {"auto", True, False}, default "auto"
        Colour-scale policy.  ``False`` renders one colorbar per panel with
        its own 99th-percentile maximum.  ``True`` forces one shared colorbar
        (common Normalize).  ``"auto"`` shares only when the panels carry the
        same intensity unit AND their dynamic ranges are within a factor of
        :data:`_AUTO_SHARED_RATIO`; otherwise it falls back to per-panel
        colorbars.
    share_energy_axis : bool, optional
        When True/False the two panels share/keep separate y axes.  By
        default they share only when the eV axes use the same energy
        reference — a purely positive (kinetic-energy) axis never shares with
        a purely non-positive (binding-energy) axis.
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
        # Render in canonical (eV, other) order regardless of input dim order.
        oriented = data.transpose("eV", xdim)
        values = np.asarray(oriented.values)
        if values.ndim != 2:
            raise ValueError(f"expected a 2D array, got {values.ndim}D data")
        return xdim, oriented.coords[xdim].values, oriented.coords["eV"].values, values

    raw_xdim, raw_x, raw_e, raw_values = panel_axes(raw)
    out_xdim, out_x, out_e, out_values = panel_axes(processed)

    raw_vmax = float(np.nanpercentile(raw_values, 99))
    out_vmax = float(np.nanpercentile(out_values, 99))

    # --- colour-scale policy -------------------------------------------------
    if shared_scale is True:
        use_shared_colorbar = True
    elif shared_scale is False:
        use_shared_colorbar = False
    else:  # "auto": same intensity unit AND comparable dynamic range
        same_unit = _data_unit(raw) and _data_unit(raw).lower() == _data_unit(processed).lower()
        ratio = max(raw_vmax, out_vmax) / max(min(raw_vmax, out_vmax), 1e-12)
        use_shared_colorbar = same_unit and ratio <= _AUTO_SHARED_RATIO
    if use_shared_colorbar:
        panel_vmax = max(raw_vmax, out_vmax)
    else:
        panel_vmax = None

    # --- y-axis policy -------------------------------------------------------
    if share_energy_axis is None:
        share_energy_axis = (
            _energy_reference(raw_e) == _energy_reference(out_e)
            or _energy_reference(raw_e) == "unknown"
            or _energy_reference(out_e) == "unknown"
        )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize,
        dpi=dpi,
        sharey=share_energy_axis,
        constrained_layout=True,
    )
    ax_raw, ax_processed = axes
    panels = (
        (ax_raw, raw_x, raw_e, raw_values, raw_xdim, title_raw),
        (ax_processed, out_x, out_e, out_values, out_xdim, title_processed),
    )
    artists: list[tuple[plt.Axes, object]] = []
    for ax, x, e, values, xdim, title in panels:
        vmax = panel_vmax if use_shared_colorbar else float(np.nanpercentile(values, 99))
        artist = ax.pcolormesh(x, e, values, vmin=0, vmax=vmax, cmap=cmap, shading="auto")
        ax.set_title(title)
        ax.set_xlabel(_x_axis_label(xdim))
        artists.append((ax, artist))
    ax_raw.set_ylabel(ylabel)
    if ef_line:
        ax_processed.axhline(0, color="w", ls="--", lw=0.6)
    if use_shared_colorbar:
        fig.colorbar(artists[-1][1], ax=axes, label=colorbar_label)
    else:
        # One colorbar per panel: each panel's colour scale is independent.
        for ax, artist in artists:
            fig.colorbar(artist, ax=ax, label=colorbar_label, shrink=0.92)

    _display_figures([fig])
    return fig
