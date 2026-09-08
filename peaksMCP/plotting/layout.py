"""Deterministic publication layout for batches of ARPES spectra."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.figure import Figure


def _unit(value: Any) -> str:
    coordinate_units = getattr(value, "units", None)
    if coordinate_units is not None:
        return str(coordinate_units)
    attrs = getattr(value, "attrs", {}) or {}
    return str(attrs.get("units") or attrs.get("unit") or "")


def _axis_label(data: xr.DataArray, dimension: str) -> str:
    unit = _unit(data.coords[dimension]) if dimension in data.coords else ""
    return f"{dimension} [{unit}]" if unit else dimension


def _data_unit(data: xr.DataArray) -> str:
    return _unit(data)


def _resolve_axis_dim(data: xr.DataArray, axis: Any) -> str | None:
    """Resolve an x=/y= plot argument (dimension or coordinate name) to the
    dimension that owns the label, or None when not statically identifiable."""
    if axis is None:
        return None
    if isinstance(axis, str):
        if axis in data.dims:
            return axis
        if axis in data.coords:
            for dim in data[axis].dims:
                return dim
    return None


def _compatible_for_shared_colorbar(items: Sequence[xr.DataArray]) -> bool:
    if not items or any(item.ndim != 2 for item in items):
        return False
    return len({_data_unit(item) for item in items}) == 1


def _ranges_comparable(items: Sequence[xr.DataArray], *, max_ratio: float = 5.0) -> bool:
    """Whether 2D panels share a comparable intensity range.

    A single colorbar is only meaningful when the panels' dynamic ranges are
    within ``max_ratio`` of each other; otherwise forcing one vmin/vmax washes
    out the low-intensity panels.  Uses a robust percentile span so one hot
    pixel does not decide compatibility.
    """
    scales: list[float] = []
    for item in items:
        values = np.asarray(item.values, dtype=float)
        if not np.isfinite(values).any():
            return False
        span = float(np.nanpercentile(values, 99) - np.nanpercentile(values, 1))
        if not np.isfinite(span) or span <= 0:
            span = float(np.nanmax(values) - np.nanmin(values))
        if not np.isfinite(span) or span <= 0:
            return False
        scales.append(span)
    if not scales:
        return False
    return max(scales) / min(scales) <= max_ratio


def _display_figures(figures: list[Figure]) -> None:
    """Render each figure inline exactly once when inside IPython/Jupyter.

    Display-then-close: matplotlib's inline backend also flushes the "current"
    figure at the end of the cell, so a bare ``display(fig)`` renders the same
    figure twice — and the second render can come out as a blank/transparent
    PNG ("drawn but not rendered").  Closing each figure right after displaying
    it leaves the backend nothing extra to flush.  The returned figure objects
    still support ``savefig`` after ``plt.close``.

    Non-interactive callers (scripts, tests) receive the returned list and keep
    control of ``plt.show`` / saving, exactly as before.
    """
    try:
        from IPython import get_ipython

        if get_ipython() is not None:
            from IPython.display import display

            for figure in figures:
                display(figure)
                plt.close(figure)
    except Exception:
        pass


def _global_limits(items: Sequence[xr.DataArray]) -> tuple[float, float] | None:
    lows: list[float] = []
    highs: list[float] = []
    for item in items:
        try:
            values = np.asarray(item.values)
            lows.append(float(np.nanmin(values)))
            highs.append(float(np.nanmax(values)))
        except (TypeError, ValueError):
            return None
    if not lows or not np.isfinite(lows + highs).all():
        return None
    return min(lows), max(highs)


def _as_dataarrays(data: Iterable[xr.DataArray] | xr.DataTree) -> list[xr.DataArray]:
    if isinstance(data, xr.DataTree):
        arrays: list[xr.DataArray] = []
        for node in data.subtree:
            dataset = node.dataset
            if dataset is not None and len(dataset.data_vars) == 1:
                arrays.append(next(iter(dataset.data_vars.values())))
        return arrays
    arrays = list(data)
    if not all(isinstance(item, xr.DataArray) for item in arrays):
        raise TypeError("plot_batch accepts DataArray items or a DataTree with one variable per leaf")
    return arrays


def plot_batch(
    data: Iterable[xr.DataArray] | xr.DataTree,
    *,
    titles: Sequence[str] | None = None,
    max_cols: int = 5,
    max_panels_per_figure: int = 20,
    sharex: bool = False,
    sharey: bool = False,
    shared_units: bool = True,
    shared_colorbar: Literal["auto"] | bool = "auto",
    figsize_per_panel: tuple[float, float] = (4.0, 3.5),
    dpi: int = 180,
    cmap: str = "viridis",
    **plot_kwargs: Any,
) -> list[Figure]:
    """Plot ARPES arrays in deterministic, paginated grids.

    Parameters
    ----------
    data : iterable of xarray.DataArray or xarray.DataTree
        One- or two-dimensional spectra. Plotting computes lazy values because Matplotlib needs
        concrete pixels.
    titles : sequence of str, optional
        Panel titles matching the number of arrays.
    max_cols : int, default 5
        Maximum number of panels in a row (values above 5 are capped to 5).
    max_panels_per_figure : int, default 20
        Page size for large batches.
    sharex, sharey : bool, default False
        Share Matplotlib axes within each page.
    shared_units : bool, default True
        Suppress repeated axis labels only when dimensions and units match.
    shared_colorbar : {"auto", True, False}, default "auto"
        Use one colorbar when every panel is two-dimensional, has the same data
        unit AND a comparable intensity range (robust 99th-percentile spans
        within 5x).  ``True`` raises only when the data units are incompatible;
        when only the ranges differ it falls back to per-panel colorbars with a
        warning instead of washing out low-intensity panels.
    figsize_per_panel : tuple of float, default (4.0, 3.5)
        Width and height in inches allocated to each panel.
    dpi : int, default 180
        Figure resolution used by Jupyter inline rendering and export.
    cmap : str, default "viridis"
        Matplotlib colormap for two-dimensional spectra.
    **plot_kwargs
        Additional keyword arguments passed to xarray plotting.

    Returns
    -------
    list of matplotlib.figure.Figure
        One figure per page. Figures are rendered inline automatically (Jupyter
        ``display``, falling back to ``plt.show``); the returned list remains
        available for saving.

    Raises
    ------
    ValueError
        If layout limits, titles or an explicitly requested shared colorbar are invalid.
    TypeError
        If an item is not an xarray DataArray.

    Examples
    --------
    >>> figures = plot_batch(scans, titles=temperatures, max_cols=5)  # renders inline
    """
    arrays = _as_dataarrays(data)
    if not arrays:
        return []
    if max_cols < 1 or max_panels_per_figure < 1:
        raise ValueError("max_cols and max_panels_per_figure must be positive")
    max_cols = min(max_cols, 5)
    if titles is not None and len(titles) != len(arrays):
        raise ValueError("titles length must match the number of arrays")

    figures: list[Figure] = []
    for start in range(0, len(arrays), max_panels_per_figure):
        page = arrays[start : start + max_panels_per_figure]
        page_titles = titles[start : start + len(page)] if titles is not None else None
        ncols = min(max_cols, len(page))
        nrows = math.ceil(len(page) / ncols)
        figure, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            sharex=sharex,
            sharey=sharey,
            squeeze=False,
            layout="constrained",
            figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
            dpi=dpi,
        )
        flat_axes = list(axes.flat)
        unit_compatible = _compatible_for_shared_colorbar(page)
        if shared_colorbar is True and not unit_compatible:
            plt.close(figure)
            raise ValueError("a shared colorbar requires compatible two-dimensional data units")
        ranges_compatible = _ranges_comparable(page) if unit_compatible else False
        if shared_colorbar is True and unit_compatible and not ranges_compatible:
            # Explicit True is a strict request: one shared colorbar or an
            # error.  Only "auto" may degrade to per-panel colorbars.
            plt.close(figure)
            raise ValueError(
                "a shared colorbar requires comparable intensity ranges across "
                "panels; pass shared_colorbar='auto' to allow per-panel colorbars"
            )
        use_shared_colorbar = unit_compatible and ranges_compatible and shared_colorbar in {True, "auto"}
        limits = _global_limits(page) if use_shared_colorbar else None
        mappable = None

        same_dims_units = len(
            {
                tuple((dim, _unit(item.coords[dim])) for dim in item.dims)
                for item in page
            }
        ) == 1
        for offset, (array, axis) in enumerate(zip(page, flat_axes, strict=False)):
            kwargs = dict(plot_kwargs)
            if array.ndim == 2:
                kwargs.setdefault("cmap", cmap)
                kwargs["add_colorbar"] = not use_shared_colorbar
                # Do not force data-range vmin/vmax when the caller pinned an
                # explicit norm (e.g. LogNorm): the auto limits would conflict
                # with the user's normalisation intent.
                if limits is not None and "norm" not in kwargs:
                    kwargs.setdefault("vmin", limits[0])
                    kwargs.setdefault("vmax", limits[1])
                artist = array.plot(ax=axis, **kwargs)
                if use_shared_colorbar:
                    mappable = artist
            elif array.ndim == 1:
                array.plot(ax=axis, **kwargs)
            else:
                plt.close(figure)
                raise ValueError(f"panel {start + offset} has ndim={array.ndim}; expected 1 or 2")
            if page_titles is not None:
                axis.set_title(str(page_titles[offset]))
            elif array.name:
                axis.set_title(str(array.name))
            if shared_units and same_dims_units:
                row, col = divmod(offset, ncols)
                if array.ndim >= 1 and row < nrows - 1:
                    axis.set_xlabel("")
                if array.ndim >= 2 and col > 0:
                    axis.set_ylabel("")
            else:
                # When the caller pins axes via x= / y= (dimension or coordinate
                # name), label those dimensions; otherwise fall back to the
                # natural dims order (last dim = x, second-to-last = y).
                x_dim = _resolve_axis_dim(array, plot_kwargs.get("x"))
                y_dim = _resolve_axis_dim(array, plot_kwargs.get("y"))
                if array.ndim >= 1:
                    axis.set_xlabel(_axis_label(array, x_dim if x_dim is not None else array.dims[-1]))
                if array.ndim >= 2:
                    axis.set_ylabel(_axis_label(array, y_dim if y_dim is not None else array.dims[-2]))

        for axis in flat_axes[len(page) :]:
            axis.set_visible(False)
        if use_shared_colorbar and mappable is not None:
            label = _data_unit(page[0])
            colorbar = figure.colorbar(mappable, ax=flat_axes[: len(page)], shrink=0.9)
            if label:
                colorbar.set_label(label)
        figures.append(figure)
    _display_figures(figures)
    return figures

