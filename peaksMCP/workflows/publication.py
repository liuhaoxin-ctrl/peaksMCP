"""Small deterministic publication workflows built on xarray and Peaks."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import xarray as xr
from matplotlib.figure import Figure

from peaksMCP.plotting import plot_batch


def validate_arpes_metadata(data: xr.DataArray) -> list[str]:
    """Return missing unit and coordinate requirements relevant to ARPES plots.

    Parameters
    ----------
    data : xarray.DataArray
        ARPES intensity with dimension coordinates and physical-unit attrs.

    Returns
    -------
    list of str
        Empty when plotting requirements are satisfied; otherwise actionable problems.

    Examples
    --------
    >>> validate_arpes_metadata(scan)
    []
    """
    issues: list[str] = []
    if not data.dims:
        issues.append("data has no dimensions")
    for dimension in data.dims:
        if dimension not in data.coords:
            issues.append(f"dimension {dimension!r} has no coordinate")
        elif not (data.coords[dimension].attrs.get("units") or data.coords[dimension].attrs.get("unit")):
            issues.append(f"coordinate {dimension!r} has no units")
    if not (data.attrs.get("units") or data.attrs.get("unit")):
        issues.append("intensity has no units")
    return issues


def _publication_grid(
    scans: Iterable[xr.DataArray],
    *,
    titles: Sequence[str] | None = None,
    max_cols: int = 5,
    dpi: int = 300,
) -> list[Figure]:
    """Create a checked, paginated grid for publication export.

    Parameters
    ----------
    scans : iterable of xarray.DataArray
        One- or two-dimensional spectra with coordinate and intensity units.
    titles : sequence of str, optional
        Panel labels in scan order.
    max_cols : int, default 5
        Maximum columns, capped at five by ``plot_batch``.
    dpi : int, default 300
        Figure resolution.

    Returns
    -------
    list of matplotlib.figure.Figure
        Publication-ready page figures shown inline by Jupyter.

    Raises
    ------
    ValueError
        If any scan lacks required units or coordinates.

    Examples
    --------
    >>> figures = _publication_grid([scan_20K, scan_40K], titles=["20 K", "40 K"])
    >>> import matplotlib.pyplot as plt
    >>> plt.show()
    """
    values = list(scans)
    problems = {index: validate_arpes_metadata(item) for index, item in enumerate(values)}
    problems = {index: issue for index, issue in problems.items() if issue}
    if problems:
        raise ValueError(f"ARPES metadata validation failed: {problems}")
    return plot_batch(values, titles=titles, max_cols=max_cols, dpi=dpi)
