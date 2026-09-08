"""Small deterministic publication workflows built on xarray and Peaks."""

from __future__ import annotations

import xarray as xr


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
