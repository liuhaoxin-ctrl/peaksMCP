"""Read L112 Scienta Omicron Igor packed experiments as xarray DataArrays."""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from igor2 import packed


def _extract_wave(
    path: str | os.PathLike[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    records, filesystem = packed.load(os.fspath(path))
    _ = records
    candidates: list[tuple[float, dict[str, Any], np.ndarray]] = []
    for value in filesystem["root"].values():
        if "WaveRecord" not in type(value).__name__:
            continue
        try:
            wave = value.wave["wave"]
            values = np.asarray(wave["wData"])
            score = len(wave.get("note", b"")) + values.size * 1e-6
            candidates.append((score, wave, values))
        except Exception:
            continue
    if not candidates:
        raise ValueError(f"{path}: no readable wave record")
    _, wave, values = max(candidates, key=lambda item: item[0])
    header = wave["wave_header"]
    units = wave.get("dimension_units", b"")
    if isinstance(units, bytes):
        units = units.decode("utf-8", errors="replace")
    return (
        values,
        np.asarray(header["nDim"], dtype=int),
        np.asarray(header["sfA"], dtype=float),
        np.asarray(header["sfB"], dtype=float),
        str(units),
    )


def _dimensions(description: str, ndim: int) -> tuple[list[str], dict[str, str]]:
    parts = re.split(r"(?<=\])(?=[A-Za-z])", description) if description else []
    dimensions: list[str] = []
    units: dict[str, str] = {}
    for position in range(ndim):
        if position == 0:
            name, unit = "eV", "eV"
        else:
            label = parts[position].lower() if position < len(parts) else ""
            if "thetax" in label or "y-scale" in label or "y_scale" in label:
                name, unit = "theta_par", "deg"
            elif "thetay" in label:
                name, unit = "deflector_perp", "deg"
            else:
                name, unit = f"dim_{position}", ""
        dimensions.append(name)
        units[name] = unit
    return dimensions, units


def load_pxt(path: str | os.PathLike[str]) -> xr.DataArray:
    """Load the spectral values and axis coordinates from one PXT file.

    Parameters
    ----------
    path : path-like
        Input Igor packed experiment file.

    Returns
    -------
    xarray.DataArray
        Float32 intensity data with energy/angle coordinates, coordinate units and minimal scan
        provenance. Physical calibration metadata is supplied by the experiment datasheet or a
        later notebook workflow.

    Examples
    --------
    >>> data = load_pxt("BP_0005.pxt")
    >>> data.dims
    ('eV', 'theta_par')
    """
    source = Path(path).expanduser().resolve()
    values, shape, scales, offsets, description = _extract_wave(source)
    dimensions, units = _dimensions(description, values.ndim)
    coordinates: dict[str, xr.DataArray] = {}
    for position, dimension in enumerate(dimensions):
        size = int(shape[position])
        coordinate = offsets[position] + scales[position] * np.arange(size)
        coordinates[dimension] = xr.DataArray(
            coordinate,
            dims=(dimension,),
            attrs={"units": units[dimension]},
        )
    return xr.DataArray(
        np.asarray(values, dtype=np.float32),
        dims=dimensions,
        coords=coordinates,
        name=source.stem,
        attrs={
            "units": "counts",
            "source_path": str(source),
            "source_format": "PXT",
            "loaded_at": datetime.now().astimezone().isoformat(),
        },
    )


class L112PXTLoader:
    """Compatibility facade exposing the direct L112 PXT loader."""

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> xr.DataArray:
        """Load one PXT file as an xarray DataArray."""
        return load_pxt(path)


def ensure_loader_available() -> bool:
    """Register an L112 PXT handler with Peaks when its registry is available.

    Returns
    -------
    bool
        Whether the ``L112_PXT`` loader is present after registration.

    Examples
    --------
    >>> isinstance(ensure_loader_available(), bool)
    True
    """
    try:
        from peaks.core.fileIO.loc_registry import LOC_REGISTRY, IdentifyLoc

        if not hasattr(IdentifyLoc, "_handler_pxt"):
            IdentifyLoc._handler_pxt = staticmethod(lambda _filename: "L112_PXT")
        if "L112_PXT" not in LOC_REGISTRY:
            LOC_REGISTRY["L112_PXT"] = L112PXTLoader
        return "L112_PXT" in LOC_REGISTRY
    except Exception:
        return False
