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
from igor2.record.wave import WaveRecord


def _load_packed(path: str | os.PathLike[str]) -> tuple[list[Any], dict[str, Any]]:
    """Load a packed experiment while accepting either Igor byte order.

    Igor2 normally infers byte order from the first versioned record.  Some
    valid packed experiments begin with an unversioned record, so inference
    is impossible until a later record.  Retrying both explicit byte orders
    keeps those files readable without changing the common fast path.
    """
    errors: list[Exception] = []
    for byte_order in (None, "<", ">"):
        try:
            return packed.load(
                os.fspath(path),
                **({} if byte_order is None else {"initial_byte_order": byte_order}),
            )
        except (OSError, ValueError) as exc:
            errors.append(exc)
    raise ValueError(f"{path}: unable to read packed experiment") from errors[0]


def _decode_text(value: Any) -> str:
    """Decode an Igor byte string or byte array without raising."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").rstrip("\x00")
    array = np.asarray(value)
    if array.dtype.kind in {"S", "U"}:
        pieces = array.ravel().tolist()
        raw = b"".join(piece for piece in pieces if isinstance(piece, bytes))
        if raw:
            return raw.decode("utf-8", errors="replace").rstrip("\x00")
        return "".join(str(piece) for piece in pieces).rstrip("\x00")
    return str(value) if value is not None else ""


def _dimension_description(wave: dict[str, Any], ndim: int) -> str:
    """Return one normalized label/unit description per wave dimension."""
    extended = _decode_text(wave.get("dimension_units", b""))
    if extended:
        return extended

    header = wave["wave_header"]
    header_units = header.get("dimUnits", [])
    labels = wave.get("labels", [])
    descriptions: list[str] = []
    for position in range(ndim):
        label = ""
        if position < len(labels) and labels[position]:
            label = _decode_text(labels[position][0])
        unit = ""
        if position < len(header_units):
            unit = _decode_text(header_units[position])
        descriptions.append(f"{label} [{unit}]".strip() if unit else label)
    return "".join(descriptions)


def _extract_wave(
    path: str | os.PathLike[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    records, filesystem = _load_packed(path)
    _ = records
    candidates: list[tuple[tuple[str, ...], dict[str, Any], np.ndarray]] = []

    def collect(dirpath: list[bytes], key: bytes, value: Any) -> None:
        if not isinstance(value, WaveRecord):
            return
        # Elettra VUV chunk format stores per-axis helper waves (chunkImage,
        # delta/dim/label/offsetInfoWave) under a ``DA_infoWaves`` folder.
        # They are instrument metadata, not experiment data: skip them so the
        # single data wave (e.g. ``chunkcube``) is selected unambiguously.
        if any(_decode_text(part).lower() == "da_infowaves" for part in dirpath):
            return
        try:
            wave = value.wave["wave"]
            values = np.asarray(wave["wData"])
            path_parts = tuple(_decode_text(part) for part in [*dirpath, key])
            candidates.append((path_parts, wave, values))
        except Exception:
            return

    packed.walk(filesystem["root"], collect)
    if not candidates:
        raise ValueError(f"{path}: no readable wave record")
    if len(candidates) > 1:
        names = ", ".join(":".join(parts) for parts, _, _ in candidates)
        raise ValueError(
            f"{path}: expected exactly one data wave, found {len(candidates)} ({names})"
        )
    _, wave, values = candidates[0]
    header = wave["wave_header"]
    shape = np.asarray(header["nDim"], dtype=int)
    declared_shape = tuple(int(size) for size in shape if size > 0)
    if declared_shape != values.shape:
        raise ValueError(
            f"{path}: wave header shape {declared_shape} does not match data shape {values.shape}"
        )
    return (
        values,
        shape,
        np.asarray(header["sfA"], dtype=float),
        np.asarray(header["sfB"], dtype=float),
        _dimension_description(wave, values.ndim),
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
    """Direct L112 PXT loader used by the conversion pipeline (``load_pxt``).

    This facade intentionally does not register into ``peaks``'s ``LOC_REGISTRY``:
    peaks' standard loading pipeline expects a full loader interface
    (``_loc_name`` / ``_load_data`` / ``_load_metadata``) which the PXT extraction
    does not provide.  The conversion path calls :func:`load_pxt` directly, which
    is self-contained.
    """

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> xr.DataArray:
        """Load one PXT file as an xarray DataArray.

        Parameters
        ----------
        path : str or os.PathLike
            Path to the ``.pxt`` file.

        Returns
        -------
        xarray.DataArray
            The loaded data.
        """
        return load_pxt(path)


def ensure_loader_available() -> bool:
    """Check that the L112 PXT loader path is importable and usable.

    Returns
    -------
    bool
        Whether the conversion pipeline (``load_pxt``) is usable.

    Examples
    --------
    >>> isinstance(ensure_loader_available(), bool)
    True
    """
    try:
        import numpy as np  # noqa: F401
        import xarray as xr  # noqa: F401

        return callable(load_pxt)
    except Exception:
        return False
