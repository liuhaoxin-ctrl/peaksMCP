"""Unified data loading facade.

Single entry for opening one data file in the notebook:

    data = load_data("BP_0020.nc")
    data = load_data("BP_0020.nc", metadata="experiment_metadata.json")
    data = load_data("raw.pxt")

Rules:
- One file per call. Directories are handled by conversion/batch facades.
- PXT input is read through the internal PXT reader; NetCDF through peaks.load.
- Optional ``metadata`` (path or dict) is embedded into
  ``attrs["experiment_metadata_json"]`` for later per-record lookups.
- Returns the DataArray unchanged in the variable; the original file is never
  modified.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import Report


class LoadReport(Report):
    """Result of load_data: which file, how it was read, dims."""

    operation = "load_data"
    status = "ok"
    path = ""
    kind = ""
    dims: dict[str, int] | None = None

    def summary_line(self) -> str:
        dims = self.dims or {}
        size = ", ".join(f"{k}={v}" for k, v in dims.items())
        return f"load_data: {self.path} ({self.kind}) dims [{size}]"


def _register_l112_once() -> None:
    try:
        from peaksMCP.pxt_utils.loader import _register_l112_loader

        _register_l112_loader()
    except Exception:
        pass  # native peaks already knows the location or registration is a no-op


def _attach_metadata(data: Any, metadata: Any) -> None:
    if metadata is None:
        return
    try:
        from peaksMCP.pxt_utils.metadata import load_metadata

        payload = load_metadata(metadata)
    except Exception:
        payload = metadata if isinstance(metadata, dict) else {}
    try:
        data.attrs["experiment_metadata_json"] = payload
    except Exception:
        pass  # object without attrs: metadata simply not embedded


def load_data(
    source: str | Path,
    *,
    lazy: bool = True,
    metadata: str | Path | dict[str, Any] | None = None,
) -> Any:
    """Open one PXT or NetCDF data file and return it as a peaks DataArray.

    Parameters
    ----------
    source : str or Path
        Path to one file (``.pxt`` PXT export or ``.nc`` NetCDF).
    lazy : bool, default True
        Passed to the underlying reader (NetCDF keeps chunks lazy).
    metadata : str, Path or dict, optional
        Experiment metadata (``experiment_metadata.json``) embedded into
        ``attrs["experiment_metadata_json"]``.

    Returns
    -------
    peaks DataArray (also used as xarray.DataArray)

    Raises
    ------
    ValueError
        For missing files or directories (load_data takes exactly one file).
    """
    path = Path(source).expanduser()
    if not path.exists():
        raise ValueError(
            f"load_data: file not found: {path}. Check the path before retrying."
        )
    if path.is_dir():
        raise ValueError(
            f"load_data: {path} is a directory; load_data opens one file. "
            "Use convert/experiment facades for directories."
        )
    suffix = path.suffix.lower()
    if suffix == ".pxt":
        from peaksMCP.pxt_utils.loader import load_pxt

        data = load_pxt(str(path))
        kind = "PXT"
    elif suffix == ".nc":
        _register_l112_once()
        from peaks import load

        data = load(str(path), lazy=lazy)
        kind = "NetCDF"
    else:
        raise ValueError(
            f"load_data: unsupported file type {suffix!r}; supported: .pxt, .nc"
        )
    _attach_metadata(data, metadata)
    report = LoadReport(path=str(path), kind=kind, dims=dict(data.sizes))
    print(report.summary_line())
    return data
