"""Save facade: nothing is persisted unless the user approves.

Flow (two calls, default is a no-write preview):

    1. report = save_result(data, "out.nc")          # prints preview; NO write
    2. report = save_result(data, "out.nc", approve=True)   # writes atomically

The preview line shows path, kind, size estimate and shape/dtype. Writing is
atomic (``.part`` + rename) and never overwrites unless ``overwrite=True`` is
explicitly set and previewed.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .models import Report


class SaveReport(Report):
    """Outcome of the preview/approve save flow."""

    operation = "save_result"
    status = "awaiting_consent"
    path = ""
    kind = ""
    approx_bytes = 0
    dims: dict[str, int] | None = None
    dtype: str | None = None

    def summary_line(self) -> str:
        dims = self.dims or {}
        shape = "x".join(str(v) for v in dims.values()) if dims else "-"
        return (
            f"save_result: {self.path} ({self.kind}, shape {shape}, "
            f"~{self.approx_bytes / 1024:.1f} KiB); status={self.status}"
        )


def _preview(data: Any, path: Path, overwrite: bool) -> SaveReport:
    dims = dict(getattr(data, "sizes", {}))
    dtype = str(getattr(data, "dtype", type(data).__name__))
    if hasattr(data, "nbytes") and data.nbytes:
        approx = int(data.nbytes)
    elif isinstance(data, dict):
        approx = len(json.dumps(data, default=str).encode("utf-8"))
    else:
        approx = len(repr(data).encode("utf-8"))
    return SaveReport(
        status="awaiting_consent",
        path=str(path),
        kind="netcdf" if str(path).endswith(".nc") else "json",
        approx_bytes=approx,
        dims=dims,
        dtype=dtype,
        overwrite=overwrite,
    )


def _atomic_write_bytes(path: Path, payload: bytes) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=path.name + ".part-", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return len(payload)


def save_result(
    data: Any,
    path: str | Path,
    *,
    overwrite: bool = False,
    approve: bool = False,
) -> SaveReport:
    """Preview then (on ``approve=True``) atomically save ``data``.

    Parameters
    ----------
    data : xarray DataArray/Dataset or JSON-serialisable object
        Result to persist.
    path : str or Path
        Destination: ``.nc`` for xarray objects, ``.json`` for dicts/lists.
    overwrite : bool, default False
        Existing files are never replaced unless this is True.
    approve : bool, default False
        The only switch that actually writes. Default returns an
        ``awaiting_consent`` preview without touching disk.

    Returns
    -------
    SaveReport
        ``status`` = ``awaiting_consent`` (default) or ``saved``.
    """
    target = Path(path).expanduser()
    if target.exists() and not overwrite:
        report = _preview(data, target, overwrite=False)
        report.status = "blocked"
        print(report.summary_line() + "; exists (pass overwrite=True after review)")
        return report

    report = _preview(data, target, overwrite=overwrite)
    print(report.summary_line())
    if not approve:
        print("save_result: no write performed; re-run with approve=True to persist.")
        return report

    if str(target).endswith(".nc") and hasattr(data, "to_netcdf"):
        import io

        buffer = io.BytesIO()
        data.to_netcdf(buffer)
        report.approx_bytes = _atomic_write_bytes(target, buffer.getvalue())
        report.status = "saved"
    elif isinstance(data, (dict, list)):
        payload = json.dumps(data, default=str, ensure_ascii=False).encode("utf-8")
        report.approx_bytes = _atomic_write_bytes(target, payload)
        report.status = "saved"
    else:
        report.status = "failed"
        report.dtype = "unsupported-type"
        raise TypeError(
            f"save_result: cannot serialize {type(data).__name__}; "
            "support: xarray DataArray/Dataset (.nc) or dict/list (.json)"
        )
    print(report.summary_line())
    return report
