"""Save facade: nothing is persisted unless a human approves the staged bytes.

Real-content consent loop (no code-level approval flag exists):

    1. report = save_result(data, "out.nc")
       The result is serialised to a hidden ``.part`` staging file and a
       one-time ticket is created (binding path / kind / size / sha256 /
       structure summary).  With an approval channel registered (the
       notebook frontend), a card showing exactly this summary is presented
       to the user; the file is only published when the user approves.
       Without a channel the call returns ``pending_consent`` and the
       gateway can finish the ticket later.

    2. gateway: ``_finalize_save`` (approval-gated) atomically renames the
       staged bytes over the target; ``_discard_save`` removes them.

The model cannot express approval in code - ``approve`` does not exist.
Consent is a runtime object (ticket) consumed by the gateway, so what the
user sees on the card is exactly what gets written (same staged bytes).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import Report

#: Minutes a staged ticket stays valid before the gateway refuses it.
_TICKET_TTL_S = 10 * 60


class SaveReport(Report):
    """Outcome of one staged save: pending_consent / saved / blocked / denied."""

    operation = "save_result"
    status = "awaiting_consent"
    path = ""
    kind = ""
    approx_bytes = 0
    dims: dict[str, int] | None = None
    dtype: str | None = None
    overwrite: bool = False
    ticket_id: str | None = None
    sha256: str | None = None

    def summary_line(self) -> str:
        dims = self.dims or {}
        shape = "x".join(str(v) for v in dims.values()) if dims else "-"
        return (
            f"save_result: {self.path} ({self.kind}, shape {shape}, "
            f"~{self.approx_bytes / 1024:.1f} KiB, sha256={str(self.sha256)[:10]})"
            f"; status={self.status}"
        )


@dataclass
class PendingSave:
    """One staged result waiting for a human decision (one-time ticket)."""

    ticket_id: str
    path: Path
    tmp_path: Path
    kind: str
    approx_bytes: int
    sha256: str
    preview: dict[str, Any]
    created_at: float = field(default_factory=time.monotonic)
    used: bool = False
    #: Set to True only by the approval channel's affirmative reply.
    authorized: bool = False

    def expired(self) -> bool:
        return time.monotonic() - self.created_at > _TICKET_TTL_S


#: In-process registry of staged tickets (one per kernel process).
_STAGED: dict[str, PendingSave] = {}

#: Optional human-approval channel installed by the notebook frontend layer.
_APPROVAL_CHANNEL: Callable[[dict[str, Any]], bool] | None = None


# The functions below are gateway plumbing, NOT model verbs: they stay
# underscore-private so discovery never surfaces them and the server (the
# code that owns the human-approval channel) imports them explicitly.  A
# ticket only becomes writable after the approval channel returned True;
# calling the gateway on an unapproved ticket raises PermissionError, so
# code in the notebook cannot self-authorise a write.


def _set_approval_channel(channel: Callable[[dict[str, Any]], bool] | None) -> None:
    """Install the human-approval channel (frontend card) or remove it.

    The channel receives the ticket's preview payload (path, kind, size,
    sha256, structure and statistics of the ACTUAL result to be written) and
    must return True only after an affirmative human decision.  Without a
    channel, :func:`save_result` returns ``pending_consent``.
    """
    global _APPROVAL_CHANNEL
    _APPROVAL_CHANNEL = channel


def _serialise(data: Any, path: Path) -> tuple[bytes, str]:
    """Serialise one result to bytes: NetCDF for xarray, JSON otherwise."""
    if str(path).endswith(".nc") and hasattr(data, "to_netcdf"):
        buffer = io.BytesIO()
        data.to_netcdf(buffer)
        return buffer.getvalue(), "netcdf"
    if isinstance(data, (dict, list)):
        return (
            json.dumps(data, default=str, ensure_ascii=False).encode("utf-8"),
            "json",
        )
    raise TypeError(
        f"save_result: cannot serialise {type(data).__name__}; support: "
        "xarray DataArray/Dataset (.nc) or dict/list (.json)"
    )


def _structure(data: Any, path: Path, kind: str) -> dict[str, Any]:
    """Header-level structure plus best-effort statistics for the card."""
    structure: dict[str, Any] = {"kind": kind}
    if kind == "netcdf":
        structure.update(
            {
                "name": getattr(data, "name", None),
                "dims": list(getattr(data, "dims", ()) or ()),
                "sizes": {
                    str(key): int(value)
                    for key, value in dict(getattr(data, "sizes", {}) or {}).items()
                },
                "dtype": str(getattr(data, "dtype", "")),
                "units": (getattr(data, "attrs", {}) or {}).get("units"),
            }
        )
        # Statistics only when the data is already materialised in memory
        # (staging serialisation reads it once anyway; we never force a lazy
        # compute here just for the card).
        if getattr(getattr(data, "data", None), "chunks", None) is None:
            try:
                import numpy as np

                values = np.asarray(data.values)
                if values.size and np.isfinite(values).all():
                    structure["stats"] = {
                        "min": float(np.nanmin(values)),
                        "max": float(np.nanmax(values)),
                        "nan_fraction": round(float(np.isnan(values).mean()), 6),
                    }
            except Exception:
                pass
    else:
        snippet = json.dumps(data, default=str, ensure_ascii=False)
        structure["json_preview"] = snippet[:800]
    return structure


def _stage(data: Any, path: Path, overwrite: bool) -> tuple[SaveReport, PendingSave | None]:
    """Serialise and stage the result; no target is ever touched here."""
    payload, kind = _serialise(data, path)
    digest = hashlib.sha256(payload).hexdigest()
    approx = len(payload)
    report = SaveReport(
        status="awaiting_consent",
        path=str(path),
        kind=kind,
        approx_bytes=approx,
        dims={str(k): int(v) for k, v in dict(getattr(data, "sizes", {}) or {}).items()},
        dtype=str(getattr(data, "dtype", type(data).__name__)),
        overwrite=overwrite,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.part-",
        suffix=".tmp",
        dir=path.parent,
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    ticket_id = uuid.uuid4().hex
    preview = {
        "path": str(path),
        "kind": kind,
        "size_bytes": approx,
        "sha256": digest,
        "ticket_id": ticket_id,
        "structure": _structure(data, path, kind),
    }
    pending = PendingSave(
        ticket_id=ticket_id,
        path=path,
        tmp_path=Path(temporary_name),
        kind=kind,
        approx_bytes=approx,
        sha256=digest,
        preview=preview,
    )
    report.ticket_id = ticket_id
    report.sha256 = digest
    _STAGED[ticket_id] = pending
    _expire_stale()
    return report, pending


def _expire_stale() -> None:
    """Drop expired tickets and their staging files (best effort)."""
    expired = [ticket for ticket, pending in _STAGED.items() if pending.expired()]
    for ticket_id in expired:
        pending = _STAGED.pop(ticket_id, None)
        if pending is not None:
            try:
                pending.tmp_path.unlink(missing_ok=True)
            except OSError:
                pass


def _pending(ticket_id: str) -> PendingSave:
    if ticket_id not in _STAGED:
        raise KeyError(f"save_result: unknown or already-used ticket {ticket_id!r}")
    pending = _STAGED[ticket_id]
    if pending.expired():
        _STAGED.pop(ticket_id, None)
        try:
            pending.tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise KeyError(f"save_result: ticket {ticket_id!r} expired")
    if pending.used:
        raise KeyError(f"save_result: ticket {ticket_id!r} was already used")
    return pending


def _finalize_save(ticket_id: str) -> SaveReport:
    """Publish one staged ticket atomically: the staged bytes become the file.

    The only gateway that writes a user-approved result, callable only after
    the approval channel returned True for this ticket (``authorized``);
    unapproved tickets are refused so notebook code cannot self-authorise.
    The bytes are exactly the ones whose summary the user approved (sha256
    bound to the ticket) — no re-serialisation drift is possible.
    """
    pending = _pending(ticket_id)
    if not pending.authorized:
        raise PermissionError(
            "save_result: ticket not authorized by a human approval"
        )
    pending.tmp_path.replace(pending.path)
    pending.used = True
    _STAGED.pop(ticket_id, None)
    report = SaveReport(
        status="saved",
        path=str(pending.path),
        kind=pending.kind,
        approx_bytes=pending.approx_bytes,
        sha256=pending.sha256,
        ticket_id=ticket_id,
        overwrite=False,
    )
    print(report.summary_line())
    return report


def _discard_save(ticket_id: str) -> None:
    """Drop one staged ticket without writing anything."""
    pending = _pending(ticket_id)
    pending.used = True
    _STAGED.pop(ticket_id, None)
    try:
        pending.tmp_path.unlink(missing_ok=True)
    except OSError:
        pass


def save_result(
    data: Any,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> SaveReport:
    """Stage one result and ask the human through the approval channel.

    Serialises the result to a hidden staging file and creates a one-time
    ticket bound to the exact staged bytes (path, kind, sha256, structure).
    With an approval channel installed this presents a card to the user and
    writes the file only on affirmative approval; otherwise the ticket stays
    ``pending_consent`` for the approval flow (or a later ``_finalize_save`` after an affirmative human reply).

    There is deliberately NO ``approve`` parameter: code cannot authorise a
    write.  Consent is a runtime object consumed by the gateway.

    Parameters
    ----------
    data : xarray DataArray/Dataset or JSON-serialisable object
        Result to persist.
    path : str or Path
        Destination: ``.nc`` for xarray objects, ``.json`` for dicts/lists.
    overwrite : bool, default False
        Existing files are never replaced unless this is True.

    Returns
    -------
    SaveReport
        ``status`` = ``saved`` (after human approval through the channel),
        ``pending_consent`` (no channel / awaiting), ``blocked`` (target
        exists without overwrite) or ``denied`` (human rejected).
    """
    target = Path(path).expanduser()
    if target.exists() and not overwrite:
        report = SaveReport(status="blocked", path=str(target), kind="", overwrite=False)
        print(
            f"save_result: {target} exists; pass overwrite=True after review."
        )
        return report

    report, pending = _stage(data, target, overwrite=overwrite)
    print(report.summary_line())
    channel = _APPROVAL_CHANNEL
    if channel is None:
        print(
            "save_result: staged and waiting for human approval "
            f"(ticket {report.ticket_id}); no frontend approval channel is "
            "installed - call _finalize_save via the approval flow only after the user "
            "confirmed the summary above."
        )
        return report
    approved = bool(channel(pending.preview))
    if approved:
        pending.authorized = True
        final = _finalize_save(pending.ticket_id)
        final.dims = report.dims
        final.dtype = report.dtype
        return final
    _discard_save(pending.ticket_id)
    report.status = "denied"
    print(
        f"save_result: the user did not approve; nothing was written "
        f"({report.path})."
    )
    return report
