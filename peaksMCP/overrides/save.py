"""Save gateway: nothing is persisted unless a human approves the staged bytes.

Every model verb that would persist files (save_result, convert_experiment,
preprocess_batch, ...) funnels through this module:

    1. stage: results are serialised to hidden ``.part`` staging files; a
       one-time ticket binds the EXACT bytes of every item (path, kind,
       size, sha256, structure) plus an idempotency note.
    2. consent: with an approval channel installed (the notebook frontend) a
       card listing the REAL content of every item is presented; approval
       publishes all staged bytes (atomic per item), rejection or expiry
       removes them.
    3. gateway: publication is a per-item atomic rename of the staged bytes
       over the target - there is no code path that writes a target without
       an affirmative human decision, and no code-level approve exists.

Consent is a runtime object, never a parameter: gateway functions stay
underscore-private and refuse unapproved tickets, so notebook code cannot
self-authorise a write at any layer.
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

import numpy as np

from .models import Report

#: Minutes a staged ticket stays valid before the gateway refuses it.
_TICKET_TTL_S = 10 * 60


class SaveReport(Report):
    """Outcome of one staged save: saved / pending_consent / blocked / denied."""

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
class PendingItem:
    """One staged file inside a ticket: exact bytes bound to the item."""

    path: Path
    tmp_path: Path
    kind: str
    approx_bytes: int
    sha256: str
    structure: dict[str, Any]
    overwrite: bool
    #: True when the target already existed at staging time (idempotent
    #: verbs skip it on publish unless overwrite=True).
    exists_at_stage: bool = False


@dataclass
class PendingBatch:
    """One-time staged batch waiting for a human decision."""

    ticket_id: str
    operation: str
    summary: str
    items: list[PendingItem]
    created_at: float = field(default_factory=time.monotonic)
    used: bool = False
    authorized: bool = False

    def expired(self) -> bool:
        return time.monotonic() - self.created_at > _TICKET_TTL_S


#: In-process registry of staged tickets (one per kernel process).
_STAGED: dict[str, PendingBatch] = {}

#: Optional human-approval channel installed by the notebook frontend layer.
_APPROVAL_CHANNEL: Callable[[dict[str, Any]], bool] | None = None


# The functions below are gateway plumbing, NOT model verbs: they stay
# underscore-private so discovery never surfaces them, and the server (the
# code that owns the human-approval channel) imports them explicitly.


def _set_approval_channel(channel: Callable[[dict[str, Any]], bool] | None) -> None:
    """Install the human-approval channel (frontend card) or remove it."""
    global _APPROVAL_CHANNEL
    _APPROVAL_CHANNEL = channel


def _netcdf_safe(data: Any) -> Any:
    """Return a copy whose attrs can be serialised to NetCDF.

    peaks.load restores physical units as pint Quantity objects and carries
    pydantic metadata models (``_scan`` etc.) in attrs; raw ``to_netcdf``
    rejects both.  The staged copy keeps values and coordinates intact,
    stringifies unit-like attrs and drops non-serialisable object attrs.
    """

    def scalar_ok(value: Any) -> bool:
        if isinstance(value, (str, bytes, bool, int, float)) or value is None:
            return True
        if isinstance(value, np.ndarray):
            return True
        if isinstance(value, (list, tuple)):
            return all(scalar_ok(item) for item in value)
        if isinstance(value, dict):
            return all(
                isinstance(key, str) and scalar_ok(item) for key, item in value.items()
            )
        return False

    def sanitize_attrs(attrs: dict[str, Any]) -> None:
        for key in [k for k in attrs.keys()]:
            value = attrs[key]
            if scalar_ok(value):
                continue
            module = type(value).__module__ or ""
            if module.startswith("pint") or module.startswith("numpy"):
                attrs[key] = str(value)
            else:
                attrs.pop(key, None)

    copy = data.copy(deep=False)
    sanitize_attrs(copy.attrs)
    for coordinate in copy.coords.values():
        sanitize_attrs(coordinate.attrs)
    variables = getattr(copy, "data_vars", None)
    if variables is not None:
        for name in variables:
            sanitize_attrs(copy[name].attrs)
    return copy


def _serialise(data: Any, path: Path) -> tuple[bytes, str]:
    """Serialise one result to bytes: NetCDF for xarray, JSON otherwise."""
    if str(path).endswith(".nc") and hasattr(data, "to_netcdf"):
        buffer = io.BytesIO()
        _netcdf_safe(data).to_netcdf(buffer)
        return buffer.getvalue(), "netcdf"
    if isinstance(data, (dict, list)):
        return (
            json.dumps(data, default=str, ensure_ascii=False).encode("utf-8"),
            "json",
        )
    raise TypeError(
        f"save: cannot serialise {type(data).__name__}; support: "
        "xarray DataArray/Dataset (.nc) or dict/list (.json)"
    )


def _structure(data: Any, kind: str) -> dict[str, Any]:
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
        if getattr(getattr(data, "data", None), "chunks", None) is None:
            try:
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


def _stage_item(data: Any, path: Path, overwrite: bool) -> PendingItem:
    """Serialise one result into a hidden staging file next to its target."""
    payload, kind = _serialise(data, path)
    digest = hashlib.sha256(payload).hexdigest()
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
    return PendingItem(
        path=path,
        tmp_path=Path(temporary_name),
        kind=kind,
        approx_bytes=len(payload),
        sha256=digest,
        structure=_structure(data, kind),
        overwrite=overwrite,
        exists_at_stage=path.exists(),
    )


def _item_payload(item: PendingItem) -> dict[str, Any]:
    return {
        "path": str(item.path),
        "kind": item.kind,
        "size_bytes": item.approx_bytes,
        "sha256": item.sha256,
        "structure": item.structure,
        "overwrite": item.overwrite,
        "exists_at_stage": item.exists_at_stage,
    }


def _create_ticket(
    operation: str,
    items: list[PendingItem],
    summary: str,
) -> PendingBatch:
    ticket = PendingBatch(
        ticket_id=uuid.uuid4().hex,
        operation=operation,
        summary=summary,
        items=items,
    )
    _STAGED[ticket.ticket_id] = ticket
    _expire_stale()
    return ticket


def _ticket_payload(ticket: PendingBatch) -> dict[str, Any]:
    return {
        "operation": ticket.operation,
        "summary": ticket.summary,
        "items": [_item_payload(item) for item in ticket.items],
    }


def _expire_stale() -> None:
    """Drop expired tickets and their staging files (best effort)."""
    expired = [ticket_id for ticket_id, batch in _STAGED.items() if batch.expired()]
    for ticket_id in expired:
        batch = _STAGED.pop(ticket_id, None)
        if batch is not None:
            _cleanup(batch)


def _cleanup(batch: PendingBatch) -> None:
    for item in batch.items:
        try:
            item.tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def _lookup(ticket_id: str) -> PendingBatch:
    batch = _STAGED.get(ticket_id)
    if batch is None:
        raise KeyError(f"save: unknown or already-used ticket {ticket_id!r}")
    if batch.expired():
        _STAGED.pop(ticket_id, None)
        _cleanup(batch)
        raise KeyError(f"save: ticket {ticket_id!r} expired")
    if batch.used:
        raise KeyError(f"save: ticket {ticket_id!r} was already used")
    return batch


def _publish_batch(ticket_id: str) -> dict[str, Any]:
    """Publish one approved ticket: every staged item is atomically renamed
    over its target.  Idempotent items whose target appeared meanwhile and
    lack overwrite are skipped and their staging file removed."""
    batch = _lookup(ticket_id)
    if not batch.authorized:
        raise PermissionError("save: ticket not authorized by a human approval")
    batch.used = True
    _STAGED.pop(ticket_id, None)
    published: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for item in batch.items:
        if item.path.exists() and not item.overwrite:
            skipped.append(
                {
                    "path": str(item.path),
                    "sha256": item.sha256,
                    "status": "exists",
                }
            )
            try:
                item.tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            continue
        try:
            item.tmp_path.replace(item.path)
            published.append(
                {
                    "path": str(item.path),
                    "sha256": item.sha256,
                    "approx_bytes": item.approx_bytes,
                }
            )
        except OSError as exc:
            skipped.append(
                {
                    "path": str(item.path),
                    "sha256": item.sha256,
                    "status": f"publish failed: {exc}",
                }
            )
    return {"published": published, "skipped": skipped}


def _discard_ticket(ticket_id: str) -> None:
    """Drop one staged ticket without writing anything."""
    batch = _lookup(ticket_id)
    batch.used = True
    _STAGED.pop(ticket_id, None)
    _cleanup(batch)


def _request_consent(ticket: PendingBatch) -> bool | None:
    """Run the human-approval step.  None = no channel installed."""
    channel = _APPROVAL_CHANNEL
    if channel is None:
        return None
    return bool(channel(_ticket_payload(ticket)))


def _run_staged(
    operation: str,
    requests: list[tuple[Any, Path, bool]],
    summary: str,
) -> dict[str, Any]:
    """Stage items, ask the human, publish or clean up (gateway primitive).

    ``requests`` are ``(data, target_path, overwrite)`` triples.  Returns a
    JSON-safe outcome: ``{"status": "saved" | "denied" | "pending_consent",
    "ticket_id": ..., ...}``.  Facades with their own result models call this
    and map the outcome onto their reports.
    """
    staged = [
        _stage_item(data, Path(path), overwrite) for data, path, overwrite in requests
    ]
    ticket = _create_ticket(operation, staged, summary)
    approved = _request_consent(ticket)
    if approved is None:
        return {"status": "pending_consent", "ticket_id": ticket.ticket_id}
    if approved:
        ticket.authorized = True
        outcome = _publish_batch(ticket.ticket_id)
        return {
            "status": "saved",
            "ticket_id": ticket.ticket_id,
            "published": outcome["published"],
            "skipped": outcome["skipped"],
        }
    _discard_ticket(ticket.ticket_id)
    return {"status": "denied", "ticket_id": ticket.ticket_id}


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
    ``pending_consent`` for the approval flow.

    There is deliberately NO ``approve`` parameter: code cannot authorise a
    write.  Consent is a runtime object consumed by the gateway.
    """
    target = Path(path).expanduser()
    if target.exists() and not overwrite:
        report = SaveReport(status="blocked", path=str(target), kind="", overwrite=False)
        print(f"save_result: {target} exists; pass overwrite=True after review.")
        return report
    staged = [_stage_item(data, target, overwrite)]
    ticket = _create_ticket(
        "save_result", staged, f"save_result: {target.name}"
    )
    report = SaveReport(
        status="awaiting_consent",
        path=str(target),
        kind=staged[0].kind,
        approx_bytes=staged[0].approx_bytes,
        dims={
            str(key): int(value)
            for key, value in (staged[0].structure.get("sizes") or {}).items()
        },
        dtype=staged[0].structure.get("dtype"),
        overwrite=overwrite,
        ticket_id=ticket.ticket_id,
        sha256=staged[0].sha256,
    )
    print(report.summary_line())
    approved = _request_consent(ticket)
    if approved is None:
        print(
            "save_result: staged and waiting for human approval "
            f"(ticket {ticket.ticket_id}); no frontend approval channel is "
            "installed."
        )
        return report
    if approved:
        ticket.authorized = True
        outcome = _publish_batch(ticket.ticket_id)
        report.status = "saved" if outcome["published"] else "denied"
        if outcome["skipped"]:
            print(f"save_result: skipped: {outcome['skipped'][0]['path']}")
        return report
    _discard_ticket(ticket.ticket_id)
    report.status = "denied"
    print(
        f"save_result: the user did not approve; nothing was written "
        f"({target})."
    )
    return report
