"""Save gateway: nothing is persisted unless a human approves the staged bytes.

Every persistence path funnels through this module:

    1. stage: the result is serialised into a hidden file in the unified
       staging area (a per-ticket temp directory with a strict TTL) - never
       next to the target, never creating the target directory early; a
       one-time ticket binds the EXACT bytes of every item (path, kind,
       size, sha256, structure).
    2. consent: with an approval channel installed (the notebook frontend) a
       card listing the REAL content of every item is presented; approval
       publishes, rejection or expiry removes the staging area.
    3. publish: only after an affirmative human decision the target directory
       is created (if needed) and the staged bytes are copied to a hidden
       ``.part`` file there, then atomically renamed over the target.  There
       is no code path that writes a target without an affirmative human
       decision, and no code-level approve exists.

Consent is a runtime object, never a parameter: gateway functions stay
underscore-private and refuse unapproved tickets, so notebook code cannot
self-authorise a write at any layer.

The model-facing save verb is the MCP tool ``save_with_consent`` (registered
by the server); ``_save_result`` here is its internal implementation and is
NOT part of the model Python surface.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tempfile
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

#: Minutes a staged ticket stays valid before the gateway refuses it.
_TICKET_TTL_S = 10 * 60


class SaveReceipt(BaseModel):
    """Receipt of one save_with_consent attempt (one variable, one file)."""

    operation: str = "save_with_consent"
    status: str = "awaiting_consent"  # saved | denied | pending_consent | blocked
    variable_name: str = ""
    path: str = ""
    kind: str = ""
    size_bytes: int | None = None
    sha256: str | None = None
    dims: dict[str, int] | None = None
    dtype: str | None = None
    units: Any = None
    overwrite: bool = False
    ticket_id: str | None = None
    structure: dict[str, Any] | None = None
    note: str | None = None


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
    #: True when the target already existed at staging time (publish skips it
    #: unless overwrite=True).
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
    """Serialise one result into the unified staging area.

    The staging file lives in its own ``peaksmcp-save-*`` temp directory under
    the platform temp area - never next to the target, and the target
    directory is created only at publish time (after human approval).
    Worker processes can stage their own items the same way; the ticket just
    collects the items.
    """
    payload, kind = _serialise(data, path)
    digest = hashlib.sha256(payload).hexdigest()
    staging_dir = Path(tempfile.mkdtemp(prefix="peaksmcp-save-"))
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"{path.name}.part-",
        suffix=".tmp",
        dir=staging_dir,
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
    """Drop expired tickets and their staging areas (best effort)."""
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
        # Each staged file lives in its own peaksmcp-save-* directory; remove
        # the directory too (best effort, only for our own staging roots).
        parent = item.tmp_path.parent
        if parent.name.startswith("peaksmcp-save-"):
            try:
                shutil.rmtree(parent, ignore_errors=True)
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
    """Publish one approved ticket: each staged item is copied into a hidden
    ``.part`` file inside its target directory and atomically renamed over the
    target.  Idempotent items whose target appeared meanwhile and lack
    overwrite are skipped and their staged bytes removed.

    The target directory is created here, only after human approval; staging
    never touches it and never leaves a ``.part`` next to the target early.
    """
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
            continue
        try:
            item.path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, local_name = tempfile.mkstemp(
                prefix=f".{item.path.name}.part-",
                suffix=".tmp",
                dir=item.path.parent,
            )
            os.close(descriptor)
            local = Path(local_name)
            try:
                shutil.copyfile(item.tmp_path, local)
                with open(local, "rb") as stream:
                    os.fsync(stream.fileno())
                local.replace(item.path)
            finally:
                local.unlink(missing_ok=True)
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
    _cleanup(batch)
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

    ``requests`` are ``(data, target_path, overwrite)`` triples with an
    optional fourth ``dict`` merged into the item structure (e.g. the
    source input path for a conversion manifest row).  Returns a JSON-safe
    outcome: ``{"status": "saved" | "denied" | "pending_consent",
    "ticket_id": ..., ...}``.  Facades with their own result models call this
    and map the outcome onto their reports.
    """
    ticket = _create_ticket(operation, [], summary)
    ticket.items = []
    for request in requests:
        data, path, overwrite, *extra = request
        item = _stage_item(data, Path(path), overwrite)
        if extra and isinstance(extra[0], dict):
            item.structure.update(extra[0])
        ticket.items.append(item)
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


def _variable_preview(value: Any, path: Path, variable_name: str) -> tuple[dict[str, Any], str]:
    """Build the normalized preview of ONE variable for the notebook record
    cell: structure (kind/dims/dtype/units) plus a canonical one-liner.

    This is the "show" step of the save flow: the human sees exactly what
    would be serialised before any consent card is shown.
    """
    kind = "netcdf" if str(path).endswith(".nc") and hasattr(value, "to_netcdf") else "json"
    structure = _structure(value, kind)
    dims = structure.get("sizes") or {}
    shape = "x".join(str(v) for v in dims.values()) if dims else "-"
    text = (
        f"save_with_consent: variable {variable_name!r} -> {path} "
        f"({kind}, shape {shape}, dtype {structure.get('dtype') or '-'}, "
        f"units {structure.get('units') or '-'})"
    )
    return {"kind": kind, "structure": structure, "summary": text}, text


def _save_result(
    data: Any,
    path: str | Path,
    *,
    overwrite: bool = False,
    variable_name: str = "",
) -> SaveReceipt:
    """Stage one result and ask the human through the approval channel.

    Internal implementation of the ``save_with_consent`` MCP tool: serialises
    the result into the unified staging area and creates a one-time ticket
    bound to the exact staged bytes (path, kind, sha256, structure).  With an
    approval channel installed this presents a card to the user and writes
    the file only on affirmative approval; otherwise the ticket stays
    ``pending_consent``.  Prints nothing - the receipt is the outcome.

    There is deliberately NO ``approve`` parameter: code cannot authorise a
    write.  Consent is a runtime object consumed by the gateway.
    """
    target = Path(path).expanduser()
    receipt = SaveReceipt(
        variable_name=variable_name,
        path=str(target),
        overwrite=overwrite,
    )
    if target.exists() and not overwrite:
        receipt.status = "blocked"
        receipt.note = f"{target} exists; pass overwrite=True after review."
        return receipt
    ticket = _create_ticket("save_with_consent", [], f"save_with_consent: {target.name}")
    staged = [_stage_item(data, target, overwrite)]
    ticket.items = staged
    receipt.kind = staged[0].kind
    receipt.size_bytes = staged[0].approx_bytes
    receipt.sha256 = staged[0].sha256
    receipt.structure = staged[0].structure
    receipt.dims = dict(staged[0].structure.get("sizes") or {})
    receipt.dtype = staged[0].structure.get("dtype")
    receipt.units = staged[0].structure.get("units")
    approved = _request_consent(ticket)
    if approved is None:
        receipt.status = "pending_consent"
        receipt.ticket_id = ticket.ticket_id
        receipt.note = (
            "staged and waiting for human approval; no frontend approval "
            "channel is installed"
        )
        return receipt
    if approved:
        ticket.authorized = True
        outcome = _publish_batch(ticket.ticket_id)
        receipt.ticket_id = ticket.ticket_id
        if outcome["published"]:
            receipt.status = "saved"
        elif outcome["skipped"]:
            receipt.status = "blocked"
            receipt.note = outcome["skipped"][0].get("status")
        return receipt
    _discard_ticket(ticket.ticket_id)
    receipt.ticket_id = ticket.ticket_id
    receipt.status = "denied"
    receipt.note = "the user did not approve; nothing was written"
    return receipt
