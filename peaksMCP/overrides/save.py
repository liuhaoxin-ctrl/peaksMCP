"""Save gateway: nothing is persisted unless a human approves the staged bytes.

Every persistence path funnels through this module's :class:`SaveGateway`:

    1. stage: the result is serialised into a hidden file in the unified
       staging area (a per-item ``peaksmcp-save-*`` temp directory) - never
       next to the target, never creating the target directory early; a
       one-time ticket binds the EXACT bytes of every item (path, kind,
       size, sha256, structure).
    2. consent: with an approval channel installed (the notebook frontend) a
       card listing the REAL content of every item is presented; approval
       publishes, rejection or expiry removes the staging area.  Without a
       channel the operation fails closed immediately (``blocked``,
       ``no_consent_channel``) - there is no unrecoverable "pending" state.
    3. publish: only after an affirmative human decision the target directory
       is created (if needed) and the staged bytes are copied to a hidden
       ``.part`` file there, then atomically renamed over the target.  Each
       item reports ``published`` / ``exists`` / ``failed`` explicitly;
       approval is never conflated with save success.

Ownership and lifecycle:

- the gateway is an instance owned by the running MCP server (installed on
  ``start()``, cleared on ``stop()``); a lock serialises all registry
  operations; expired tickets are scavenged on every gateway touch and on
  installation, and staging failures clean up in ``try/finally``.
- worker processes (batch staging) stage their own items with the same
  stateless helpers; the ticket only collects the items.
- consent is a runtime object, never a parameter: code cannot authorise a
  write at any layer, and no code-level approve exists.

The model-facing save verb is the MCP tool ``save_with_consent``; the module
helpers are its internal implementation, NOT part of the model Python
surface.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

#: Minutes a staged ticket stays valid (enforced at every gateway touch).
_TICKET_TTL_S = 10 * 60


class SaveReceipt(BaseModel):
    """Receipt of one save_with_consent attempt (one variable, one file)."""

    operation: str = "save_with_consent"
    status: str = "blocked"  # saved | denied | blocked | failed
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
    items: list[PendingItem] = field(default_factory=list)
    created_at: float = field(default_factory=time.monotonic)
    used: bool = False
    authorized: bool = False

    def expired(self) -> bool:
        return time.monotonic() - self.created_at > _TICKET_TTL_S


class SaveGateway:
    """Server-owned staged-save registry (locked, scavenging, fail-closed).

    One gateway instance exists per kernel; the running MCP server owns it
    (installed on start, cleared on stop).  All registry operations run under
    one lock; expired tickets are scavenged at every touch and on
    installation; staging failures clean up via ``try/finally``.
    """

    def __init__(self) -> None:
        self._staged: dict[str, PendingBatch] = {}
        self._lock = threading.RLock()
        self._channel: Callable[[dict[str, Any]], bool | None] | None = None
        self.scavenge()

    # -- channel ownership ---------------------------------------------------
    def set_approval_channel(self, channel: Callable[[dict[str, Any]], bool | None] | None) -> None:
        """Install the human-approval channel (frontend card) or remove it."""
        with self._lock:
            self._channel = channel

    @property
    def has_consent_channel(self) -> bool:
        with self._lock:
            return self._channel is not None

    def _request_consent(self, ticket: PendingBatch) -> bool | None:
        channel = self._channel
        if channel is None:
            return None
        answer = channel(self.ticket_payload(ticket))
        # ``None`` from the channel = the approver was unreachable (frontend
        # offline); that is a blocked disposition, not a human refusal.
        return None if answer is None else bool(answer)

    # -- registry ------------------------------------------------------------
    def scavenge(self) -> None:
        """Drop expired tickets and their staging areas (every gateway touch)."""
        with self._lock:
            expired = [ticket_id for ticket_id, batch in self._staged.items() if batch.expired()]
            for ticket_id in expired:
                self._drop_locked(ticket_id)

    def reset(self) -> None:
        """Clear every staged ticket and the approval channel (server stop)."""
        with self._lock:
            for ticket_id in list(self._staged):
                self._drop_locked(ticket_id)
            self._channel = None

    def create_ticket(
        self,
        operation: str,
        items: list[PendingItem],
        summary: str,
    ) -> PendingBatch:
        self.scavenge()
        ticket = PendingBatch(
            ticket_id=uuid.uuid4().hex,
            operation=operation,
            summary=summary,
            items=list(items),
        )
        with self._lock:
            self._staged[ticket.ticket_id] = ticket
        return ticket

    def _lookup_locked(self, ticket_id: str) -> PendingBatch:
        batch = self._staged.get(ticket_id)
        if batch is None:
            raise KeyError(f"save: unknown or already-used ticket {ticket_id!r}")
        if batch.expired():
            self._drop_locked(ticket_id)
            raise KeyError(f"save: ticket {ticket_id!r} expired")
        if batch.used:
            raise KeyError(f"save: ticket {ticket_id!r} was already used")
        return batch

    def _drop_locked(self, ticket_id: str) -> None:
        batch = self._staged.pop(ticket_id, None)
        if batch is not None:
            self._cleanup(batch)

    @staticmethod
    def _cleanup(batch: PendingBatch) -> None:
        """Remove every staged file and its per-item staging directory.

        Each staged file lives in its own ``peaksmcp-save-*`` temp directory;
        cleanup removes the file first, then the directory (best effort, only
        for our own staging roots - never anything next to a target).
        """
        for item in batch.items:
            try:
                item.tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            parent = item.tmp_path.parent
            if parent.name.startswith("peaksmcp-save-"):
                try:
                    shutil.rmtree(parent, ignore_errors=True)
                except OSError:
                    pass

    def ticket_payload(self, ticket: PendingBatch) -> dict[str, Any]:
        return {
            "ticket_id": ticket.ticket_id,
            "operation": ticket.operation,
            "summary": ticket.summary,
            "items": [item_payload(item) for item in ticket.items],
        }

    def discard(self, ticket_id: str) -> None:
        """Drop one staged ticket without writing anything."""
        with self._lock:
            batch = self._lookup_locked(ticket_id)
            batch.used = True
            self._drop_locked(ticket_id)

    def publish(self, ticket_id: str) -> dict[str, Any]:
        """Publish one approved ticket (authorization-gated, atomic per item).

        Each staged item is copied into a hidden ``.part`` file inside its
        target directory and atomically renamed over the target; the outcome
        reports every item explicitly as ``published`` / ``exists`` /
        ``failed`` - approval alone never counts as save success.  The target
        directory is created here, only after human approval.  Returns
        ``{"status": "saved" | "exists", "items": [...]}``.
        """
        with self._lock:
            batch = self._lookup_locked(ticket_id)
            if not batch.authorized:
                raise PermissionError("save: ticket not authorized by a human approval")
            batch.used = True
            # Single-use: leave the registry now, but clean the staging files
            # only AFTER they have been copied onto their targets below.
            self._staged.pop(ticket_id, None)
        results: list[dict[str, Any]] = []
        for item in batch.items:
            try:
                if item.path.exists() and not item.overwrite:
                    results.append(
                        {
                            "path": str(item.path),
                            "sha256": item.sha256,
                            "status": "exists",
                            "approx_bytes": item.approx_bytes,
                        }
                    )
                    continue
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
                results.append(
                    {
                        "path": str(item.path),
                        "sha256": item.sha256,
                        "status": "published",
                        "approx_bytes": item.approx_bytes,
                    }
                )
            except OSError as exc:
                results.append(
                    {
                        "path": str(item.path),
                        "sha256": item.sha256,
                        "status": "failed",
                        "error": str(exc),
                    }
                )
        self._cleanup(batch)
        all_existing = bool(results) and all(item["status"] == "exists" for item in results)
        return {
            "status": "exists" if all_existing else "saved",
            "items": results,
        }

    def run_staged(
        self,
        operation: str,
        requests: list[tuple[Any, Path, bool]],
        summary: str,
    ) -> dict[str, Any]:
        """Stage items, ask the human, publish or clean up (gateway primitive).

        ``requests`` are ``(data, target_path, overwrite)`` triples with an
        optional fourth ``dict`` merged into the item structure (e.g. the
        source input path for a conversion manifest row).  Returns a JSON-safe
        outcome: ``{"status": "saved" | "denied" | "blocked", "items": [...]}``
        plus ``reason`` for blocked.  Without an approval channel the staged
        items are cleaned up and the outcome is ``blocked``
        (``no_consent_channel``) - no unrecoverable pending state exists.
        """
        ticket = self.create_ticket(operation, [], summary)
        try:
            for request in requests:
                data, path, overwrite, *extra = request
                item = _stage_item(data, Path(path), overwrite)
                if extra and isinstance(extra[0], dict):
                    item.structure.update(extra[0])
                ticket.items.append(item)
            if not self._channel:
                self.discard(ticket.ticket_id)
                return {
                    "status": "blocked",
                    "reason": "no_consent_channel",
                    "ticket_id": ticket.ticket_id,
                }
            approved = self._request_consent(ticket)
            if approved is None:
                self.discard(ticket.ticket_id)
                return {
                    "status": "blocked",
                    "reason": "no_consent_channel",
                    "ticket_id": ticket.ticket_id,
                }
            if approved:
                with self._lock:
                    ticket.authorized = True
                outcome = self.publish(ticket.ticket_id)
                return {
                    "status": outcome["status"],
                    "ticket_id": ticket.ticket_id,
                    "items": outcome["items"],
                }
            self.discard(ticket.ticket_id)
            return {"status": "denied", "ticket_id": ticket.ticket_id, "items": []}
        except Exception:
            self.discard(ticket.ticket_id)
            raise


# --------------------------------------------------------------------------- #
# Stateless serialisation / staging helpers (module level, safe for workers)  #
# --------------------------------------------------------------------------- #

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
        # Mappings are NOT storable: NetCDF attributes accept only str, numbers,
        # ndarray, list, tuple and bytes.  Accepting them here made every save
        # of a scan carrying `experiment_metadata_json` fail at write time.
        return False

    def netcdf_value(value: Any) -> Any | None:
        """NetCDF-storable form of one attribute value, or None to drop it."""
        if isinstance(value, (dict, list, tuple)) and not scalar_ok(value):
            try:
                # The embedded metadata document (and anything else structured)
                # is stored as JSON - the form the conversion header reader
                # already restores with ``json.loads``.
                return json.dumps(value, default=str)
            except (TypeError, ValueError):
                return None
        if scalar_ok(value):
            return list(value) if isinstance(value, tuple) else value
        module = type(value).__module__ or ""
        if module.startswith("pint") or module.startswith("numpy"):
            return str(value)
        return None

    def sanitize_attrs(attrs: dict[str, Any]) -> None:
        for key in [k for k in attrs.keys()]:
            replacement = netcdf_value(attrs[key])
            if replacement is None:
                attrs.pop(key, None)
            else:
                attrs[key] = replacement

    copy = data.copy(deep=False)
    sanitize_attrs(copy.attrs)
    for coordinate in copy.coords.values():
        sanitize_attrs(coordinate.attrs)
    variables = getattr(copy, "data_vars", None)
    if variables is not None:
        for name in variables:
            sanitize_attrs(copy[name].attrs)
    return copy


def _serialisation_kind(data: Any, path: Path) -> str:
    """Resolve the serialisation kind for one (value, target) pair.

    Raises TypeError (with a model-readable message) for unsupported pairs:
    xarray results require a ``.nc`` target, matplotlib figures an image
    suffix, plain dict/list any suffix (json).  This single decision is
    shared by the precheck, the preview and the actual serialisation, so the
    save tool never stages something the writer would refuse.
    """
    suffix = str(path).lower()
    if suffix.endswith(".nc") and hasattr(data, "to_netcdf"):
        return "netcdf"
    if isinstance(data, (dict, list)):
        return "json"
    if suffix.endswith((".png", ".svg", ".pdf", ".jpg", ".jpeg")) and _is_figure(data):
        return "figure"
    if hasattr(data, "to_netcdf"):
        raise TypeError(
            f"save: cannot serialise xarray {type(data).__name__} to {path.name}; "
            "xarray results require a .nc target"
        )
    if _is_figure(data):
        raise TypeError(
            f"save: cannot serialise a matplotlib Figure to {path.name}; "
            "figures require an image target (.png/.svg/.pdf/.jpg)"
        )
    raise TypeError(
        f"save: cannot serialise {type(data).__name__} to {path.name}; support: "
        "xarray DataArray/Dataset (.nc), dict/list (.json), matplotlib "
        "Figure (.png/.svg/.pdf/.jpg)"
    )


def _serialise(data: Any, path: Path) -> tuple[bytes, str]:
    """Serialise one result to bytes: NetCDF / JSON / matplotlib Figure."""
    kind = _serialisation_kind(data, path)
    if kind == "netcdf":
        buffer = io.BytesIO()
        _netcdf_safe(data).to_netcdf(buffer)
        return buffer.getvalue(), kind
    if kind == "json":
        return (
            json.dumps(data, default=str, ensure_ascii=False).encode("utf-8"),
            kind,
        )
    return _figure_bytes(data, path), kind


def _is_figure(data: Any) -> bool:
    """True for a matplotlib Figure (checked by module, no import required)."""
    return type(data).__module__.startswith("matplotlib.figure") and hasattr(data, "savefig")


def _figure_bytes(figure: Any, path: Path) -> bytes:
    """Render a matplotlib figure to bytes (Agg rendering, no display side effect)."""
    import matplotlib

    backend = matplotlib.get_backend()
    if backend.lower() != "agg":
        matplotlib.use("Agg", force=True)
    buffer = io.BytesIO()
    figure.savefig(buffer, format=path.suffix.lstrip(".").lower() or "png", dpi=figure.get_dpi() or 150)
    return buffer.getvalue()


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
    elif kind == "figure":
        structure.update(
            {
                "figure": str(type(data).__name__),
                "n_axes": len(getattr(data, "axes", []) or []),
                "dpi": getattr(data, "dpi", None),
                "size_inches": [
                    float(value)
                    for value in getattr(data, "get_size_inches", lambda: [])()
                ],
            }
        )
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


def item_payload(item: PendingItem) -> dict[str, Any]:
    return {
        "path": str(item.path),
        "kind": item.kind,
        "size_bytes": item.approx_bytes,
        "sha256": item.sha256,
        "structure": item.structure,
        "overwrite": item.overwrite,
        "exists_at_stage": item.exists_at_stage,
    }


def _variable_preview(value: Any, path: Path, variable_name: str) -> tuple[dict[str, Any], str]:
    """Build the normalized preview of ONE variable for the notebook record
    cell: structure (kind/dims/dtype/units) plus a canonical one-liner.

    This is the "show" step of the save flow: the human sees exactly what
    would be serialised before any consent card is shown.  Figure targets
    preview as figure metadata (axes count / size); JSON as a bounded snippet.
    """
    kind = _serialisation_kind(value, path)
    structure = _structure(value, kind)
    dims = structure.get("sizes") or {}
    shape = "x".join(str(v) for v in dims.values()) if dims else "-"
    if kind == "figure":
        detail = (
            f"figure {structure.get('figure')}, {structure.get('n_axes')} axes, "
            f"dpi {structure.get('dpi')}"
        )
    elif kind == "netcdf":
        detail = f"shape {shape}, dtype {structure.get('dtype') or '-'}, units {structure.get('units') or '-'}"
    else:
        detail = "json"
    text = (
        f"save_with_consent: variable {variable_name!r} -> {path} "
        f"({kind}: {detail})"
    )
    return {"kind": kind, "structure": structure, "summary": text}, text


# --------------------------------------------------------------------------- #
# Active-gateway delegation (the running server installs its gateway here)    #
# --------------------------------------------------------------------------- #

#: Process-wide gateway.  The MCP server owns an instance and installs it on
#: start (cleared on stop); cell-level code (convert_experiment /
#: batch staging) routes through the installed gateway.  Before a server
#: exists a fresh default gateway is used (fail-closed: no channel).
_GATEWAY: SaveGateway | None = None
_GATEWAY_LOCK = threading.Lock()


def _active_gateway() -> SaveGateway:
    global _GATEWAY
    if _GATEWAY is None:
        with _GATEWAY_LOCK:
            if _GATEWAY is None:
                _GATEWAY = SaveGateway()
    return _GATEWAY


def install_gateway(gateway: SaveGateway) -> None:
    """Install a server-owned gateway (server start); clears the previous one."""
    global _GATEWAY
    with _GATEWAY_LOCK:
        if _GATEWAY is not None:
            _GATEWAY.reset()
        _GATEWAY = gateway
        gateway.scavenge()


def uninstall_gateway() -> None:
    """Stop owning the gateway: clear tickets + channel, fall back to a fresh
    fail-closed default (no channel)."""
    global _GATEWAY
    with _GATEWAY_LOCK:
        if _GATEWAY is not None:
            _GATEWAY.reset()
            _GATEWAY = None


def _set_approval_channel(channel: Callable[[dict[str, Any]], bool] | None) -> None:
    """Install/remove the approval channel on the active gateway."""
    _active_gateway().set_approval_channel(channel)


def _create_ticket(operation: str, items: list[PendingItem], summary: str) -> PendingBatch:
    return _active_gateway().create_ticket(operation, items, summary)


def _ticket_payload(ticket: PendingBatch) -> dict[str, Any]:
    return _active_gateway().ticket_payload(ticket)


def _request_consent(ticket: PendingBatch) -> bool | None:
    return _active_gateway()._request_consent(ticket)


def _publish_batch(ticket_id: str) -> dict[str, Any]:
    return _active_gateway().publish(ticket_id)


def _discard_ticket(ticket_id: str) -> None:
    return _active_gateway().discard(ticket_id)


def _run_staged(operation: str, requests: list[tuple[Any, Path, bool]], summary: str) -> dict[str, Any]:
    return _active_gateway().run_staged(operation, requests, summary)


def _save_result(
    data: Any,
    path: str | Path,
    *,
    overwrite: bool = False,
    variable_name: str = "",
) -> SaveReceipt:
    """Stage one result and ask the human through the approval channel.

    Internal implementation of the ``save_with_consent`` MCP tool.  Returns a
    receipt; prints nothing.  There is deliberately NO ``approve`` parameter:
    consent is a runtime object consumed by the gateway, and without an
    approval channel the receipt is ``blocked`` (``no_consent_channel``) with
    nothing staged.
    """
    gateway = _active_gateway()
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
    if not gateway.has_consent_channel:
        receipt.status = "blocked"
        receipt.note = "no approval channel is installed; nothing was staged"
        return receipt
    ticket = gateway.create_ticket("save_with_consent", [], f"save_with_consent: {target.name}")
    try:
        staged = [_stage_item(data, target, overwrite)]
        ticket.items = staged
        receipt.kind = staged[0].kind
        receipt.size_bytes = staged[0].approx_bytes
        receipt.sha256 = staged[0].sha256
        receipt.structure = staged[0].structure
        receipt.dims = dict(staged[0].structure.get("sizes") or {})
        receipt.dtype = staged[0].structure.get("dtype")
        receipt.units = staged[0].structure.get("units")
        approved = gateway._request_consent(ticket)
        if approved is None:
            gateway.discard(ticket.ticket_id)
            receipt.status = "blocked"
            receipt.note = (
                "no approver was reachable (no approval channel, or the notebook "
                "frontend is not connected); nothing was written"
            )
            return receipt
        if not approved:
            gateway.discard(ticket.ticket_id)
            receipt.ticket_id = ticket.ticket_id
            receipt.status = "denied"
            receipt.note = "the user did not approve; nothing was written"
            return receipt
        with gateway._lock:
            ticket.authorized = True
        outcome = gateway.publish(ticket.ticket_id)
        receipt.ticket_id = ticket.ticket_id
        first = outcome["items"][0] if outcome["items"] else {}
        if first.get("status") == "published":
            receipt.status = "saved"
        elif first.get("status") == "exists":
            receipt.status = "blocked"
            receipt.note = f"{target} exists at publish; pass overwrite=True."
        else:
            receipt.status = "failed"
            receipt.note = first.get("error") or "publish failed"
        return receipt
    except Exception:
        gateway.discard(ticket.ticket_id)
        raise
