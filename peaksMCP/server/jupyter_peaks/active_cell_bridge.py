"""Kernel-to-JupyterLab Comm bridge for active-cell and consent operations."""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any


class CommBridge:
    """Track a Jupyter Comm and provide request/reply operations."""

    target_name = "peaksMCP:frontend"

    #: A frontend is considered disconnected only after this long without a
    #: heartbeat. The frontend heartbeats every 2s, but browsers throttle
    #: ``setInterval`` in background tabs to ~1/min, so a tight timeout would
    #: falsely drop a live but backgrounded notebook. 120s covers throttled
    #: heartbeats and self-heals: the Comm stays open and reconnects on the next
    #: heartbeat when the tab regains focus.
    STALE_AFTER_S = 120.0

    def __init__(self, state: Any) -> None:
        self.state = state
        self.comm: Any | None = None
        self._pending: dict[str, tuple[threading.Event, dict[str, Any]]] = {}
        self._lock = threading.RLock()
        self.last_seen: float | None = None

    @property
    def connected(self) -> bool:
        """Return whether a live frontend Comm is attached."""
        with self._lock:
            recently_seen = (
                self.last_seen is not None
                and time.time() - self.last_seen < self.STALE_AFTER_S
            )
            return self.comm is not None and not getattr(self.comm, "_closed", False) and recently_seen

    def register(self) -> None:
        """Register the Comm target with the current IPython kernel."""
        kernel = getattr(self.state.ipython, "kernel", None)
        manager = getattr(kernel, "comm_manager", None)
        if manager is not None:
            manager.register_target(self.target_name, self._on_open)

    def _on_open(self, comm: Any, message: dict[str, Any]) -> None:
        with self._lock:
            if self.comm is not comm:
                self._fail_pending("JupyterLab Comm connection was replaced; operation outcome is unknown, check the notebook before retrying")
            self.comm = comm
            self.last_seen = time.time()
            # Bind callbacks to this specific connection. An old tab can keep
            # delivering queued messages after a newer Comm has taken over.
            comm.on_msg(lambda msg: self._on_message(comm, msg))
            comm.on_close(lambda _msg: self._on_close(comm))
            data = message.get("content", {}).get("data", {})
            if data:
                self._update_state(comm, data)
            comm.send({"type": "kernel_ready", "protocol": 1})

    def _fail_pending(self, error: str) -> None:
        """Wake unfinished requests from the outgoing Comm while holding _lock."""
        for event, holder in self._pending.values():
            if not event.is_set():
                holder.update(ok=False, error=error)
                event.set()
        self._pending.clear()

    def _on_close(self, comm: Any) -> None:
        with self._lock:
            if self.comm is comm:
                self.comm = None
                self.last_seen = None
                self._fail_pending("JupyterLab Comm connection was closed; operation outcome is unknown, check the notebook before retrying")

    def _on_message(self, comm: Any, message: dict[str, Any]) -> None:
        with self._lock:
            if comm is not self.comm:
                return
            data = message.get("content", {}).get("data", {}) or {}
            self.last_seen = time.time()
            request_id = data.get("request_id")
            if request_id:
                pending = self._pending.get(request_id)
                if pending is not None:
                    event, holder = pending
                    if not event.is_set():
                        holder.update(data)
                        event.set()
                return
            self._update_state(comm, data)

    def _update_state(self, comm: Any, data: dict[str, Any]) -> None:
        message_type = data.get("type")
        if message_type == "frontend_closing":
            self._on_close(comm)
            return
        if message_type in {"active_cell", "notebook_state"}:
            cell = data.get("cell") or data.get("active_cell") or {}
            if isinstance(cell, dict):
                self.state.active_cell = cell
            outputs = data.get("outputs")
            if isinstance(outputs, list):
                self.state.active_cell_output = outputs

    def request(self, operation: str, payload: dict[str, Any] | None = None, timeout: float = 30) -> dict[str, Any]:
        """Send a request on the current Comm and wait for its correlated reply.

        Parameters
        ----------
        operation : str
            Frontend operation name.
        payload : dict, optional
            Operation-specific arguments.
        timeout : float, default 30
            Maximum seconds to wait for the frontend reply.

        Returns
        -------
        dict
            Result received from the same connection that received the request.

        Raises
        ------
        RuntimeError
            The connection is unavailable, replaced or closed, or the frontend
            reports an operation error.
        TimeoutError
            No reply arrived before the deadline.

        Notes
        -----
        Connection loss wakes pending callers immediately. Operations are not
        replayed: a mutation may already have executed before its reply was lost.

        Examples
        --------
        >>> bridge.request("read_active_cell", timeout=5)  # doctest: +SKIP
        {"id": "cell-1", "source": "data.plot()"}
        """
        request_id = uuid.uuid4().hex
        event = threading.Event()
        holder: dict[str, Any] = {}
        try:
            with self._lock:
                if not self.connected:
                    raise RuntimeError("JupyterLab Comm bridge is not connected")
                # Capture and send under the same lock as replacement/closure.
                comm = self.comm
                self._pending[request_id] = (event, holder)
                comm.send({"type": "request", **(payload or {}), "operation": operation, "request_id": request_id})
            if not event.wait(timeout):
                raise TimeoutError(f"frontend operation {operation!r} timed out")
            if holder.get("ok") is False:
                raise RuntimeError(holder.get("error") or f"frontend operation {operation!r} failed")
            return holder.get("result") or holder
        finally:
            with self._lock:
                self._pending.pop(request_id, None)


def register_comm_target(state: Any) -> CommBridge:
    """Create and register the peaksMCP frontend bridge."""
    bridge = CommBridge(state)
    state.bridge = bridge
    bridge.register()
    return bridge
