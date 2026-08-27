"""Kernel-to-JupyterLab Comm bridge for active-cell and consent operations."""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any


class CommBridge:
    """Track a Jupyter Comm and provide request/reply operations."""

    target_name = "peaksMCP:frontend"

    def __init__(self, state: Any) -> None:
        self.state = state
        self.comm: Any | None = None
        self._pending: dict[str, tuple[threading.Event, dict[str, Any]]] = {}
        self._lock = threading.RLock()
        self.last_seen: float | None = None

    @property
    def connected(self) -> bool:
        """Return whether a live frontend Comm is attached."""
        recently_seen = self.last_seen is not None and time.time() - self.last_seen < 10
        return self.comm is not None and not getattr(self.comm, "_closed", False) and recently_seen

    def register(self) -> None:
        """Register the Comm target with the current IPython kernel."""
        kernel = getattr(self.state.ipython, "kernel", None)
        manager = getattr(kernel, "comm_manager", None)
        if manager is not None:
            manager.register_target(self.target_name, self._on_open)

    def _on_open(self, comm: Any, message: dict[str, Any]) -> None:
        with self._lock:
            self.comm = comm
            self.last_seen = time.time()
        comm.on_msg(self._on_message)
        comm.on_close(lambda _msg: self._on_close(comm))
        data = message.get("content", {}).get("data", {})
        if data:
            self._update_state(data)
        comm.send({"type": "kernel_ready", "protocol": 1})

    def _on_close(self, comm: Any) -> None:
        with self._lock:
            if self.comm is comm:
                self.comm = None

    def _on_message(self, message: dict[str, Any]) -> None:
        data = message.get("content", {}).get("data", {}) or {}
        self.last_seen = time.time()
        request_id = data.get("request_id")
        if request_id and request_id in self._pending:
            event, holder = self._pending[request_id]
            holder.update(data)
            event.set()
            return
        self._update_state(data)

    def _update_state(self, data: dict[str, Any]) -> None:
        message_type = data.get("type")
        if message_type == "frontend_closing":
            self._on_close(self.comm)
            return
        if message_type in {"active_cell", "notebook_state"}:
            cell = data.get("cell") or data.get("active_cell") or {}
            if isinstance(cell, dict):
                self.state.active_cell = cell
            outputs = data.get("outputs")
            if isinstance(outputs, list):
                self.state.active_cell_output = outputs

    def request(self, operation: str, payload: dict[str, Any] | None = None, timeout: float = 30) -> dict[str, Any]:
        """Send a frontend request and wait for its correlated response."""
        if not self.connected:
            raise RuntimeError("JupyterLab Comm bridge is not connected")
        request_id = uuid.uuid4().hex
        event = threading.Event()
        holder: dict[str, Any] = {}
        with self._lock:
            self._pending[request_id] = (event, holder)
        try:
            self.comm.send({"type": "request", **(payload or {}), "operation": operation, "request_id": request_id})
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
