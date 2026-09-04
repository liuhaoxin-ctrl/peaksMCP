"""State shared by all backends in one Jupyter kernel."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


def ensure_fresh_index(state: SharedState) -> Any:
    """Return the live API index, hot-rebuilding it in the kernel when stale.

    Replaces the old fail-fast ``INDEX_STALE_RESTART_REQUIRED`` error: if the
    installed Peaks / peaksMCP source changed after the index was built, the
    index is rebuilt (under the shared state lock) so searches and code
    execution keep working without a kernel restart.  Only the actual rebuild
    holds the lock; the staleness check and searches are lock-free.
    """
    from peaksMCP.discovery.index import build_index

    if state.api_index is None:
        with state.lock:
            if state.api_index is None:
                state.api_index = build_index()
    elif state.api_index.is_stale():
        with state.lock:
            # Re-check under the lock: another call may have rebuilt already.
            if state.api_index.is_stale():
                state.api_index = build_index()
    return state.api_index


class ExecutionMode(StrEnum):
    """Security mode selecting the consent policy for mutation tools.

    The exposed tool surface is identical in every mode; the mode only changes
    how strictly mutation tools ask for consent when the ``require_consent``
    master switch is enabled (see :mod:`peaksMCP.server.jupyter_peaks`).
    """

    SAFE = "safe"
    UNSAFE = "unsafe"
    DANGEROUS = "dangerous"


@dataclass(slots=True)
class SharedState:
    """Mutable runtime state owned by one IPython kernel.

    A Python module does not persist data by itself. The IPython extension creates one
    ``SharedState`` object and keeps references to it in the server and every backend. The object
    remains alive for as long as those references (normally the kernel process) remain alive.
    """

    ipython: Any
    mode: ExecutionMode = ExecutionMode.SAFE
    require_consent: bool = False
    bridge: Any | None = None
    api_index: Any | None = None
    kernel_state: str = "idle"
    busy_since: float | None = None
    active_cell: dict[str, Any] = field(default_factory=dict)
    active_cell_output: list[dict[str, Any]] = field(default_factory=list)
    cell_outputs: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    last_execution_cell_id: str | None = None
    lock: threading.RLock = field(default_factory=threading.RLock)
    started_at: float = field(default_factory=time.time)
    kernel_instance_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    mcp_instance_id: str | None = None
    read_plot_resources: bool = False
    #: API names proven real by a successful ``peaks_get_api`` this session
    #: (canonical name plus search aliases).  Unknown write references that hit
    #: this set are unlocked instead of blocked.
    verified_peaks_names: set[str] = field(default_factory=set)
    #: Per-name count of unverifiable write attempts, driving the advisory ->
    #: hard-refusal escalation until the name is proven with peaks_get_api.
    unknown_api_attempts: dict[str, int] = field(default_factory=dict)

    @property
    def namespace(self) -> dict[str, Any]:
        """Return the live IPython user namespace."""
        return self.ipython.user_ns

    def mark_busy(self, info: Any | None = None) -> None:
        """Record that the kernel began executing a cell."""
        with self.lock:
            self.kernel_state = "busy"
            self.busy_since = time.time()
            raw = getattr(info, "raw_cell", None)
            if raw:
                self.active_cell["source"] = raw

    def mark_idle(self, _result: Any | None = None) -> None:
        """Record that the kernel completed cell execution."""
        with self.lock:
            self.kernel_state = "idle"
            self.busy_since = None
