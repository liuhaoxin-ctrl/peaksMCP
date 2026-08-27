"""State shared by all backends in one Jupyter kernel."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ExecutionMode(StrEnum):
    """Security mode controlling which notebook tools are exposed."""

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
    bridge: Any | None = None
    api_index: Any | None = None
    kernel_state: str = "idle"
    busy_since: float | None = None
    active_cell: dict[str, Any] = field(default_factory=dict)
    active_cell_output: list[dict[str, Any]] = field(default_factory=list)
    lock: threading.RLock = field(default_factory=threading.RLock)
    started_at: float = field(default_factory=time.time)
    kernel_instance_id: str = field(default_factory=lambda: uuid.uuid4().hex)

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
