"""Append-only JSONL audit log for MCP tool calls and consent decisions."""

from __future__ import annotations

import json
import os
import threading
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

#: 当前 MCP tool call 的 operation_id（tools wrapper 设置）。审计写入会自动带上，
#: 使一次调用内部的 scanner/consent/approve 等事件与 wrapper 的
#: called→blocked/executed/saved 事件共享同一 id —— Observability 链路可重建。
operation_context: ContextVar[str | None] = ContextVar("peaksmcp_operation_id", default=None)


class AuditLogger:
    """Write privacy-conscious structured events to a user-owned log."""

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        home = Path(os.environ.get("PEAKSMCP_HOME", str(Path.home() / ".peaksMCP")))
        self.path = Path(path) if path else home / "audit" / "tool_audit.log"
        self._lock = threading.Lock()

    def write(self, tool: str, outcome: str, details: dict[str, Any] | None = None) -> None:
        """Append one audit event, creating the parent directory on demand.

        When the event does not already carry an ``operation_id`` and one is
        active in the context (a tool call in flight), it is injected so
        internal events (scanner blocks, consent decisions, gateway writes)
        join the same operation chain as the wrapper's called/result lines.
        """
        payload = dict(details or {})
        if "operation_id" not in payload:
            active = operation_context.get()
            if active:
                payload["operation_id"] = active
        event = {"timestamp": datetime.now(UTC).isoformat(), "tool": tool, "outcome": outcome, "details": payload}
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
            try:
                os.chmod(self.path, 0o600)
            except OSError:
                pass

