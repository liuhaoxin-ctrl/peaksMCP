"""Append-only JSONL audit log for MCP tool calls and consent decisions."""

from __future__ import annotations

import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class AuditLogger:
    """Write privacy-conscious structured events to a user-owned log."""

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self.path = Path(path) if path else Path.home() / ".peaksMCP" / "audit" / "tool_audit.log"
        self._lock = threading.Lock()

    def write(self, tool: str, outcome: str, details: dict[str, Any] | None = None) -> None:
        """Append one audit event, creating the parent directory on demand."""
        event = {"timestamp": datetime.now(UTC).isoformat(), "tool": tool, "outcome": outcome, "details": details or {}}
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
            try:
                os.chmod(self.path, 0o600)
            except OSError:
                pass

