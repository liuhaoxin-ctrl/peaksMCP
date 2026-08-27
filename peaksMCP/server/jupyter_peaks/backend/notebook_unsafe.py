"""Notebook mutation operations mediated by scanning and frontend consent."""

from __future__ import annotations

from typing import Any

from ..security import AuditLogger, ConsentManager, scan_code
from .base import ExecutionMode, SharedState


class UnsafeNotebookBackend:
    """Execute or mutate notebook cells after security checks and consent."""

    def __init__(self, state: SharedState, consent: ConsentManager, audit: AuditLogger) -> None:
        self.state = state
        self.consent = consent
        self.audit = audit

    def _authorize(self, operation: str, code: str = "") -> None:
        scan = scan_code(code) if code else None
        if scan and scan.blocked:
            self.audit.write(operation, "blocked", {"scan": scan.to_dict()})
            raise PermissionError(scan.block_reason or "code was blocked by security scanner")
        if self.state.mode is not ExecutionMode.DANGEROUS:
            approved = self.consent.request(operation, {"code": code[:4000], "scan": scan.to_dict() if scan else None})
            if not approved:
                self.audit.write(operation, "denied", {})
                raise PermissionError("user did not approve the notebook operation")
        self.audit.write(operation, "approved", {})

    def execute_code(self, code: str, timeout: float = 120.0) -> dict[str, Any]:
        self._authorize("notebook_execute_code", code)
        return self.state.bridge.request("execute_code", {"code": code}, timeout=timeout)

    def execute_active_cell(self, timeout: float = 120.0) -> dict[str, Any]:
        code = str(self.state.active_cell.get("source") or "")
        self._authorize("notebook_execute_active_cell", code)
        return self.state.bridge.request("execute_active_cell", timeout=timeout)

    def add_cell(self, source: str = "", cell_type: str = "code", position: str = "below") -> dict[str, Any]:
        if cell_type not in {"code", "markdown", "raw"}:
            raise ValueError("cell_type must be code, markdown, or raw")
        self._authorize("notebook_add_cell", source if cell_type == "code" else "")
        return self.state.bridge.request("add_cell", {"source": source, "cell_type": cell_type, "position": position})

    def delete_cell(self, index: int | None = None) -> dict[str, Any]:
        self._authorize("notebook_delete_cell")
        return self.state.bridge.request("delete_cell", {"index": index})

    def apply_patch(self, index: int, source: str) -> dict[str, Any]:
        self._authorize("notebook_apply_patch", source)
        return self.state.bridge.request("apply_patch", {"index": index, "source": source})

