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

    def _authorize(self, operation: str, code: str = "", force_consent: bool = False, cell: dict[str, Any] | None = None) -> None:
        scan = scan_code(code) if code else None
        if scan and scan.blocked:
            self.audit.write(operation, "blocked", {"scan": scan.to_dict()})
            raise PermissionError(scan.block_reason or "code was blocked by security scanner")
        # Patterns such as ``plt.savefig``, and destructive cell operations
        # (delete / patch existing cells), require an explicit, informed consent
        # in every mode (including dangerous): existing cells must never be
        # deleted or overwritten unless the user actively approves it.
        requires_consent = force_consent or bool(scan and scan.requires_explicit_consent)
        if self.state.mode is not ExecutionMode.DANGEROUS or requires_consent:
            details: dict[str, Any] = {"code": code[:4000], "scan": scan.to_dict() if scan else None}
            if cell is not None:
                details["cell"] = cell
            approved = self.consent.request(operation, details)
            if not approved:
                self.audit.write(operation, "denied", {})
                raise PermissionError("user did not approve the notebook operation")
        self.audit.write(operation, "approved", {})

    def execute_code(self, code: str, timeout: float = 120.0) -> dict[str, Any]:
        self._authorize("notebook_execute_code", code)
        return self.state.bridge.request("execute_code", {"code": code}, timeout=timeout)

    def execute_active_cell(self, timeout: float = 120.0) -> dict[str, Any]:
        # Read the LIVE cell source through the frontend instead of trusting the
        # cached ``state.active_cell`` (which goes stale when the user edits a
        # cell without switching away from it).  The very same source is used
        # for scanning, authorisation and execution, and the frontend re-checks
        # the cell id + source before running to close the TOCTOU window.
        fresh = self.state.bridge.request("read_active_cell", timeout=10)
        source = str(fresh.get("source") or "")
        cell_id = fresh.get("id")
        self._authorize("notebook_execute_active_cell", source)
        return self.state.bridge.request(
            "execute_active_cell",
            {"expected_id": cell_id, "expected_source": source},
            timeout=timeout,
        )

    def add_cell(self, source: str = "", cell_type: str = "code", position: str = "below") -> dict[str, Any]:
        if cell_type not in {"code", "markdown", "raw"}:
            raise ValueError("cell_type must be code, markdown, or raw")
        self._authorize("notebook_add_cell", source if cell_type == "code" else "")
        return self.state.bridge.request("add_cell", {"source": source, "cell_type": cell_type, "position": position})

    def delete_cell(self, index: int | None = None) -> dict[str, Any]:
        # Deleting an existing cell is destructive: always require explicit
        # user approval, even in dangerous mode, and state which cell is targeted.
        self._authorize(
            "notebook_delete_cell",
            force_consent=True,
            cell={"index": index, "action": "delete"},
        )
        return self.state.bridge.request("delete_cell", {"index": index})

    def apply_patch(self, index: int, source: str) -> dict[str, Any]:
        # Overwriting an existing cell is destructive: always require explicit
        # user approval, even in dangerous mode, and state which cell is targeted.
        self._authorize(
            "notebook_apply_patch",
            source,
            force_consent=True,
            cell={"index": index, "action": "overwrite"},
        )
        return self.state.bridge.request("apply_patch", {"index": index, "source": source})

