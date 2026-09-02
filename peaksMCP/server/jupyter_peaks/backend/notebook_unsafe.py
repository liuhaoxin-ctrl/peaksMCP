"""Notebook mutation operations mediated by scanning and frontend consent."""

from __future__ import annotations

from typing import Any

from ..security import AuditLogger, ConsentManager, scan_code
from ..security.api_allowlists import BUILTIN_NAMES, GENERIC_METHODS, GENERIC_MODULES
from ..security.api_provenance import CallTarget, analyze_provenance, extract_call_targets
from .base import ExecutionMode, SharedState


class UnsafeNotebookBackend:
    """Execute or mutate notebook cells after security checks and consent."""

    _PYTHON_EXECUTION_OPERATIONS = {
        "notebook_write_with_api_check",
    }

    def __init__(self, state: SharedState, consent: ConsentManager, audit: AuditLogger) -> None:
        self.state = state
        self.consent = consent
        self.audit = audit

    def _authorize(
        self,
        operation: str,
        code: str = "",
        force_consent: bool = False,
        cell: dict[str, Any] | None = None,
    ) -> None:
        scan = scan_code(code) if code else None
        if scan and scan.blocked:
            self.audit.write(operation, "blocked", {"scan": scan.to_dict()})
            raise PermissionError(scan.block_reason or "code was blocked by security scanner")
        # AST scanning is a useful early rejection layer, but it cannot prove
        # arbitrary Python safe: reflection, import side effects and higher-order
        # calls can hide behavior from static name matching.  Therefore every
        # operation that actually executes Python normally requires informed
        # consent, and dangerous only relaxes consent for non-executing,
        # append-only mutations such as adding a cell.
        #
        # ``state.require_consent`` (profile ``mcp.require_consent``) is the
        # global master switch: when False (default) no mutation asks for consent
        # at all (the scanner still hard-blocks dangerous code and every call is
        # audit-logged); the operator can flip it back to True to re-enable
        # consent for every write/execute.
        executes_python = operation in self._PYTHON_EXECUTION_OPERATIONS
        requires_consent = (
            executes_python
            or force_consent
            or bool(scan and scan.requires_explicit_consent)
        )
        if self.state.require_consent and (
            self.state.mode is not ExecutionMode.DANGEROUS or requires_consent
        ):
            details: dict[str, Any] = {"code": code[:4000], "scan": scan.to_dict() if scan else None}
            if cell is not None:
                details["cell"] = cell
            approved = self.consent.request(operation, details)
            if not approved:
                self.audit.write(operation, "denied", {})
                raise PermissionError("user did not approve the notebook operation")
        self.audit.write(operation, "approved", {})

    def execute_code(self, code: str, timeout: float = 120.0) -> dict[str, Any]:
        self._authorize("notebook_write_with_api_check", code)
        return self.state.bridge.request("execute_code", {"code": code}, timeout=timeout)

    def _probe_ns(self, name: str, generic_roots: set[str]) -> str | None:
        """Resolve a name's receiver type from the live kernel namespace.

        Only resolves names (``type(ns[name])``); it never executes user code.
        Returns ``None`` when the name is absent or its type is not a module or
        an xarray object.
        """
        obj = self.state.namespace.get(name)
        if obj is None:
            return None
        if isinstance(obj, __import__("types").ModuleType):
            module = getattr(obj, "__name__", "") or ""
            if module.split(".")[0] in generic_roots or name in generic_roots:
                return "module"
            return None
        xarray = __import__("xarray")
        if isinstance(obj, xarray.DataArray):
            return "dataarray"
        if isinstance(obj, getattr(xarray, "Dataset", ())):
            return "dataset"
        if isinstance(obj, getattr(xarray, "DataTree", ())):
            return "datatree"
        return None

    def _receiver_type(
        self,
        target: CallTarget,
        tags: dict[str, str],
        generic_roots: set[str],
    ) -> str:
        """Return the receiver type tag for one call site."""
        root = target.root_id
        if root is None:
            return "unknown"
        if root in generic_roots:
            return "module"
        tag = tags.get(root)
        if tag is None:
            probed = self._probe_ns(root, generic_roots)
            if probed is not None:
                tag = probed
        if tag is None:
            tag = "unknown"
        if target.subscripted:
            # ``da[i]`` inherits the receiver type of ``da``.
            return tag if tag in {"dataarray", "dataset", "datatree"} else "unknown"
        if target.accessor:
            return "accessor" if tag in {"dataarray", "dataset", "datatree"} else "unknown"
        return tag

    def write_with_api_check(self, code: str, timeout: float = 120.0) -> dict[str, Any]:
        """Write and execute ``code`` after checking Peaks API references.

        This is the model-generated-code entry point: it appends a new notebook
        cell, executes it, and verifies every Peaks API reference against the
        live index (agents are expected to ``peaks_search_api`` /
        ``peaks_get_api`` first, but that exploration is not a hard gate).
        Every call site is classified by receiver origin and leaf name:

        - ``verified_peaks_apis``: exact Peaks index hits whose scope matches the
          receiver (DataArray/Dataset/DataTree or accessor like ``metadata``);
        - ``generic_refs``: calls on generic modules (numpy, xarray, matplotlib,
          stdlib, ...), Python builtins, and names defined/imported by ``code``;
        - ``unknown_refs``: leaves that are neither.  Their presence HARD-BLOCKS
          execution (fail-closed): an unverifiable name is usually an invented or
          typo'd Peaks API (e.g. ``correct_EF``).  Receivers whose origin cannot
          be proven do not grant a free pass — the leaf must still verify.

        The live API index is re-checked before classifying; a stale index
        blocks with a kernel-restart request.
        """
        index = self.state.api_index
        if index is None:
            return {
                "success": False, "executed": False, "blocked": True,
                "message": "The Peaks API index is not available; restart the kernel.",
            }
        if index.is_stale():
            self.audit.write(
                "notebook_write_with_api_check", "blocked", {"reason": "index stale"}
            )
            return {
                "success": False, "executed": False, "blocked": True,
                "message": (
                    "INDEX_STALE_RESTART_REQUIRED: the installed Peaks or peaksMCP "
                    "source changed after the API index was built. Restart the kernel "
                    "to rebuild the index before writing code."
                ),
            }

        targets = extract_call_targets(code)
        tags, generic_roots, defined, imported = analyze_provenance(
            code, set(GENERIC_MODULES)
        )
        verified: list[dict[str, Any]] = []
        generic: list[str] = []
        unknown: list[dict[str, Any]] = []
        for target in targets:
            receiver = self._receiver_type(target, tags, generic_roots)
            matches = index.search(target.leaf, "all", 5) if index else []
            exact = [m for m in matches if m.get("name") == target.leaf]

            if receiver == "module":
                # External module call (``np.linalg.svd``, ``fig.savefig``): never
                # a Peaks API, never matched by name against the Peaks index.
                generic.append(f"{target.root_id}.{target.leaf}")
                continue
            if receiver == "accessor":
                scope = target.accessor
            elif receiver in {"dataarray", "dataset", "datatree"}:
                scope = receiver
            else:
                scope = None

            if scope is not None:
                # Receiver type is known: only scope-compatible Peaks APIs verify.
                scoped = [m for m in exact if m.get("scope") == scope]
                if scoped:
                    verified.append({"name": target.leaf, "matches": [str(m["id"]) for m in scoped]})
                elif target.leaf in GENERIC_METHODS or target.leaf in BUILTIN_NAMES:
                    generic.append(target.leaf)
                else:
                    unknown.append({"name": target.leaf, "suggested": [str(m.get("id")) for m in matches]})
                continue

            if target.root_id == target.leaf:
                # Bare call: builtins, code-defined helpers, imported functions
                # and helpers already defined in the live namespace are generic;
                # exact Peaks APIs verify; anything else is unknown.
                in_namespace = callable(self.state.namespace.get(target.leaf))
                if (
                    target.leaf in BUILTIN_NAMES
                    or target.leaf in defined
                    or target.leaf in imported
                    or in_namespace
                ):
                    generic.append(target.leaf)
                elif exact:
                    verified.append({"name": target.leaf, "matches": [str(m["id"]) for m in exact]})
                else:
                    unknown.append({"name": target.leaf, "suggested": [str(m.get("id")) for m in matches]})
                continue

            # Unknown receiver: fail closed.  The leaf must still resolve to a
            # verified Peaks API or a known generic method/builtin; an
            # unverifiable leaf (``make().correct_EF()``, ``da[i].correct_EF()``)
            # is blocked.  Suggest splitting the chain into an intermediate
            # variable when the receiver is complex.
            if exact:
                verified.append({"name": target.leaf, "matches": [str(m["id"]) for m in exact]})
            elif target.leaf in GENERIC_METHODS or target.leaf in BUILTIN_NAMES:
                generic.append(target.leaf)
            else:
                unknown.append({"name": target.leaf, "suggested": [str(m.get("id")) for m in matches]})

        api_check: dict[str, Any] = {
            "verified_peaks_apis": verified,
            "generic_refs": generic,
            "unknown_refs": [u["name"] for u in unknown],
            "suggestions": {
                u["name"]: u["suggested"] for u in unknown if u["suggested"]
            },
            "rule": "Only use non-Peaks functions when Peaks has no corresponding API; "
            "unrecognised method names block execution.",
        }
        if unknown:
            self.audit.write(
                "notebook_write_with_api_check", "blocked", {"unknown_refs": unknown}
            )
            return {
                "success": False,
                "executed": False,
                "blocked": True,
                "unknown_refs": [u["name"] for u in unknown],
                "message": (
                    "Execution blocked: unverifiable API reference(s) "
                    f"{[u['name'] for u in unknown]}. None of these resolve to a Peaks "
                    "API by exact name. Use peaks_search_api to find the correct API "
                    f"(candidates: {api_check['suggestions']}) or fix the typo. If the "
                    "receiver is complex (e.g. a function return value), assign it to "
                    "an intermediate variable first."
                ),
                "api_check": api_check,
            }
        result = self.execute_code(code, timeout)
        result["api_check"] = api_check
        return result

    def add_cell(self, source: str = "", cell_type: str = "code", position: str = "below") -> dict[str, Any]:
        if cell_type not in {"code", "markdown", "raw"}:
            raise ValueError("cell_type must be code, markdown, or raw")
        self._authorize("notebook_add_cell", source if cell_type == "code" else "")
        return self.state.bridge.request("add_cell", {"source": source, "cell_type": cell_type, "position": position})

    def delete_cell(self, index: int | None = None) -> dict[str, Any]:
        # Bind the consent to the actual cell identity (id + current content),
        # not just the index: concurrent edits / other tabs can shift indices
        # between authorisation and execution.
        cell = self.state.bridge.request("read_cell_at", {"index": index}, timeout=10)
        cell_id = cell.get("id")
        self._authorize(
            "notebook_delete_cell",
            force_consent=True,
            cell={
                "index": index,
                "action": "delete",
                "id": cell_id,
                "current_source": str(cell.get("source", ""))[:200],
            },
        )
        return self.state.bridge.request("delete_cell", {"index": index, "expected_id": cell_id})
