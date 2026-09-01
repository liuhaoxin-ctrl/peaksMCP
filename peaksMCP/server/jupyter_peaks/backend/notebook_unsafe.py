"""Notebook mutation operations mediated by scanning and frontend consent."""

from __future__ import annotations

import ast
from typing import Any

from ..security import AuditLogger, ConsentManager, scan_code
from .base import ExecutionMode, SharedState


def _extract_call_targets(code: str) -> list[tuple[str | None, str]]:
    """Return ``(root, leaf)`` for every call in ``code``.

    ``np.linalg.svd(x)`` -> ``("np", "svd")``
    ``da.k_convert()`` -> ``("da", "k_convert")``
    ``fit_gold(data)`` -> ``("fit_gold", "fit_gold")``
    ``da[i].mean()`` -> ``(None, "mean")``  (receiver is not a plain name)

    ``root`` is the leftmost plain identifier of the receiver chain, or ``None``
    when it is not a plain name.  ``leaf`` is the invoked method/function name.
    Returns an empty list on syntax errors.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    targets: list[tuple[str | None, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        leaf: str | None = None
        root: str | None = None
        while isinstance(func, ast.Attribute):
            if leaf is None:
                # Outermost attribute is the invoked method; inner attributes
                # are part of the receiver chain.
                leaf = func.attr
            func = func.value
        if isinstance(func, ast.Name):
            root = func.id
            if leaf is None:
                leaf = func.id
        if leaf:
            targets.append((root, leaf))
    return targets


class UnsafeNotebookBackend:
    """Execute or mutate notebook cells after security checks and consent."""

    _PYTHON_EXECUTION_OPERATIONS = {
        "notebook_execute_code",
        "notebook_execute_active_cell",
    }

    # Python builtins that appear as bare calls (``print``, ``len``, ``range``,
    # ``str``, ...).  They are never Peaks APIs and never block execution.
    # ``exec``/``eval``/``compile``/``getattr`` are deliberately absent: the
    # security scanner hard-blocks them, and keeping them unverifiable adds a
    # second rejection layer.
    # Modules whose members are treated as generic (never block).
    _GENERIC_MODULES = frozenset({
        "np", "numpy", "scipy", "plt", "matplotlib", "xr", "xarray",
        "pd", "pandas", "os", "sys", "json", "math", "re", "time",
        "glob", "shutil", "pathlib", "Path", "warnings", "pickle",
        "copy", "itertools", "functools", "collections", "datetime",
        "csv", "io", "tempfile", "uuid", "hashlib", "string", "traceback",
        "threading", "dataclasses", "enum", "abc", "typing", "contextlib",
        "calendar", "random", "statistics", "numbers", "decimal", "fractions",
    })

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
        self._authorize("notebook_execute_code", code)
        return self.state.bridge.request("execute_code", {"code": code}, timeout=timeout)

    def execute_with_api_check(
        self, code: str, timeout: float = 120.0, strict: bool = True
    ) -> dict[str, Any]:
        """Execute ``code`` after verifying every Peaks API reference against the live index.

        The ONLY code-execution tool: it makes the ``peaks_search_api`` /
        ``peaks_get_api`` step mandatory in one call.  All method references in
        ``code`` are extracted via AST and classified:

        - ``verified_peaks_apis``: names resolved to Peaks APIs (index hit);
        - ``generic_refs``: names from known generic libraries (numpy, xarray,
          matplotlib, stdlib, ...) — acceptable because Peaks has no API for them;
        - ``unknown_refs``: attribute names that are neither.  With
          ``strict=True`` (default) the presence of any unknown reference
          HARD-BLOCKS execution and returns an error instead of running, because
          an unknown method name is usually an invented or typo'd Peaks API
          (e.g. ``correct_EF``).  Non-Peaks functions are only allowed when the
          name is a recognised generic one.
        """

        method_targets = _extract_call_targets(code)
        index = self.state.api_index
        verified: list[dict[str, Any]] = []
        generic: list[str] = []
        unknown: list[dict[str, Any]] = []
        for root, leaf in sorted(method_targets):
            # A call rooted on a known generic module (``np``, ``xr``, ``plt``,
            # ``pd``, ``os``, ...) is a library call, not a Peaks API: it never
            # blocks regardless of the leaf name (``np.linalg.svd``, ``np.fft``,
            # ``scipy.signal.savgol_filter``, ...).
            if root is not None and root in self._GENERIC_MODULES:
                # Receiver is an imported generic module (``np``, ``plt``, ...):
                # library call, never a Peaks API.
                generic.append(f"{root}.{leaf}" if root != leaf else leaf)
                continue
            if root is None:
                # Receiver is not a plain identifier (``da[i].mean()``,
                # ``result.squeeze()``): not a Peaks API call shape. Relax.
                generic.append(leaf)
                continue
            matches = index.search(leaf, "all", 5) if index else []
            # Only an EXACT name hit counts as verified; fuzzy matches (e.g.
            # "correct_EF" matching correct_isolated_bad_pixels) are suggested
            # but do NOT verify the name, so invented/typo'd APIs stay blocked.
            exact = [m for m in matches if m.get("name") == leaf]
            if exact:
                verified.append(
                    {"name": leaf, "matches": [str(m.get("id")) for m in exact]}
                )
            elif root == leaf:
                # Module-level function call (``print(...)``, ``len(...)``,
                # user helpers): not in the index -> builtin/user code, relax.
                generic.append(leaf)
            else:
                # ``obj.<method>`` on a data receiver with no Peaks API ->
                # invented/typo'd API, hard-block.
                unknown.append(
                    {
                        "name": leaf,
                        "suggested": [str(m.get("id")) for m in matches],
                    }
                )

        # Task-level select-then-run gate: the first execution of a task must be
        # preceded by at least TWO peaks_search_api / peaks_get_api calls, so
        # the agent genuinely explores before writing/running code.
        if strict and self.state.exploration_count < 2:
            self.audit.write(
                "notebook_execute_with_api_check",
                "blocked",
                {"reason": "task not selected"},
            )
            return {
                "success": False,
                "executed": False,
                "blocked": True,
                "message": (
                    "Execution blocked by the select-then-run rule: no Peaks API has "
                    "been explored yet. First call peaks_search_api to find the APIs "
                    "this task needs, then peaks_get_api to read their signatures and "
                    "return conventions, then execute."
                ),
            }

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
        if strict and unknown:
            self.audit.write(
                "notebook_execute_with_api_check", "blocked", {"unknown_refs": unknown}
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
                    f"(candidates: {api_check['suggestions']}) or fix the typo."
                ),
                "api_check": api_check,
            }
        result = self.execute_code(code, timeout)
        result["api_check"] = api_check
        return result

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
