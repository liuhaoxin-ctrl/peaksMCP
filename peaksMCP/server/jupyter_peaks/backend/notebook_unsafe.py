"""Notebook mutation operations mediated by scanning and frontend consent."""

from __future__ import annotations

from typing import Any

from peaksMCP.config import prompts as _load_prompts

from ..security import AuditLogger, ConsentManager, scan_code
from ..security.api_allowlists import BUILTIN_NAMES, GENERIC_METHODS, GENERIC_MODULES
from ..security.api_provenance import CallTarget, analyze_provenance, extract_call_targets
from .base import SharedState, ensure_fresh_index

#: Curated hard-block reply text (config/prompts.yaml), read once at import.
_PROMPTS = _load_prompts().get("notebook_unsafe") or {}


def _refused(message: str, *, requires_search: bool = False) -> dict[str, Any]:
    """Build the standard refusal payload for a non-executed cell."""
    payload: dict[str, Any] = {
        "success": False,
        "executed": False,
        "blocked": True,
        "message": message,
    }
    if requires_search:
        payload["requires_search"] = True
    return payload


class UnsafeNotebookBackend:
    """Execute or mutate notebook cells after security checks and consent."""

    def __init__(self, state: SharedState, consent: ConsentManager, audit: AuditLogger) -> None:
        self.state = state
        self.consent = consent
        self.audit = audit

    def _authorize(
        self,
        operation: str,
        code: str = "",
    ) -> None:
        scan = scan_code(code) if code else None
        if scan and scan.blocked:
            self.audit.write(operation, "blocked", {"scan": scan.to_dict()})
            raise PermissionError(scan.block_reason or "code was blocked by security scanner")
        # AST scanning is a useful early rejection layer, but it cannot prove
        # arbitrary Python safe: reflection, import side effects and higher-order
        # calls can hide behavior from static name matching.  Every operation
        # that actually executes Python is therefore gated by informed consent.
        #
        # Two independent consent triggers:
        # 1. ``state.require_consent`` (profile ``mcp.require_consent``) is the
        #    master switch for plain execution: when False (default) no consent
        #    is asked for ordinary analysis cells (the scanner still hard-blocks
        #    dangerous code and every call is audit-logged).
        # 2. Write-to-disk / network intents reported by the scanner
        #    (``requires_explicit_consent``: SAVE001 savefig, SAVE002 file
        #    writers, FILE002 unclear file mode, NET001 egress) ALWAYS require
        #    explicit user approval in the notebook, regardless of the switch:
        #    no result is persisted unless the user sees the exact cell and
        #    approves it.  This is the requirement-first save gate.
        requires_consent = self.state.require_consent or bool(
            scan and scan.requires_explicit_consent
        )
        if requires_consent:
            details: dict[str, Any] = {"code": code[:4000], "scan": scan.to_dict() if scan else None}
            approved = self.consent.request(operation, details)
            if not approved:
                self.audit.write(operation, "denied", {"reason": "requires_explicit_consent"})
                raise PermissionError("user did not approve the notebook operation")
        self.audit.write(operation, "approved", {})

    def execute_code(self, code: str, timeout: float = 120.0) -> dict[str, Any]:
        self._authorize("notebook_write_with_api_check", code)
        return self.state.bridge.request("execute_code", {"code": code}, timeout=timeout)

    def _check_project_imports(
        self, provenance: Any, index: Any
    ) -> dict[str, Any] | None:
        """Reject unverifiable peaks/peaksMCP imports before classification.

        ``from peaksMCP.overrides import ghost`` binds a name that is neither a
        real export nor generic; executing it would only fail in the kernel.
        Star imports from peaks/peaksMCP are refused outright (importing every
        name defeats the API check), and every other from-imported name must
        resolve to a top-level/module index entry or to a real indexed module
        (``from peaksMCP.pxt_utils import converter``).
        """
        star = sorted(provenance.star_sources & {"peaks", "peaksMCP"})
        if star:
            self.audit.write("notebook_write_with_api_check", "blocked", {"reason": "star_import"})
            return _refused(
                _PROMPTS["star_import_rejected"].format(module=", ".join(star)),
                requires_search=True,
            )
        project_imports = {
            name: qualified
            for name, qualified in provenance.from_sources.items()
            if (qualified or "").split(".")[0] in {"peaks", "peaksMCP"}
        }
        if not project_imports:
            return None
        importable_names = {
            entry["name"]
            for entry in index.entries
            if entry.get("scope") in {"top_level", "module"}
        }
        indexed_modules = {entry["module"] for entry in index.entries}
        # ``qualified`` is ``<module>.<original export>``, so renames
        # (``from ... import load_data as ld``) still check the real export.
        missing = sorted(
            qualified.rsplit(".", 1)[-1]
            for name, qualified in project_imports.items()
            if qualified.rsplit(".", 1)[-1] not in importable_names
            and qualified not in indexed_modules
        )
        if not missing:
            return None
        self.audit.write(
            "notebook_write_with_api_check",
            "blocked",
            {"reason": "unverified_import", "names": missing},
        )
        return _refused(
            _PROMPTS["import_not_exported"].format(names=missing),
            requires_search=True,
        )

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

        The live API index is hot-rebuilt in the kernel when the source changed, so
        no kernel restart is needed.
        """
        try:
            index = ensure_fresh_index(self.state)
        except Exception:
            return _refused(_PROMPTS["index_build_failed"])

        targets = extract_call_targets(code)
        provenance = analyze_provenance(code, set(GENERIC_MODULES))
        tags, generic_roots, defined, imported = (
            provenance.tags,
            provenance.generic,
            provenance.defined,
            provenance.imported,
        )
        blocked = self._check_project_imports(provenance, index)
        if blocked:
            return blocked
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

        # A name the model already proved with a successful peaks_get_api this
        # session counts as verified-by-probe: drop it from the unknown set so
        # it no longer blocks.  Only canonical executable names are recorded —
        # aliases never unlock Python symbols (see _record_verified_api).
        verified_names = self.state.verified_peaks_names
        unlocked = [u for u in unknown if u["name"] in verified_names]
        if unlocked:
            verified.extend(
                {"name": u["name"], "matches": list(u.get("suggested") or [])}
                for u in unlocked
            )
        unknown = [u for u in unknown if u["name"] not in verified_names]

        api_check: dict[str, Any] = {
            "verified_peaks_apis": verified,
            "generic_refs": generic,
            "unknown_refs": [u["name"] for u in unknown],
            "suggestions": {
                u["name"]: u["suggested"] for u in unknown if u["suggested"]
            },
            "rule": _PROMPTS["api_check_rule"],
        }
        if unknown:
            # Escalation: an unverifiable name must be proven with a successful
            # peaks_get_api.  First occurrence is advisory (with candidates); a
            # repeated occurrence of the same unproven name is a hard refusal
            # until the model actually gets the API.
            attempts = self.state.unknown_api_attempts
            for name in [u["name"] for u in unknown]:
                attempts[name] = attempts.get(name, 0) + 1
            hard_names = [u["name"] for u in unknown if attempts[u["name"]] >= 2]
            if hard_names:
                self.audit.write(
                    "notebook_write_with_api_check",
                    "blocked",
                    {"unknown_refs": hard_names, "reason": "unverified_retry"},
                )
                return {
                    "success": False,
                    "executed": False,
                    "blocked": True,
                    "requires_search": True,
                    "hard_refusal": True,
                    "unknown_refs": hard_names,
                    "message": _PROMPTS["unknown_api_retry"].format(names=hard_names),
                    "api_check": api_check,
                }
            self.audit.write(
                "notebook_write_with_api_check",
                "blocked",
                {
                    "unknown_refs": [u["name"] for u in unknown],
                    "reason": "unknown_first",
                },
            )
            return {
                "success": False,
                "executed": False,
                "blocked": True,
                "requires_search": True,
                "unknown_refs": [u["name"] for u in unknown],
                "message": _PROMPTS["unknown_api_first"].format(
                    names=[u["name"] for u in unknown],
                    suggestions=api_check["suggestions"],
                ),
                "api_check": api_check,
            }
        result = self.execute_code(code, timeout)
        result["api_check"] = api_check
        return result

    def add_cell(self, source: str = "", cell_type: str = "code") -> dict[str, Any]:
        """Append one cell at the END of the notebook (append-only log).

        There is intentionally no way to insert in the middle, edit or delete an
        existing cell: the notebook preserves the agent's full work history in
        top-to-bottom order, so nothing can be silently overwritten or removed.
        """
        if cell_type not in {"code", "markdown", "raw"}:
            raise ValueError("cell_type must be code, markdown, or raw")
        self._authorize("notebook_add_cell", source if cell_type == "code" else "")
        return self.state.bridge.request("add_cell", {"source": source, "cell_type": cell_type})
