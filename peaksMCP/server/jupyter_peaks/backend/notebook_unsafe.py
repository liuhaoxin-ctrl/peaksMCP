"""Notebook mutation operations mediated by scanning and frontend consent."""

from __future__ import annotations

import time
from typing import Any

from peaksMCP.config import prompts as _load_prompts

from ..security import AuditLogger, ConsentManager, scan_code
from ..security.api_allowlists import BUILTIN_NAMES, GENERIC_METHODS, GENERIC_MODULES
from ..security.api_provenance import CallTarget, analyze_provenance, extract_call_targets
from .base import SharedState, ensure_fresh_index

#: Curated hard-block reply text (config/prompts.yaml), read once at import.
_PROMPTS = _load_prompts().get("notebook_unsafe") or {}
#: Persistence-policy block text (top-level prompt key, single source).
_RUN_CELL_PERSIST_BLOCKED = (
    _load_prompts().get("run_cell_persist_blocked")
    or "Execution blocked: this cell writes a file; results persist only "
    "through save_with_consent / convert_experiment."
)


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
        # Persistence policy: run_cell is the analysis execution entry point,
        # NEVER a persistence path.  File-write intents (SAVE001 savefig,
        # SAVE002 file writers, FILE002 unclear file mode) are hard-blocked:
        # results persist exclusively through save_with_consent (results) and
        # convert_experiment (PXT -> NetCDF), both of which stage + consent.
        # Notebook autosave (frontend save_notebook) is Run provenance, not
        # analysis-result persistence, and is not affected.
        if operation in {"run_cell", "add_cell"} and scan:
            persist_issues = [
                issue
                for issue in scan.requires_explicit_consent
                if issue.rule_id in {"SAVE001", "SAVE002", "FILE002"}
            ]
            if persist_issues:
                self.audit.write(
                    operation, "blocked",
                    {"reason": "single_persistence_owner",
                     "issues": [issue.to_dict() for issue in persist_issues]},
                )
                raise PermissionError(_RUN_CELL_PERSIST_BLOCKED)
        # Two independent consent triggers:
        # 1. ``state.require_consent`` (profile ``mcp.require_consent``) is the
        #    master switch for plain execution: when False (default) no consent
        #    is asked for ordinary analysis cells (the scanner still hard-blocks
        #    dangerous code and every call is audit-logged).
        # 2. Network egress (NET001) reported by the scanner ALWAYS requires
        #    explicit user approval in the notebook, regardless of the switch.
        requires_consent = self.state.require_consent or bool(
            scan
            and any(issue.rule_id == "NET001" for issue in scan.requires_explicit_consent)
        )
        if requires_consent:
            details: dict[str, Any] = {"code": code[:4000], "scan": scan.to_dict() if scan else None}
            approved = self.consent.request(operation, details)
            if not approved:
                self.audit.write(
                    operation,
                    "denied" if approved is False else "blocked",
                    {
                        "reason": (
                            "requires_explicit_consent"
                            if approved is False
                            else "no_consent_channel"
                        )
                    },
                )
                raise PermissionError("user did not approve the notebook operation")
        self.audit.write(operation, "approved", {})

    def execute_code(self, code: str, timeout: float = 120.0) -> dict[str, Any]:
        """执行一个 cell；超时不等于停止（kernel 可能仍在跑）。

        超时时返回结构化提示而不是抛错，让 agent 明确知道：这一格已被提交到
        kernel，可能仍在执行；后续 run 会排队。正确做法是先 inspect_notebook
        确认实际状态再决定重试/继续。
        """
        self._authorize("run_cell", code)
        try:
            return self.state.bridge.request("execute_code", {"code": code}, timeout=timeout)
        except TimeoutError:
            return {
                "execution_timed_out": True,
                "executed": False,
                "note": (
                    "run_cell timed out waiting for the kernel reply. TIMEOUT IS NOT "
                    "STOP: the cell was submitted and the kernel may still be running; "
                    "later cells will queue behind it. Inspect the notebook "
                    "(inspect_notebook cells/cell) before retrying."
                ),
            }

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
            self.audit.write("run_cell", "blocked", {"reason": "star_import"})
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
            "run_cell",
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

    def write_with_api_check(
        self,
        code: str,
        timeout: float = 120.0,
        api_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Write and execute ``code`` after checking Peaks API references.

        Every reply - blocked, refused or executed - carries ``kernel_state``
        (``busy``/``idle``) and ``kernel_busy_s``: a timeout does NOT stop the
        kernel, so the model must be able to see whether it is still working
        before it retries or queues further cells.
        """
        return self._with_kernel_state(
            self._write_with_api_check(code=code, timeout=timeout, api_ids=api_ids)
        )

    def _with_kernel_state(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Attach the live kernel disposition to one run_cell reply."""
        state = self.state
        busy_since = getattr(state, "busy_since", None)
        payload.setdefault("kernel_state", getattr(state, "kernel_state", None))
        payload.setdefault(
            "kernel_busy_s",
            round(time.time() - busy_since, 3) if busy_since else None,
        )
        return payload

    def _write_with_api_check(
        self,
        code: str,
        timeout: float = 120.0,
        api_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Write and execute ``code`` after checking Peaks API references.

        This is the model-generated-code entry point: it appends a new notebook
        cell, executes it, and verifies every Peaks API reference against the
        live API index.  Canonical-API proof is a HARD gate: an exact-name
        Peaks call is executed only when a successful ``get`` already recorded
        its canonical id in the proof ledger AND the ledger entry's scope
        matches the call site (same-name APIs in different modules/scopes
        cannot be confused).  Optional ``api_ids`` declares the canonical ids
        this cell relies on; every declared id must already be in the ledger
        (an unfetched or never-`get`-ed id is refused, never silently
        ignored).

        - ``verified_peaks_apis``: exact Peaks index hits whose canonical id
          was proven via ``get`` (scope-compatible);
        - ``generic_refs``: calls on generic modules (numpy, xarray,
          matplotlib, stdlib, ...), Python builtins, names defined by the
          cell, and live-namespace helpers that are NOT exact index names;
        - ``unknown_refs``: leaves that are neither - usually invented or
          typo'd Peaks APIs, or exact names whose proof is still missing
          (``get`` first).  Their presence HARD-BLOCKS execution.

        The live API index is hot-rebuilt in the kernel when the source
        changed, so no kernel restart is needed.
        """
        try:
            index = ensure_fresh_index(self.state)
        except Exception:
            return _refused(_PROMPTS["index_build_failed"])

        # Declared canonical ids must be proven already (get happened).
        if api_ids:
            declared = [api_id for api_id in api_ids if api_id not in self.state.verified_apis]
            if declared:
                return {
                    "success": False,
                    "executed": False,
                    "blocked": True,
                    "unproven_api_ids": declared,
                    "message": (
                        "run_cell: api_ids not proven this session - call get "
                        "with each canonical id first: "
                        + ", ".join(str(api_id) for api_id in declared)
                    ),
                }

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
        # Exact-name Peaks candidates that still need a canonical proof.
        needs_proof: list[dict[str, Any]] = []
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
                # Receiver type known: only scope-compatible Peaks APIs apply,
                # and they must be proven via the ledger.
                scoped = [m for m in exact if m.get("scope") == scope]
                if scoped:
                    needs_proof.append(
                        {
                            "name": target.leaf,
                            "scope": scope,
                            "candidates": [str(m["id"]) for m in scoped],
                        }
                    )
                elif target.leaf in GENERIC_METHODS or target.leaf in BUILTIN_NAMES:
                    generic.append(target.leaf)
                else:
                    unknown.append({"name": target.leaf, "suggested": [str(m.get("id")) for m in matches]})
                continue

            if target.root_id == target.leaf:
                # Bare call: builtins and code-defined helpers are generic.
                # A peaks/peaksMCP import or an exact index name is NOT
                # automatically generic - it must be proven with get.
                if target.leaf in BUILTIN_NAMES or target.leaf in defined:
                    generic.append(target.leaf)
                    continue
                imported_from = provenance.from_sources.get(target.leaf)
                import_root = str(imported_from or "").split(".")[0]
                if exact and not (target.leaf in imported and import_root not in {"peaks", "peaksMCP"}):
                    needs_proof.append(
                        {
                            "name": target.leaf,
                            "scope": "bare",
                            "candidates": [str(m["id"]) for m in exact],
                        }
                    )
                    continue
                if target.leaf in imported or callable(self.state.namespace.get(target.leaf)):
                    generic.append(target.leaf)
                    continue
                unknown.append({"name": target.leaf, "suggested": [str(m.get("id")) for m in matches]})
                continue

            # Unknown receiver: fail closed.  The leaf must still resolve to a
            # proven Peaks API or a known generic method/builtin; an
            # unverifiable leaf (``make().correct_EF()``, ``da[i].correct_EF()``)
            # is blocked.  Suggest splitting the chain into an intermediate
            # variable when the receiver is complex.  Generic xarray methods
            # take precedence over same-named index entries (``da[i].mean()``
            # is xarray's mean, not a peaks module function).
            if target.leaf in GENERIC_METHODS or target.leaf in BUILTIN_NAMES:
                generic.append(target.leaf)
            elif exact:
                needs_proof.append(
                    {
                        "name": target.leaf,
                        "scope": "unknown_receiver",
                        "candidates": [str(m["id"]) for m in exact],
                    }
                )
            else:
                unknown.append({"name": target.leaf, "suggested": [str(m.get("id")) for m in matches]})

        # Canonical proof resolution: unlock ONLY through ledger ids whose
        # name matches AND whose scope is compatible with the call site.
        for candidate in needs_proof:
            leaf = candidate["name"]
            call_scope = candidate["scope"]
            proven_ids = self._proven_ids(leaf, call_scope)
            if proven_ids:
                verified.append({"name": leaf, "matches": proven_ids})
            else:
                unknown.append({"name": leaf, "suggested": candidate["candidates"]})

        if unknown:
            # Escalation: an unverifiable name must be proven with a successful
            # get.  First occurrence is advisory (with candidates); a
            # repeated occurrence of the same unproven name is a hard refusal
            # until the model actually gets the API.
            ledger_names = {
                str(snapshot.get("name"))
                for snapshot in self.state.verified_apis.values()
            }
            scope_mismatch = sorted({
                name for name in ledger_names
                if any(unproven["name"] == name for unproven in unknown)
            })
            scope_hint = ""
            if scope_mismatch:
                scope_hint = "\n" + _PROMPTS["proven_scope_mismatch"].format(names=", ".join(scope_mismatch))
            attempts = self.state.unknown_api_attempts
            for name in [u["name"] for u in unknown]:
                attempts[name] = attempts.get(name, 0) + 1
            hard_names = [u["name"] for u in unknown if attempts[u["name"]] >= 2]
            if hard_names:
                self.audit.write(
                    "run_cell",
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
                    "message": _PROMPTS["unknown_api_retry"].format(names=hard_names) + scope_hint,
                    "api_check": {
                        "verified_peaks_apis": verified,
                        "generic_refs": generic,
                        "unknown_refs": [u["name"] for u in unknown],
                        "suggestions": {
                            u["name"]: u["suggested"] for u in unknown if u["suggested"]
                        },
                        "rule": _PROMPTS["api_check_rule"],
                    },
                }
            self.audit.write(
                "run_cell",
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
                    suggestions={
                        u["name"]: u["suggested"] for u in unknown if u["suggested"]
                    },
                ) + scope_hint,
                "api_check": {
                    "verified_peaks_apis": verified,
                    "generic_refs": generic,
                    "unknown_refs": [u["name"] for u in unknown],
                    "suggestions": {
                        u["name"]: u["suggested"] for u in unknown if u["suggested"]
                    },
                    "rule": _PROMPTS["api_check_rule"],
                },
            }
        result = self.execute_code(code, timeout)
        result["api_check"] = {
            "verified_peaks_apis": verified,
            "generic_refs": generic,
            "unknown_refs": [],
            "suggestions": {},
            "rule": _PROMPTS["api_check_rule"],
        }
        return result

    def _proven_ids(self, leaf: str, call_scope: str | None) -> list[str]:
        """Canonical ids in the proof ledger matching a leaf + call scope.

        The receiver's scope is only enforceable when the call site HAS one: a
        known DataArray/Dataset/accessor receiver accepts a proof in that very
        scope, so ``da.plot_bz(...)`` (a module-level API) stays blocked even
        after ``plot_bz`` was fetched, and the scope-mismatch hint stays
        meaningful.

        A bare call or an untyped receiver carries no scope of its own, and
        ``search`` returns exactly ONE row per name - the index's module-scope
        twin of a DataArray method (``module:peaks.core.process.k_conversion:
        k_convert``) is not discoverable through the model-facing tools.  The
        old rule demanded precisely that twin for those call sites, so the
        advice "search for the canonical id of this call's scope and get it
        again" pointed at an id the caller could never find: the real-model
        trials spent 11 blocked calls on it.  Untyped call sites therefore
        accept any proven id for the name.  What the gate still guarantees is
        unchanged - the name must resolve to a real Peaks API and must have
        been fetched with ``get`` in this session.
        """
        untyped = call_scope in {None, "bare", "unknown_receiver"}
        proven: list[str] = []
        for snapshot in self.state.verified_apis.values():
            if snapshot.get("name") != leaf:
                continue
            if not untyped and snapshot.get("scope") != call_scope:
                continue
            proven.append(str(snapshot["id"]))
        return sorted(set(proven))

    def add_cell(self, source: str = "", cell_type: str = "code") -> dict[str, Any]:
        """Append one cell at the END of the notebook (append-only log).

        There is intentionally no way to insert in the middle, edit or delete an
        existing cell: the notebook preserves the agent's full work history in
        top-to-bottom order, so nothing can be silently overwritten or removed.
        """
        if cell_type not in {"code", "markdown", "raw"}:
            raise ValueError("cell_type must be code, markdown, or raw")
        self._authorize("add_cell", source if cell_type == "code" else "")
        return self.state.bridge.request("add_cell", {"source": source, "cell_type": cell_type})

    def append_record_cell(self, source: str, cell_type: str = "markdown") -> dict[str, Any]:
        """Append an INTERNAL record cell at the END of the notebook.

        Internal plumbing (save intents/outcomes, archival notes) appends
        directly through the frontend bridge - deliberately NOT through any
        model-facing add-cell consent gate, so an operation that already has
        its own consent (e.g. the save card) never triggers a second
        confirmation when ``require_consent`` is on.  The append-only log
        contract is unchanged; the cell is just not a model-requested
        mutation.
        """
        if cell_type not in {"code", "markdown", "raw"}:
            raise ValueError("cell_type must be code, markdown, or raw")
        return self.state.bridge.request("add_cell", {"source": source, "cell_type": cell_type})

    def save_with_consent(
        self,
        variable_name: str,
        path: str,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """Persist ONE notebook variable through the staged approval flow.

        Order of operations (the save contract):

        1. precheck: the variable exists and is serialisable; the target path
           policy is decided up front (existing target without overwrite =
           blocked, nothing staged);
        2. an INTENT record cell (normalized preview: kind/shape/dtype/units/
           target) is appended through the internal record path - the human
           sees exactly what would be written BEFORE any staging happens, and
           no second consent dialog is triggered (require_consent applies to
           model mutations, not to internal record cells);
        3. stage: the variable is serialised into the unified staging area
           (strict TTL, server-owned gateway, lock + try/finally cleanup);
        4. consent: the save card asks the human; without an approval channel
           the operation fails closed (blocked: no_consent_channel) and the
           staging area is cleaned up - there is no unrecoverable pending
           state;
        5. publish: only an affirmative decision atomically writes the exact
           staged bytes;
        6. an OUTCOME record cell is appended and the SaveReceipt returned.

        One variable, one file, per call - there is no batch-save and no
        code-level approve anywhere in the flow.
        """
        from pathlib import Path

        from peaksMCP.overrides.save import (
            SaveReceipt,
            _save_result,
            _serialisation_kind,
            _variable_preview,
        )

        value = self.state.namespace.get(variable_name)
        if value is None:
            raise KeyError(
                f"save_with_consent: variable {variable_name!r} does not exist "
                "in the kernel namespace"
            )
        target = Path(path).expanduser()
        # Precheck: reject a (value, target) pair the serializer would refuse
        # BEFORE any record cell or staging happens.
        try:
            _kind = _serialisation_kind(value, target)
        except TypeError as exc:
            raise ValueError(f"save_with_consent: {exc}") from None
        _preview, line = _variable_preview(value, target, variable_name)

        def _fill_preview(receipt: SaveReceipt) -> None:
            """Populate preview metadata (kind/dims/dtype/units) on receipts
            that never staged (blocked before serialisation)."""
            if receipt.kind:
                return
            structure = _preview["structure"]
            receipt.kind = _preview["kind"]
            receipt.dims = dict(structure.get("sizes") or {})
            receipt.dtype = structure.get("dtype")
            receipt.units = structure.get("units")

        # 1) precheck: nothing is staged for a refused overwrite.
        if target.exists() and not overwrite:
            receipt = SaveReceipt(
                status="blocked",
                variable_name=variable_name,
                path=str(target),
                overwrite=False,
                note=f"{target} exists; pass overwrite=True after review.",
            )
            _fill_preview(receipt)
            self.append_record_cell(
                f"**save_with_consent** - blocked: {target} already exists; "
                f"nothing was written. (pass overwrite=True after review)"
            )
            return receipt.model_dump(mode="json")

        # 2) intent record cell BEFORE staging/consent (human sees the preview
        #    of exactly what would be serialised).
        try:
            self.append_record_cell(
                f"**save_with_consent** - request recorded before approval.\n\n"
                f"{line}\n\n"
                f"The file is written ONLY after your approval on the save card."
            )
        except Exception:
            self.audit.write(
                "save_with_consent", "error",
                {"error_type": "record_cell_failed",
                 "error": "could not append the intent record cell"},
            )

        # 3-5) stage -> consent -> publish (fail-closed when no channel).
        receipt = _save_result(
            value, target, overwrite=overwrite, variable_name=variable_name
        )

        if receipt.status in {"blocked", "failed"} and not receipt.kind:
            _fill_preview(receipt)

        # 6) outcome record cell (archival; never blocks the receipt).
        try:
            if receipt.status == "saved":
                note = f"approved and written: {receipt.path} ({receipt.size_bytes} bytes, sha256 {str(receipt.sha256)[:10]}...)"
            elif receipt.status == "denied":
                note = f"the user did not approve; nothing was written ({receipt.path})."
            elif receipt.status == "blocked":
                note = f"blocked: {receipt.note or receipt.path}"
            else:
                note = f"failed: {receipt.note or receipt.path}"
            self.append_record_cell(f"**save_with_consent** - outcome: {note}")
        except Exception:
            self.audit.write(
                "save_with_consent", "error",
                {"error_type": "record_cell_failed",
                 "error": "could not append the outcome record cell"},
            )
        return receipt.model_dump(mode="json")
