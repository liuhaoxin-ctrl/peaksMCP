"""Source-first signature and documentation extraction for Peaks APIs."""

from __future__ import annotations

import ast
import importlib
import inspect
import os
from pathlib import Path
from typing import Any

MAX_DOC_CHARS = 6000

# Scopes whose entries are xarray accessor methods (bound: receiver dropped).
_ACCESSOR_SCOPES = frozenset({"dataarray", "dataset", "datatree"})

# Parameter names that act as the implicit receiver of an accessor method or
# accessor-class method and are dropped from the bound signature.
_RECEIVER_NAMES = frozenset({"self", "cls", "data", "da", "ds", "dt"})

# Runtime introspection imports the module to read ``inspect`` metadata.  Only
# allow modules owned by this project or the installed peaks package; an index
# entry pointing elsewhere must not trigger an arbitrary import on the FastMCP
# background thread (import-lock risk and unexpected side effects).
_ALLOWED_MODULE_PREFIXES = ("peaks.", "peaksMCP.")


def _source_path(package_dir: str, module: str) -> Path | None:
    relative = module.split(".", 1)[1] if module.startswith("peaks.") else module
    candidate = Path(package_dir, *relative.split(".")).with_suffix(".py")
    if candidate.is_file():
        return candidate
    package_init = candidate.with_suffix("") / "__init__.py"
    return package_init if package_init.is_file() else None


def _find_node(tree: ast.Module, name: str) -> ast.AST | None:
    return next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name == name
        ),
        None,
    )


def extract_from_source(package_dir: str, module: str, name: str) -> dict[str, Any] | None:
    """Extract an API signature and docstring from its defining source file."""
    path = _source_path(package_dir, module)
    if path is None:
        return None
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
    except (OSError, SyntaxError):
        return None
    node = _find_node(tree, name)
    if node is None:
        return None
    doc = ast.get_docstring(node, clean=True) or ""
    if isinstance(node, ast.ClassDef):
        init = next(
            (child for child in node.body if isinstance(child, ast.FunctionDef) and child.name == "__init__"),
            None,
        )
        args = ast.unparse(init.args) if init else ""
        methods = [
            child.name
            for child in node.body
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and not child.name.startswith("_")
        ]
        return {"signature": f"{name}({args})", "docstring": doc[:MAX_DOC_CHARS], "methods": methods, "source_path": str(path)}
    return {
        "signature": f"{name}({ast.unparse(node.args)})",
        "docstring": doc[:MAX_DOC_CHARS],
        "source_path": str(path),
    }


def _inspect_runtime(entry: dict[str, Any]) -> dict[str, Any] | None:
    module_name = str(entry.get("module") or "")
    name = str(entry.get("func_name") or entry.get("name") or "")
    if not module_name or not name:
        return None
    if not module_name.startswith(_ALLOWED_MODULE_PREFIXES):
        return None
    try:
        module = importlib.import_module(module_name)
        # Accessor-class members (e.g. ``Metadata.set_EF_correction`` from
        # ``da.metadata``) live on the class, not at module level.
        cls_name = str(entry.get("accessor_class") or "")
        obj = getattr(getattr(module, cls_name), name) if cls_name else getattr(module, name)
        signature = str(inspect.signature(obj)) if callable(obj) else None
        doc = inspect.getdoc(obj) or ""
        return {
            "signature": f"{name}{signature}" if signature else None,
            "docstring": doc[:MAX_DOC_CHARS],
            "source_path": inspect.getsourcefile(obj),
        }
    except Exception:
        return None


def _bound(signature: str | None, name: str, scope: str) -> str | None:
    """Bind a signature to its owner: drop the leading ``self``/``data`` argument.

    The binding rule is driven by the signature itself, not by scope or field
    metadata, so it stays correct for every accessor class (``metadata``,
    ``quick_fit``, ``history``, ...):

    - a leading ``self`` is always a class method (bound);
    - a leading ``data``/``da``/``ds``/``dt`` is dropped only for xarray
      accessor scopes, because module-level functions (e.g. ``save(data, ...)``)
      have a real ``data`` argument that must be kept.
    """
    if not signature:
        return signature
    inside = signature.partition("(")[2].rpartition(")")[0]
    parts = [part.strip() for part in inside.split(",") if part.strip()]
    if not parts:
        return f"{name}()"
    first = parts[0].split("=", 1)[0].strip()
    if first == "self" or (
        first in _RECEIVER_NAMES and scope in _ACCESSOR_SCOPES
    ):
        parts.pop(0)
    return f"{name}({', '.join(parts)})"


def _check_contract_inputs(
    name: str,
    signature: inspect.Signature | None,
    inputs: Any,
) -> tuple[bool, list[str], list[str]]:
    """Compare a manifest v5 ``inputs`` list with the real signature.

    Returns ``(ok, issues, undocumented)``.  ``issues`` are contract defects
    that must block ``get``: a declared parameter that does not exist (typo or
    rename) or a required real parameter the contract forgot to declare.
    ``undocumented`` lists optional real parameters the contract omits —
    informational, since parameter names are what the model calls.
    """
    if not isinstance(inputs, list) or not inputs:
        return False, ["inputs are not a non-empty list"], []
    if signature is None:
        return False, ["signature could not be resolved"], []
    declared = [
        str(item.get("name")) for item in inputs if isinstance(item, dict) and item.get("name")
    ]
    parameters = signature.parameters.values()
    named = {
        parameter.name
        for parameter in parameters
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    variadic = {
        parameter.name
        for parameter in parameters
        if parameter.kind in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
    }
    required = {
        parameter.name
        for parameter in parameters
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    issues = [
        f"declared input {parameter_name!r} is not a parameter of {name}()"
        for parameter_name in declared
        if parameter_name not in named | variadic
    ]
    issues += [
        f"required parameter {parameter_name!r} is missing from the declared inputs"
        for parameter_name in sorted(required - set(declared))
    ]
    undocumented = sorted(named - set(declared))
    return not issues, issues, undocumented


def _describe_contract_api(entry: dict[str, Any]) -> dict[str, Any]:
    """Describe a manifest (v5) project API from its contract, not its source.

    The export (``peaksMCP.overrides.<name>``) is imported and the signature
    read via :func:`inspect.signature` — the declared surface is runtime-
    verified against the real object on every detail call.  Failure to import
    or resolve the export is reported as ``signature_resolved: false`` (the
    get tool turns that into an error: manifest/implementation drift).

    The returned dict is a whitelist: canonical identity, tier/module context,
    the verified signature and the structured manifest contract.  Never source
    paths, implementation modules, aliases or the raw index entry.
    """
    name = str(entry.get("name") or "")
    export = str(entry.get("export") or f"{entry.get('module')}.{name}")
    signature: str | None = None
    signature_object: inspect.Signature | None = None
    resolved = False
    module_name, _, attr = export.rpartition(".")
    if attr and module_name.startswith(_ALLOWED_MODULE_PREFIXES):
        try:
            module = importlib.import_module(module_name)
            obj = getattr(module, attr)
            if callable(obj):
                signature_object = inspect.signature(obj)
                signature = f"{name}{signature_object}"
                resolved = True
        except Exception:
            resolved = False
    inputs_ok, input_issues, undocumented = _check_contract_inputs(
        name, signature_object, (entry.get("contract") or {}).get("inputs")
    )
    scope = str(entry.get("scope") or "module")
    return {
        "id": str(entry.get("id") or ""),
        "name": name,
        "module": entry.get("module"),
        "scope": scope,
        "tier": entry.get("tier"),
        "exposure": entry.get("exposure"),
        "kind": entry.get("kind"),
        "export": export,
        "signature": signature,
        "signature_resolved": resolved,
        "signature_bound": _bound(signature, name, scope) if signature else None,
        # v5: the declared inputs are checked against the real signature, so a
        # typo'd or renamed parameter can never reach the model as a contract.
        "contract_inputs_ok": inputs_ok,
        "contract_input_issues": input_issues,
        "contract_undocumented_params": undocumented,
        "contract": dict(entry.get("contract") or {}),
        "project_added": True,
    }


def describe_api(entry: dict[str, Any], package_dir: str | None = None) -> dict[str, Any]:
    """Return detailed, source-backed metadata for an indexed API.

    Project (manifest v5) entries take the contract path: signature verified
    by importing the declared ``export`` plus the structured contract.  Native
    entries keep the source-first extraction with runtime fallback.

    Parameters
    ----------
    entry : dict
        One canonical record from the dynamic API index.
    package_dir : str, optional
        Peaks package root; discovered from the installed package by default.

    Returns
    -------
    dict
        Whitelisted detail: identity, tier, signature, docstring/contract.
        Never source paths or aliases.

    Examples
    --------
    >>> details = describe_api(index.entries[0])
    >>> "signature_bound" in details
    True
    """
    if entry.get("project_added") or entry.get("contract") is not None:
        return _describe_contract_api(entry)
    if package_dir is None:
        import peaks

        package_dir = os.path.dirname(peaks.__file__)
    name = str(entry.get("name") or "")
    func_name = str(entry.get("func_name") or name)
    details = extract_from_source(package_dir, str(entry.get("module") or ""), func_name)
    if details is None:
        details = _inspect_runtime(entry) or {}
    note = str(entry.get("docstring_note") or "").strip()
    if note:
        doc = details.get("docstring") or ""
        details["docstring"] = f"{note}\n\n{doc}".strip()
    signature = details.get("signature") or entry.get("signature")
    # Model-facing detail is a whitelist: the caller gets the canonical id,
    # tier/module context, the signature and a bounded docstring.  Never
    # source paths, aliases, legacy ids or the full index entry - get is the
    # detail step AFTER search returned the canonical id.
    docstring = str(details.get("docstring") or "").strip()
    return {
        "id": str(entry.get("id") or ""),
        "name": str(entry.get("name") or ""),
        "module": entry.get("module"),
        "scope": entry.get("scope"),
        "tier": entry.get("tier"),
        "exposure": entry.get("exposure"),
        "kind": entry.get("kind"),
        "signature": signature,
        "signature_bound": _bound(signature, name, str(entry.get("scope") or "")),
        "docstring": docstring[:_DESCRIBE_DOCSTRING_MAX],
        "project_added": bool(entry.get("project_added")),
    }


#: Detail docstrings are trimmed to a review-friendly size (full text stays
#: available in the notebook/source).
_DESCRIBE_DOCSTRING_MAX = 3000
