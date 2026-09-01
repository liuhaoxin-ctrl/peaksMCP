"""Source-first signature and documentation extraction for Peaks APIs."""

from __future__ import annotations

import ast
import importlib
import inspect
import os
from pathlib import Path
from typing import Any

MAX_DOC_CHARS = 6000

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
        obj = getattr(module, name)
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
    if not signature or scope not in {"dataarray", "dataset", "datatree"}:
        return signature
    inside = signature.partition("(")[2].rpartition(")")[0]
    parts = [part.strip() for part in inside.split(",") if part.strip()]
    if parts and parts[0].split("=", 1)[0].strip() in {"self", "data", "da", "ds", "dt"}:
        parts.pop(0)
    return f"{name}({', '.join(parts)})"


def describe_api(entry: dict[str, Any], package_dir: str | None = None) -> dict[str, Any]:
    """Return detailed, source-backed metadata for an indexed API.

    Parameters
    ----------
    entry : dict
        One canonical record from the dynamic API index.
    package_dir : str, optional
        Peaks package root; discovered from the installed package by default.

    Returns
    -------
    dict
        Merged signature, bound accessor signature, docstring and source location.

    Examples
    --------
    >>> details = describe_api(index.entries[0])
    >>> "signature_bound" in details
    True
    """
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
    return {
        **entry,
        **details,
        "signature": signature,
        "signature_bound": _bound(signature, name, str(entry.get("scope") or "")),
    }
