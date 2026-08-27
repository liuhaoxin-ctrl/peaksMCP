#!/usr/bin/env python3
"""Audit public peaksMCP callables for minimum NumPy-style documentation."""

from __future__ import annotations

import ast
import json
from pathlib import Path

SECTIONS = {"Parameters", "Returns", "Examples"}


def exported_names(root: Path) -> set[str]:
    """Collect symbols explicitly exported by package ``__all__`` declarations."""
    names: set[str] = set()
    for path in root.rglob("__init__.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if not any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets):
                continue
            try:
                names.update(ast.literal_eval(node.value))
            except (ValueError, TypeError):
                pass
    return names


def audit(root: Path) -> list[dict[str, object]]:
    """Return documentation findings for public functions and classes."""
    findings: list[dict[str, object]] = []
    exports = exported_names(root)
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts or "node_modules" in path.parts or path.name == "__init__.py":
            continue
        relative = path.relative_to(root)
        public_module = relative.parts[0] == "workflows" or "backend" in relative.parts
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) or node.name.startswith("_"):
                continue
            if node.name not in exports and not public_module:
                continue
            doc = ast.get_docstring(node, clean=True) or ""
            missing = []
            if not doc.splitlines():
                missing.append("summary")
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                positional = [arg.arg for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs) if arg.arg not in {"self", "cls"}]
                if positional and "Parameters\n----------" not in doc:
                    missing.append("Parameters")
                if node.returns is not None and "None" not in ast.unparse(node.returns) and "Returns\n-------" not in doc:
                    missing.append("Returns")
                if "Examples\n--------" not in doc:
                    missing.append("Examples")
            if missing:
                findings.append({"file": str(path), "line": node.lineno, "name": node.name, "missing": missing})
    return findings


def main() -> int:
    """Print findings as JSON and return a CI-friendly exit status."""
    root = Path(__file__).resolve().parents[1] / "peaksMCP"
    findings = audit(root)
    print(json.dumps({"ok": not findings, "findings": findings}, indent=2))
    return int(bool(findings))


if __name__ == "__main__":
    raise SystemExit(main())
