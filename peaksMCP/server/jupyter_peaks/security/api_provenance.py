"""AST extraction and variable-origin analysis for the API check.

These helpers turn a code snippet into (a) every call site (receiver root,
accessor segment and invoked leaf, ordered by source position) and (b) a small
flow-sensitive map of variable origins used to decide whether a receiver is a
generic module, an xarray object, a Peaks accessor, or unknown.
"""

from __future__ import annotations

import ast
from typing import NamedTuple

# Accessor names whose members are Peaks APIs callable as ``da.<accessor>.<method>``
# (e.g. ``da.metadata.set_EF_correction``).
ACCESSOR_NAMES = frozenset({"metadata", "quick_fit", "history", "dt", "tr", "ML", "xps"})

# Known peaks/xarray constructors and the receiver type they produce, used to
# propagate variable origins (``data = load(...)`` -> DataArray).
CONSTRUCTOR_TYPES = {
    "load": "dataarray",
    "DataArray": "dataarray",
    "Dataset": "dataset",
    "DataTree": "datatree",
    "concat": "dataarray",
    "open_dataset": "dataset",
    "open_dataarray": "dataarray",
    "from_dict": "datatree",
}


class CallTarget(NamedTuple):
    """One call site: the receiver root identifier (or None), the accessor
    segment (``da.metadata.xxx`` -> ``metadata``) when present, and the invoked
    leaf name, positioned by source location for a stable, crash-free order."""

    root_id: str | None
    accessor: str | None
    leaf: str
    lineno: int
    col_offset: int
    subscripted: bool = False


def extract_call_targets(code: str) -> list[CallTarget]:
    """Return every call site in ``code``, sorted by source position.

    ``np.linalg.svd(x)``              -> root="np",  leaf="svd"
    ``da.k_convert()``                -> root="da",  leaf="k_convert"
    ``da.metadata.set_EF_correction(x)`` -> root="da", accessor="metadata", leaf="set_EF_correction"
    ``fit_gold(data)``                -> root="fit_gold", leaf="fit_gold"
    ``da[i].mean()``                  -> root="da",  leaf="mean", subscripted=True
    ``make().correct_EF()``           -> root=None,  leaf="correct_EF"

    Sorting by ``(lineno, col_offset)`` avoids comparing ``None`` roots with
    string roots (a TypeError), and makes the reported order deterministic.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    targets: list[CallTarget] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        chain: list[str] = []
        while isinstance(func, ast.Attribute):
            chain.append(func.attr)
            func = func.value
        base = func
        leaf = chain[0] if chain else None
        root_id: str | None = None
        subscripted = False
        if isinstance(base, ast.Name):
            root_id = base.id
            if leaf is None:
                leaf = base.id  # bare call (``print(...)``, ``fit_gold(data)``)
        elif isinstance(base, ast.Subscript):
            # ``da[i].method()`` -> receiver inherits from ``da``.
            inner = base.value
            if isinstance(inner, ast.Name):
                root_id = inner.id
                subscripted = True
        if leaf is None:
            continue
        accessor = chain[1] if len(chain) >= 2 and chain[1] in ACCESSOR_NAMES else None
        targets.append(
            CallTarget(root_id, accessor, leaf, node.lineno, node.col_offset, subscripted)
        )
    targets.sort(key=lambda target: (target.lineno, target.col_offset))
    return targets


def _receiver_module(node: ast.AST, imports: dict[str, str], generic: set[str]) -> str | None:
    """Return the module name for an attribute chain rooted on a module name.

    ``plt.savefig`` with ``import matplotlib.pyplot as plt`` resolves to
    ``matplotlib`` (generic).  Returns ``None`` when the root is not a known
    generic module.
    """
    base = node
    while isinstance(base, ast.Attribute):
        base = base.value
    if isinstance(base, ast.Name):
        real = imports.get(base.id, base.id)
        if base.id in generic or real in generic:
            return real
    return None


def _assign_tag(
    value: ast.AST | None,
    tags: dict[str, str],
    imports: dict[str, str],
    generic: set[str],
) -> str:
    """Return the receiver-type tag for an assignment RHS."""
    if value is None:
        return "unknown"
    if isinstance(value, ast.Name):
        return tags.get(value.id, "unknown")
    if isinstance(value, ast.Subscript):
        base = value.value
        return tags.get(base.id, "unknown") if isinstance(base, ast.Name) else "unknown"
    if isinstance(value, ast.Call):
        func = value.func
        if isinstance(func, ast.Name):
            name = func.id
            if _receiver_module(func, imports, generic) or (
                imports.get(name) and imports[name] in generic
            ):
                return "generic"  # bare alias call into a generic module
            if name in CONSTRUCTOR_TYPES:
                return CONSTRUCTOR_TYPES[name]  # ``data = load(...)``
            return "unknown"
        if isinstance(func, ast.Attribute):
            if _receiver_module(func, imports, generic):
                return "generic"  # ``fig = plt.figure()``
            # peaks / xarray constructors: ``data = pks.load(...)``, ``x = xr.concat(...)``
            base = func
            while isinstance(base, ast.Attribute):
                base = base.value
            if isinstance(base, ast.Name):
                real = imports.get(base.id, base.id)
                if real in {"peaks", "pks"} or base.id in {"xr", "xarray", "pks", "peaks"}:
                    if func.attr in CONSTRUCTOR_TYPES:
                        return CONSTRUCTOR_TYPES[func.attr]
            return "unknown"
    if isinstance(value, ast.Attribute):
        if _receiver_module(value, imports, generic):
            return "generic"
        return "unknown"
    return "unknown"


def analyze_provenance(
    code: str, generic_roots: set[str]
) -> tuple[dict[str, str], set[str], set[str], set[str]]:
    """Return ``(tags, generic, defined, imported)`` for ``code``.

    ``tags`` maps variable names to a receiver-type tag by walking statements in
    order: import aliases resolve generic modules (``import numpy as n``),
    assignments propagate origins (``fig = plt.figure()`` -> generic,
    ``data = load(...)`` -> dataarray), and simple copies inherit their source.
    ``defined``/``imported`` record names bound by the code (bare user helpers)
    and names brought in by imports, so bare calls to them are treated as
    generic rather than unknown.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {}, set(generic_roots), set(), set()
    tags: dict[str, str] = {}
    imports: dict[str, str] = {}
    defined: set[str] = set()
    imported: set[str] = set()
    generic = set(generic_roots)
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name.split(".")[0]
                real = alias.name.split(".")[0]
                imports[name] = real
                defined.add(name)
                imported.add(name)
                if real in generic:
                    # Alias of a generic module (``import numpy as n``): the
                    # alias itself becomes a generic root too.
                    tags[name] = "module"
                    generic.add(name)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    continue
                name = alias.asname or alias.name
                defined.add(name)
                imported.add(name)
                parent = node.module.split(".")[0] if node.module else ""
                if parent in generic or name in generic:
                    tags[name] = "module"
                    generic.add(name)
        elif isinstance(node, ast.FunctionDef):
            defined.add(node.name)
        elif isinstance(node, ast.Assign):
            tag = _assign_tag(node.value, tags, imports, generic)
            for target in node.targets:
                if isinstance(target, ast.Name):
                    tags[target.id] = tag
                    defined.add(target.id)
        elif isinstance(node, ast.AnnAssign):
            tag = _assign_tag(node.value, tags, imports, generic)
            if isinstance(node.target, ast.Name):
                tags[node.target.id] = tag
                defined.add(node.target.id)
    return tags, generic, defined, imported