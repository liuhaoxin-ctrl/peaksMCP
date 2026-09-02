"""Notebook mutation operations mediated by scanning and frontend consent."""

from __future__ import annotations

import ast
from typing import Any, NamedTuple

from ..security import AuditLogger, ConsentManager, scan_code
from .base import ExecutionMode, SharedState

# Accessor names whose members are Peaks APIs callable as ``da.<accessor>.<method>``
# (e.g. ``da.metadata.set_EF_correction``).
_ACCESSOR_NAMES = frozenset({"metadata", "quick_fit", "history", "dt", "tr", "ML", "xps"})

# Known peaks/xarray constructors and the receiver type they produce, used to
# propagate variable origins (``data = load(...)`` -> DataArray).
_CONSTRUCTOR_TYPES = {
    "load": "dataarray",
    "DataArray": "dataarray",
    "Dataset": "dataset",
    "DataTree": "datatree",
    "concat": "dataarray",
    "open_dataset": "dataset",
    "open_dataarray": "dataarray",
    "from_dict": "datatree",
}


class _CallTarget(NamedTuple):
    """One call site: the receiver root identifier (or None), the accessor
    segment (``da.metadata.xxx`` -> ``metadata``) when present, and the invoked
    leaf name, positioned by source location for a stable, crash-free order."""

    root_id: str | None
    accessor: str | None
    leaf: str
    lineno: int
    col_offset: int
    subscripted: bool = False


def _extract_call_targets(code: str) -> list[_CallTarget]:
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
    targets: list[_CallTarget] = []
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
        accessor = chain[1] if len(chain) >= 2 and chain[1] in _ACCESSOR_NAMES else None
        targets.append(
            _CallTarget(root_id, accessor, leaf, node.lineno, node.col_offset, subscripted)
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
            if name in _CONSTRUCTOR_TYPES:
                return _CONSTRUCTOR_TYPES[name]  # ``data = load(...)``
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
                    if func.attr in _CONSTRUCTOR_TYPES:
                        return _CONSTRUCTOR_TYPES[func.attr]
            return "unknown"
    if isinstance(value, ast.Attribute):
        if _receiver_module(value, imports, generic):
            return "generic"
        return "unknown"
    return "unknown"


def _analyze_provenance(
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


class UnsafeNotebookBackend:
    """Execute or mutate notebook cells after security checks and consent."""

    _PYTHON_EXECUTION_OPERATIONS = {
        "notebook_write_with_api_check",
    }

    # Python builtins that appear as bare calls (``print``, ``len``, ``range``,
    # ``str``, ...).  They are never Peaks APIs and never block execution.
    # ``exec``/``eval``/``compile``/``getattr`` are deliberately absent: the
    # security scanner hard-blocks them, and keeping them unverifiable adds a
    # second rejection layer.
    _BUILTIN_NAMES = frozenset({
        "abs", "all", "any", "ascii", "bin", "bool", "breakpoint", "bytearray",
        "bytes", "callable", "chr", "classmethod", "complex", "delattr", "dict",
        "dir", "divmod", "enumerate", "filter", "float", "format", "frozenset",
        "hash", "hasattr", "help", "hex", "id", "input", "int", "isinstance",
        "issubclass", "iter", "len", "list", "locals", "map", "max", "memoryview",
        "min", "next", "object", "oct", "ord", "pow", "print", "property", "range",
        "repr", "reversed", "round", "set", "setattr", "slice", "sorted",
        "staticmethod", "str", "sum", "super", "tuple", "type", "vars", "zip",
    })
    # Ordinary methods on data receivers (xarray / pandas / numpy / matplotlib /
    # stdlib).  These are NOT Peaks APIs and never block execution; anything else
    # used as ``obj.<name>`` that is neither a Peaks API nor in this set is an
    # unverifiable API reference and hard-blocks execution.
    _GENERIC_METHODS = frozenset({
        "sel", "isel", "mean", "median", "max", "min", "sum", "std", "var",
        "count", "cumsum", "diff", "argmax", "argmin", "clip", "abs", "round",
        "plot", "plot_fit", "plot_residuals", "assign_coords", "drop_vars",
        "rename", "stack", "unstack", "transpose", "squeeze", "expand_dims",
        "swap_dims", "values", "data", "item", "to_dataset", "to_netcdf",
        "compute", "load", "where", "fillna", "interp", "groupby", "resample",
        "shift", "rolling", "coarsen", "quantile", "copy", "dims", "sizes",
        "coords", "attrs", "metadata", "history", "pint", "real", "imag",
        "astype", "reshape", "flatten", "tolist", "split",
        "join", "replace", "strip", "lower", "upper", "format", "append",
        "extend", "pop", "keys", "items", "get", "setdefault", "update",
        "add", "remove", "discard", "union", "difference", "intersection",
        "dumps", "loads", "dump", "reader", "writer", "DictReader",
        "DictWriter", "read_csv", "read_excel", "read_json", "to_csv",
        "to_excel", "to_json", "to_numpy", "DataFrame", "Series",
        "value_counts", "dropna", "describe", "head", "tail", "iloc", "loc",
        "set_index", "reset_index", "sort_values", "sort_index", "merge",
        "concat", "pivot", "melt", "iterrows", "apply", "map", "unique",
        "isin", "sample", "drop", "insert", "sort", "reverse", "index",
        "read", "write", "close", "open", "readline", "readlines",
        "writelines", "seek", "tell", "flush", "truncate", "read_text",
        "write_text", "read_bytes", "write_bytes", "exists", "mkdir",
        "unlink", "glob", "rglob", "iterdir",
        "is_file", "is_dir", "resolve", "joinpath", "with_suffix", "stem",
        "suffix", "parent", "name", "listdir", "makedirs", "walk", "getcwd",
        "chdir", "getenv", "path", "splitext", "dirname", "basename",
        "startswith", "endswith", "find", "zfill", "encode",
        "decode", "isdigit", "isalpha", "isalnum", "isnumeric", "isspace",
        "splitlines", "rstrip", "lstrip", "capitalize", "title", "partition",
        "rpartition", "center", "expandtabs", "swapcase", "casefold",
        "popitem", "fromkeys", "clear", "symmetric_difference", "issubset",
        "issuperset", "any", "all", "next", "iter", "sorted",
        "reversed", "enumerate", "zip", "filter", "hash", "id", "repr",
        "array", "arange", "linspace", "zeros", "ones", "full", "eye",
        "zeros_like", "ones_like", "empty", "empty_like", "full_like",
        "meshgrid", "concatenate", "vstack", "hstack", "ravel", "dot",
        "matmul", "prod", "sqrt", "exp", "log", "log10", "log2", "sin",
        "cos", "tan", "asin", "acos", "atan", "atan2", "gradient",
        "poly1d", "polyfit", "polyval", "percentile", "nanpercentile",
        "nanmean", "nanmax", "nanmin", "histogram", "bincount", "loadtxt",
        "genfromtxt", "savetxt", "save", "savez", "fromfile", "fromstring",
        "asarray", "asanyarray", "seed", "random", "normal", "uniform",
        "randint", "rand", "randn", "choice", "flip",
        "roll", "argsort", "take", "repeat", "tile",
        "pad", "corrcoef", "cov", "apply_along_axis", "vectorize",
        "isnan", "isinf", "isfinite", "nonzero", "mod", "floor",
        "ceil", "around", "sign", "power", "square", "maximum", "minimum",
        "amax", "amin", "ptp", "trapz", "convolve", "correlate", "fft",
        "ifft", "rfft", "irfft", "conjugate", "angle", "hypot", "diag",
        "tril", "triu", "cumprod", "figure", "subplots", "scatter", "imshow",
        "pcolormesh", "bar", "hist", "errorbar", "fill_between", "colorbar",
        "xlabel", "ylabel", "legend", "tight_layout", "show", "savefig",
        "subplots_adjust", "set_title", "set_xlabel", "set_ylabel",
        "axhline", "axvline", "set_ylim", "set_xlim", "xticks", "yticks",
        "grid", "axis", "contour", "contourf", "text", "annotate", "rcParams",
        "set_visible", "get_figure", "convert_to", "twinx",
        "twiny", "loglog", "semilogx", "semilogy", "suptitle", "clf", "cla",
        "gcf", "gca", "xlim", "ylim", "set_xscale", "set_yscale", "cm",
        "search", "match", "findall", "finditer", "sub", "subn", "compile",
        "fullmatch", "escape", "pi", "e", "tau", "inf", "nan", "isclose",
        "fabs", "fmod", "gcd", "factorial", "degrees", "radians", "strftime",
        "strptime", "isoformat", "now", "today", "utcnow", "timestamp",
        "fromtimestamp", "combine", "timedelta", "date", "time", "datetime",
    })
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
        target: _CallTarget,
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
        cell, executes it, and makes the ``peaks_search_api`` / ``peaks_get_api``
        step mandatory.  Every call site is classified by receiver origin and
        leaf name:

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

        targets = _extract_call_targets(code)
        tags, generic_roots, defined, imported = _analyze_provenance(
            code, set(self._GENERIC_MODULES)
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
                elif target.leaf in self._GENERIC_METHODS or target.leaf in self._BUILTIN_NAMES:
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
                    target.leaf in self._BUILTIN_NAMES
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
            elif target.leaf in self._GENERIC_METHODS or target.leaf in self._BUILTIN_NAMES:
                generic.append(target.leaf)
            else:
                unknown.append({"name": target.leaf, "suggested": [str(m.get("id")) for m in matches]})

        # Task-level select-then-run gate: the first execution of a task must be
        # preceded by at least TWO peaks_search_api / peaks_get_api calls, so
        # the agent genuinely explores before writing/running code.
        if self.state.exploration_count < 2:
            self.audit.write(
                "notebook_write_with_api_check",
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
