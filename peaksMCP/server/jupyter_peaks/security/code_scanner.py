"""Alias-aware, AST-semantic scanner for dangerous notebook code.

The scanner matches canonical AST attribute chains, including common import,
assignment and constructor aliases. It is a best-effort guard, not a Python
sandbox: arbitrary runtime reflection and existing kernel objects cannot be
fully described by source-only analysis.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from peaksMCP.config import prompts as _load_prompts

from .ipython_scanner import scan_ipython


class RiskLevel(StrEnum):
    """Severity attached to a security finding."""

    HIGH = "high"
    CRITICAL = "critical"


#: Curated issue-description templates (config/prompts.yaml). Wording lives in
#: YAML; rule ids and risk levels stay here. Loaded once at import.
_SCANNER_TEXT = _load_prompts().get("scanner") or {}


def _desc(key: str, **values: object) -> str:
    """Return one formatted scanner description from the curated prompts YAML.

    Parameters
    ----------
    key : str
        Template key under ``prompts.scanner`` in ``config/prompts.yaml``.
    values
        Placeholders to format into the template.

    Returns
    -------
    str
        Rendered description text.

    Raises
    ------
    KeyError
        If the template key is missing — a loud failure beats shipping an
        empty or stale description to the model or the consent dialog.
    """
    template = _SCANNER_TEXT.get(key)
    if template is None:
        raise KeyError(f"scanner prompt template {key!r} missing from config/prompts.yaml")
    return template.format(**values) if values else template


@dataclass(slots=True)
class SecurityIssue:
    """One blocked code pattern."""

    rule_id: str
    description: str
    level: RiskLevel
    line: int = 0
    code: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"rule_id": self.rule_id, "description": self.description, "level": self.level.value, "line": self.line, "code": self.code[:200]}


@dataclass(slots=True)
class ScanResult:
    """Combined IPython and Python AST scan result."""

    blocked: bool
    issues: list[SecurityIssue] = field(default_factory=list)
    block_reason: str | None = None
    syntax_error: dict[str, Any] | None = None
    # Patterns that are not hard-blocked but require an explicit, informed user
    # consent when consent is enabled (``state.require_consent``), e.g.
    # ``plt.savefig`` — figures are shown inline by default and must not be
    # written unless the user asks.
    requires_explicit_consent: list[SecurityIssue] = field(default_factory=list)

    @property
    def is_safe(self) -> bool:
        return not self.blocked

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_safe": self.is_safe, "blocked": self.blocked, "block_reason": self.block_reason,
            "syntax_error": self.syntax_error, "issues": [issue.to_dict() for issue in self.issues],
            "requires_explicit_consent": [issue.to_dict() for issue in self.requires_explicit_consent],
        }


class _Aliases(ast.NodeVisitor):
    """Record alias bindings at each node, in source order.

    A later reassignment must not erase the meaning of an earlier call. Function
    and class bodies are inspected in separate scopes rather than changing the
    surrounding bindings.
    """

    def __init__(self, names: dict[str, str] | None = None) -> None:
        self.names = dict(names or {})
        self._snapshots: dict[int, dict[str, str]] = {}
        # Roots imported anywhere in the snippet (not scoped: module objects are
        # process-global, so assigning an attribute onto them is smuggling no
        # matter where the import happened).
        self.imported_roots: set[str] = set()

    def visit(self, node: ast.AST) -> None:
        self._snapshots[id(node)] = self.names.copy()
        super().visit(node)

    def at(self, node: ast.AST) -> _Aliases:
        return _Aliases(self._snapshots.get(id(node), {}))

    def visit_Import(self, node: ast.Import) -> None:
        for item in node.names:
            # ``import os.path`` binds ``os``, not a local named ``os.path``.
            self.names[item.asname or item.name.split(".")[0]] = (
                item.name if item.asname else item.name.split(".")[0]
            )
            self.imported_roots.add(item.asname or item.name.split(".")[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        for item in node.names:
            self.names[item.asname or item.name] = f"{module}.{item.name}"
            if module and not module.startswith("."):
                self.imported_roots.add(module.split(".")[0])

    def resolve(self, name: str) -> str:
        return self.names.get(name, name)

    def _reference(self, value: ast.AST) -> str | None:
        if isinstance(value, (ast.Name, ast.Attribute)):
            return ".".join(_chain(value, self))
        if isinstance(value, ast.Call):
            name = _call_name(value, self)
            if name in _PATH_CONSTRUCTORS | _NETWORK_CLIENTS:
                return name
            if name in {"getattr", "builtins.getattr"} and len(value.args) >= 2:
                if isinstance(value.args[1], ast.Constant):
                    return ".".join(_chain(value, self))
            # Path-valued methods preserve the receiver's origin.
            if isinstance(value.func, ast.Attribute) and value.func.attr in {
                "absolute", "cwd", "home", "expanduser", "resolve", "joinpath", "with_name",
                "with_stem", "with_suffix",
            }:
                receiver = self._reference(value.func.value)
                if receiver and _has_path_origin(receiver.split(".")):
                    return receiver
        if isinstance(value, ast.BinOp) and isinstance(value.op, ast.Div):
            receiver = self._reference(value.left)
            if receiver and _has_path_origin(receiver.split(".")):
                return receiver
        return None

    def _bind(self, target: ast.AST, value: ast.AST, source: _Aliases | None = None) -> None:
        # Resolve all RHS expressions before updating targets, including swaps
        # such as ``f, g = print, f`` and chained assignments.
        source = source or _Aliases(self.names)
        if isinstance(target, ast.Name):
            reference = source._reference(value)
            if reference is None:
                self.names.pop(target.id, None)
            else:
                self.names[target.id] = reference
        elif isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, (ast.Tuple, ast.List)):
            if len(target.elts) == len(value.elts):
                for child, item in zip(target.elts, value.elts, strict=True):
                    self._bind(child, item, source)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.generic_visit(node)
        source = _Aliases(self.names)
        for target in node.targets:
            self._bind(target, node.value, source)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.generic_visit(node)
        if node.value is not None:
            self._bind(node.target, node.value)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.generic_visit(node)
        self._bind(node.target, node.value)

    def visit_With(self, node: ast.With | ast.AsyncWith) -> None:
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars is not None:
                self._bind(item.optional_vars, item.context_expr)
        for statement in node.body:
            self.visit(statement)

    visit_AsyncWith = visit_With

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda) -> None:
        arguments = node.args
        parameters = [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]
        parameters.extend(arg for arg in (arguments.vararg, arguments.kwarg) if arg)
        # Defaults, annotations and decorators execute in the surrounding scope.
        outer_expressions = [*arguments.defaults, *arguments.kw_defaults]
        outer_expressions.extend(arg.annotation for arg in parameters)
        if not isinstance(node, ast.Lambda):
            outer_expressions.extend([*node.decorator_list, node.returns])
        for expression in outer_expressions:
            if expression is not None:
                self.visit(expression)
        outer = self.names.copy()
        for parameter in parameters:
            self.names.pop(parameter.arg, None)
        positional = [*arguments.posonlyargs, *arguments.args]
        defaults = zip(positional[len(positional) - len(arguments.defaults):], arguments.defaults, strict=True)
        keyword_defaults = zip(arguments.kwonlyargs, arguments.kw_defaults, strict=True)
        for parameter, default in [*defaults, *keyword_defaults]:
            if default is not None:
                reference = _Aliases(outer)._reference(default)
                if reference is not None:
                    self.names[parameter.arg] = reference
        body = [node.body] if isinstance(node, ast.Lambda) else node.body
        for statement in body:
            self.visit(statement)
        self.names = outer
        if not isinstance(node, ast.Lambda):
            self.names.pop(node.name, None)

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function
    visit_Lambda = _visit_function

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for expression in [*node.bases, *node.keywords, *node.decorator_list]:
            self.visit(expression)
        outer = self.names.copy()
        for statement in node.body:
            self.visit(statement)
        self.names = outer
        self.names.pop(node.name, None)


# --------------------------------------------------------------------------- #
# Canonical attribute-chain extraction                                         #
# --------------------------------------------------------------------------- #
def _func_text(call: ast.Call, aliases: _Aliases) -> str:
    """Best-effort canonical name of a callable expression inside a Call node."""
    func = call.func
    if isinstance(func, ast.Name):
        return aliases.resolve(func.id)
    if isinstance(func, ast.Attribute):
        chain = _chain(func, aliases)
        return ".".join(chain) if chain else "(call)"
    if isinstance(func, ast.Subscript):
        return "(subscript)"
    return "(call)"


def _chain(node: ast.AST, aliases: _Aliases) -> list[str]:
    """Canonical attribute chain of an expression with aliases resolved.

    ``os.environ.update``            -> ``["os", "environ", "update"]``
    ``Path('x').unlink`` (func part) -> ``["Path", "unlink"]``
    ``o.environ['A']`` (subscript)   -> ``["os", "environ"]``
    ``getattr(builtins, 'exec')``   -> ``["builtins", "exec"]``
    """
    if isinstance(node, ast.Attribute):
        return [*_chain(node.value, aliases), node.attr]
    if isinstance(node, ast.Name):
        # Normalize ``from os import environ`` to the same chain as os.environ.
        return aliases.resolve(node.id).split(".")
    if isinstance(node, ast.Call):
        name = _func_text(node, aliases)
        if name in {"getattr", "builtins.getattr"} and len(node.args) >= 2:
            attribute = node.args[1]
            if isinstance(attribute, ast.Constant) and isinstance(attribute.value, str):
                return [*_chain(node.args[0], aliases), attribute.value]
        return name.split(".")
    if isinstance(node, ast.Subscript):
        return _chain(node.value, aliases)
    if isinstance(node, ast.NamedExpr):
        return _chain(node.value, aliases)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return _chain(node.left, aliases)
    if isinstance(node, ast.Constant):
        return [repr(node.value)]
    return [f"({type(node).__name__})"]


def _call_name(node: ast.Call, aliases: _Aliases) -> str:
    """Canonical dotted name of the called function, aliases resolved."""
    return ".".join(_chain(node.func, aliases))


# --------------------------------------------------------------------------- #
# Capability domains                                                          #
# --------------------------------------------------------------------------- #
# An ARPES analysis notebook uses a small, well-defined capability surface.
# Classification is per MODULE capability, not per method: any call routed
# through a module outside the sandbox is flagged uniformly, so star imports,
# ctypes, reflection and deserialization cannot hide behind an unlisted method
# name.  Mixed modules (``os``, ``pathlib``, ``builtins``) stay method-level
# below, because they legitimately carry read-only members too.

# Pure-compute modules: fully allowed (also the only safe star-import source).
_COMPUTE_MODULES = frozenset({
    "xarray", "numpy", "matplotlib", "pandas", "scipy", "peaks", "peaksMCP",
    "math", "datetime", "re", "copy", "itertools", "functools", "collections",
    "dataclasses", "enum", "numbers", "statistics", "decimal", "fractions",
    "random", "typing", "contextlib", "abc", "string", "traceback", "operator",
    "warnings", "calendar", "hashlib", "uuid",
})

# Network egress: any access requires explicit consent (never a hard block).
_NETWORK_MODULES = frozenset({
    "requests", "httpx", "urllib", "urllib3", "socket", "aiohttp", "http",
    "websockets",
})

# Process / native-code / dynamic-import / deserialization modules: hard block
# on any access — the code-execution sinks of an analysis sandbox.
_BLOCK_MODULES = frozenset({"subprocess", "ctypes", "importlib", "pickle", "joblib"})

# Reflection bases: fetching anything from them is a hard block (``globals()``,
# ``locals()``, ``vars()``, ``__builtins__`` can reach any callable).
_REFLECTION_BASES = frozenset({"globals", "locals", "vars", "__builtins__"})

# Dunder/reflection attributes: reading them is how a source-only scanner is
# defeated (``os.__dict__['system']``, ``''.__class__.__mro__[-1]``).  Any
# callable whose chain crosses one of these cannot be tracked statically.
_REFLECTION_ATTRS = frozenset({
    "__dict__", "__globals__", "__builtins__", "__class__", "__mro__",
    "__bases__", "__subclasses__", "__getattribute__", "__getattr__",
    "__setattr__", "__delattr__", "__init_subclass__",
})


# --------------------------------------------------------------------------- #
# Method-level rules for mixed modules                                         #
# --------------------------------------------------------------------------- #
_CRITICAL_CALLS = {"exec", "eval", "compile", "builtins.exec", "builtins.eval", "builtins.compile"}
_SYSTEM_CALLS = {
    "os.system", "os.popen", "os.remove", "os.unlink", "os.rmdir", "os.removedirs",
    "os.replace", "os.rename", "os.fork", "os.kill", "os.putenv", "os.unsetenv",
    "os.execv", "os.execve", "os.execvp", "os.execvpe", "os.spawnl", "os.spawnle",
    "os.spawnlp", "os.spawnlpe", "os.spawnv", "os.spawnve", "os.spawnvp", "os.spawnvpe",
    "os.posix_spawn", "os.chmod", "os.chown", "os.truncate", "os.ftruncate",
    "os.symlink", "os.mkfifo", "os.mknod", "shutil.rmtree", "shutil.rmdir",
    "shutil.move",
}
# Raw-descriptor file writers: the ``open('w')`` equivalent that does not route
# through ``open`` (``os.open`` + ``os.write``/``os.fdopen``) and would
# otherwise bypass FILE001.
_RAW_FILE_WRITERS = {"os.open", "os.write", "os.fdopen"}
_DYNAMIC_IMPORTS = {"__import__", "builtins.__import__"}
_PATH_METHODS = {"unlink", "write_text", "write_bytes", "rmdir", "rename", "replace", "symlink_to", "hardlink_to"}
_PATH_CONSTRUCTORS = {
    "Path", "PosixPath", "WindowsPath", "pathlib.Path", "pathlib.PosixPath", "pathlib.WindowsPath",
}
_ENV_MUTATORS = {"update", "setdefault", "pop", "popitem", "clear", "__setitem__", "__delitem__"}
_DYN_BASES = {"globals", "locals", "vars", "__builtins__", "builtins"}
_INDIRECT_TARGETS = {"exec", "eval", "compile", "__import__"}
# Pickle-family deserialization hidden inside an otherwise-allowed module:
# ``pd.read_pickle`` is a code-execution sink just like ``pickle.loads``.
_DESERIALIZE_SINKS = {"pd.read_pickle", "pandas.read_pickle"}

# Dangerous callables fetched indirectly (getattr / globals()['x'] / .get()).
_SYSTEM_METHOD_NAMES = {name.rsplit(".", 1)[-1] for name in _SYSTEM_CALLS} | {"system", "popen", "run"}
_FILE_WRITERS = {
    "np.save", "numpy.save", "np.savetxt", "numpy.savetxt", "np.savez", "numpy.savez",
    "np.savez_compressed", "numpy.savez_compressed", "plt.imsave", "matplotlib.pyplot.imsave",
    "pd.to_csv", "pandas.DataFrame.to_csv", "xr.to_netcdf", "xarray.DataArray.to_netcdf",
    "xarray.Dataset.to_netcdf", "json.dump",
}
_FILE_WRITE_METHOD_NAMES = {"save", "savetxt", "savez", "savez_compressed", "imsave", "to_csv", "to_netcdf", "dump", "to_pickle"}
# Constructors that propagate a network origin through aliases
# (``s = requests.Session(); client = s; client.get(...)``).
_NETWORK_CLIENTS = {"requests.Session", "requests.sessions.Session", "httpx.Client", "httpx.AsyncClient"}


def _has_path_origin(chain: list[str]) -> bool:
    return any(part in {"Path", "PosixPath", "WindowsPath"} for part in chain)


def _is_path_method_call(node: ast.Call, aliases: _Aliases) -> bool:
    """``Path(...).unlink()`` / ``pathlib.Path('x').write_text(...)`` and friends."""
    chain = _chain(node.func, aliases)
    if not chain or chain[-1] not in _PATH_METHODS:
        return False
    return _has_path_origin(chain)


def _is_env_mutation_call(node: ast.Call, aliases: _Aliases) -> bool:
    """``os.environ.update/pop/clear/...`` regardless of aliasing."""
    chain = _chain(node.func, aliases)
    if len(chain) < 3 or chain[:2] != ["os", "environ"]:
        return False
    return chain[-1] in _ENV_MUTATORS


def _is_env_assignment(target: ast.AST, aliases: _Aliases) -> bool:
    """``os.environ[...] = ...`` / ``o.environ['A'] += ...`` / ``del os.environ['A']``."""
    # Rebinding/deleting a local alias (``env = {}``) does not mutate environ.
    return isinstance(target, (ast.Attribute, ast.Subscript)) and _chain(target, aliases)[:2] == ["os", "environ"]


def _open_modes(node: ast.Call, mode_position: int = 1) -> str | None:
    """Classify an ``open`` mode: ``"w"`` (modifying), ``"r"`` (read-only) or
    ``None`` when the mode is not statically resolvable (e.g. a variable) and
    read-only-ness cannot be confirmed."""
    flags = "wax+"

    def classify(value: Any) -> str | None:
        text = str(value)
        if any(flag in text for flag in flags):
            return "w"
        return "r"

    if len(node.args) > mode_position:
        arg = node.args[mode_position]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return classify(arg.value)
        return None  # variable / expression mode: cannot confirm read-only
    for keyword in node.keywords:
        if keyword.arg == "mode":
            if isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
                return classify(keyword.value.value)
            return None  # variable / expression mode
    return None


def _indirect_call_target(node: ast.Call, aliases: _Aliases) -> str | None:
    """Name of a callable fetched indirectly: ``getattr(obj, 'x')()``,
    ``globals()['x']()``, ``globals().get('x')()``, ``__builtins__['x']()``."""
    func = node.func
    if isinstance(func, ast.Call) and _call_name(func, aliases) in {"getattr", "builtins.getattr"}:
        if len(func.args) >= 2 and isinstance(func.args[1], ast.Constant):
            return str(func.args[1].value)
    if isinstance(func, ast.Subscript):
        base = ".".join(_chain(func.value, aliases))
        if base in _DYN_BASES and isinstance(func.slice, ast.Constant):
            return str(func.slice.value)
    if isinstance(func, ast.Attribute) and func.attr == "get" and isinstance(func.value, ast.Call):
        base = ".".join(_chain(func.value, aliases))
        if base in _DYN_BASES and len(node.args) >= 1 and isinstance(node.args[0], ast.Constant):
            return str(node.args[0].value)
    return None


def _is_savefig(node: ast.Call, aliases: _Aliases) -> bool:
    """Detect figure saving: ``plt.savefig`` / ``fig.savefig`` / ``figure.savefig``.

    Figures are shown inline by default and must not be written to disk unless
    the user explicitly asks.  This is reported as ``requires_explicit_consent``
    (never a hard block) so the user can still approve a deliberate save.
    """
    chain = _chain(node.func, aliases)
    if not chain or chain[-1] != "savefig":
        return False
    keywords = ("pyplot", "plt", "figure", "fig", "matplotlib")
    return any(any(keyword in part for keyword in keywords) for part in chain)


def _dangerous_reference(name: str) -> bool:
    """True when a stored alias/attribute target resolves to a dangerous sink."""
    if not name:
        return False
    if name in _SYSTEM_CALLS or name in _CRITICAL_CALLS or name in _DYNAMIC_IMPORTS:
        return True
    if name in _RAW_FILE_WRITERS or name in _DESERIALIZE_SINKS or name in _FILE_WRITERS:
        return True
    return name.rsplit(".", 1)[-1] in (
        _SYSTEM_METHOD_NAMES | _PATH_METHODS | _FILE_WRITE_METHOD_NAMES | {"read_pickle"}
    )


def _smuggling_rule_id(
    target: ast.AST,
    value_chain: str | None,
    aliases: _Aliases,
) -> str | None:
    """Return ``"SMUG001"`` when an attribute target smuggles a dangerous callable.

    Module attribute assignment (``os.run_id = os.system``), assignment through
    reflection chains and assignments whose value resolves to a dangerous sink
    all defeat alias/name tracking and are never legitimate analysis code.
    """
    chain = _chain(target, aliases)
    if len(chain) < 2:
        return None
    if chain[0] in aliases.imported_roots:
        return "SMUG001"
    if any(part in _REFLECTION_ATTRS for part in chain[:-1]):
        return "SMUG001"
    if value_chain and _dangerous_reference(value_chain):
        return "SMUG001"
    return None


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #
def _classify_call(
    node: ast.Call,
    name: str,
    aliases: _Aliases,
    issues: list[SecurityIssue],
    consent_issues: list[SecurityIssue],
) -> None:
    """Method-level rules for calls whose module is inside the sandbox.

    ``name`` is the alias-resolved canonical call name.  Runs after the
    capability-module layer, so it only sees allowed/mixed modules (``os``,
    ``pathlib``, ``builtins``, numpy, ...).
    """
    if name in {"getattr", "builtins.getattr"} and (
        len(node.args) < 2
        or not (
            isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        )
    ):
        # A computed attribute name (``getattr(os, 'sy' + 'stem')``) makes the
        # fetched member unknowable: hard block rather than guess.
        issues.append(SecurityIssue(
            "REF003", _desc("ref_getattr_dynamic"),
            RiskLevel.HIGH, getattr(node, "lineno", 0), ast.unparse(node),
        ))
    if name in {"setattr", "delattr", "builtins.setattr", "builtins.delattr"}:
        attribute = node.args[1] if len(node.args) >= 2 else None
        receiver_chain = _chain(node.args[0], aliases) if node.args else []
        if not (
            isinstance(attribute, ast.Constant) and isinstance(attribute.value, str)
        ):
            issues.append(SecurityIssue(
                "REF003", _desc("ref_dynamic_attr", name=name),
                RiskLevel.HIGH, getattr(node, "lineno", 0), ast.unparse(node),
            ))
        elif any(part in _REFLECTION_ATTRS for part in receiver_chain) or (
            receiver_chain
            and (
                receiver_chain[0] in aliases.imported_roots
                or receiver_chain[0] in {"builtins", "__builtins__"}
            )
        ):
            issues.append(SecurityIssue(
                "SMUG002", _desc("smug_attr_mutation", name=name),
                RiskLevel.HIGH, getattr(node, "lineno", 0), ast.unparse(node),
            ))
        elif name.rsplit(".", 1)[-1] == "setattr" and len(node.args) >= 3:
            value_name = ".".join(_chain(node.args[2], aliases))
            if _dangerous_reference(value_name):
                issues.append(SecurityIssue(
                    "SMUG002", _desc("smug_setattr_dangerous", value_name=value_name),
                    RiskLevel.HIGH, getattr(node, "lineno", 0), ast.unparse(node),
                ))
    if name in _RAW_FILE_WRITERS:
        issues.append(SecurityIssue(
            "FILE003", _desc("file_raw_write", name=name),
            RiskLevel.HIGH, getattr(node, "lineno", 0), ast.unparse(node),
        ))
    elif name in _DESERIALIZE_SINKS or name.rsplit(".", 1)[-1] == "read_pickle":
        issues.append(SecurityIssue(
            "DES001", _desc("deserialize_unsafe", name=name),
            RiskLevel.HIGH, getattr(node, "lineno", 0), ast.unparse(node),
        ))
    if name in _CRITICAL_CALLS or (name.split(".")[-1] in _CRITICAL_CALLS and name.startswith("builtins.")):
        issues.append(SecurityIssue("EXEC001", _desc("exec_dynamic", name=name), RiskLevel.CRITICAL, getattr(node, "lineno", 0), ast.unparse(node)))
    elif name in _SYSTEM_CALLS:
        issues.append(SecurityIssue("SYS001", _desc("sys_destructive", name=name), RiskLevel.HIGH, getattr(node, "lineno", 0), ast.unparse(node)))
    elif name in _DYNAMIC_IMPORTS:
        issues.append(SecurityIssue("IMPORT001", _desc("import_dynamic", name=name), RiskLevel.HIGH, getattr(node, "lineno", 0), ast.unparse(node)))
    elif name in {"open", "builtins.open", "io.open"} and _open_modes(node) == "w":
        issues.append(SecurityIssue("FILE001", _desc("file_modifying"), RiskLevel.HIGH, getattr(node, "lineno", 0), ast.unparse(node)))
    elif name in {"open", "builtins.open", "io.open"} and _open_modes(node) is None:
        # mode is a variable/expression: read-only-ness cannot be
        # confirmed, so it always requires explicit consent.
        consent_issues.append(SecurityIssue("FILE002", _desc("file_mode_unclear"), RiskLevel.HIGH, getattr(node, "lineno", 0), ast.unparse(node)))
    elif name.endswith(".open") and _has_path_origin(_chain(node.func, aliases)) and _open_modes(node, mode_position=0) == "w":
        issues.append(SecurityIssue("FILE001", _desc("file_modifying_path"), RiskLevel.HIGH, getattr(node, "lineno", 0), ast.unparse(node)))
    elif _is_path_method_call(node, aliases):
        issues.append(SecurityIssue("SYS001", _desc("path_destructive", name=name), RiskLevel.HIGH, getattr(node, "lineno", 0), ast.unparse(node)))
    elif _is_env_mutation_call(node, aliases):
        issues.append(SecurityIssue("ENV001", _desc("env_mutation"), RiskLevel.HIGH, getattr(node, "lineno", 0), ast.unparse(node)))
    elif _is_savefig(node, aliases):
        consent_issues.append(SecurityIssue("SAVE001", _desc("savefig_consent"), RiskLevel.HIGH, getattr(node, "lineno", 0), ast.unparse(node)))
    elif name in _FILE_WRITERS or name.rsplit(".", 1)[-1] in _FILE_WRITE_METHOD_NAMES:
        consent_issues.append(SecurityIssue("SAVE002", _desc("file_write_consent", name=name), RiskLevel.HIGH, getattr(node, "lineno", 0), ast.unparse(node)))
    target = _indirect_call_target(node, aliases)
    if target:
        if target in _INDIRECT_TARGETS:
            issues.append(SecurityIssue("EXEC001", _desc("exec_indirect", target=target), RiskLevel.CRITICAL, getattr(node, "lineno", 0), ast.unparse(node)))
        elif target in _SYSTEM_METHOD_NAMES | _PATH_METHODS:
            issues.append(SecurityIssue("SYS001", _desc("sys_destructive_indirect", target=target), RiskLevel.HIGH, getattr(node, "lineno", 0), ast.unparse(node)))
        elif target == "open" and _open_modes(node) == "w":
            issues.append(SecurityIssue("FILE001", _desc("file_modifying_indirect"), RiskLevel.HIGH, getattr(node, "lineno", 0), ast.unparse(node)))
        elif target == "open" and _open_modes(node) is None:
            consent_issues.append(SecurityIssue("FILE002", _desc("file_mode_unclear_indirect"), RiskLevel.HIGH, getattr(node, "lineno", 0), ast.unparse(node)))
        elif target in _FILE_WRITE_METHOD_NAMES:
            consent_issues.append(SecurityIssue("SAVE002", _desc("file_write_consent_indirect", target=target), RiskLevel.HIGH, getattr(node, "lineno", 0), ast.unparse(node)))


def scan_code(code: str) -> ScanResult:
    """Detect the same broad classes of dangerous code guarded by instrMCP.

    Parameters
    ----------
    code : str
        Python or IPython source proposed for Kernel execution.

    Returns
    -------
    ScanResult
        Blocking decision, structured issues, reason and optional syntax detail.

    Examples
    --------
    >>> scan_code("value = 1 + 2").blocked
    False
    >>> scan_code("import os; os.system('whoami')").blocked
    True
    """
    issues: list[SecurityIssue] = []
    consent_issues: list[SecurityIssue] = []
    ipython = scan_ipython(code)
    for item in ipython.issues:
        issues.append(SecurityIssue(item.rule_id, item.description, RiskLevel.CRITICAL, code=item.matched))
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        detail = {"message": exc.msg, "line": exc.lineno, "offset": exc.offset, "text": exc.text}
        return ScanResult(True, issues, "code contains a Python syntax error", detail)

    bindings = _Aliases()
    bindings.visit(tree)
    for node in ast.walk(tree):
        aliases = bindings.at(node)
        if isinstance(node, ast.ImportFrom) and any(item.name == "*" for item in node.names):
            root = (node.module or "").split(".")[0]
            if root not in _COMPUTE_MODULES:
                # Star imports defeat all name-based tracking: pull in an
                # unlisted ``os.system`` etc. that would never be seen.
                issues.append(SecurityIssue(
                    "CAP003", _desc("cap_star_import", module=node.module),
                    RiskLevel.HIGH, getattr(node, "lineno", 0), ast.unparse(node),
                ))
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Subscript):
                # d['p'](...) / os.__dict__['system'](...) / cls[idx](...): the
                # callee is unknowable from source, which is exactly how exec /
                # subprocess are smuggled through containers.
                issues.append(SecurityIssue(
                    "IND002", _desc("ind_subscript_call"),
                    RiskLevel.HIGH, getattr(node, "lineno", 0), ast.unparse(node),
                ))
            elif any(part in _REFLECTION_ATTRS for part in _chain(node.func, aliases)):
                # __dict__ / __class__.__mro__ / __subclasses__ chains can reach
                # any object in the interpreter; a call through them is unbounded.
                issues.append(SecurityIssue(
                    "REF001", _desc("ref_chain_call"),
                    RiskLevel.HIGH, getattr(node, "lineno", 0), ast.unparse(node),
                ))
            name = _call_name(node, aliases)
            root = name.split(".")[0]
            if root in _BLOCK_MODULES:
                # subprocess / ctypes / importlib / pickle / joblib: process,
                # native-code, dynamic-import and deserialization sinks.
                issues.append(SecurityIssue(
                    "CAP001", _desc("cap_sandbox_via", name=name),
                    RiskLevel.CRITICAL, getattr(node, "lineno", 0), ast.unparse(node),
                ))
            elif root in _REFLECTION_BASES:
                # globals/locals/vars/__builtins__ can reach any callable.
                issues.append(SecurityIssue(
                    "CAP002", _desc("cap_reflection_via", name=name),
                    RiskLevel.CRITICAL, getattr(node, "lineno", 0), ast.unparse(node),
                ))
            elif root in _NETWORK_MODULES:
                consent_issues.append(SecurityIssue(
                    "NET001", _desc("network_consent", name=name),
                    RiskLevel.HIGH, getattr(node, "lineno", 0), ast.unparse(node),
                ))
            else:
                _classify_call(node, name, aliases, issues, consent_issues)
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value_chain = (
                ".".join(_chain(node.value, aliases))
                if isinstance(node, ast.Assign)
                and isinstance(node.value, (ast.Name, ast.Attribute))
                else None
            )
            for target in targets:
                if _is_env_assignment(target, aliases):
                    issues.append(SecurityIssue("ENV001", _desc("env_mutation"), RiskLevel.HIGH, getattr(node, "lineno", 0), ast.unparse(node)))
                elif (
                    isinstance(node, ast.AugAssign)
                    and isinstance(target, ast.Name)
                    and ".".join(_chain(target, aliases)) == "os.environ"
                ):
                    # ``env = os.environ; env |= {...}`` mutates the process
                    # environment through an aliased Name target.
                    issues.append(SecurityIssue("ENV001", _desc("env_mutation_aliased"), RiskLevel.HIGH, getattr(node, "lineno", 0), ast.unparse(node)))
                if isinstance(target, ast.Attribute):
                    rule = _smuggling_rule_id(target, value_chain, aliases)
                    if rule:
                        issues.append(SecurityIssue(
                            rule,
                            _desc("smug_hidden_callable"),
                            RiskLevel.HIGH, getattr(node, "lineno", 0), ast.unparse(node),
                        ))
        if isinstance(node, ast.Delete):
            for target in node.targets:
                if _is_env_assignment(target, aliases):
                    issues.append(SecurityIssue("ENV001", _desc("env_mutation"), RiskLevel.HIGH, getattr(node, "lineno", 0), ast.unparse(node)))
                if isinstance(target, ast.Attribute) and _smuggling_rule_id(target, None, aliases):
                    issues.append(SecurityIssue(
                        "SMUG001",
                        _desc("smug_attribute_delete"),
                        RiskLevel.HIGH, getattr(node, "lineno", 0), ast.unparse(node),
                    ))
    reason = "; ".join(issue.description for issue in issues[:3]) if issues else None
    return ScanResult(bool(issues), issues, reason, requires_explicit_consent=consent_issues)


def call_names(code: str) -> list[str]:
    """Return the alias-resolved canonical name of every call expression.

    Uses the same alias tracking as :func:`scan_code`, so
    ``import matplotlib.pyplot as plt; plt.subplots()`` yields
    ``matplotlib.pyplot.subplots``.  Subscript calls (``d['fn'](...)``) surface
    their base chain because the callee itself is untrackable.

    Returns an empty list when the code does not parse — callers should treat
    unparsable input as having no matches and let the normal syntax-error path
    handle it.

    Parameters
    ----------
    code : str
        Python source proposed for execution.

    Returns
    -------
    list of str
        Canonical call names, in source order.
    """
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError:
        return []
    bindings = _Aliases()
    bindings.visit(tree)
    names: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Subscript):
            names.append(".".join(_chain(node.func.value, bindings.at(node))))
        else:
            names.append(_call_name(node, bindings.at(node)))
    return names
