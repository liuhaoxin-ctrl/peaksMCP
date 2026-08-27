"""Alias-aware, AST-semantic scanner for dangerous notebook code.

The scanner works on the parsed AST rather than on ``ast.unparse`` text so
that aliased imports, keyword arguments, method calls on constructed objects
(e.g. ``Path(...).unlink()``) and indirect fetches (``getattr(obj, "exec")``)
cannot slip past the rules.  All matching is done on a canonical attribute
chain with import aliases resolved.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .ipython_scanner import scan_ipython


class RiskLevel(StrEnum):
    """Severity attached to a security finding."""

    HIGH = "high"
    CRITICAL = "critical"


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
    # Patterns that are not hard-blocked but require an explicit, informed
    # user consent even in dangerous mode (e.g. ``plt.savefig`` — figures are
    # shown inline by default and must not be written unless the user asks).
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
    """Collect ``import`` aliases so ``import os as o`` resolves to ``os``."""

    def __init__(self) -> None:
        self.names: dict[str, str] = {}

    def visit_Import(self, node: ast.Import) -> None:
        for item in node.names:
            self.names[item.asname or item.name] = item.name

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        for item in node.names:
            self.names[item.asname or item.name] = f"{module}.{item.name}"

    def resolve(self, name: str) -> str:
        return self.names.get(name, name)

    def collect_assignments(self, tree: ast.AST) -> None:
        """Track assignment aliases such as ``f = os.system`` so that ``f("id")``
        resolves to ``os.system``.  Iterated to a fixpoint for chains like
        ``g = f; f = os.system``."""
        for _ in range(3):
            changed = False
            for node in ast.walk(tree):
                if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                    continue
                target = node.targets[0]
                if not isinstance(target, ast.Name):
                    continue
                if isinstance(node.value, ast.Attribute):
                    resolved = ".".join(_chain(node.value, self))
                elif isinstance(node.value, ast.Name):
                    resolved = self.resolve(node.value.id)
                else:
                    continue
                if self.names.get(target.id) != resolved:
                    self.names[target.id] = resolved
                    changed = True
            if not changed:
                break


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
    ``getattr(b, 'exec')`` (func)    -> ``["getattr", "exec"]``
    """
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(aliases.resolve(cur.id))
    elif isinstance(cur, ast.Call):
        parts.append(_func_text(cur, aliases))
    elif isinstance(cur, ast.Subscript):
        # Fold os.environ['A'] -> os.environ; keep value chain only.
        return _chain(cur.value, aliases)
    elif isinstance(cur, ast.Constant):
        parts.append(repr(cur.value))
    elif isinstance(cur, ast.Starred):
        parts.append("(starred)")
    else:
        parts.append(f"({type(cur).__name__})")
    parts.reverse()
    return parts


def _call_name(node: ast.Call, aliases: _Aliases) -> str:
    """Canonical dotted name of the called function, aliases resolved."""
    return ".".join(_chain(node.func, aliases))


# --------------------------------------------------------------------------- #
# Rule sets                                                                    #
# --------------------------------------------------------------------------- #
_CRITICAL_CALLS = {"exec", "eval", "compile", "builtins.exec", "builtins.eval", "builtins.compile"}
_SYSTEM_CALLS = {
    "os.system", "os.popen", "os.remove", "os.unlink", "os.rmdir", "os.removedirs",
    "os.replace", "os.rename",
    "subprocess.run", "subprocess.call", "subprocess.Popen", "subprocess.check_call",
    "subprocess.check_output", "subprocess.getoutput", "subprocess.getstatusoutput",
    "shutil.rmtree", "shutil.rmdir", "shutil.move",
}
_DYNAMIC_IMPORTS = {"__import__", "builtins.__import__", "importlib.import_module"}
_PATH_METHODS = {"unlink", "write_text", "write_bytes", "rmdir", "rename", "replace", "symlink_to", "hardlink_to"}
_ENV_MUTATORS = {"update", "setdefault", "pop", "clear", "__setitem__", "__delitem__"}
_DYN_BASES = {"globals", "locals", "vars", "__builtins__", "builtins"}
_INDIRECT_TARGETS = {"exec", "eval", "compile", "__import__"}

# Dangerous callables fetched indirectly (getattr / globals()['x'] / .get()).
_SYSTEM_METHOD_NAMES = {name.rsplit(".", 1)[-1] for name in _SYSTEM_CALLS} | {"system", "popen", "run"}
_FILE_WRITERS = {
    "np.save", "numpy.save", "np.savetxt", "numpy.savetxt", "np.savez", "numpy.savez",
    "np.savez_compressed", "numpy.savez_compressed", "plt.imsave", "matplotlib.pyplot.imsave",
    "pd.to_csv", "pandas.DataFrame.to_csv", "xr.to_netcdf", "xarray.DataArray.to_netcdf",
    "xarray.Dataset.to_netcdf", "json.dump", "pickle.dump", "joblib.dump",
}
_FILE_WRITE_METHOD_NAMES = {"save", "savetxt", "savez", "savez_compressed", "imsave", "to_csv", "to_netcdf", "dump"}
_NETWORK_CALLS = {
    "requests.get", "requests.post", "requests.put", "requests.patch", "requests.delete",
    "requests.head", "requests.options", "urllib.request.urlopen", "urllib.request.Request",
    "httpx.get", "httpx.post", "httpx.put", "httpx.patch", "httpx.delete",
}


def _is_path_method_call(node: ast.Call, aliases: _Aliases) -> bool:
    """``Path(...).unlink()`` / ``pathlib.Path('x').write_text(...)`` and friends."""
    chain = _chain(node.func, aliases)
    if not chain or chain[-1] not in _PATH_METHODS:
        return False
    # Chain forms: ['Path','unlink'], ['pathlib.Path','unlink'] or ['pathlib','Path','unlink']
    return any(e == "Path" or e == "pathlib.Path" or e.endswith(".Path") for e in chain)


def _is_env_mutation_call(node: ast.Call, aliases: _Aliases) -> bool:
    """``os.environ.update/pop/clear/...`` regardless of aliasing."""
    chain = _chain(node.func, aliases)
    if len(chain) < 3 or chain[:2] != ["os", "environ"]:
        return False
    return chain[-1] in _ENV_MUTATORS


def _is_env_assignment(target: ast.AST, aliases: _Aliases) -> bool:
    """``os.environ[...] = ...`` / ``o.environ['A'] += ...`` / ``del os.environ['A']``."""
    return _chain(target, aliases)[:2] == ["os", "environ"]


def _open_modes(node: ast.Call) -> bool:
    """Flag ``open`` in a modifying mode, positional or ``mode=`` keyword."""
    flags = "wax+"
    if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
        if any(flag in str(node.args[1].value) for flag in flags):
            return True
    for keyword in node.keywords:
        if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
            if any(flag in str(keyword.value.value) for flag in flags):
                return True
    return False


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


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #
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

    aliases = _Aliases()
    aliases.visit(tree)
    aliases.collect_assignments(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _call_name(node, aliases)
            if name in _CRITICAL_CALLS or (name.split(".")[-1] in _CRITICAL_CALLS and name.startswith("builtins.")):
                issues.append(SecurityIssue("EXEC001", f"dynamic code execution via {name}", RiskLevel.CRITICAL, getattr(node, "lineno", 0), ast.unparse(node)))
            elif name in _SYSTEM_CALLS:
                issues.append(SecurityIssue("SYS001", f"system or destructive operation via {name}", RiskLevel.HIGH, getattr(node, "lineno", 0), ast.unparse(node)))
            elif name in _DYNAMIC_IMPORTS:
                issues.append(SecurityIssue("IMPORT001", f"dynamic import via {name}", RiskLevel.HIGH, getattr(node, "lineno", 0), ast.unparse(node)))
            elif name == "open" and _open_modes(node):
                issues.append(SecurityIssue("FILE001", "file opened in a modifying mode", RiskLevel.HIGH, getattr(node, "lineno", 0), ast.unparse(node)))
            elif _is_path_method_call(node, aliases):
                issues.append(SecurityIssue("SYS001", f"destructive path operation via {name}", RiskLevel.HIGH, getattr(node, "lineno", 0), ast.unparse(node)))
            elif _is_env_mutation_call(node, aliases):
                issues.append(SecurityIssue("ENV001", "process environment modification", RiskLevel.HIGH, getattr(node, "lineno", 0), ast.unparse(node)))
            elif _is_savefig(node, aliases):
                consent_issues.append(SecurityIssue("SAVE001", "figure save (savefig); figures are shown inline by default — approve only to write to disk", RiskLevel.HIGH, getattr(node, "lineno", 0), ast.unparse(node)))
            elif name in _FILE_WRITERS or name.rsplit(".", 1)[-1] in _FILE_WRITE_METHOD_NAMES:
                consent_issues.append(SecurityIssue("SAVE002", f"file write via {name}; approve only to write to disk", RiskLevel.HIGH, getattr(node, "lineno", 0), ast.unparse(node)))
            elif name in _NETWORK_CALLS:
                consent_issues.append(SecurityIssue("NET001", f"network request via {name}; approve only to send data externally", RiskLevel.HIGH, getattr(node, "lineno", 0), ast.unparse(node)))
            target = _indirect_call_target(node, aliases)
            if target:
                if target in _INDIRECT_TARGETS:
                    issues.append(SecurityIssue("EXEC001", f"dynamic code execution via indirect fetch of {target}", RiskLevel.CRITICAL, getattr(node, "lineno", 0), ast.unparse(node)))
                elif target in _SYSTEM_METHOD_NAMES:
                    issues.append(SecurityIssue("SYS001", f"system or destructive operation via indirect fetch of {target}", RiskLevel.HIGH, getattr(node, "lineno", 0), ast.unparse(node)))
                elif target == "open" and _open_modes(node):
                    issues.append(SecurityIssue("FILE001", "file opened in a modifying mode via indirect fetch", RiskLevel.HIGH, getattr(node, "lineno", 0), ast.unparse(node)))
                elif target in _FILE_WRITE_METHOD_NAMES:
                    consent_issues.append(SecurityIssue("SAVE002", f"file write via indirect fetch of {target}; approve only to write to disk", RiskLevel.HIGH, getattr(node, "lineno", 0), ast.unparse(node)))
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if _is_env_assignment(target, aliases):
                    issues.append(SecurityIssue("ENV001", "process environment modification", RiskLevel.HIGH, getattr(node, "lineno", 0), ast.unparse(node)))
        if isinstance(node, ast.Delete):
            for target in node.targets:
                if _is_env_assignment(target, aliases):
                    issues.append(SecurityIssue("ENV001", "process environment modification", RiskLevel.HIGH, getattr(node, "lineno", 0), ast.unparse(node)))
    reason = "; ".join(issue.description for issue in issues[:3]) if issues else None
    return ScanResult(bool(issues), issues, reason, requires_explicit_consent=consent_issues)
