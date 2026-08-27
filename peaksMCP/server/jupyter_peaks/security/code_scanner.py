"""Alias-aware AST scanner for dangerous notebook code."""

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

    @property
    def is_safe(self) -> bool:
        return not self.blocked

    def to_dict(self) -> dict[str, Any]:
        return {"is_safe": self.is_safe, "blocked": self.blocked, "block_reason": self.block_reason, "syntax_error": self.syntax_error, "issues": [issue.to_dict() for issue in self.issues]}


class _Aliases(ast.NodeVisitor):
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
        root, dot, tail = name.partition(".")
        return self.names.get(root, root) + (dot + tail if dot else "")


def _call_name(node: ast.Call, aliases: _Aliases) -> str:
    try:
        return aliases.resolve(ast.unparse(node.func))
    except Exception:
        return ""


_CRITICAL_CALLS = {"exec", "eval", "compile", "builtins.exec", "builtins.eval", "builtins.compile"}
_SYSTEM_CALLS = {
    "os.system", "os.popen", "subprocess.run", "subprocess.call", "subprocess.Popen",
    "subprocess.check_call", "subprocess.check_output", "shutil.rmtree", "pathlib.Path.unlink",
    "pathlib.Path.rmdir", "os.remove", "os.unlink", "os.rmdir", "os.removedirs",
}
_DYNAMIC_IMPORTS = {"__import__", "importlib.import_module"}


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
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _call_name(node, aliases)
            if name in _CRITICAL_CALLS:
                issues.append(SecurityIssue("EXEC001", f"dynamic code execution via {name}", RiskLevel.CRITICAL, getattr(node, "lineno", 0), ast.unparse(node)))
            elif name in _SYSTEM_CALLS:
                issues.append(SecurityIssue("SYS001", f"system or destructive operation via {name}", RiskLevel.HIGH, getattr(node, "lineno", 0), ast.unparse(node)))
            elif name in _DYNAMIC_IMPORTS:
                issues.append(SecurityIssue("IMPORT001", f"dynamic import via {name}", RiskLevel.HIGH, getattr(node, "lineno", 0), ast.unparse(node)))
            elif name == "open" and len(node.args) > 1 and isinstance(node.args[1], ast.Constant) and any(flag in str(node.args[1].value) for flag in "wax+"):
                issues.append(SecurityIssue("FILE001", "file opened in a modifying mode", RiskLevel.HIGH, getattr(node, "lineno", 0), ast.unparse(node)))
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            rendered = ast.unparse(node)
            if "os.environ" in rendered:
                issues.append(SecurityIssue("ENV001", "process environment modification", RiskLevel.HIGH, getattr(node, "lineno", 0), rendered))
    reason = "; ".join(issue.description for issue in issues[:3]) if issues else None
    return ScanResult(bool(issues), issues, reason)
