"""Pre-AST scanner for IPython syntax that Python's parser cannot safely model."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_SHELL = re.compile(
    r"^\s*!|^\s*%%(?:bash|sh|script|capture)|get_ipython\(\)\.(?:system|run_line_magic|run_cell_magic)",
    re.MULTILINE,
)
_ENV = re.compile(r"^\s*%(?:env|set_env|pip|conda|load_ext|run)\b", re.MULTILINE)


@dataclass(slots=True)
class IPythonIssue:
    """One unsafe IPython construct."""

    rule_id: str
    description: str
    matched: str


@dataclass(slots=True)
class IPythonScanResult:
    """Result of scanning IPython-only syntax."""

    issues: list[IPythonIssue] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return bool(self.issues)


def scan_ipython(code: str) -> IPythonScanResult:
    """Find shell escapes, dangerous cell magics and environment mutations."""
    issues: list[IPythonIssue] = []
    for rule_id, pattern, description in (
        ("IPY001", _SHELL, "shell escape or indirect IPython command execution"),
        ("IPY002", _ENV, "environment or extension modifying IPython magic"),
    ):
        match = pattern.search(code)
        if match:
            issues.append(IPythonIssue(rule_id, description, match.group(0)))
    return IPythonScanResult(issues)
