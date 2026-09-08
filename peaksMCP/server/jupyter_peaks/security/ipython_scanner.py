"""Pre-AST scanner for IPython syntax that Python's parser cannot safely model."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from peaksMCP.config import prompts as _load_prompts

_SHELL = re.compile(
    r"^\s*!|^\s*%%(?:bash|sh|script|capture)|get_ipython\(\)\.(?:system|run_line_magic|run_cell_magic)",
    re.MULTILINE,
)
_ENV = re.compile(r"^\s*%(?:env|set_env|pip|conda|load_ext|run)\b", re.MULTILINE)

#: Curated issue-description templates (config/prompts.yaml).
_IPYTHON_TEXT = _load_prompts().get("ipython") or {}


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
    for rule_id, pattern, key in (
        ("IPY001", _SHELL, "ipy_shell"),
        ("IPY002", _ENV, "ipy_env_magic"),
    ):
        match = pattern.search(code)
        if match:
            issues.append(IPythonIssue(rule_id, _IPYTHON_TEXT[key], match.group(0)))
    return IPythonScanResult(issues)
