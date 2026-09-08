"""Base report shape shared by every peaksMCP operation.

All reports converge on the common vocabulary below; typed subclasses add
fields only where the model or the user reads them.

Common fields (where present on a report):
- operation : str   — verb that produced the report (e.g. ``load_data``).
- status    : str   — ``ok`` / ``partial`` / ``skipped`` / ``failed``.
- partial   : bool  — True when only part of the requested work ran.
- warnings  : list[str]
- selection : dict  — input selection that produced the report.
- provenance: list[str] — steps applied to the data.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any


class Report:
    """Marker/base for operation reports; dataclasses may inherit it freely."""

    operation = ""
    status = ""
    partial = False

    def __init__(self, **kwargs: Any) -> None:
        """Plain-object initialiser; dataclass/pydantic subclasses override it."""
        for key, value in kwargs.items():
            setattr(self, key, value)

    def summary_line(self) -> str:
        """One-line human summary (Show convention)."""
        text = f"{self.operation}: {self.status}" if self.operation else str(self.status)
        if bool(getattr(self, "partial", False)):
            text += " (partial)"
        warnings = getattr(self, "warnings", []) or []
        if warnings:
            text += f"; {len(warnings)} warning(s)"
        return text


def _report_dict(report: Any) -> dict[str, Any]:
    """JSON-safe dict of a report, merging its dataclass fields with the
    common vocabulary (works for dataclasses and plain Report subclasses).

    Module helpers are underscore-private on purpose: discovery scans
    ``peaksMCP.overrides`` for model-facing verbs, and report plumbing is not
    a verb.  ``peaksMCP.overrides`` re-exports the public names below.
    """
    payload: dict[str, Any] = {}
    if is_dataclass(report) and not isinstance(report, type):
        payload = asdict(report)
    for key in ("operation", "status", "partial", "warnings", "selection", "provenance"):
        value = getattr(report, key, None)
        if value is not None and key not in payload:
            payload[key] = value
    return payload


def _report_summary(report: Any) -> str:
    """Short canonical line for model-facing output."""
    summary = getattr(report, "summary_line", None)
    if callable(summary):
        return summary()
    return _report_dict(report).get("status", "ok")


# Public aliases for code that consumes reports (server/Show layer, tests).
report_dict = _report_dict
report_summary = _report_summary
