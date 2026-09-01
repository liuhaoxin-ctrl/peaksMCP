"""Structured batch execution results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class BatchItemResult:
    """Result of processing one batch input."""

    index: int
    input: Any
    status: str
    output: Any = None
    error_type: str | None = None
    error: str | None = None
    duration_s: float = 0.0


@dataclass(slots=True)
class BatchResult:
    """Aggregate result and resource statistics for a batch."""

    items: list[BatchItemResult] = field(default_factory=list)
    duration_s: float = 0.0
    average_cpu_percent: float = 0.0
    peak_cpu_percent: float = 0.0
    peak_moving_average_cpu_percent: float = 0.0
    cpu_budget_percent: float = 60.0
    cpu_budget_strategy: str = "best_effort_progressive"
    cpu_budget_exceeded: bool = False
    cancelled: bool = False

    @property
    def completed(self) -> int:
        """Return the number of successful items."""
        return sum(item.status == "completed" for item in self.items)

    @property
    def failed(self) -> int:
        """Return the number of failed items."""
        return sum(item.status == "failed" for item in self.items)

    @property
    def skipped(self) -> int:
        """Return the number of skipped or cancelled items."""
        return sum(item.status in {"skipped", "cancelled"} for item in self.items)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report."""
        return {
            "completed": self.completed,
            "failed": self.failed,
            "skipped": self.skipped,
            "duration_s": self.duration_s,
            "average_cpu_percent": self.average_cpu_percent,
            "peak_cpu_percent": self.peak_cpu_percent,
            "peak_moving_average_cpu_percent": self.peak_moving_average_cpu_percent,
            "cpu_budget_percent": self.cpu_budget_percent,
            "cpu_budget_strategy": self.cpu_budget_strategy,
            "cpu_budget_exceeded": self.cpu_budget_exceeded,
            "cancelled": self.cancelled,
            "items": [
                {
                    "index": item.index,
                    "input": str(item.input),
                    "status": item.status,
                    "output": item.output,
                    "error_type": item.error_type,
                    "error": item.error,
                    "duration_s": item.duration_s,
                }
                for item in self.items
            ],
        }
