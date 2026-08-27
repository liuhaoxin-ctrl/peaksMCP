"""Failure-isolated process pool governed by :class:`ResourceBudget`."""

from __future__ import annotations

import concurrent.futures
import threading
import time
from collections.abc import Callable, Iterable
from typing import Any

from threadpoolctl import threadpool_limits

from .models import BatchItemResult, BatchResult
from .resource_budget import ResourceBudget


def _run_one(function: Callable[[Any], Any], item: Any) -> tuple[Any, float]:
    ResourceBudget.configure_worker_threads()
    started = time.monotonic()
    with threadpool_limits(limits=1):
        result = function(item)
    return result, time.monotonic() - started


class BatchExecutor:
    """Execute independent items with CPU budgeting and failure isolation."""

    def __init__(self, budget: ResourceBudget | None = None) -> None:
        self.budget = budget or ResourceBudget()
        self.cancel_event = threading.Event()
        # Futures of the currently running batch, so ``cancel()`` can stop
        # queued (not yet started) tasks immediately instead of waiting for
        # the next completed task to trigger a scan.
        self._pending_futures: dict[concurrent.futures.Future[tuple[Any, float]], tuple[int, Any]] = {}

    def cancel(self) -> None:
        """Request cancellation of tasks that have not yet been submitted.

        Python's ``ProcessPoolExecutor`` marks a task ``RUNNING`` as soon as it
        is placed on the worker queue (``set_running_or_notify_cancel``), so
        submitted tasks cannot be interrupted or cancelled.  ``cancel()`` stops
        the submission loop for remaining items; in-flight and queued tasks run
        to completion (failure isolation keeps a bad task from blocking the
        rest of the batch).
        """
        self.cancel_event.set()

    def run(
        self,
        function: Callable[[Any], Any],
        items: Iterable[Any],
        *,
        progress: Callable[[BatchItemResult], None] | None = None,
    ) -> BatchResult:
        """Process items and return a structured batch report.

        Parameters
        ----------
        function : callable
            Pickleable top-level function accepting one item.
        items : iterable
            Independent batch inputs.
        progress : callable, optional
            Callback invoked in the parent process after each result.

        Returns
        -------
        BatchResult
            Ordered per-item results and CPU statistics.
        """
        values = list(items)
        started = time.monotonic()
        results: dict[int, BatchItemResult] = {}
        self.budget.start()
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.budget.max_workers) as pool:
                for index, item in enumerate(values):
                    if self.cancel_event.is_set():
                        results[index] = BatchItemResult(index, item, "cancelled")
                        continue
                    if not self.budget.wait_for_capacity(timeout=60):
                        results[index] = BatchItemResult(index, item, "skipped", error="CPU budget wait timed out")
                        continue
                    future = pool.submit(_run_one, function, item)
                    self._pending_futures[future] = (index, item)
                for future in concurrent.futures.as_completed(self._pending_futures):
                    index, item = self._pending_futures[future]
                    try:
                        output, duration = future.result()
                        result = BatchItemResult(index, item, "completed", output=output, duration_s=duration)
                    except Exception as exc:  # batch intentionally continues
                        result = BatchItemResult(
                            index,
                            item,
                            "failed",
                            error_type=type(exc).__name__,
                            error=str(exc),
                        )
                    results[index] = result
                    if progress:
                        progress(result)
        finally:
            self.budget.stop()
            self._pending_futures.clear()
        return BatchResult(
            items=[results[index] for index in range(len(values))],
            duration_s=time.monotonic() - started,
            average_cpu_percent=self.budget.average_cpu_percent,
            peak_cpu_percent=self.budget.peak_cpu_percent,
            cancelled=self.cancel_event.is_set(),
        )

