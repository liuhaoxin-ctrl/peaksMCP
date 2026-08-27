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
        """Stop future submissions and cancel futures not yet running."""
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
                next_index = 0
                while next_index < len(values) or self._pending_futures:
                    while (
                        next_index < len(values)
                        and len(self._pending_futures) < self.budget.max_workers
                        and not self.cancel_event.is_set()
                    ):
                        item = values[next_index]
                        if not self.budget.wait_for_capacity(
                            timeout=60,
                            cancel_event=self.cancel_event,
                        ):
                            if self.cancel_event.is_set():
                                break
                            result = BatchItemResult(
                                next_index,
                                item,
                                "skipped",
                                error="CPU budget wait timed out",
                            )
                            results[next_index] = result
                            if progress:
                                progress(result)
                            next_index += 1
                            continue
                        future = pool.submit(_run_one, function, item)
                        self._pending_futures[future] = (next_index, item)
                        next_index += 1

                    if self.cancel_event.is_set():
                        while next_index < len(values):
                            result = BatchItemResult(next_index, values[next_index], "cancelled")
                            results[next_index] = result
                            if progress:
                                progress(result)
                            next_index += 1
                        for future, (index, item) in list(self._pending_futures.items()):
                            if future.cancel():
                                result = BatchItemResult(index, item, "cancelled")
                                results[index] = result
                                if progress:
                                    progress(result)
                                self._pending_futures.pop(future)

                    if not self._pending_futures:
                        continue
                    completed, _pending = concurrent.futures.wait(
                        tuple(self._pending_futures),
                        timeout=0.1,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    for future in completed:
                        index, item = self._pending_futures.pop(future)
                        if future.cancelled():
                            result = BatchItemResult(index, item, "cancelled")
                            results[index] = result
                            if progress:
                                progress(result)
                            continue
                        try:
                            output, duration = future.result()
                            result = BatchItemResult(
                                index,
                                item,
                                "completed",
                                output=output,
                                duration_s=duration,
                            )
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
