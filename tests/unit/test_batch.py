from __future__ import annotations

import os

from peaksMCP.batch import BatchExecutor, ResourceBudget


def square_or_fail(value):
    if value == 2:
        raise ValueError("isolated")
    return value * value


def test_worker_bound_and_native_thread_environment():
    budget = ResourceBudget(cpu_limit_percent=60, resume_percent=50)
    assert 1 <= budget.max_workers <= max(1, (os.cpu_count() or 1) // 2)
    budget.configure_worker_threads()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        assert os.environ[name] == "1"


def test_failure_isolation_results_and_cpu_statistics():
    budget = ResourceBudget(cpu_limit_percent=60, resume_percent=50, sample_interval_s=0.02, worker_fraction=0.25)
    result = BatchExecutor(budget).run(square_or_fail, [1, 2, 3])
    assert result.completed == 2
    assert result.failed == 1
    assert [item.status for item in result.items] == ["completed", "failed", "completed"]
    assert result.peak_cpu_percent <= 100


def test_pre_cancel_marks_every_item_cancelled():
    executor = BatchExecutor(ResourceBudget(sample_interval_s=0.02))
    executor.cancel()
    result = executor.run(square_or_fail, [1, 3])
    assert result.cancelled
    assert result.skipped == 2


def _blocking_task(payload):
    """Worker task that blocks until a release marker file appears (pickle-safe)."""
    import time
    from pathlib import Path

    started_path, release_path, item = payload
    Path(started_path).touch()
    deadline = time.time() + 20
    while not Path(release_path).exists() and time.time() < deadline:
        time.sleep(0.05)
    return item


def test_cancel_mid_run_keeps_submitted_tasks_isolated(tmp_path):
    """Cancelling mid-run does not interrupt already-submitted tasks
    (``ProcessPoolExecutor`` marks queued tasks RUNNING), but every submitted
    task still completes with failure isolation."""
    import threading as th
    import time

    started = tmp_path / "started"
    release = tmp_path / "release"
    payloads = [(str(started), str(release), item) for item in range(1, 6)]
    budget = ResourceBudget(sample_interval_s=0.02, worker_fraction=0.1)
    executor = BatchExecutor(budget)
    holder: dict[str, object] = {}

    def run_batch():
        holder["result"] = executor.run(_blocking_task, payloads)

    thread = th.Thread(target=run_batch)
    thread.start()
    deadline = time.time() + 10
    while not started.exists() and time.time() < deadline:
        time.sleep(0.05)
    assert started.exists(), "first task did not start"
    time.sleep(0.5)
    executor.cancel()
    release.touch()
    thread.join(timeout=25)
    result = holder["result"]
    statuses = [item.status for item in result.items]
    # Submitted tasks always run to completion (cancellable only before submit).
    assert statuses == ["completed"] * 5, statuses
    assert result.cancelled

