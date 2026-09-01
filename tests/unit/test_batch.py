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


def test_cpu_gate_uses_trailing_window_and_hysteresis():
    budget = ResourceBudget(cpu_limit_percent=60, resume_percent=50, moving_window_s=10)
    budget._record_sample(80, timestamp=0)
    assert not budget._gate.is_set()
    budget._record_sample(40, timestamp=11)
    assert budget._gate.is_set()
    assert budget.average_cpu_percent == 40


def test_worker_submission_ramps_up_and_reduces_after_cpu_pressure(monkeypatch):
    monkeypatch.setattr("peaksMCP.batch.resource_budget.os.cpu_count", lambda: 8)
    budget = ResourceBudget(
        cpu_limit_percent=60,
        resume_percent=50,
        sample_interval_s=1,
        worker_fraction=0.5,
    )
    budget._allowed_workers = 1
    budget._last_ramp_at = 0

    budget._record_sample(20, timestamp=1)
    assert budget.submission_limit == 2
    budget._record_sample(20, timestamp=2)
    assert budget.submission_limit == 3

    budget._record_sample(80, timestamp=3)
    assert budget.submission_limit == 2
    assert not budget._gate.is_set()
    budget._record_sample(80, timestamp=4)
    assert budget.submission_limit == 1


def test_batch_report_states_best_effort_cpu_budget_semantics():
    budget = ResourceBudget(
        cpu_limit_percent=60,
        resume_percent=50,
        sample_interval_s=0.02,
        worker_fraction=0.0001,
    )
    result = BatchExecutor(budget).run(square_or_fail, [1])
    payload = result.to_dict()
    assert payload["cpu_budget_percent"] == 60
    assert payload["cpu_budget_strategy"] == "best_effort_progressive"
    assert payload["cpu_budget_exceeded"] == (
        payload["peak_moving_average_cpu_percent"]
        >= payload["cpu_budget_percent"]
    )


def test_start_takes_a_real_initial_cpu_sample(monkeypatch):
    calls: list[float | None] = []

    def sample(interval=None):
        calls.append(interval)
        return 75.0

    monkeypatch.setattr("peaksMCP.batch.resource_budget.psutil.cpu_percent", sample)
    budget = ResourceBudget(
        cpu_limit_percent=60,
        resume_percent=50,
        sample_interval_s=10,
    )
    budget.start()
    try:
        assert calls[0] is not None and calls[0] > 0
        assert not budget._gate.is_set()
    finally:
        budget.stop()


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


def test_cancel_mid_run_stops_future_submissions(tmp_path):
    """Cancellation lets one in-flight task finish and marks the remainder cancelled."""
    import threading as th
    import time

    started = tmp_path / "started"
    release = tmp_path / "release"
    payloads = [(str(started), str(release), item) for item in range(1, 6)]
    budget = ResourceBudget(sample_interval_s=0.02, worker_fraction=0.0001)
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
    assert statuses[0] == "completed"
    assert statuses[1:] == ["cancelled"] * 4
    assert result.cancelled
