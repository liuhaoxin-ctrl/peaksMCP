"""CPU budget and native thread controls for scientific batch jobs."""

from __future__ import annotations

import os
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import psutil

_THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@dataclass(slots=True)
class ResourceBudget:
    """Monitor CPU use and gate submission of new batch tasks.

    Parameters
    ----------
    cpu_limit_percent : float, default 60
        System CPU threshold at which new tasks stop being submitted.
    resume_percent : float, default 50
        Lower hysteresis threshold used to resume submissions.
    sample_interval_s : float, default 1
        Sampling period for CPU statistics.
    worker_fraction : float, default 0.5
        Fraction of logical CPUs available to the process pool.
    """

    cpu_limit_percent: float = 60.0
    resume_percent: float = 50.0
    sample_interval_s: float = 1.0
    worker_fraction: float = 0.5
    moving_window_s: float = 10.0
    _samples: deque[tuple[float, float]] = field(default_factory=deque, init=False)
    _all_samples: list[float] = field(default_factory=list, init=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _gate: threading.Event = field(default_factory=threading.Event, init=False)
    _sample_lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __post_init__(self) -> None:
        if not 1 <= self.cpu_limit_percent <= 100:
            raise ValueError("cpu_limit_percent must be between 1 and 100")
        if not 0 <= self.resume_percent < self.cpu_limit_percent:
            raise ValueError("resume_percent must be lower than cpu_limit_percent")
        if self.moving_window_s <= 0:
            raise ValueError("moving_window_s must be positive")
        self._gate.set()

    @property
    def max_workers(self) -> int:
        """Return the bounded default process count."""
        count = os.cpu_count() or 1
        return max(1, min(count, int(count * self.worker_fraction)))

    @property
    def average_cpu_percent(self) -> float:
        """Return the trailing moving-window CPU average."""
        with self._sample_lock:
            values = [sample for _timestamp, sample in self._samples]
        return statistics.fmean(values) if values else 0.0

    @property
    def peak_cpu_percent(self) -> float:
        """Return the highest monitor sample."""
        with self._sample_lock:
            return max(self._all_samples, default=0.0)

    def start(self) -> None:
        """Start background CPU monitoring with an immediate first sample.

        The gate is set according to the CURRENT CPU load so a busy machine is
        throttled from the very first submission (previously the gate started
        open and the whole worker pool was submitted before any sample).
        """
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        # Seed the gate with a real first sample instead of defaulting to open.
        # (Record manually — _record_sample takes the lock itself, so calling
        # it from inside a locked block would deadlock.)
        initial = float(psutil.cpu_percent(interval=None))
        with self._sample_lock:
            self._samples.clear()
            self._all_samples.clear()
            self._samples.append((time.monotonic(), initial))
            self._all_samples.append(initial)
        if initial >= self.cpu_limit_percent:
            self._gate.clear()
        else:
            self._gate.set()
        self._thread = threading.Thread(target=self._monitor, name="peaksMCP-cpu-budget", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop CPU monitoring and release any waiting submitter."""
        self._stop.set()
        self._gate.set()
        if self._thread:
            self._thread.join(timeout=max(2.0, self.sample_interval_s * 2))

    def wait_for_capacity(
        self,
        timeout: float | None = None,
        cancel_event: threading.Event | None = None,
    ) -> bool:
        """Wait until CPU capacity is available or cancellation is requested."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if cancel_event and cancel_event.is_set():
                return False
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                return False
            if self._gate.wait(0.1 if remaining is None else min(0.1, remaining)):
                return True

    def _record_sample(self, sample: float, timestamp: float | None = None) -> None:
        """Record one sample and update the hysteresis gate."""
        now = time.monotonic() if timestamp is None else timestamp
        with self._sample_lock:
            self._samples.append((now, sample))
            self._all_samples.append(sample)
            cutoff = now - self.moving_window_s
            while self._samples and self._samples[0][0] < cutoff:
                self._samples.popleft()
            moving_average = statistics.fmean(value for _time, value in self._samples)
        if sample >= self.cpu_limit_percent or moving_average >= self.cpu_limit_percent:
            self._gate.clear()
        elif moving_average <= self.resume_percent:
            self._gate.set()

    def _monitor(self) -> None:
        psutil.cpu_percent(interval=None)
        while not self._stop.wait(self.sample_interval_s):
            sample = float(psutil.cpu_percent(interval=None))
            self._record_sample(sample)

    @staticmethod
    def configure_worker_threads() -> None:
        """Limit common native scientific runtimes to one worker thread."""
        for key in _THREAD_ENV_KEYS:
            os.environ[key] = "1"
