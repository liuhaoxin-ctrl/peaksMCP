"""CPU budget and native thread controls for scientific batch jobs."""

from __future__ import annotations

import os
import statistics
import threading
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
    _samples: list[float] = field(default_factory=list, init=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _gate: threading.Event = field(default_factory=threading.Event, init=False)

    def __post_init__(self) -> None:
        if not 1 <= self.cpu_limit_percent <= 100:
            raise ValueError("cpu_limit_percent must be between 1 and 100")
        if not 0 <= self.resume_percent < self.cpu_limit_percent:
            raise ValueError("resume_percent must be lower than cpu_limit_percent")
        self._gate.set()

    @property
    def max_workers(self) -> int:
        """Return the bounded default process count."""
        count = os.cpu_count() or 1
        return max(1, min(count, int(count * self.worker_fraction)))

    @property
    def average_cpu_percent(self) -> float:
        """Return the arithmetic mean of monitor samples."""
        return statistics.fmean(self._samples) if self._samples else 0.0

    @property
    def peak_cpu_percent(self) -> float:
        """Return the highest monitor sample."""
        return max(self._samples, default=0.0)

    def start(self) -> None:
        """Start background CPU monitoring."""
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._monitor, name="peaksMCP-cpu-budget", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop CPU monitoring and release any waiting submitter."""
        self._stop.set()
        self._gate.set()
        if self._thread:
            self._thread.join(timeout=max(2.0, self.sample_interval_s * 2))

    def wait_for_capacity(self, timeout: float | None = None) -> bool:
        """Wait until system CPU is below the resume threshold."""
        return self._gate.wait(timeout)

    def _monitor(self) -> None:
        psutil.cpu_percent(interval=None)
        while not self._stop.wait(self.sample_interval_s):
            sample = float(psutil.cpu_percent(interval=None))
            self._samples.append(sample)
            if sample >= self.cpu_limit_percent:
                self._gate.clear()
            elif sample <= self.resume_percent:
                self._gate.set()

    @staticmethod
    def configure_worker_threads() -> None:
        """Limit common native scientific runtimes to one worker thread."""
        for key in _THREAD_ENV_KEYS:
            os.environ[key] = "1"

