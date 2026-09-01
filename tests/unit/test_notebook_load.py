"""Run the actual generated Load control code in a simulated kernel namespace."""

from __future__ import annotations

import builtins
from types import SimpleNamespace

import pytest

from peaksMCP.app import api


class _Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class _Kernel:
    def __init__(self, *, finish_after=1, load_error=None, bridge_error=None, reply_override=None):
        self.finish_after = finish_after
        self.load_error = load_error
        self.bridge_error = bridge_error
        self.reply_override = reply_override or {}
        self.queued = []
        self.codes = []
        self.cells = []
        self.paths = []
        self.slots = []
        self.polls = 0
        self.old_data = object()
        self.new_data = object()
        self.namespace = {"data": self.old_data}
        self.namespace["get_ipython"] = lambda: SimpleNamespace(user_ns=self.namespace)
        self.namespace["__builtins__"] = {**vars(builtins), "__import__": self._import}

    def _import(self, name, *args, **kwargs):
        if name == "threading":
            def thread(*, target, daemon):
                return SimpleNamespace(start=lambda: self.queued.append(target))
            return SimpleNamespace(Thread=thread)
        if name.endswith("jupyter_mcp_extension"):
            return SimpleNamespace(get_server=lambda: SimpleNamespace(state=SimpleNamespace(bridge=self)))
        if name == "peaks":
            return SimpleNamespace(load=self._load)
        return builtins.__import__(name, *args, **kwargs)

    def _load(self, path):
        self.paths.append(path)
        if self.load_error:
            raise self.load_error
        return self.new_data

    def request(self, operation, payload, timeout):
        assert operation == "execute_code"
        if self.bridge_error:
            raise self.bridge_error
        code = payload["code"]
        self.cells.append(code)
        outputs = []
        try:
            exec(compile(code, "<visible-load-cell>", "exec"), self.namespace)
            success = True
        except Exception as exc:
            success = False
            outputs = [{"output_type": "error", "evalue": str(exc)}]
        return {
            "id": "cell-1", "cell_type": "code", "source": code,
            "execution_success": success, "saved": True, "outputs": outputs,
            **self.reply_override,
        }

    def execute_kernel(self, code, timeout=10):
        self.codes.append(code)
        if code.startswith("if get_ipython"):
            self.polls += 1
            if self.finish_after is not None and self.polls > self.finish_after and self.queued:
                self.queued.pop(0)()
        try:
            exec(compile(code, "<supervisor-control>", "exec"), self.namespace)
        except Exception as exc:
            raise RuntimeError(str(exc)) from exc
        if code.startswith("def _peaksMCP_start_load"):
            self.slots.extend(key for key in self.namespace if key.startswith("_peaksMCP_load_"))
        return {"status": "ok"}


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    simulated = _Clock()
    monkeypatch.setattr(api, "time", simulated)
    return simulated


def _assert_clean(kernel):
    assert not any(key.startswith("_peaksMCP_load_") for key in kernel.namespace)
    assert "_peaksMCP_start_load" not in kernel.namespace


def test_load_waits_for_its_cell_even_when_old_data_exists():
    kernel = _Kernel(finish_after=2)
    path = '/data/测量 "quoted"\\scan.nc'
    assert api.load_into_notebook(kernel, path, timeout=2)
    assert kernel.polls == 3
    assert kernel.namespace["data"] is kernel.new_data
    assert kernel.paths == [path]
    assert len(kernel.cells) == 1
    assert kernel.cells[0].startswith("from peaks import load\ndata = load(")
    _assert_clean(kernel)


@pytest.mark.parametrize("failure", [FileNotFoundError("missing.nc"), ValueError("corrupt NetCDF")])
def test_failed_load_does_not_report_success_from_previous_data(failure):
    kernel = _Kernel(load_error=failure)
    assert not api.load_into_notebook(kernel, "/data/missing.nc", timeout=2)
    assert kernel.namespace["data"] is kernel.old_data
    assert len(kernel.cells) == 1
    _assert_clean(kernel)


@pytest.mark.parametrize("failure", [RuntimeError("Comm disconnected"), TimeoutError("Comm timed out")])
def test_bridge_failure_cannot_report_load_success(failure):
    kernel = _Kernel(bridge_error=failure)
    assert not api.load_into_notebook(kernel, "/data/file.nc", timeout=2)
    assert kernel.cells == []
    _assert_clean(kernel)


@pytest.mark.parametrize("reply_override", [
    {"execution_success": False}, {"execution_success": None},
    {"source": "data = old_data"}, {"cell_type": "markdown"}, {"id": None},
    {"saved": False, "save_error": "disk full"},
    {"outputs": [{"output_type": "error", "evalue": "failed"}]},
])
def test_load_requires_successful_reply_for_the_exact_code_cell(reply_override):
    kernel = _Kernel(reply_override=reply_override)
    assert not api.load_into_notebook(kernel, "/data/file.nc", timeout=2)
    _assert_clean(kernel)


def test_load_timeout_cleans_its_job_without_replaying_or_cancelling_the_cell():
    kernel = _Kernel(finish_after=None)
    assert not api.load_into_notebook(kernel, "/data/file.nc", timeout=1)
    assert kernel.namespace["data"] is kernel.old_data
    assert len(kernel.queued) == 1
    _assert_clean(kernel)
    # A late execution may still complete; it must not resurrect a stale job.
    kernel.queued.pop()()
    assert kernel.namespace["data"] is kernel.new_data
    assert len(kernel.cells) == 1
    _assert_clean(kernel)


def test_second_load_cannot_reuse_the_first_operations_success():
    kernel = _Kernel(finish_after=0)
    assert api.load_into_notebook(kernel, "/data/first.nc", timeout=2)
    kernel.load_error = ValueError("second load failed")
    assert not api.load_into_notebook(kernel, "/data/second.nc", timeout=2)
    assert len(set(kernel.slots)) == 2
    assert kernel.paths == ["/data/first.nc", "/data/second.nc"]
    _assert_clean(kernel)


def test_busy_kernel_probe_can_time_out_before_load_completes(monkeypatch):
    kernel = _Kernel(finish_after=0)
    execute = kernel.execute_kernel
    calls = []

    def temporarily_busy(code, timeout):
        if code.startswith("if get_ipython") and not calls:
            calls.append("busy")
            raise TimeoutError("kernel is executing the Load cell")
        return execute(code, timeout)

    monkeypatch.setattr(kernel, "execute_kernel", temporarily_busy)
    assert api.load_into_notebook(kernel, "/data/file.nc", timeout=2)
    assert calls == ["busy"]
    _assert_clean(kernel)


@pytest.mark.parametrize("reply", [{}, {"status": "aborted"}])
def test_invalid_kernel_acknowledgement_is_not_success(reply):
    kernel = SimpleNamespace(execute_kernel=lambda *_args, **_kwargs: reply)
    assert not api.load_into_notebook(kernel, "/data/file.nc", timeout=1)
