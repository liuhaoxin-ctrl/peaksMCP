from __future__ import annotations

import pytest

from peaksMCP.app.profiles import Profile
from peaksMCP.app.runtime import RuntimeSupervisor


class _KernelResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, str]:
        return {"execution_state": "idle"}


def test_restart_all_passes_previous_kernel_generation(monkeypatch):
    supervisor = RuntimeSupervisor(Profile())
    supervisor.kernel_id = "kernel-1"
    observed: dict[str, object] = {}

    monkeypatch.setattr(supervisor, "_mcp_generation", lambda: "old-generation")
    monkeypatch.setattr(supervisor, "_comm_connected", lambda: True)
    monkeypatch.setattr(supervisor, "execute_kernel", lambda *_args, **_kwargs: {})

    def wait_ready(*, timeout, require_comm, previous_generation):
        observed.update(
            timeout=timeout,
            require_comm=require_comm,
            previous_generation=previous_generation,
        )
        return {"ready": True}

    monkeypatch.setattr(supervisor, "wait_ready", wait_ready)
    assert supervisor.restart_kernel(timeout=30, require_comm=True)["ready"]
    assert observed == {
        "timeout": 30,
        "require_comm": True,
        "previous_generation": "old-generation",
    }


def test_restart_all_falls_back_to_rest_when_frontend_offline(monkeypatch):
    """require_comm=True with no live Comm must degrade to a REST restart instead
    of waiting for a frontend coordination that can never happen."""
    supervisor = RuntimeSupervisor(Profile())
    supervisor.kernel_id = "kernel-1"
    observed: dict[str, object] = {}

    monkeypatch.setattr(supervisor, "_mcp_generation", lambda: "old-generation")
    monkeypatch.setattr(supervisor, "_comm_connected", lambda: False)

    class _RestResponse:
        def raise_for_status(self) -> None:
            return None

    monkeypatch.setattr(
        "peaksMCP.app.runtime.httpx.post",
        lambda *_args, **_kwargs: _RestResponse(),
    )
    monkeypatch.setattr(supervisor, "execute_kernel", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("execute_kernel must not run on REST fallback")))

    def wait_ready(*, timeout, require_comm, previous_generation):
        observed.update(
            timeout=timeout,
            require_comm=require_comm,
            previous_generation=previous_generation,
        )
        return {"ready": True}

    monkeypatch.setattr(supervisor, "wait_ready", wait_ready)
    assert supervisor.restart_kernel(timeout=30, require_comm=True)["ready"]
    assert observed == {
        "timeout": 30,
        "require_comm": False,
        "previous_generation": "old-generation",
    }


def test_wait_ready_rejects_old_mcp_generation(monkeypatch):
    supervisor = RuntimeSupervisor(Profile())
    supervisor.kernel_id = "kernel-1"
    generations = iter(("old-generation", "new-generation"))

    async def probe(*_args, **_kwargs):
        generation = next(generations)
        return {
            "ok": True,
            "tool_count": 12,
            "status": {
                "kernel_instance_id": generation,
                "extension_loaded": True,
                "comm_connected": True,
            },
        }

    monkeypatch.setattr("peaksMCP.app.runtime.httpx.get", lambda *_args, **_kwargs: _KernelResponse())
    monkeypatch.setattr("peaksMCP.app.runtime.check_http_mcp_server", probe)
    monkeypatch.setattr("peaksMCP.app.runtime.time.sleep", lambda _seconds: None)

    result = supervisor.wait_ready(
        timeout=2,
        require_comm=True,
        previous_generation="old-generation",
    )
    assert result["ready"] is True
    assert result["kernel_instance_id"] == "new-generation"
    assert result["stages"]["kernel_restarted"] is True


def test_dashboard_bind_failure_is_propagated(monkeypatch):
    supervisor = RuntimeSupervisor(Profile())

    class FailingServer:
        started = False
        should_exit = False

        def __init__(self, _config) -> None:
            pass

        def run(self) -> None:
            raise SystemExit(1)

    monkeypatch.setattr("peaksMCP.app.runtime.uvicorn.Server", FailingServer)
    with pytest.raises(RuntimeError, match="dashboard failed to start"):
        supervisor._start_dashboard(timeout=0.5)


def test_remote_dashboard_requires_explicit_opt_in():
    supervisor = RuntimeSupervisor(Profile(dashboard={"host": "0.0.0.0", "port": 8765}))
    with pytest.raises(ValueError, match="allow_remote=true"):
        supervisor._start_dashboard(timeout=0.1)
