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


def test_mcp_server_refuses_non_loopback_binding():
    """The in-kernel MCP listener has no auth; binding to a non-loopback host
    must be refused unless ``allow_remote`` is explicitly enabled."""
    from peaksMCP.server.jupyter_peaks.backend import SharedState
    from peaksMCP.server.jupyter_peaks.mcp_server import JupyterPeaksMCPServer

    class FakeIPython:
        user_ns = {}

    state = SharedState(FakeIPython())
    server = JupyterPeaksMCPServer(state, host="0.0.0.0", port=9999)
    with pytest.raises(RuntimeError, match="refuses to bind"):
        server.start()
    # Loopback is fine even without allow_remote.
    JupyterPeaksMCPServer(state, host="127.0.0.1", port=9998)
    # allow_remote=True lets the operator bind (they accept auth/TLS duty).
    JupyterPeaksMCPServer(state, host="0.0.0.0", port=9997, allow_remote=True)


def test_kernelspec_reinstalled_when_profile_mode_changes(monkeypatch):
    """A profile switched from dangerous back to safe must reinstall the
    kernelspec so the old dangerous startup script is not reused."""

    supervisor = RuntimeSupervisor(Profile(mcp={"mode": "safe"}))
    supervisor.kernel_id = "kernel-1"
    calls: list[str] = []

    monkeypatch.setattr("peaksMCP.app.runtime.kernel_installed", lambda _name: True)
    monkeypatch.setattr(
        "peaksMCP.app.runtime.kernel_spec_state",
        lambda _name: {"mode": "dangerous", "autostart": True},  # stale: dangerous
    )

    def fake_install(profile, replace=False):
        calls.append(f"install:replace={replace}")

    monkeypatch.setattr("peaksMCP.app.runtime.install_kernel", fake_install)

    # Stop before start() reaches the JupyterLab spawn.
    class _Stop(RuntimeError):
        pass

    def _stop(*_args, **_kwargs):
        raise _Stop()

    monkeypatch.setattr(supervisor, "_wait_jupyter", _stop)
    with pytest.raises(_Stop):
        supervisor.start()
    assert calls == ["install:replace=True"]


def test_startup_script_honors_autostart_and_allow_remote():
    """autostart:false must not start the MCP (no mode/start magic), and
    allow_remote must be embedded as a string env var (not a bare bool)."""
    import ast

    from peaksMCP.app.kernel import _startup

    off = Profile(mcp={"mode": "dangerous", "autostart": False, "allow_remote": False})
    src = _startup(off)
    ast.parse(src)  # generated script must compile
    assert "peaksMCP_start" not in src
    assert "peaksMCP_dangerous" not in src
    assert 'setdefault("PEAKSMCP_AUTOSTART", "false")' in src
    assert 'setdefault("PEAKSMCP_ALLOW_REMOTE", "false")' in src
    assert "load_ext" in src  # magics remain available

    on = Profile(mcp={"autostart": True})
    src_on = _startup(on)
    assert "peaksMCP_start" in src_on
    assert 'setdefault("PEAKSMCP_AUTOSTART", "true")' in src_on
