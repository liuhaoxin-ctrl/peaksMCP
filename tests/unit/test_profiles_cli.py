from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from peaksMCP.app.profiles import Profile, list_profiles, load_profile
from peaksMCP.cli import main


def test_default_profile_and_strict_validation():
    profile = load_profile()
    assert profile.mcp.port == 8123
    assert profile.mcp.allow_remote is False
    assert "default" in list_profiles()
    with pytest.raises(ValidationError):
        Profile.model_validate({"name": "bad", "unknown": True})


def test_version_and_profile_cli(capsys):
    main(["version"])
    assert capsys.readouterr().out.strip() == "0.1.0"
    main(["profiles", "show", "default"])
    assert json.loads(capsys.readouterr().out)["name"] == "default"


def test_restart_without_component_restarts_whole_stack(monkeypatch, capsys):
    """`peaksMCP restart` (no component) stops the dashboard host and starts a
    fresh one — the host then manages JupyterLab/kernel/MCP internally."""
    from peaksMCP import cli

    calls: list[str] = []

    def fake_kill(pid, _sig):
        calls.append(f"kill:{pid}")
        raise ProcessLookupError()  # process already gone -> kill loop exits

    monkeypatch.setattr(cli.os, "kill", fake_kill)

    def fake_read_runfile():
        return {"pid": 4242, "stale": False}

    def fake_ensure(_args):
        calls.append("ensure")
        return {"dashboard_url": "http://127.0.0.1:8765", "kernel_id": None}

    monkeypatch.setattr("peaksMCP.observability.read_runfile", fake_read_runfile)
    monkeypatch.setattr(cli, "_ensure_host", fake_ensure)

    main(["restart"])
    assert calls[0] == "kill:4242"
    assert calls[-1] == "ensure"
    assert json.loads(capsys.readouterr().out)["ready"] is True


def test_restart_with_component_still_posts_to_dashboard(monkeypatch, capsys):
    """`peaksMCP restart kernel` keeps the kernel-side component restart."""
    from peaksMCP import cli

    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"ready": True}

    def fake_runfile(_required=True):
        return {"dashboard_url": "http://127.0.0.1:8765", "dashboard_token": "tok"}

    def fake_post(*_args, **_kwargs):
        return _Response()

    monkeypatch.setattr(cli, "_runfile", fake_runfile)
    monkeypatch.setattr(cli.httpx, "post", fake_post)
    main(["restart", "kernel"])
    assert json.loads(capsys.readouterr().out)["ready"] is True


def test_dash_timeout_terminates_spawned_host(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from peaksMCP import cli

    class Process:
        pid = 99999
        returncode = None

        def __init__(self):
            self.terminated = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            return 0

    process = Process()
    monkeypatch.setenv("PEAKSMCP_HOME", str(tmp_path))
    monkeypatch.setattr(cli, "_runfile", lambda _required=False: None)
    monkeypatch.setattr(cli, "_recover_discovery", lambda _args: None)
    monkeypatch.setattr(cli, "_listener_on_port", lambda _port: [])
    monkeypatch.setattr(cli.subprocess, "Popen", lambda *_a, **_k: process)
    monkeypatch.setattr(cli, "_terminate_spawned_supervisor", lambda p: p.terminate())
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)

    args = SimpleNamespace(profile="default", timeout=0.0, notebook=None)
    with pytest.raises(SystemExit, match="did not answer"):
        cli._ensure_host(args)
    assert process.terminated is True


def test_dash_opens_once_dashboard_answers(monkeypatch, capsys):
    """peaksMCP dash waits only for the dashboard (host) to answer — not for
    Jupyter readiness — then opens the operator console."""
    from peaksMCP import cli

    class _Response:
        status_code = 200

    runfile = {
        "pid": 4242,
        "dashboard_url": "http://127.0.0.1:8765",
        "dashboard_token": "secret",
        "stale": False,
        "token": "jupyter-token",
        "notebook_path": "peaksMCP-runtime.ipynb",
        "jupyter_url": "http://127.0.0.1:8888",
    }
    monkeypatch.setattr(cli, "_runfile", lambda _required=False: runfile)
    monkeypatch.setattr(cli.httpx, "get", lambda *_a, **_k: _Response())
    opened: list[str] = []
    monkeypatch.setattr(cli.webbrowser, "open", lambda url: opened.append(url))
    monkeypatch.setattr(
        cli.subprocess,
        "Popen",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("an existing host must not be spawned again")
        ),
    )

    main(["dash", "--timeout", "1"])
    assert opened and "8765" in opened[0]
    assert "token=secret" in capsys.readouterr().out


def test_dash_replaces_live_host_when_notebook_is_explicitly_requested(
    monkeypatch, tmp_path
):
    from types import SimpleNamespace

    from peaksMCP import cli

    class Response:
        status_code = 200

    class Process:
        returncode = None

        @staticmethod
        def poll():
            return None

    old = {
        "pid": 4242,
        "profile": "default",
        "notebook_path": "peaksMCP-runtime.ipynb",
        "stale": False,
    }
    requested = {
        "pid": 5252,
        "profile": "default",
        "dashboard_url": "http://127.0.0.1:8765",
        "dashboard_token": "dashboard-token",
        "jupyter_url": "http://127.0.0.1:8888",
        "token": "jupyter-token",
        "notebook_path": "peaksMCP-snapshot-123.ipynb",
        "stale": False,
    }
    reads = iter((old, requested))
    stopped: list[int] = []
    spawned: list[str | None] = []

    monkeypatch.setenv("PEAKSMCP_HOME", str(tmp_path))
    monkeypatch.setattr(cli, "_runfile", lambda _required=False: next(reads))
    monkeypatch.setattr(cli, "_recover_discovery", lambda _args: None)
    monkeypatch.setattr(cli, "_listener_on_port", lambda _port: [])
    monkeypatch.setattr(cli, "_terminate_supervisor", lambda pid: stopped.append(pid))
    monkeypatch.setattr(
        cli,
        "_spawn_host_process",
        lambda args: (spawned.append(args.notebook), Process())[1],
    )
    monkeypatch.setattr(cli.httpx, "get", lambda *_args, **_kwargs: Response())

    args = SimpleNamespace(
        profile="default",
        timeout=1.0,
        notebook="peaksMCP-snapshot-123.ipynb",
    )
    data = cli._ensure_host(args)

    assert stopped == [4242]
    assert spawned == ["peaksMCP-snapshot-123.ipynb"]
    assert data["notebook_path"] == "peaksMCP-snapshot-123.ipynb"


def test_dash_reuses_live_host_for_same_requested_notebook(monkeypatch):
    from types import SimpleNamespace

    from peaksMCP import cli

    class Response:
        status_code = 200

    current = {
        "pid": 4242,
        "profile": "default",
        "dashboard_url": "http://127.0.0.1:8765",
        "dashboard_token": "dashboard-token",
        "jupyter_url": "http://127.0.0.1:8888",
        "token": "jupyter-token",
        "notebook_path": "peaksMCP-snapshot-123.ipynb",
        "stale": False,
    }
    monkeypatch.setattr(cli, "_runfile", lambda _required=False: current)
    monkeypatch.setattr(cli.httpx, "get", lambda *_args, **_kwargs: Response())
    monkeypatch.setattr(
        cli,
        "_terminate_supervisor",
        lambda _pid: (_ for _ in ()).throw(
            AssertionError("a matching host must not be stopped")
        ),
    )
    monkeypatch.setattr(
        cli,
        "_spawn_host_process",
        lambda _args: (_ for _ in ()).throw(
            AssertionError("a matching host must not be replaced")
        ),
    )

    args = SimpleNamespace(
        profile="default",
        timeout=1.0,
        notebook="peaksMCP-snapshot-123.ipynb",
    )

    assert cli._ensure_host(args) is current


def test_dash_replaces_live_host_for_requested_profile(monkeypatch, tmp_path):
    """A live host on a different profile is replaced, never silently reused."""
    from types import SimpleNamespace

    from peaksMCP import cli

    class Response:
        status_code = 200

    class Process:
        returncode = None

        @staticmethod
        def poll():
            return None

    old = {
        "pid": 4242,
        "profile": "research",
        "notebook_path": "peaksMCP-runtime.ipynb",
        "stale": False,
    }
    requested = {
        "pid": 5252,
        "profile": "default",
        "dashboard_url": "http://127.0.0.1:8765",
        "dashboard_token": "dashboard-token",
        "jupyter_url": "http://127.0.0.1:8888",
        "token": "jupyter-token",
        "notebook_path": "peaksMCP-runtime.ipynb",
        "stale": False,
    }
    reads = iter((old, requested))
    stopped: list[int] = []
    spawned: list[str] = []

    monkeypatch.setenv("PEAKSMCP_HOME", str(tmp_path))
    monkeypatch.setattr(cli, "_runfile", lambda _required=False: next(reads))
    monkeypatch.setattr(cli, "_recover_discovery", lambda _args: None)
    monkeypatch.setattr(cli, "_listener_on_port", lambda _port: [])
    monkeypatch.setattr(cli, "_terminate_supervisor", lambda pid: stopped.append(pid))
    monkeypatch.setattr(
        cli,
        "_spawn_host_process",
        lambda args: (spawned.append(args.notebook), Process())[1],
    )
    monkeypatch.setattr(cli.httpx, "get", lambda *_args, **_kwargs: Response())

    args = SimpleNamespace(profile="default", timeout=1.0, notebook=None)
    data = cli._ensure_host(args)

    assert stopped == [4242]
    assert spawned == [None]
    assert data["pid"] == 5252


@pytest.mark.parametrize(
    "requested",
    [
        "peaksMCP-snapshot-123.ipynb",
        "./peaksMCP-snapshot-123.ipynb",
        # Absolute path to the same checkout file must also count as a match.
        pytest.param(str(Path(__file__).resolve().parents[2] / "peaksMCP-snapshot-123.ipynb")),
    ],
)
def test_dash_reuses_host_for_equivalent_notebook_spelling(
    monkeypatch, requested
):
    """Equivalent spellings of the same workspace notebook must not restart
    the host (./x.ipynb, x.ipynb and the absolute path all match)."""
    from types import SimpleNamespace

    from peaksMCP import cli

    class Response:
        status_code = 200

    current = {
        "pid": 4242,
        "profile": "default",
        "dashboard_url": "http://127.0.0.1:8765",
        "dashboard_token": "dashboard-token",
        "jupyter_url": "http://127.0.0.1:8888",
        "token": "jupyter-token",
        "notebook_path": "peaksMCP-snapshot-123.ipynb",
        "stale": False,
    }
    monkeypatch.setattr(cli, "_runfile", lambda _required=False: current)
    monkeypatch.setattr(cli.httpx, "get", lambda *_args, **_kwargs: Response())
    monkeypatch.setattr(
        cli,
        "_terminate_supervisor",
        lambda _pid: (_ for _ in ()).throw(
            AssertionError("an equivalent workspace must not be replaced")
        ),
    )
    monkeypatch.setattr(
        cli,
        "_spawn_host_process",
        lambda _args: (_ for _ in ()).throw(
            AssertionError("an equivalent workspace must not be spawned")
        ),
    )

    args = SimpleNamespace(profile="default", timeout=1.0, notebook=requested)

    assert cli._ensure_host(args) is current


def test_dash_adopts_live_host_without_runfile(monkeypatch, capsys):
    """A live host whose runfile was removed is adopted (no duplicate spawn):
    ``_recover_discovery`` republishes discovery, so dash must not spawn."""
    from types import SimpleNamespace

    from peaksMCP import cli

    class _Response:
        status_code = 200

    runfile = {
        "pid": 4242,
        "dashboard_url": "http://127.0.0.1:8765",
        "dashboard_token": "secret",
        "stale": False,
        "token": "jupyter-token",
        "notebook_path": "peaksMCP-runtime.ipynb",
        "jupyter_url": "http://127.0.0.1:8888",
    }
    # First read finds no runfile; the recovered host then publishes one.
    reads = iter([None, runfile])
    monkeypatch.setattr(cli, "_runfile", lambda _required=False: next(reads))
    recovered: list[dict] = []

    def fake_recover(_args):
        recovered.append(True)
        return runfile

    monkeypatch.setattr(cli, "_recover_discovery", fake_recover)
    monkeypatch.setattr(cli.httpx, "get", lambda *_a, **_k: _Response())
    monkeypatch.setattr(
        cli.subprocess,
        "Popen",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("an adoptable live host must not be spawned again")
        ),
    )

    args = SimpleNamespace(profile="default", timeout=1.0, notebook=None)
    data = cli._ensure_host(args)
    assert data["pid"] == 4242
    assert recovered == [True]


def test_dash_busy_port_without_runfile_raises_clear_error(monkeypatch):
    """No runfile + an unadoptable port holder -> clear message, never a
    duplicate host that would crash on the already-bound dashboard port."""
    from types import SimpleNamespace

    from peaksMCP import cli

    monkeypatch.setattr(cli, "_runfile", lambda _required=False: None)
    monkeypatch.setattr(cli, "_recover_discovery", lambda _args: None)
    monkeypatch.setattr(
        cli,
        "_listener_on_port",
        lambda _port: [(4242, "/usr/bin/python -m peaksMCP _serve --profile default")],
    )

    args = SimpleNamespace(profile="default", timeout=1.0, notebook=None)
    with pytest.raises(SystemExit, match="already in use by pid 4242"):
        cli._ensure_host(args)


def test_stale_jupyter_cleanup_requires_matching_process_create_time(monkeypatch):
    from peaksMCP import cli

    signals: list[tuple[int, int]] = []

    class Process:
        @staticmethod
        def create_time():
            return 1234.5

    monkeypatch.setattr(cli.psutil, "Process", lambda _pid: Process())
    monkeypatch.setattr(cli.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(cli.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)

    cli._cleanup_stale_jupyter_tree(
        {"jupyter_pid": 4242, "jupyter_process_create_time": 1234.5}
    )

    assert signals == [
        (4242, cli.signal.SIGTERM),
        (4242, cli.signal.SIGKILL),
    ]


@pytest.mark.parametrize(
    "state",
    [
        {"jupyter_pid": 4242},
        {"jupyter_pid": 4242, "jupyter_process_create_time": 9999.0},
    ],
)
def test_stale_jupyter_cleanup_rejects_missing_or_mismatched_timestamp(
    monkeypatch, state
):
    from peaksMCP import cli

    class Process:
        @staticmethod
        def create_time():
            return 1234.5

    monkeypatch.setattr(cli.psutil, "Process", lambda _pid: Process())
    monkeypatch.setattr(
        cli.os,
        "killpg",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("an unverified process must not be signalled")
        ),
    )

    cli._cleanup_stale_jupyter_tree(state)


def test_stale_jupyter_cleanup_revalidates_before_sigkill(monkeypatch):
    from peaksMCP import cli

    create_times = iter((1234.5, 9999.0))
    signals: list[tuple[int, int]] = []

    class Process:
        @staticmethod
        def create_time():
            return next(create_times)

    monkeypatch.setattr(cli.psutil, "Process", lambda _pid: Process())
    monkeypatch.setattr(cli.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(cli.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)

    cli._cleanup_stale_jupyter_tree(
        {"jupyter_pid": 4242, "jupyter_process_create_time": 1234.5}
    )

    assert signals == [(4242, cli.signal.SIGTERM)]


def test_status_adopts_live_host_without_runfile(monkeypatch, capsys):
    """``peaksMCP status`` reports the recovered host instead of STOPPED when
    the runfile is missing but the dashboard host is still alive."""
    from peaksMCP import cli

    class _Response:
        def json(self):
            return {"status": "RUNNING", "profile": "default"}

    runfile = {
        "pid": 4242,
        "dashboard_url": "http://127.0.0.1:8765",
        "dashboard_token": "secret",
        "stale": False,
    }
    monkeypatch.setattr(cli, "_runfile", lambda _required=False: None)
    monkeypatch.setattr(cli, "_recover_discovery", lambda _args: runfile)
    monkeypatch.setattr(cli.httpx, "get", lambda *_a, **_k: _Response())

    main(["status"])
    assert json.loads(capsys.readouterr().out)["status"] == "RUNNING"


@pytest.mark.parametrize(
    "autostart,mcp_state,expected",
    [(True, "error", False), (True, "ready", True), (False, "error", True)],
)
def test_launch_readiness_respects_autostart(
    monkeypatch, autostart, mcp_state, expected
):
    from peaksMCP import cli

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "components": {
                    "supervisor": {"state": "ready"},
                    "jupyter": {"state": "ready"},
                    "kernel": {"state": "ready"},
                    "extension": {"state": "ready"},
                    "comm": {"state": "degraded"},
                    "mcp": {"state": mcp_state},
                }
            }

    monkeypatch.setattr(cli.httpx, "get", lambda *_args, **_kwargs: Response())
    ready, _status = cli._launch_readiness(
        {
            "dashboard_url": "http://127.0.0.1:8765",
            "dashboard_token": "secret",
            "mcp_autostart": autostart,
        }
    )
    assert ready is expected
