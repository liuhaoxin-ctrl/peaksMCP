from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from peaksMCP.app.profiles import Profile, list_profiles, load_profile
from peaksMCP.cli import main


def test_default_profile_and_strict_validation():
    profile = load_profile()
    assert profile.mcp.port == 8123
    assert profile.mcp.mode == "safe"
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
    """`peaksMCP restart` (no component) must stop the running supervisor and
    launch a fresh stack — the same level as `launch`."""
    from peaksMCP import cli

    calls: list[str] = []

    def fake_kill(pid, _sig):
        calls.append(f"kill:{pid}")
        raise ProcessLookupError()  # process already gone -> kill loop exits

    monkeypatch.setattr(cli.os, "kill", fake_kill)

    def fake_read_runfile():
        return {"pid": 4242, "stale": False}

    monkeypatch.setattr("peaksMCP.observability.read_runfile", fake_read_runfile)
    monkeypatch.setattr(cli, "command_launch", lambda args: calls.append("launch"))

    main(["restart"])
    assert calls[0] == "kill:4242"
    assert calls[-1] == "launch"


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


def test_launch_timeout_terminates_spawned_supervisor(monkeypatch, tmp_path):
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
    monkeypatch.setattr(cli.subprocess, "Popen", lambda *_a, **_k: process)

    with pytest.raises(SystemExit, match="did not become ready"):
        main(["launch", "--timeout", "0"])
    assert process.terminated is True


def test_launch_waits_for_mcp_readiness_even_when_runfile_exists(
    monkeypatch, capsys
):
    """A runfile proves process ownership, not that MCP initialization finished."""
    from peaksMCP import cli

    runfile = {
        "pid": 4242,
        "profile": "default",
        "dashboard_url": "http://127.0.0.1:8765",
        "dashboard_token": "secret",
        "mcp_autostart": True,
        "stale": False,
    }
    probes = iter(
        [
            (False, {"components": {"mcp": {"state": "error"}}}),
            (
                True,
                {
                    "components": {
                        name: {"state": "ready"}
                        for name in (
                            "supervisor",
                            "jupyter",
                            "kernel",
                            "extension",
                            "mcp",
                        )
                    }
                },
            ),
        ]
    )
    monkeypatch.setattr(cli, "_runfile", lambda _required=False: runfile)
    monkeypatch.setattr(cli, "_launch_readiness", lambda _data: next(probes))
    monkeypatch.setattr(
        cli.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("an existing supervisor must not be spawned again")
        ),
    )
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)

    main(["launch", "--timeout", "1"])

    payload = json.loads(capsys.readouterr().out)
    assert payload["ready"] is True
    assert payload["components"]["mcp"]["state"] == "ready"
    assert "dashboard_token" not in payload


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
