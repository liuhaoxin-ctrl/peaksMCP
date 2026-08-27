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

