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
    assert "default" in list_profiles()
    with pytest.raises(ValidationError):
        Profile.model_validate({"name": "bad", "unknown": True})


def test_version_and_profile_cli(capsys):
    main(["version"])
    assert capsys.readouterr().out.strip() == "0.1.0"
    main(["profiles", "show", "default"])
    assert json.loads(capsys.readouterr().out)["name"] == "default"

