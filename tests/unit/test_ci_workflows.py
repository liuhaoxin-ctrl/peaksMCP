"""Exercise the CI drift gate in tiny temporary repositories, without a build."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[2]
FRONTEND = Path("peaksMCP/extensions/jupyterlab")


def _workflow(name):
    return yaml.safe_load((ROOT / ".github/workflows" / name).read_text())


@pytest.mark.parametrize("change", ["clean", "modified", "deleted", "new_chunk", "staged", "lib", "unrelated"])
def test_ci_detects_tracked_and_untracked_frontend_drift(tmp_path, change):
    if shutil.which("git") is None or shutil.which("bash") is None:
        pytest.skip("git and bash are required to exercise the workflow")
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)

    def git(*args):
        return subprocess.run(["git", *args], cwd=tmp_path, env=env, check=True, capture_output=True, text=True)

    git("init", "-q")
    git("config", "core.hooksPath", os.devnull)
    bundle = tmp_path / FRONTEND / "labextension/static/bundle.js"
    library = tmp_path / FRONTEND / "lib/index.js"
    for file in (bundle, library):
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("// baseline\n")
    git("add", ".")
    git("-c", "user.name=peaksMCP Tests", "-c", "user.email=tests@example.invalid", "-c", "commit.gpgsign=false", "commit", "-qm", "fixture")
    if change in {"modified", "staged"}:
        bundle.write_text("// changed\n")
        if change == "staged":
            git("add", ".")
    elif change == "deleted":
        bundle.unlink()
    elif change == "new_chunk":
        bundle.with_name("new-hash.js").write_text("// new chunk\n")
    elif change == "lib":
        library.write_text("// changed\n")
    elif change == "unrelated":
        (tmp_path / "notes.txt").write_text("unrelated\n")

    steps = _workflow("ci.yml")["jobs"]["test"]["steps"]
    gate = next(step for step in steps if step.get("id") == "check_frontend_bundle")
    assert gate.get("working-directory", ".") == "."
    assert any(step.get("working-directory") == str(FRONTEND) for step in steps)
    result = subprocess.run(["bash", "-e", "-c", gate["run"]], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=5)
    assert (result.returncode == 0) == (change in {"clean", "unrelated"}), result.stdout + result.stderr


def test_plugin_release_preserves_required_hidden_configuration():
    steps = _workflow("release.yml")["jobs"]["build"]["steps"]
    upload = next(step for step in steps if step.get("with", {}).get("name") == "peaksMCP-claude-plugin")
    settings = upload["with"]
    assert settings["include-hidden-files"] is True
    assert settings["if-no-files-found"] == "error"
    assert settings["path"].rstrip("/") == "claude_plugin"
    plugin = ROOT / settings["path"]
    assert json.loads((plugin / ".claude-plugin/plugin.json").read_text())["name"] == "peaksMCP"
    assert "peaksMCP" in json.loads((plugin / ".mcp.json").read_text())["mcpServers"]
    assert (plugin / "skills/peaks-analysis/SKILL.md").is_file()
