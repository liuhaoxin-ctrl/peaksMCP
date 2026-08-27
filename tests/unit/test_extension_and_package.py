from __future__ import annotations

import json
from pathlib import Path

from peaksMCP.config import tool_metadata
from peaksMCP.ext_install import extension_source


def test_prebuilt_extension_has_manifest_and_remote_entry():
    source = extension_source()
    manifest = json.loads((source / "package.json").read_text())
    assert manifest["name"] == "peaksmcp-jupyterlab"
    assert any((source / "static").glob("remoteEntry.*.js"))


def test_frontend_bridge_releases_inactive_notebook_bindings():
    root = Path(__file__).parents[2]
    source = root.joinpath("peaksMCP/extensions/jupyterlab/src/index.ts").read_text()
    assert "let activePanel: NotebookPanel | null" in source
    assert "cleanupPanelBindings();" in source
    assert "active notebook changed" in source
    assert "kernelChanged.disconnect" in source
    assert "statusChanged.disconnect" in source
    assert "activeCellChanged.disconnect" in source
    assert "outputs.changed.disconnect" in source
    assert "_peaksMCPAttached" not in source


def test_every_declared_tool_has_curated_metadata():
    names = {
        "peaks_search_api", "peaks_get_api", "askuserquestion", "notebook_list_variables",
        "notebook_read_variable", "notebook_read_active_cell", "notebook_read_active_cell_output",
        "notebook_read_content", "notebook_move_cursor", "notebook_server_status",
        "notebook_kernel_status", "notebook_wait_for_kernel", "notebook_execute_code",
        "notebook_execute_active_cell", "notebook_add_cell", "notebook_delete_cell", "notebook_apply_patch",
    }
    for name in names:
        metadata = tool_metadata(name)
        assert metadata["title"] and metadata["description"]


def test_claude_plugin_and_skill_are_self_contained():
    root = Path(__file__).parents[2]
    plugin = json.loads(root.joinpath("claude_plugin/.claude-plugin/plugin.json").read_text())
    assert plugin["name"] == "peaksMCP"
    assert "stdio-proxy" in root.joinpath("claude_plugin/.mcp.json").read_text()
    assert "peaks_search_api" in root.joinpath("claude_plugin/skills/peaks-analysis/SKILL.md").read_text()
