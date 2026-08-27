from __future__ import annotations

import asyncio

from peaksMCP.server.jupyter_peaks.backend import SharedState
from peaksMCP.server.jupyter_peaks.mcp_server import JupyterPeaksMCPServer


class FakeIPython:
    user_ns = {}


def names(server):
    return asyncio.run(server.tool_names())


def test_exact_tool_surface_all_exposed():
    server = JupyterPeaksMCPServer(SharedState(FakeIPython()))
    assert names(server) == sorted([
        "peaks_search_api", "peaks_get_api", "askuserquestion", "notebook_list_variables",
        "notebook_read_variable", "notebook_read_active_cell", "notebook_read_active_cell_output",
        "notebook_read_content", "notebook_move_cursor", "notebook_server_status",
        "notebook_kernel_status", "notebook_wait_for_kernel",
        "notebook_execute_code", "notebook_execute_active_cell", "notebook_add_cell",
        "notebook_delete_cell", "notebook_apply_patch",
    ])


def test_mode_changes_consent_policy_not_tool_surface():
    server = JupyterPeaksMCPServer(SharedState(FakeIPython()))
    assert len(names(server)) == 17
    for mode in ("unsafe", "dangerous", "safe"):
        server.set_mode(mode)
        assert len(names(server)) == 17
        assert {"notebook_execute_code", "notebook_execute_active_cell", "notebook_add_cell",
                "notebook_delete_cell", "notebook_apply_patch"} <= set(names(server))


def test_tool_metadata_is_nonempty():
    server = JupyterPeaksMCPServer(SharedState(FakeIPython()))
    tools = asyncio.run(server.mcp.list_tools())
    assert all(tool.title and tool.description for tool in tools)


def test_server_status_exposes_index_stale():
    """server_status must expose ``index_stale`` so the model can self-diagnose
    the INDEX_STALE_RESTART_REQUIRED condition instead of only search/get."""
    from unittest.mock import MagicMock

    from peaksMCP.server.jupyter_peaks.backend.notebook import NotebookBackend

    state = SharedState(FakeIPython())
    notebook = NotebookBackend(state)
    status = notebook.server_status()
    assert "index_stale" in status
    # No index built yet -> not stale.
    assert status["index_stale"] is False
    # An index whose fingerprint no longer matches is reported stale.
    stale_index = MagicMock()
    stale_index.entries = []
    stale_index.is_stale.return_value = True
    state.api_index = stale_index
    assert notebook.server_status()["index_stale"] is True
    # A fresh index reports not stale.
    fresh_index = MagicMock()
    fresh_index.entries = [object()]
    fresh_index.is_stale.return_value = False
    state.api_index = fresh_index
    status = notebook.server_status()
    assert status["index_stale"] is False
    assert status["api_count"] == 1

