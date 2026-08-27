from __future__ import annotations

import asyncio

from peaksMCP.server.jupyter_peaks.backend import SharedState
from peaksMCP.server.jupyter_peaks.mcp_server import JupyterPeaksMCPServer


class FakeIPython:
    user_ns = {}


def names(server):
    return asyncio.run(server.tool_names())


def test_exact_safe_tool_surface():
    server = JupyterPeaksMCPServer(SharedState(FakeIPython()))
    assert names(server) == sorted([
        "peaks_search_api", "peaks_get_api", "askuserquestion", "notebook_list_variables",
        "notebook_read_variable", "notebook_read_active_cell", "notebook_read_active_cell_output",
        "notebook_read_content", "notebook_move_cursor", "notebook_server_status",
        "notebook_kernel_status", "notebook_wait_for_kernel",
    ])


def test_unsafe_and_dangerous_add_exactly_five_tools():
    server = JupyterPeaksMCPServer(SharedState(FakeIPython()))
    server.set_mode("unsafe")
    unsafe = names(server)
    assert len(unsafe) == 17
    assert {"notebook_execute_code", "notebook_execute_active_cell", "notebook_add_cell", "notebook_delete_cell", "notebook_apply_patch"} <= set(unsafe)
    server.set_mode("safe")
    assert len(names(server)) == 12
    server.set_mode("dangerous")
    assert len(names(server)) == 17


def test_tool_metadata_is_nonempty():
    server = JupyterPeaksMCPServer(SharedState(FakeIPython()))
    tools = asyncio.run(server.mcp.list_tools())
    assert all(tool.title and tool.description for tool in tools)

