from __future__ import annotations

import base64

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from peaksMCP.server.jupyter_peaks.backend import SharedState
from peaksMCP.server.jupyter_peaks.mcp_server import JupyterPeaksMCPServer


class FakeIPython:
    user_ns = {"answer": 42}


@pytest.mark.asyncio
async def test_initialize_list_and_safe_tool_calls():
    server = JupyterPeaksMCPServer(SharedState(FakeIPython()))
    async with Client(server.mcp) as client:
        tools = await client.list_tools()
        assert len(tools) == 12
        question = await client.call_tool("askuserquestion", {"prompt": "Photon energy?", "options": ["21.2 eV", "40.8 eV"]})
        assert question.data["status"] == "needs_input"
        variables = await client.call_tool("notebook_list_variables", {})
        assert variables.data["variables"][0]["name"] == "answer"
        search = await client.call_tool("peaks_search_api", {"query": "动量转换", "limit": 3})
        assert search.data["matches"][0]["name"] == "k_convert"


@pytest.mark.asyncio
async def test_inline_png_becomes_native_image_content():
    state = SharedState(FakeIPython())
    state.active_cell_output = [{"output_type": "display_data", "data": {"image/png": base64.b64encode(b"png").decode(), "text/plain": "figure"}}]
    server = JupyterPeaksMCPServer(state)
    async with Client(server.mcp) as client:
        result = await client.call_tool("notebook_read_active_cell_output", {})
    assert any(block.type == "image" and block.mimeType == "image/png" for block in result.content)


@pytest.mark.asyncio
async def test_invalid_tool_arguments_are_rejected():
    server = JupyterPeaksMCPServer(SharedState(FakeIPython()))
    async with Client(server.mcp) as client:
        with pytest.raises(ToolError):
            await client.call_tool("peaks_search_api", {"unknown": True})
