from __future__ import annotations

import base64

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from peaksMCP.config.metadata import tool_names
from peaksMCP.server.jupyter_peaks.backend import SharedState
from peaksMCP.server.jupyter_peaks.mcp_server import JupyterPeaksMCPServer


class FakeIPython:
    user_ns = {"answer": 42}


def _fake_bridge(state):
    """A connected bridge whose execute_code mirrors the frontend contract."""
    class FakeBridge:
        connected = True

        def request(self, operation, payload=None, timeout=30.0):
            if operation != "execute_code":
                raise AssertionError(f"unexpected operation {operation!r}")
            outputs = [
                {"output_type": "display_data",
                 "data": {"image/png": base64.b64encode(b"png").decode(), "text/plain": "figure"}},
                {"output_type": "stream", "text": "later text"},
            ]
            return {
                "id": "cell-9", "index": 4, "cell_type": "code",
                "source": payload["code"], "execution_success": True,
                "saved": True, "outputs": outputs,
            }

    return FakeBridge()


@pytest.mark.asyncio
async def test_initialize_list_and_safe_tool_calls():
    server = JupyterPeaksMCPServer(SharedState(FakeIPython()))
    async with Client(server.mcp) as client:
        tools = await client.list_tools()
        assert len(tools) == len(tool_names())
        names = {tool.name for tool in tools}
        assert names == set(tool_names())
        assert "notebook_execute_active_cell" not in names
        assert "notebook_delete_cell" not in names
        assert "notebook_write_with_api_check" in names
        # The read-back / cursor / status tools were removed with the output
        # normalisation pass (the write tool returns the normalised output).
        for removed in (
            "notebook_read_active_cell_output", "notebook_read_content",
            "notebook_move_cursor", "notebook_kernel_status", "notebook_wait_for_kernel",
        ):
            assert removed not in names
        question = await client.call_tool("askuserquestion", {"prompt": "Photon energy?", "options": ["21.2 eV", "40.8 eV"]})
        assert question.data["status"] == "needs_input"
        variables = await client.call_tool("notebook_list_variables", {})
        assert variables.data["variables"][0]["name"] == "answer"
        search = await client.call_tool("peaks_search_api", {"query": "动量转换", "limit": 3})
        assert search.data["matches"][0]["name"] == "k_convert"


@pytest.mark.asyncio
async def test_write_returns_normalised_output_without_raw_outputs():
    """The write tool is the single execution channel: it returns the normalised
    output list (figure marker, errors), never the raw cell outputs."""
    state = SharedState(FakeIPython())
    state.bridge = _fake_bridge(state)
    server = JupyterPeaksMCPServer(state)
    async with Client(server.mcp) as client:
        result = await client.call_tool(
            "notebook_write_with_api_check", {"code": "fig, ax = plt.subplots()"}
        )
    data = result.data
    assert data["id"] == "cell-9"
    assert data["execution_success"] is True
    assert "outputs" not in data  # raw outputs never leave the kernel
    text = "\n".join(data["output"])
    assert "Inline figure rendered" in text
    assert "<Figure>" not in text
    assert "later text" not in text  # text is not echoed to the model


@pytest.mark.asyncio
async def test_invalid_tool_arguments_are_rejected():
    server = JupyterPeaksMCPServer(SharedState(FakeIPython()))
    async with Client(server.mcp) as client:
        with pytest.raises(ToolError):
            await client.call_tool("peaks_search_api", {"unknown": True})
