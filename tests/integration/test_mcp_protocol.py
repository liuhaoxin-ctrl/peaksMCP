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
        assert "run_cell" in names
        # The read-back / cursor / status tools were removed with the output
        # normalisation pass (the write tool returns the normalised output).
        for removed in (
            "notebook_read_active_cell_output", "notebook_read_content",
            "notebook_move_cursor", "notebook_kernel_status", "notebook_wait_for_kernel",
        ):
            assert removed not in names
        # The final five-tool surface: search/get/inspect_notebook/run_cell/
        # save_with_consent - nothing else is model-registered.
        assert set(tool_names()) == {
            "search", "get", "inspect_notebook", "run_cell", "save_with_consent",
        }
        for removed in (
            "askuserquestion", "notebook_list_variables", "notebook_read_variable",
            "notebook_read_active_cell", "notebook_server_status",
            "notebook_add_cell", "peaks_search_api", "peaks_get_api",
            "notebook_write_with_api_check",
        ):
            assert removed not in names, removed
        variables = await client.call_tool("inspect_notebook", {"target": "variables"})
        assert variables.data["variables"][0]["name"] == "answer"
        search = await client.call_tool("search", {"query": "动量转换", "limit": 3})
        assert search.data["matches"][0]["name"] == "k_convert"


@pytest.mark.asyncio
async def test_write_returns_normalised_output_without_raw_outputs():
    """run_cell is the single execution channel: it returns the normalised
    output list (figure marker, errors), never the raw cell outputs."""
    state = SharedState(FakeIPython())
    state.bridge = _fake_bridge(state)
    server = JupyterPeaksMCPServer(state)
    async with Client(server.mcp) as client:
        result = await client.call_tool(
            "run_cell", {"code": "fig, ax = plt.subplots()"}
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
            await client.call_tool("search", {"unknown": True})


def _save_fake_bridge(state, cells=None):
    """A connected bridge whose execute/add_cell mirror the frontend contract."""
    class FakeBridge:
        connected = True

        def request(self, operation, payload=None, timeout=30.0):
            if operation == "add_cell":
                if cells is not None:
                    cells.append(payload.get("source", ""))
                return {"id": "cell-1", "cell_type": payload.get("cell_type"),
                        "source": payload.get("source", "")}
            raise AssertionError(f"unexpected operation {operation!r}")

    return FakeBridge()


@pytest.mark.asyncio
async def test_save_with_consent_is_the_only_persistence_verb(tmp_path):
    """save_with_consent: preview record cell first, staged ticket, receipt.
    With no approval channel the save fails closed (blocked,
    no_consent_channel) and nothing is written; the model Python surface has
    no save_result export."""
    import xarray as xr

    from peaksMCP.overrides import __all__ as overrides_all

    assert "save_result" not in overrides_all

    namespace = {"scan": xr.DataArray([[1.0, 2.0]], dims=("eV", "kx"),
                                      attrs={"units": "counts"})}
    cells: list[str] = []
    state = SharedState(type("_IP", (), {"user_ns": namespace})())
    state.bridge = _save_fake_bridge(state, cells)
    server = JupyterPeaksMCPServer(state)
    async with Client(server.mcp) as client:
        result = await client.call_tool(
            "save_with_consent",
            {"variable_name": "scan", "path": str(tmp_path / "scan.nc")},
        )
    data = result.data
    assert data["operation"] == "save_with_consent"
    assert data["status"] == "blocked"
    assert data["note"] and "no approval channel" in data["note"]
    assert data["kind"] == "netcdf"
    assert data["dims"] == {"eV": 1, "kx": 2}
    assert not data["sha256"] and not data["ticket_id"]
    assert not (tmp_path / "scan.nc").exists()
    assert cells and "save_with_consent" in cells[0]  # intent record cell


@pytest.mark.asyncio
async def test_save_with_consent_blocks_existing_target_without_overwrite(tmp_path):
    import xarray as xr

    target = tmp_path / "scan.nc"
    target.write_bytes(b"existing")
    state = SharedState(type("_IP", (), {"user_ns": {"scan": xr.DataArray([1], dims="eV")}})())
    state.bridge = _save_fake_bridge(state)
    server = JupyterPeaksMCPServer(state)
    async with Client(server.mcp) as client:
        result = await client.call_tool(
            "save_with_consent",
            {"variable_name": "scan", "path": str(target)},
        )
    data = result.data
    assert data["status"] == "blocked"
    assert "exists" in (data["note"] or "")
    assert target.read_bytes() == b"existing"  # untouched


@pytest.mark.asyncio
async def test_save_with_consent_rejects_unknown_variable_and_bad_kind(tmp_path):
    import xarray as xr
    from fastmcp.exceptions import ToolError

    state = SharedState(type("_IP", (), {"user_ns": {"scan": xr.DataArray([1], dims="eV")}})())
    state.bridge = _save_fake_bridge(state)
    server = JupyterPeaksMCPServer(state)
    async with Client(server.mcp) as client:
        with pytest.raises(ToolError, match="does not exist"):
            await client.call_tool(
                "save_with_consent",
                {"variable_name": "ghost", "path": str(tmp_path / "x.nc")},
            )
        with pytest.raises(ToolError, match="cannot serialise"):
            # .txt is neither netcdf nor json-able for an xarray object.
            await client.call_tool(
                "save_with_consent",
                {"variable_name": "scan", "path": str(tmp_path / "x.txt")},
            )
