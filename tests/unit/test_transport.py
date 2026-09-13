from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastmcp import Client, FastMCP

from peaksMCP.transport.stdio_proxy import (
    EXPECTED_TOOL_NAMES,
    SERVER_INSTRUCTIONS,
    check_http_mcp_server,
    create_stdio_proxy,
)


def test_stdio_proxy_forwards_the_curated_server_instructions():
    proxy = create_stdio_proxy()
    normalized = " ".join(proxy.instructions.split())

    assert proxy.instructions == SERVER_INSTRUCTIONS
    assert "Start with load_experiment" in normalized
    assert "get pxt2nc before observing a non-empty needs_conversion" in normalized


def test_stdio_proxy_initializes_and_forwards_tools(monkeypatch):
    """Exercise the real proxy provider across two in-memory MCP sessions."""
    upstream = FastMCP("test upstream", instructions="upstream-only instructions")

    @upstream.tool
    def echo(value: str) -> dict[str, str]:
        return {"echo": value}

    monkeypatch.setattr(
        "peaksMCP.transport.stdio_proxy.endpoint",
        lambda _host, _port: upstream,
    )
    proxy = create_stdio_proxy()

    async def exercise_proxy():
        async with Client(proxy) as client:
            tools = await client.list_tools()
            result = await client.call_tool("echo", {"value": "forwarded"})
            initialized = client.initialize_result or await client.initialize()
            return initialized.instructions, tools, result

    instructions, tools, result = asyncio.run(exercise_proxy())

    assert instructions == SERVER_INSTRUCTIONS
    assert [tool.name for tool in tools] == ["echo"]
    assert result.is_error is False
    assert result.data == {"echo": "forwarded"}


@pytest.mark.parametrize(
    "extra,missing,duplicate,expected_ok",
    [
        ([], [], [], True),
        (["rogue_tool"], [], [], False),
        ([], ["get"], [], False),
        ([], [], ["search"], False),
    ],
)
def test_http_probe_requires_exact_tool_inventory(
    monkeypatch, extra, missing, duplicate, expected_ok
):
    names = sorted(EXPECTED_TOOL_NAMES.difference(missing)) + extra + duplicate

    class FakeClient:
        def __init__(self, *_args, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def list_tools(self):
            return [SimpleNamespace(name=name) for name in names]

        async def call_tool(self, _name, _arguments):
            return SimpleNamespace(data={"status": "ready"})

    monkeypatch.setattr("peaksMCP.transport.stdio_proxy.Client", FakeClient)
    result = asyncio.run(check_http_mcp_server())

    assert result["ok"] is expected_ok
    assert result["missing_tools"] == sorted(missing)
    assert result["unexpected_tools"] == sorted(extra)
    assert result["duplicate_tools"] == sorted(set(duplicate))
