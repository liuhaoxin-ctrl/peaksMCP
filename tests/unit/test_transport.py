from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from peaksMCP.transport.stdio_proxy import EXPECTED_TOOL_NAMES, check_http_mcp_server


@pytest.mark.parametrize(
    "extra,missing,duplicate,expected_ok",
    [
        ([], [], [], True),
        (["rogue_tool"], [], [], False),
        ([], ["peaks_get_api"], [], False),
        ([], [], ["peaks_search_api"], False),
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
