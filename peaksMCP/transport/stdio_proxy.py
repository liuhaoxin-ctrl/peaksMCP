"""Transparent Claude Desktop STDIO to kernel HTTP MCP proxy."""

from __future__ import annotations

from collections import Counter
from typing import Any

from fastmcp import Client, FastMCP
from fastmcp.server import create_proxy

from peaksMCP.config import prompts as load_prompts
from peaksMCP.config.metadata import tool_names

#: Single source of truth is the ``tools:`` block in metadata_baseline.yaml; the
#: proxy only verifies the live kernel server exposes exactly that set.
EXPECTED_TOOL_NAMES = tool_names()
SERVER_INSTRUCTIONS = load_prompts()["server_instructions"]


def endpoint(host: str = "127.0.0.1", port: int = 8123) -> str:
    """Return the canonical kernel MCP endpoint."""
    return f"http://{host}:{int(port)}/mcp"


def create_stdio_proxy(host: str = "127.0.0.1", port: int = 8123) -> FastMCP:
    """Create a proxy that mirrors all kernel tools over standard input/output.

    Parameters
    ----------
    host : str, default "127.0.0.1"
        Kernel MCP listener host.
    port : int, default 8123
        Kernel MCP listener port.

    Returns
    -------
    fastmcp.FastMCP
        STDIO-capable proxy server for Claude Desktop.

    Examples
    --------
    >>> proxy = create_stdio_proxy(port=8123)
    >>> proxy.name
    'peaksMCP Claude Proxy'
    """
    # The proxy mirrors upstream tools, but its initialize response does not
    # inherit the HTTP server's instructions. Pi asks this STDIO endpoint for
    # them explicitly, so attach the same curated contract here.
    return create_proxy(
        endpoint(host, port),
        name="peaksMCP Claude Proxy",
        instructions=SERVER_INSTRUCTIONS,
    )


async def check_http_mcp_server(host: str = "127.0.0.1", port: int = 8123) -> dict[str, Any]:
    """Initialize the live HTTP server and verify its tool inventory.

    Observability reads the PRIVATE loopback ``/healthz`` endpoint for
    readiness (kernel/comm/index payload) - the model tool surface never
    exposes status, so this check must not depend on any tool.

    Parameters
    ----------
    host : str, default "127.0.0.1"
        Kernel MCP listener host.
    port : int, default 8123
        Kernel MCP listener port.

    Returns
    -------
    dict
        Endpoint health, tool inventory and the private health payload.
    """
    try:
        status: Any = None
        try:
            import httpx

            async with httpx.AsyncClient(timeout=4) as http:
                response = await http.get(f"http://{host}:{int(port)}/healthz")
                response.raise_for_status()
                status = response.json()
        except Exception:
            status = None
        async with Client(endpoint(host, port), timeout=8) as client:
            tools = await client.list_tools()
            names = sorted(tool.name for tool in tools)
            counts = Counter(names)
            actual_names = set(counts)
            missing_tools = sorted(EXPECTED_TOOL_NAMES.difference(actual_names))
            unexpected_tools = sorted(actual_names.difference(EXPECTED_TOOL_NAMES))
            duplicate_tools = sorted(
                name for name, count in counts.items() if count > 1
            )
            return {
                "ok": not missing_tools and not unexpected_tools and not duplicate_tools,
                "endpoint": endpoint(host, port),
                "tool_count": len(names),
                "tools": names,
                "missing_tools": missing_tools,
                "unexpected_tools": unexpected_tools,
                "duplicate_tools": duplicate_tools,
                "status": status,
            }
    except Exception as exc:
        return {"ok": False, "endpoint": endpoint(host, port), "error_type": type(exc).__name__, "error": str(exc)}


def run_stdio_proxy(host: str = "127.0.0.1", port: int = 8123) -> None:
    """Run the proxy for Claude Desktop without writing protocol noise to stdout."""
    create_stdio_proxy(host, port).run(transport="stdio", show_banner=False)
