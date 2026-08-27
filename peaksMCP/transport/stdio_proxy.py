"""Transparent Claude Desktop STDIO to kernel HTTP MCP proxy."""

from __future__ import annotations

from typing import Any

from fastmcp import Client, FastMCP
from fastmcp.server.providers.proxy import ProxyClient


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
    return FastMCP.as_proxy(ProxyClient(endpoint(host, port)), name="peaksMCP Claude Proxy")


async def check_http_mcp_server(host: str = "127.0.0.1", port: int = 8123) -> dict[str, Any]:
    """Initialize the live HTTP server and verify its tool inventory.

    Parameters
    ----------
    host : str, default "127.0.0.1"
        Kernel MCP listener host.
    port : int, default 8123
        Kernel MCP listener port.

    Returns
    -------
    dict
        Endpoint health, tool inventory and notebook status result.

    Examples
    --------
    >>> result = await check_http_mcp_server(port=8123)
    >>> "ok" in result
    True
    """
    try:
        async with Client(endpoint(host, port), timeout=8) as client:
            tools = await client.list_tools()
            names = sorted(tool.name for tool in tools)
            status: Any = None
            if "notebook_server_status" in names:
                result = await client.call_tool("notebook_server_status", {})
                status = getattr(result, "data", None) or str(result)
            return {"ok": True, "endpoint": endpoint(host, port), "tool_count": len(names), "tools": names, "status": status}
    except Exception as exc:
        return {"ok": False, "endpoint": endpoint(host, port), "error_type": type(exc).__name__, "error": str(exc)}


def run_stdio_proxy(host: str = "127.0.0.1", port: int = 8123) -> None:
    """Run the proxy for Claude Desktop without writing protocol noise to stdout."""
    create_stdio_proxy(host, port).run(transport="stdio", show_banner=False)
