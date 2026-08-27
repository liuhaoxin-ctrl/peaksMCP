"""FastMCP HTTP server hosted inside the active Jupyter kernel."""

from __future__ import annotations

import threading

import uvicorn
from fastmcp import FastMCP

from peaksMCP import __version__

from .backend import ExecutionMode, NotebookBackend, SharedState, UnsafeNotebookBackend
from .core import register_safe_tools, register_unsafe_tools
from .security import AuditLogger, ConsentManager


class JupyterPeaksMCPServer:
    """Own FastMCP, notebook backends and its in-kernel HTTP listener."""

    def __init__(self, state: SharedState, host: str = "127.0.0.1", port: int = 8123) -> None:
        self.state = state
        self.host = host
        self.port = int(port)
        self.audit = AuditLogger()
        self.consent = ConsentManager(state.bridge)
        self.notebook = NotebookBackend(state)
        self.unsafe = UnsafeNotebookBackend(state, self.consent, self.audit)
        self.mcp = self._build_mcp()
        self._thread: threading.Thread | None = None
        self._uvicorn: uvicorn.Server | None = None

    def _build_mcp(self) -> FastMCP:
        mcp = FastMCP(
            "peaksMCP Jupyter Kernel",
            version=__version__,
            instructions=(
                "Use peaks_search_api and peaks_get_api before writing unfamiliar Peaks code. "
                "Inspect xarray variables before analysis and preserve units in every figure."
            ),
            strict_input_validation=True,
        )
        register_safe_tools(mcp, self.state, self.notebook)
        if self.state.mode is not ExecutionMode.SAFE:
            register_unsafe_tools(mcp, self.unsafe)
        return mcp

    def set_mode(self, mode: str | ExecutionMode) -> None:
        """Change the exposed tool set; restarting HTTP is not required."""
        target = ExecutionMode(mode)
        if target is self.state.mode:
            return
        previous = self.state.mode
        self.state.mode = target
        if previous is not ExecutionMode.SAFE:
            for name in (
                "notebook_execute_code", "notebook_execute_active_cell", "notebook_add_cell",
                "notebook_delete_cell", "notebook_apply_patch",
            ):
                self.mcp.local_provider.remove_tool(name)
        if target is not ExecutionMode.SAFE:
            register_unsafe_tools(self.mcp, self.unsafe)

    def start(self) -> None:
        """Start the HTTP MCP server on a daemon thread."""
        if self.is_running():
            return
        app = self.mcp.http_app(path="/mcp", stateless_http=False)
        config = uvicorn.Config(app, host=self.host, port=self.port, log_level="warning", lifespan="on")
        self._uvicorn = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._uvicorn.run, name="peaksMCP-http", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 10) -> None:
        """Request a graceful HTTP shutdown and join its thread."""
        if self._uvicorn is not None:
            self._uvicorn.should_exit = True
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def is_running(self) -> bool:
        """Return whether the HTTP server thread is alive."""
        return bool(self._thread and self._thread.is_alive())

    async def tool_names(self) -> list[str]:
        """Return currently exposed tool names for readiness checks."""
        return sorted(tool.name for tool in await self.mcp.list_tools())
