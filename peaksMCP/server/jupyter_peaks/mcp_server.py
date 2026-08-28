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

    _LOOPBACK = {"127.0.0.1", "localhost", "::1"}

    def __init__(self, state: SharedState, host: str = "127.0.0.1", port: int = 8123, allow_remote: bool = False) -> None:
        self.state = state
        self.host = host
        self.port = int(port)
        self.allow_remote = allow_remote
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
                "Inspect xarray variables before analysis and preserve units in every figure. "
                "Never save figures to disk (plt.savefig / fig.savefig) unless the user "
                "explicitly asks for a saved file — figures are shown inline in the notebook. "
                "Executing or adding notebook cells (notebook_execute_code, notebook_add_cell) "
                "always appends at the end of the notebook and requires explicit consent shown "
                "in the notebook; notebook_delete_cell removes a cell only with consent. "
                "Existing cells are never overwritten."
            ),
            strict_input_validation=True,
        )
        register_safe_tools(mcp, self.state, self.notebook, self.audit)
        # All tools are always exposed to the model. The security mode only
        # controls whether execution/editing asks for in-notebook consent
        # (safe/unsafe) or auto-approves after the static scan (dangerous).
        register_unsafe_tools(mcp, self.unsafe, self.audit)
        return mcp

    def set_mode(self, mode: str | ExecutionMode) -> None:
        """Switch the consent policy without changing the exposed tool set."""
        self.state.mode = ExecutionMode(mode)

    def start(self) -> None:
        """Start the HTTP MCP server on a daemon thread."""
        if self.is_running():
            return
        if self.host not in self._LOOPBACK and not self.allow_remote:
            raise RuntimeError(
                f"MCP refuses to bind to non-loopback host {self.host!r}: the in-kernel "
                "listener has no authentication and would expose the notebook to the "
                "network. Set `mcp.allow_remote: true` only with auth/TLS in front."
            )
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
