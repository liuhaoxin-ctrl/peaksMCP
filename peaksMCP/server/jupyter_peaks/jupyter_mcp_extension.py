"""IPython extension that owns peaksMCP lifecycle inside a notebook kernel."""

from __future__ import annotations

import os
from typing import Any

from IPython.core.magic import Magics, line_magic, magics_class

from .active_cell_bridge import register_comm_target
from .backend import ExecutionMode, SharedState
from .mcp_server import JupyterPeaksMCPServer

_server: JupyterPeaksMCPServer | None = None
_state: SharedState | None = None


def get_server() -> JupyterPeaksMCPServer | None:
    """Return the server owned by the current kernel, if loaded."""
    return _server


def _start(ipython: Any, host: str | None = None, port: int | None = None) -> JupyterPeaksMCPServer:
    global _server, _state
    if _state is None:
        _state = SharedState(ipython=ipython)
        register_comm_target(_state)
        ipython.events.register("pre_run_cell", _state.mark_busy)
        ipython.events.register("post_run_cell", _state.mark_idle)
    if _server is None:
        _server = JupyterPeaksMCPServer(
            _state,
            host=host or os.environ.get("PEAKSMCP_HOST", "127.0.0.1"),
            port=port or int(os.environ.get("PEAKSMCP_PORT", "8123")),
        )
    _server.start()
    return _server


@magics_class
class PeaksMCPMagics(Magics):
    """Lifecycle and security-mode magics for interactive recovery."""

    @line_magic
    def peaksMCP_start(self, line: str = "") -> dict[str, Any]:
        parts = line.split()
        port = int(parts[0]) if parts else None
        server = _start(self.shell, port=port)
        return {"running": server.is_running(), "host": server.host, "port": server.port, "mode": server.state.mode.value}

    @line_magic
    def peaksMCP_stop(self, _line: str = "") -> dict[str, Any]:
        if _server:
            _server.stop()
        return {"running": bool(_server and _server.is_running())}

    @line_magic
    def peaksMCP_restart(self, _line: str = "") -> dict[str, Any]:
        global _server
        if _server:
            host, port, mode = _server.host, _server.port, _server.state.mode
            _server.stop()
            _server = JupyterPeaksMCPServer(_state, host=host, port=port)
            _server.state.mode = mode
            _server.mcp = _server._build_mcp()
        return self.peaksMCP_start("")

    @line_magic
    def peaksMCP_status(self, _line: str = "") -> dict[str, Any]:
        return {
            "loaded": _state is not None,
            "running": bool(_server and _server.is_running()),
            "mode": _state.mode.value if _state else None,
            "comm_connected": bool(_state and _state.bridge and _state.bridge.connected),
        }

    def _mode(self, mode: str) -> dict[str, Any]:
        server = _start(self.shell)
        server.set_mode(ExecutionMode(mode))
        return {"mode": server.state.mode.value}

    @line_magic
    def peaksMCP_safe(self, _line: str = "") -> dict[str, Any]:
        return self._mode("safe")

    @line_magic
    def peaksMCP_unsafe(self, _line: str = "") -> dict[str, Any]:
        return self._mode("unsafe")

    @line_magic
    def peaksMCP_dangerous(self, _line: str = "") -> dict[str, Any]:
        return self._mode("dangerous")


def load_ipython_extension(ipython: Any) -> None:
    """Register magics, Comm target and start the HTTP MCP server."""
    ipython.register_magics(PeaksMCPMagics)
    _start(ipython)


def unload_ipython_extension(ipython: Any) -> None:
    """Stop MCP and unregister kernel event callbacks."""
    global _server, _state
    if _server:
        _server.stop()
    if _state:
        for event, callback in (("pre_run_cell", _state.mark_busy), ("post_run_cell", _state.mark_idle)):
            try:
                ipython.events.unregister(event, callback)
            except Exception:
                pass
    _server = None
    _state = None

