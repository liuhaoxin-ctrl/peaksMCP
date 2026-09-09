"""FastMCP HTTP server hosted inside the active Jupyter kernel."""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any

import uvicorn
from fastmcp import FastMCP

from peaksMCP import __version__
from peaksMCP.config import prompts as _load_prompts
from peaksMCP.discovery.index import build_index

from .backend import NotebookBackend, SharedState, UnsafeNotebookBackend
from .core import register_safe_tools, register_unsafe_tools
from .security import AuditLogger, ConsentManager

#: Always-on server instruction prompt, delivered on every session through the
#: FastMCP server instructions. Wording lives in config/prompts.yaml
#: (``server_instructions``); edit it there, not here.
_SERVER_INSTRUCTIONS = _load_prompts()["server_instructions"]


class JupyterPeaksMCPServer:
    """Own FastMCP, notebook backends and its in-kernel HTTP listener."""

    _LOOPBACK = {"127.0.0.1", "localhost", "::1"}

    def __init__(self, state: SharedState, host: str = "127.0.0.1", port: int = 8123, allow_remote: bool = False) -> None:
        self.state = state
        self.state.mcp_instance_id = uuid.uuid4().hex
        self.host = host
        self.port = int(port)
        self.allow_remote = allow_remote
        self.audit = AuditLogger()
        self.consent = ConsentManager(state.bridge)
        self.notebook = NotebookBackend(state)
        self.unsafe = UnsafeNotebookBackend(state, self.consent, self.audit)
        #: Server-owned save gateway; installed in start(), released in
        #: stop().  None while the server is not running.
        self.save_gateway = None
        # Pre-build the Peaks API index at server startup instead of lazily on
        # the first search: the dashboard can then report api_index_ready /
        # api_count immediately (and index_stale reflects the current source).
        # A failure keeps api_index None so searches still fall back to lazy
        # build inside _require_index.
        if state.api_index is None:
            try:
                state.api_index = build_index()
            except Exception:
                state.api_index = None
        self.mcp = self._build_mcp()
        self._thread: threading.Thread | None = None
        self._uvicorn: uvicorn.Server | None = None


    def _install_save_approval_channel(self) -> None:
        """Route staged-ticket approvals (save_with_consent / conversion) to the
        frontend card.

        The gateway is server-owned: an instance is created here, installed as
        the active gateway on ``start()`` and released on ``stop()`` (tickets
        and channel cleared), so no module-global approval state can leak into
        a later kernel or test.  The gateway refuses tickets that never went
        through this channel.
        """
        from peaksMCP.overrides.save import SaveGateway, install_gateway

        gateway = SaveGateway()
        gateway.set_approval_channel(self._approve_save_card)
        install_gateway(gateway)
        self.save_gateway = gateway

    def _approve_save_card(self, preview: dict) -> bool:
        approved = self.consent.request("save_ticket", dict(preview))
        items = preview.get("items") or []
        first = items[0] if items else {}
        self.audit.write(
            "save_consent",
            "approved" if approved else "denied",
            {
                "operation": preview.get("operation"),
                "ticket_id": preview.get("ticket_id"),
                "item_count": len(items),
                "first_path": first.get("path"),
                "first_sha256": str(first.get("sha256") or "")[:16],
            },
        )
        return approved

    def _build_mcp(self) -> FastMCP:
        mcp = FastMCP(
            "peaksMCP Jupyter Kernel",
            version=__version__,
            instructions=_SERVER_INSTRUCTIONS,
            strict_input_validation=True,
        )

        register_safe_tools(mcp, self.state, self.notebook, self.audit)
        # All tools are always exposed to the model. Consent for the two mutation
        # tools is governed solely by the ``require_consent`` master switch
        # (profile ``mcp.require_consent``): when off, the scanner still
        # hard-blocks known-dangerous patterns and every call is audit-logged,
        # but no in-notebook consent is requested. The AST scanner is an early
        # rejection layer, not a complete security boundary for arbitrary Python.
        register_unsafe_tools(mcp, self.unsafe, self.audit)
        return mcp

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
        # Private loopback health endpoint: observability (supervisor / stdio
        # proxy / dashboard) reads readiness here; the model NEVER sees it -
        # there is no status tool in the MCP surface.
        app.add_route("/healthz", self._healthz, methods=["GET"], name="healthz")
        config = uvicorn.Config(app, host=self.host, port=self.port, log_level="warning", lifespan="on")
        self._uvicorn = uvicorn.Server(config)
        self._install_save_approval_channel()
        self._thread = threading.Thread(target=self._uvicorn.run, name="peaksMCP-http", daemon=True)
        self._thread.start()

    def _healthz(self, request: Any) -> Any:
        """Loopback-only health payload (same fields the old status tool had).

        Returns the readiness JSON without ever exposing it through the model
        tool surface: observability consumes this endpoint, agents cannot see
        it.  The response is intentionally small and kernel-local.
        """
        from starlette.responses import JSONResponse

        state = self.state
        index = state.api_index
        payload = {
            "status": "ready",
            "uptime_s": round(time.time() - state.started_at, 3),
            "kernel_instance_id": state.kernel_instance_id,
            "mcp_instance_id": state.mcp_instance_id,
            "extension_loaded": True,
            "comm_connected": bool(state.bridge and state.bridge.connected),
            "api_index_ready": index is not None,
            "api_count": len(index.entries) if index else 0,
            "index_stale": bool(index and index.is_stale()),
        }
        return JSONResponse(payload)

    def stop(self, timeout: float = 10) -> None:
        """Request a graceful HTTP shutdown, release the server-owned save
        gateway (staged tickets + approval channel cleared) and join the
        server thread.  Nothing save-related may leak into later kernels or
        tests after the server stops."""
        from peaksMCP.overrides.save import uninstall_gateway

        uninstall_gateway()
        self.save_gateway = None
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
