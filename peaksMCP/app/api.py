"""Static dashboard API, Inspector and conversion task endpoints."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING

from fastmcp import Client
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket

from peaksMCP.pxt_utils import convert_path, translate_datasheet
from peaksMCP.transport import check_http_mcp_server

if TYPE_CHECKING:
    from .runtime import RuntimeSupervisor


def create_app(supervisor: RuntimeSupervisor) -> Starlette:
    """Build the local-only Dashboard application."""
    web = Path(__file__).with_name("webapp")

    async def index(_request: Request) -> FileResponse:
        return FileResponse(web / "index.html")

    async def asset(request: Request) -> FileResponse:
        name = request.path_params["name"]
        if name not in {"app.js", "style.css"}:
            raise FileNotFoundError(name)
        return FileResponse(web / name)

    async def status(_request: Request) -> JSONResponse:
        base = supervisor.status()
        mcp = await check_http_mcp_server(supervisor.profile.mcp.host, supervisor.profile.mcp.port)
        kernel_state = "unknown"
        try:
            response = await asyncio.to_thread(
                __import__("httpx").get,
                f"{supervisor.jupyter_url}/api/kernels/{supervisor.kernel_id}",
                headers=supervisor._headers(), timeout=2,
            )
            kernel_state = response.json().get("execution_state", "unknown")
        except Exception:
            pass
        mcp_status = mcp.get("status") if isinstance(mcp.get("status"), dict) else {}
        components = {
            "supervisor": {"state": "ready", "detail": f"PID {base['pid']}"},
            "jupyter": {"state": "ready" if supervisor.jupyter and supervisor.jupyter.poll() is None else "error", "detail": supervisor.jupyter_url},
            "kernel": {"state": "ready" if kernel_state in {"idle", "busy"} else "degraded", "detail": kernel_state},
            "extension": {"state": "ready" if mcp.get("ok") else "error", "detail": "IPython extension loaded" if mcp.get("ok") else mcp.get("error", "not detected")},
            "comm": {"state": "ready" if mcp_status.get("comm_connected") else "degraded", "detail": "JupyterLab connected" if mcp_status.get("comm_connected") else "Open the managed notebook"},
            "mcp": {"state": "ready" if mcp.get("ok") else "error", "detail": f"{mcp.get('tool_count', 0)} tools · {base['mcp_url']}"},
        }
        states = {item["state"] for item in components.values()}
        aggregate = "error" if "error" in states else "degraded" if "degraded" in states else "ready"
        return JSONResponse({
            **base,
            "notebook_open_url": f"{base['notebook_url']}?token={supervisor.token}",
            "aggregate": aggregate,
            "components": components,
            "mcp": mcp,
        })

    async def logs(_request: Request) -> JSONResponse:
        return JSONResponse({"logs": list(supervisor.logs)})

    async def doctor(_request: Request) -> JSONResponse:
        from peaksMCP.ext_install import extension_source, locate_installed_extension

        from .kernel import kernel_installed
        checks = [
            {"name": "JupyterLab", "status": "ok" if supervisor.jupyter and supervisor.jupyter.poll() is None else "fail", "detail": supervisor.jupyter_url},
            {"name": "Kernel spec", "status": "ok" if kernel_installed(supervisor.profile.jupyter.kernel_name) else "fail", "detail": supervisor.profile.jupyter.kernel_name},
            {"name": "JupyterLab extension", "status": "ok" if (extension_source() / "static").is_dir() and locate_installed_extension() else "fail", "detail": ", ".join(map(str, locate_installed_extension())) or "not installed"},
            {"name": "MCP", "status": "ok" if (await check_http_mcp_server(supervisor.profile.mcp.host, supervisor.profile.mcp.port)).get("ok") else "fail", "detail": base_url(supervisor)},
        ]
        return JSONResponse({"ok": all(item["status"] == "ok" for item in checks), "checks": checks})

    async def profiles(_request: Request) -> JSONResponse:
        from .profiles import list_profiles
        return JSONResponse({"profiles": list_profiles(), "active": supervisor.profile.name})

    async def restart(request: Request) -> JSONResponse:
        component = request.path_params["component"]
        try:
            if component == "mcp":
                result = await asyncio.to_thread(supervisor.restart_mcp)
            elif component == "kernel":
                result = await asyncio.to_thread(supervisor.restart_kernel)
            elif component == "all":
                result = await asyncio.to_thread(supervisor.restart_kernel, 90, True)
            else:
                return JSONResponse({"error": "component must be mcp, kernel or all"}, status_code=400)
            return JSONResponse(result)
        except Exception as exc:
            return JSONResponse({"error_type": type(exc).__name__, "error": str(exc)}, status_code=500)

    async def tool_call(request: Request) -> JSONResponse:
        body = await request.json()
        name = str(body.get("name") or "")
        arguments = body.get("arguments") or {}
        allowed = {"peaks_search_api", "peaks_get_api", "notebook_server_status", "notebook_kernel_status", "notebook_list_variables", "notebook_read_variable"}
        if name not in allowed:
            return JSONResponse({"error": f"Dashboard Inspector does not allow {name!r}"}, status_code=403)
        try:
            async with Client(f"http://{supervisor.profile.mcp.host}:{supervisor.profile.mcp.port}/mcp", timeout=30) as client:
                result = await client.call_tool(name, arguments)
            data = getattr(result, "data", None)
            if data is None:
                data = [item.model_dump(mode="json") for item in getattr(result, "content", [])]
            return JSONResponse({"ok": True, "result": data})
        except Exception as exc:
            return JSONResponse({"ok": False, "error_type": type(exc).__name__, "error": str(exc)}, status_code=502)

    async def translate(request: Request) -> JSONResponse:
        body = await request.json()
        try:
            result = await asyncio.to_thread(translate_datasheet, body["csv"], body.get("output"))
            return JSONResponse(result.model_dump(mode="json"))
        except Exception as exc:
            return JSONResponse({"error_type": type(exc).__name__, "error": str(exc)}, status_code=400)

    async def convert(request: Request) -> JSONResponse:
        body = await request.json()
        try:
            result = await asyncio.to_thread(
                convert_path, body["input"], body.get("output"), metadata_path=body.get("metadata"),
                substring=body.get("filter", ""), force=bool(body.get("force")),
                cpu_limit_percent=float(body.get("cpu_limit", 60)),
            )
            return JSONResponse(result.model_dump(mode="json"))
        except Exception as exc:
            return JSONResponse({"error_type": type(exc).__name__, "error": str(exc)}, status_code=400)

    async def log_socket(websocket: WebSocket) -> None:
        await websocket.accept()
        cursor = 0
        try:
            while True:
                entries = list(supervisor.logs)
                if cursor < len(entries):
                    await websocket.send_text(json.dumps({"logs": entries[cursor:]}))
                    cursor = len(entries)
                await asyncio.sleep(0.5)
        except Exception:
            await websocket.close()

    return Starlette(routes=[
        Route("/", index), Route("/assets/{name}", asset), Route("/api/status", status),
        Route("/api/logs", logs), Route("/api/doctor", doctor), Route("/api/profiles", profiles),
        Route("/api/restart/{component}", restart, methods=["POST"]),
        Route("/api/mcp/tool", tool_call, methods=["POST"]),
        Route("/api/metadata/translate", translate, methods=["POST"]),
        Route("/api/convert", convert, methods=["POST"]), WebSocketRoute("/ws/logs", log_socket),
    ])


def base_url(supervisor: RuntimeSupervisor) -> str:
    """Return the configured MCP URL for dashboard diagnostics."""
    return f"http://{supervisor.profile.mcp.host}:{supervisor.profile.mcp.port}/mcp"
