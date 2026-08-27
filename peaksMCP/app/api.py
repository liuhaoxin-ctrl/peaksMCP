"""Operator-console API + static webapp, served inside the supervisor process.

The dashboard is co-hosted by the supervisor (``peaksMCP launch`` is the single
startup entry) and binds the live :class:`RuntimeSupervisor`, so it can both
*monitor* (JupyterLab, managed kernel, in-kernel MCP, Comm) and *control*
(start/restart MCP, restart kernel, open the notebook, stop the stack).
"""

from __future__ import annotations

import asyncio
import json
import secrets
import subprocess
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import httpx
from fastmcp import Client
from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, PlainTextResponse, RedirectResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket

from peaksMCP.pxt_utils import convert_path, translate_datasheet

from .profiles import list_profiles

if TYPE_CHECKING:
    from .runtime import RuntimeSupervisor

_INSPECTOR_ALLOWED = {
    "peaks_search_api", "peaks_get_api", "notebook_server_status",
    "notebook_kernel_status", "notebook_list_variables", "notebook_read_variable",
    "notebook_read_active_cell", "notebook_read_active_cell_output",
}
_DASHBOARD_COOKIE = "peaksmcp_dashboard"


def _auto_load_into_notebook(supervisor: RuntimeSupervisor, nc_path: str | None) -> bool:
    """Load a converted NetCDF into the notebook as a visible cell.

    Asks the kernel to insert and run ``from peaks import load\ndata = load(...)``
    through the frontend Comm bridge (so the cell appears in the notebook and is
    saved), instead of silently loading into the kernel namespace.  Returns
    ``False`` when the frontend/kernel is unavailable.
    """
    if not nc_path:
        return False
    load_code = f"from peaks import load\ndata = load({json.dumps(str(nc_path))})"
    # Real multi-line source (a single-line ``def _do(): try:`` chain is invalid
    # Python and would be silently swallowed by the except below).
    bridge_code = f'''import threading as _t
from peaksMCP.server.jupyter_peaks.jupyter_mcp_extension import get_server as _gs

def _do():
    try:
        _gs().state.bridge.request('execute_code', {{'code': {json.dumps(load_code)}}}, timeout=30)
    except Exception as _e:
        print('auto-load skipped:', _e)

_t.Thread(target=_do, daemon=True).start()
'''
    try:
        compile(bridge_code, "<peaksMCP-auto-load>", "exec")
    except SyntaxError:
        return False
    try:
        supervisor.execute_kernel(bridge_code, timeout=10)
        return True
    except Exception:
        return False


async def _jupyter_kernel_state(supervisor: RuntimeSupervisor) -> str:
    """Return the managed kernel execution state via the Jupyter REST API."""
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            response = await client.get(
                f"{supervisor.jupyter_url}/api/kernels/{supervisor.kernel_id}",
                headers={"Authorization": f"token {supervisor.token}"},
            )
            response.raise_for_status()
            return str(response.json().get("execution_state", "unknown"))
    except Exception:
        return "unknown"


async def _mcp_probe(supervisor: RuntimeSupervisor) -> dict[str, Any]:
    """Probe the in-kernel MCP endpoint; never raises."""
    from peaksMCP.transport import check_http_mcp_server

    try:
        return await check_http_mcp_server(
            supervisor.profile.mcp.host, supervisor.profile.mcp.port
        )
    except Exception as exc:  # pragma: no cover - defensive
        return {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}


async def status_payload(supervisor: RuntimeSupervisor) -> dict[str, Any]:
    """Compose the live operator-console status snapshot."""
    base = supervisor.status()
    mcp, kernel_state = await asyncio.gather(
        _mcp_probe(supervisor),
        _jupyter_kernel_state(supervisor),
    )
    mcp_status = mcp.get("status") if isinstance(mcp.get("status"), dict) else {}
    jupyter_up = bool(supervisor.jupyter and supervisor.jupyter.poll() is None)
    # The in-kernel MCP only runs inside the notebook kernel, so ``mcp.ok`` is
    # the strongest proof the kernel is alive and usable (Jupyter's REST
    # execution_state can remain "starting" even while the kernel serves cells).
    kernel_ready = bool(mcp.get("ok")) or kernel_state in {"idle", "busy"}
    if mcp.get("ok"):
        extension = {"loaded": True, "detail": "IPython extension loaded"}
    elif kernel_ready:
        extension = await asyncio.to_thread(supervisor.extension_status)
    else:
        extension = {"loaded": False, "detail": "kernel unavailable"}
    components = {
        "supervisor": {"state": "ready", "detail": f"PID {base['pid']}"},
        "jupyter": {"state": "ready" if jupyter_up else "error", "detail": supervisor.jupyter_url},
        "kernel": {"state": "ready" if kernel_ready else "degraded", "detail": kernel_state},
        "extension": {
            "state": "ready" if extension["loaded"] else "degraded",
            "detail": extension["detail"],
        },
        "comm": {"state": "ready" if mcp_status.get("comm_connected") else "degraded", "detail": "JupyterLab connected" if mcp_status.get("comm_connected") else "Open the managed notebook"},
        "mcp": {"state": "ready" if mcp.get("ok") else "error", "detail": f"{mcp.get('tool_count', 0)} tools · {base['mcp_url']}"},
    }
    states = {item["state"] for item in components.values()}
    aggregate = "error" if "error" in states else "degraded" if "degraded" in states else "ready"
    return {
        **base,
        "status": "RUNNING" if jupyter_up else "STOPPED",
        "supervisor_running": True,
        "kernel_state": kernel_state,
        "aggregate": aggregate,
        "components": components,
        "mcp": mcp,
        "notebook_open_url": "/open-notebook",
    }


def create_app(supervisor: RuntimeSupervisor) -> Starlette:
    """Build the operator-console application bound to a live supervisor."""
    web = Path(__file__).with_name("webapp")

    def token_valid(request: Request) -> bool:
        authorization = request.headers.get("authorization", "")
        bearer = authorization.removeprefix("Bearer ") if authorization.startswith("Bearer ") else ""
        supplied = bearer or request.cookies.get(_DASHBOARD_COOKIE, "")
        return bool(supplied) and secrets.compare_digest(supplied, supervisor.dashboard_token)

    def origin_valid(request: Request) -> bool:
        origin = request.headers.get("origin")
        if not origin:
            return True
        parsed = urlsplit(origin)
        expected_host = supervisor.profile.dashboard.host
        allowed_hosts = {expected_host}
        if expected_host in {"127.0.0.1", "localhost", "::1"}:
            allowed_hosts.update({"127.0.0.1", "localhost", "::1"})
        elif request.url.hostname:
            allowed_hosts.add(request.url.hostname)
        return (
            parsed.scheme in {"http", "https"}
            and parsed.hostname in allowed_hosts
            and (parsed.port or (443 if parsed.scheme == "https" else 80))
            == supervisor.profile.dashboard.port
        )

    def require_auth(request: Request, *, mutation: bool = False) -> None:
        if not token_valid(request):
            raise HTTPException(401, "open the console with `peaksMCP open`")
        if mutation and not origin_valid(request):
            raise HTTPException(403, "cross-origin dashboard control request rejected")

    async def index(request: Request):
        supplied = request.query_params.get("token", "")
        if supplied and secrets.compare_digest(supplied, supervisor.dashboard_token):
            response = RedirectResponse("/", status_code=303)
            response.set_cookie(
                _DASHBOARD_COOKIE,
                supervisor.dashboard_token,
                httponly=True,
                samesite="strict",
                secure=request.url.scheme == "https",
            )
            return response
        if not token_valid(request):
            return PlainTextResponse(
                "peaksMCP operator console authentication required; run `peaksMCP open`.",
                status_code=401,
            )
        return FileResponse(web / "index.html")

    async def asset(request: Request) -> FileResponse:
        name = request.path_params["name"]
        if name not in {"app.js", "style.css"}:
            raise FileNotFoundError(name)
        return FileResponse(web / name)

    async def status(_request: Request) -> JSONResponse:
        require_auth(_request)
        return JSONResponse(await status_payload(supervisor))

    async def logs(_request: Request) -> JSONResponse:
        require_auth(_request)
        return JSONResponse({"logs": list(supervisor.logs)})

    async def doctor(_request: Request) -> JSONResponse:
        require_auth(_request)
        from peaksMCP.ext_install import extension_source, locate_installed_extension

        from .kernel import kernel_installed

        mcp = await _mcp_probe(supervisor)
        checks = [
            {"name": "Kernel spec", "status": "ok" if kernel_installed(supervisor.profile.jupyter.kernel_name) else "fail", "detail": supervisor.profile.jupyter.kernel_name},
            {"name": "JupyterLab extension", "status": "ok" if (extension_source() / "static").is_dir() and locate_installed_extension() else "fail", "detail": ", ".join(map(str, locate_installed_extension())) or "not installed"},
            {"name": "Supervisor", "status": "ok", "detail": f"PID {supervisor.status()['pid']}"},
            {"name": "JupyterLab", "status": "ok" if supervisor.jupyter and supervisor.jupyter.poll() is None else "fail", "detail": supervisor.jupyter_url},
            {"name": "MCP", "status": "ok" if mcp.get("ok") else "fail", "detail": f"{supervisor.profile.mcp.host}:{supervisor.profile.mcp.port}/mcp"},
        ]
        return JSONResponse({"ok": all(item["status"] == "ok" for item in checks), "checks": checks})

    async def profiles(_request: Request) -> JSONResponse:
        require_auth(_request)
        return JSONResponse({"profiles": list_profiles(), "active": supervisor.profile.name})

    async def start_mcp(_request: Request) -> JSONResponse:
        require_auth(_request, mutation=True)
        try:
            return JSONResponse(await asyncio.to_thread(supervisor.start_mcp))
        except Exception as exc:
            return JSONResponse({"error_type": type(exc).__name__, "error": str(exc)}, status_code=409)

    async def restart(request: Request) -> JSONResponse:
        require_auth(request, mutation=True)
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
            return JSONResponse({"error_type": type(exc).__name__, "error": str(exc)}, status_code=409)

    async def tool_call(request: Request) -> JSONResponse:
        require_auth(request, mutation=True)
        body = await request.json()
        name = str(body.get("name") or "")
        arguments = body.get("arguments") or {}
        if name not in _INSPECTOR_ALLOWED:
            return JSONResponse({"error": f"Dashboard Inspector does not allow {name!r}"}, status_code=403)
        try:
            async with Client(
                f"http://{supervisor.profile.mcp.host}:{supervisor.profile.mcp.port}/mcp", timeout=30
            ) as client:
                result = await client.call_tool(name, arguments)
            data = getattr(result, "data", None)
            if data is None:
                data = [item.model_dump(mode="json") for item in getattr(result, "content", [])]
            return JSONResponse({"ok": True, "result": data})
        except Exception as exc:
            return JSONResponse({"ok": False, "error_type": type(exc).__name__, "error": str(exc)}, status_code=502)

    async def translate(request: Request) -> JSONResponse:
        require_auth(request, mutation=True)
        body = await request.json()
        try:
            result = await asyncio.to_thread(translate_datasheet, body["csv"], body.get("output"))
            return JSONResponse(result.model_dump(mode="json"))
        except Exception as exc:
            return JSONResponse({"error_type": type(exc).__name__, "error": str(exc)}, status_code=400)

    async def convert(request: Request) -> JSONResponse:
        require_auth(request, mutation=True)
        body = await request.json()
        try:
            result = await asyncio.to_thread(
                convert_path, body["input"], body.get("output"), metadata_path=body.get("metadata"),
                substring=body.get("filter", ""), force=bool(body.get("force")),
                cpu_limit_percent=float(body.get("cpu_limit", 60)),
            )
        except Exception as exc:
            return JSONResponse({"error_type": type(exc).__name__, "error": str(exc)}, status_code=400)
        payload = result.model_dump(mode="json")
        # After a successful conversion, automatically load the first converted
        # NetCDF into the notebook as a visible cell (``data = load(...)``) so
        # the data appears in the analysis history and can be inspected.
        converted = [item for item in result.items if item.status == "converted"]
        loaded = False
        if converted:
            loaded = await asyncio.to_thread(_auto_load_into_notebook, supervisor, converted[0].output)
        payload["auto_loaded"] = loaded
        if loaded:
            payload["loaded_in_notebook"] = converted[0].output
        return JSONResponse(payload)

    async def stop(request: Request) -> JSONResponse:
        """Stop the whole stack (supervisor + JupyterLab + kernel + MCP + dashboard)."""
        require_auth(request, mutation=True)

        def _shutdown() -> None:
            try:
                supervisor.stop()
            except Exception:
                pass

        threading.Thread(target=_shutdown, daemon=True).start()
        return JSONResponse({"stopping": True})

    async def snapshot_notebook(request: Request) -> JSONResponse:
        """Save the current notebook as a timestamped snapshot without touching
        the original file (no delete / no overwrite), so a half-finished session
        can be continued later from the snapshot."""
        require_auth(request, mutation=True)
        notebook_name = supervisor.notebook_path
        content_url = f"{supervisor.jupyter_url}/api/contents/{notebook_name}"
        try:
            current = httpx.get(content_url, headers=supervisor._headers(), timeout=15)
            current.raise_for_status()
            snapshot = f"peaksMCP-snapshot-{time.strftime('%Y%m%d-%H%M%S')}.ipynb"
            created = httpx.put(
                f"{supervisor.jupyter_url}/api/contents/{snapshot}",
                headers=supervisor._headers(),
                json=current.json(),
                timeout=20,
            )
            created.raise_for_status()
            return JSONResponse({
                "snapshot": snapshot,
                "original": notebook_name,
                "open_url": f"{supervisor.jupyter_url}/lab/tree/{snapshot}?token={supervisor.token}",
            })
        except Exception as exc:
            return JSONResponse({"error_type": type(exc).__name__, "error": str(exc)}, status_code=400)

    async def choose_folder(_request: Request) -> JSONResponse:
        """Open the native macOS folder picker (Finder) and return the selected path."""
        require_auth(_request, mutation=True)
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                ["osascript", "-e", 'POSIX path of (choose folder with prompt "Select PXT folder")'],
                capture_output=True, text=True, timeout=120,
            )
        except FileNotFoundError:
            return JSONResponse({"ok": False, "error": "osascript is not available (non-macOS?)"})
        except subprocess.TimeoutExpired:
            return JSONResponse({"ok": False, "error": "folder picker timed out"})
        if result.returncode != 0:
            message = (result.stderr or "").strip()
            return JSONResponse({"ok": False, "error": message or "cancelled"})
        path = result.stdout.strip()
        if not path:
            return JSONResponse({"ok": False, "error": "cancelled"})
        return JSONResponse({"ok": True, "path": path})

    async def open_notebook(request: Request):
        require_auth(request)
        return RedirectResponse(
            f"{supervisor.jupyter_url}/lab/tree/{supervisor.notebook_path}?token={supervisor.token}",
            status_code=303,
        )

    async def log_socket(websocket: WebSocket) -> None:
        supplied = websocket.cookies.get(_DASHBOARD_COOKIE, "")
        origin = websocket.headers.get("origin")
        origin_ok = True
        if origin:
            parsed = urlsplit(origin)
            expected_host = supervisor.profile.dashboard.host
            allowed_hosts = {expected_host}
            if expected_host in {"127.0.0.1", "localhost", "::1"}:
                allowed_hosts.update({"127.0.0.1", "localhost", "::1"})
            else:
                request_host = websocket.headers.get("host", "").rsplit(":", 1)[0]
                if request_host:
                    allowed_hosts.add(request_host.strip("[]"))
            origin_ok = (
                parsed.hostname in allowed_hosts
                and (parsed.port or (443 if parsed.scheme == "https" else 80))
                == supervisor.profile.dashboard.port
            )
        if not supplied or not secrets.compare_digest(supplied, supervisor.dashboard_token) or not origin_ok:
            await websocket.close(code=4401)
            return
        await websocket.accept()
        cursor = 0
        try:
            while True:
                entries = list(supervisor.logs)
                pending = [entry for entry in entries if int(entry.get("sequence", 0)) > cursor]
                if pending:
                    await websocket.send_text(json.dumps({"logs": pending}))
                    cursor = max(int(entry["sequence"]) for entry in pending)
                await asyncio.sleep(0.5)
        except Exception:
            await websocket.close()

    return Starlette(routes=[
        Route("/", index), Route("/open-notebook", open_notebook),
        Route("/assets/{name}", asset), Route("/api/status", status),
        Route("/api/logs", logs), Route("/api/doctor", doctor), Route("/api/profiles", profiles),
        Route("/api/start-mcp", start_mcp, methods=["POST"]),
        Route("/api/restart/{component}", restart, methods=["POST"]),
        Route("/api/stop", stop, methods=["POST"]),
        Route("/api/mcp/tool", tool_call, methods=["POST"]),
        Route("/api/metadata/translate", translate, methods=["POST"]),
        Route("/api/convert", convert, methods=["POST"]),
        Route("/api/choose-folder", choose_folder, methods=["POST"]),
        Route("/api/notebook/snapshot", snapshot_notebook, methods=["POST"]),
        WebSocketRoute("/ws/logs", log_socket),
    ])
