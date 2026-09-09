"""Operator-console API + static webapp, served inside the supervisor process.

The dashboard is co-hosted by the supervisor (``peaksMCP launch`` is the single
startup entry) and binds the live :class:`RuntimeSupervisor`, so it can both
*monitor* (JupyterLab, managed kernel, in-kernel MCP, Comm) and *control*
(start/restart MCP, restart kernel, open the notebook, stop the stack).
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import httpx
from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, PlainTextResponse, RedirectResponse
from starlette.routing import Route

if TYPE_CHECKING:
    from .runtime import RuntimeSupervisor

_DASHBOARD_COOKIE = "peaksmcp_dashboard"


def _flush_frontend_save(supervisor: RuntimeSupervisor) -> None:
    """Ask the frontend Comm bridge to persist the notebook.

    Raises when the frontend is unavailable or ``context.save()`` did not
    confirm success.  Callers must never snapshot stale on-disk content while
    presenting the operation as successful.
    """
    code = (
        "from peaksMCP.server.jupyter_peaks.jupyter_mcp_extension import get_server as _gs;"
        "b = _gs().state.bridge;"
        "r = b.request('save_notebook', {}, timeout=10);"
        "assert r.get('saved') is True, "
        "('Notebook save was not confirmed: ' + str(r.get('error') or r.get('save_error') or r))"
    )
    result = supervisor.execute_kernel(code, timeout=12)
    if result.get("status") != "ok":
        raise RuntimeError(f"Notebook save failed: {result}")


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
    """Compose the live operator-console status snapshot.

    Tolerates a still-starting stack: while Jupyter/kernel/MCP come up the
    individual probes may fail, but the status endpoint must always answer 200
    so the dashboard (and ``peaksMCP dash`` readiness) can show "starting".
    """
    base = supervisor.status()
    try:
        mcp, kernel_state = await asyncio.gather(
            _mcp_probe(supervisor),
            _jupyter_kernel_state(supervisor),
        )
    except Exception:
        mcp, kernel_state = {}, "unknown"
    mcp_status = mcp.get("status") if isinstance(mcp.get("status"), dict) else {}
    jupyter_up = bool(supervisor.jupyter and supervisor.jupyter.poll() is None)
    # Readiness comes from the declared group state, not process existence:
    # jupyter_state only becomes "running" once JupyterLab answers HTTP *and*
    # the managed kernel session exists.  "starting" is a real state (never an
    # error), and a process that died after being declared running is an error
    # rather than silently "ready".
    declared = base.get("jupyter_state")
    if declared is None:  # legacy supervisors without the group state
        declared = "running" if jupyter_up else "stopped"
    if declared == "running":
        jupyter_ui = "ready" if jupyter_up else "error"
    elif declared == "starting":
        jupyter_ui = "starting"
    else:
        jupyter_ui = "stopped"
    # The in-kernel MCP only runs inside the notebook kernel, so ``mcp.ok`` is
    # the strongest proof the kernel is alive and usable (Jupyter's REST
    # execution_state can remain "starting" even while the kernel serves cells).
    kernel_ready = bool(mcp.get("ok")) or kernel_state in {"idle", "busy"}
    extension: dict[str, Any] = {"loaded": False, "detail": "kernel unavailable"}
    try:
        if mcp.get("ok"):
            extension = {"loaded": True, "detail": "IPython extension loaded"}
        elif kernel_ready:
            extension = await asyncio.to_thread(supervisor.extension_status)
    except Exception:
        pass  # kernel is still coming up; report extension as unavailable
    components = {
        "supervisor": {"state": "ready", "detail": f"PID {base['pid']}"},
        "jupyter": {"state": jupyter_ui, "detail": supervisor.jupyter_url},
        "kernel": {"state": "ready" if kernel_ready else "degraded", "detail": kernel_state},
        "extension": {
            "state": "ready" if extension["loaded"] else "degraded",
            "detail": extension["detail"],
        },
        "comm": {"state": "ready" if mcp_status.get("comm_connected") else "degraded", "detail": "JupyterLab connected" if mcp_status.get("comm_connected") else "Open the managed notebook"},
        "mcp": {"state": "ready" if mcp.get("ok") else "error", "detail": f"{mcp.get('tool_count', 0)} tools · {base['mcp_url']}"},
    }
    states = {item["state"] for item in components.values()}
    aggregate = (
        "error"
        if "error" in states
        else "ready"
        if states <= {"ready"}
        else "degraded"
    )
    payload = {
        **base,
        "status": (
            "RUNNING" if jupyter_ui == "ready"
            else "STARTING" if declared == "starting"
            else "STOPPED"
        ),
        "supervisor_running": True,
        "kernel_state": kernel_state,
        "aggregate": aggregate,
        "components": components,
        "mcp": mcp,
    }
    # The managed notebook can only be opened once Jupyter is truly usable;
    # exposing the URL while stopped/starting is what made the dashboard's
    # "Open Notebook" control look available too early.
    if jupyter_ui == "ready":
        payload["notebook_open_url"] = "/open-notebook"
    return payload


async def _create_notebook_snapshot(supervisor: RuntimeSupervisor) -> str:
    """Persist and copy the managed notebook without blocking the event loop.

    The destination combines UTC microseconds with a random suffix and is
    checked before creation.  The Contents API payload contains only the
    writable notebook fields, so the original path/name cannot leak into the
    copy request.
    """
    await asyncio.to_thread(_flush_frontend_save, supervisor)
    headers = supervisor._headers()
    notebook_name = supervisor.notebook_path
    content_url = f"{supervisor.jupyter_url}/api/contents/{notebook_name}"
    async with httpx.AsyncClient(timeout=20) as client:
        current = await client.get(content_url, headers=headers)
        current.raise_for_status()
        model = current.json()
        payload = {
            key: model[key]
            for key in ("type", "format", "content")
            if key in model
        }
        for _attempt in range(5):
            stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
            snapshot = f"peaksMCP-snapshot-{stamp}-{secrets.token_hex(4)}.ipynb"
            target_url = f"{supervisor.jupyter_url}/api/contents/{snapshot}"
            existing = await client.get(target_url, headers=headers)
            if existing.status_code == 404:
                created = await client.put(target_url, headers=headers, json=payload)
                created.raise_for_status()
                return snapshot
            existing.raise_for_status()
        raise FileExistsError("could not allocate a unique notebook snapshot name")


#: 最近操作视图的读取参数（dashboard 只读审计日志，展示最近的活动链）。
_RECENT_ACTIVITY_MAX_CHAINS = 12
_RECENT_ACTIVITY_MAX_EVENTS = 600


def audit_log_path() -> Path:
    """Host 侧审计日志路径（dashboard 进程与 kernel 共享 PEAKSMCP_HOME）。"""
    home = Path(os.environ.get("PEAKSMCP_HOME", str(Path.home() / ".peaksMCP")))
    return home / "audit" / "tool_audit.log"


def recent_audit_chains(
    path: Path | None = None,
    *,
    max_events: int = _RECENT_ACTIVITY_MAX_EVENTS,
    max_chains: int = _RECENT_ACTIVITY_MAX_CHAINS,
) -> dict[str, Any]:
    """读审计日志尾部并聚合出最近的 operation 链。

    每条链 = 同一个 operation_id 下的全部事件（called → blocked/executed/
    saved/…），终端事件与票据信息（ticket_id / sha256 前缀 / cell_id /
    api_ids）都聚合出来，供 dashboard 直接渲染"一次操作干了什么"。
    旧版审计（无 operation_id）退化为单事件行，附在 ``unattached`` 里。
    """
    path = path or audit_log_path()
    raw: list[dict[str, Any]] = []
    if path.is_file():
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    raw = raw[-max_events:]

    chains: dict[str, dict[str, Any]] = {}
    unattached: list[dict[str, Any]] = []
    for event in raw:
        operation_id = str((event.get("details") or {}).get("operation_id") or "")
        if not operation_id:
            unattached.append(_compact_event(event))
            continue
        chain = chains.setdefault(operation_id, {
            "operation_id": operation_id,
            "first_at": event.get("timestamp"),
            "last_at": event.get("timestamp"),
            "tools": [],
            "outcomes": [],
            "cell_ids": [],
            "api_ids": [],
            "ticket_id": None,
            "sha256": None,
            "target_path": None,
            "events": [],
        })
        chain["last_at"] = event.get("timestamp", chain.get("last_at"))
        tool = str(event.get("tool") or "")
        outcome = str(event.get("outcome") or "")
        if tool and tool not in chain["tools"]:
            chain["tools"].append(tool)
        if outcome and outcome not in chain["outcomes"]:
            chain["outcomes"].append(outcome)
        details = event.get("details") or {}
        cell_id = details.get("cell_id")
        if cell_id and cell_id not in chain["cell_ids"]:
            chain["cell_ids"].append(str(cell_id))
        api_ids = details.get("api_ids")
        if isinstance(api_ids, list):
            for api_id in api_ids:
                if api_id not in chain["api_ids"]:
                    chain["api_ids"].append(str(api_id))
        if details.get("ticket_id"):
            chain["ticket_id"] = str(details["ticket_id"])
        if details.get("sha256"):
            chain["sha256"] = str(details["sha256"])
        if outcome == "called":
            args = details.get("args") or {}
            if args.get("path"):
                chain["target_path"] = str(args["path"])
        chain["events"].append(_compact_event(event))

    ordered = sorted(chains.values(), key=lambda c: str(c.get("last_at") or ""), reverse=True)
    return {
        "chains": ordered[:max_chains],
        "unattached": unattached[-10:],
        "source": str(path),
    }


def _compact_event(event: dict[str, Any]) -> dict[str, Any]:
    """一条事件的可展示摘要（只读，不展开 args 里的大对象）。"""
    details = event.get("details") or {}
    compact = {
        "timestamp": event.get("timestamp"),
        "tool": event.get("tool"),
        "outcome": event.get("outcome"),
    }
    args = details.get("args")
    if isinstance(args, dict) and "code" in args:
        code = str(args["code"])
        compact["code_head"] = code[:160]
    for key in ("cell_id", "ticket_id", "sha256", "operation_id", "error_type", "error"):
        if details.get(key):
            compact[key] = details[key]
    return compact


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
        response = FileResponse(web / "index.html")
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
            "script-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
        )
        return response

    async def asset(request: Request) -> FileResponse:
        name = request.path_params["name"]
        if name not in {"app.js", "style.css"}:
            raise FileNotFoundError(name)
        response = FileResponse(web / name)
        # The webapp is a live operator console bound to a changing token; never
        # let browsers cache stale JS/CSS (a stale app.js referencing removed
        # elements would crash and blank the dashboard).
        response.headers["Cache-Control"] = "no-store"
        return response

    async def status(_request: Request) -> JSONResponse:
        require_auth(_request)
        return JSONResponse(await status_payload(supervisor))

    async def start_mcp(_request: Request) -> JSONResponse:
        require_auth(_request, mutation=True)
        try:
            return JSONResponse(await asyncio.to_thread(supervisor.start_mcp))
        except Exception as exc:
            return JSONResponse({"error_type": type(exc).__name__, "error": str(exc)}, status_code=409)

    async def jupyter_control(request: Request) -> JSONResponse:
        require_auth(request, mutation=True)
        action = request.path_params["action"]
        method = {
            "start": supervisor.start_jupyter,
            "stop": supervisor.stop_jupyter,
            "restart": supervisor.restart_jupyter,
        }.get(action)
        if method is None:
            return JSONResponse({"error": "action must be start, stop or restart"}, status_code=400)
        try:
            return JSONResponse(await asyncio.to_thread(method))
        except Exception as exc:
            return JSONResponse({"error_type": type(exc).__name__, "error": str(exc)}, status_code=409)

    async def stop_mcp(_request: Request) -> JSONResponse:
        require_auth(_request, mutation=True)
        try:
            return JSONResponse(await asyncio.to_thread(supervisor.stop_mcp))
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

    async def snapshot_notebook(request: Request) -> JSONResponse:
        """Save the current notebook as a timestamped snapshot without touching
        the original file (no delete / no overwrite), so a half-finished session
        can be continued later from the snapshot."""
        require_auth(request, mutation=True)
        try:
            snapshot = await _create_notebook_snapshot(supervisor)
            return JSONResponse({
                "snapshot": snapshot,
                "original": supervisor.notebook_path,
                "open_url": f"{supervisor.jupyter_url}/lab/tree/{snapshot}?token={supervisor.token}",
            })
        except Exception as exc:
            return JSONResponse({"error_type": type(exc).__name__, "error": str(exc)}, status_code=400)

    async def recent_activity(_request: Request) -> JSONResponse:
        require_auth(_request)
        payload = await asyncio.to_thread(recent_audit_chains)
        return JSONResponse(payload)

    async def open_notebook(request: Request):
        require_auth(request)
        return RedirectResponse(
            f"{supervisor.jupyter_url}/lab/tree/{supervisor.notebook_path}?token={supervisor.token}",
            status_code=303,
        )

    return Starlette(routes=[
        Route("/", index), Route("/open-notebook", open_notebook),
        Route("/assets/{name}", asset), Route("/api/status", status),
        Route("/api/start-mcp", start_mcp, methods=["POST"]),
        Route("/api/mcp/stop", stop_mcp, methods=["POST"]),
        Route("/api/jupyter/{action}", jupyter_control, methods=["POST"]),
        Route("/api/restart/{component}", restart, methods=["POST"]),
        # Conversion / datasheet translation / loading a scan are deliberately
        # NOT endpoints: they run as notebook cells through the MCP tools so
        # the scanner, API check and consent gate apply. The console only
        # controls processes (Jupyter / kernel / MCP) and snapshots.
        Route("/api/notebook/snapshot", snapshot_notebook, methods=["POST"]),
        Route("/api/activity/recent", recent_activity),
    ])
