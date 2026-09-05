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
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import httpx
from fastmcp import Client
from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, PlainTextResponse, RedirectResponse
from starlette.routing import Route

from peaksMCP.pxt_utils import convert_path, translate_datasheet

if TYPE_CHECKING:
    from .runtime import RuntimeSupervisor

_INSPECTOR_ALLOWED = {
    "peaks_search_api", "peaks_get_api", "notebook_server_status",
    "notebook_kernel_status", "notebook_list_variables", "notebook_read_variable",
    "notebook_read_active_cell", "notebook_read_active_cell_output",
}
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


def load_into_notebook(
    supervisor: RuntimeSupervisor,
    nc_path: str | list[str] | None,
    *,
    timeout: float = 45,
) -> bool:
    """Load converted NetCDF file(s) into the notebook as one visible cell.

    Asks the kernel to insert and run a single code cell that loads all paths,
    e.g. ``from peaks import load\ndata = load(...)\ndata_2 = load(...)``,
    through the frontend Comm bridge (so the cell appears in the notebook and is
    saved), instead of silently loading into the kernel namespace or creating
    one cell per file.

    Parameters
    ----------
    supervisor : RuntimeSupervisor
        Supervisor connected to the managed kernel.
    nc_path : str, list of str, or None
        NetCDF file(s) passed to ``peaks.load`` in the visible cell.  A single
        path is loaded as ``data``; additional paths become ``data_2``,
        ``data_3``, ... in the same cell (no ``data`` overwrite).
    timeout : float, default 45
        Seconds allowed to confirm this Load operation.

    Returns
    -------
    bool
        True only after the matching code cell reports successful execution.
        False means failure or an unconfirmed outcome, not cancellation.

    Notes
    -----
    Existing ``data`` variables do not count as evidence of this operation.
    Failed/timed-out loads are not automatically replayed.

    Examples
    --------
    >>> load_into_notebook(supervisor, "/data/BP_0001.nc")  # doctest: +SKIP
    True
    """

    if not nc_path or timeout <= 0:
        return False
    # Accept a single path (legacy) or a list of paths.  All paths are loaded in
    # ONE visible cell with distinct variable names (``data``, ``data_2``, ...)
    # instead of one cell per file overwriting ``data``.
    paths = [nc_path] if isinstance(nc_path, str) else [p for p in nc_path if p]
    if not paths:
        return False
    load_lines = []
    for i, p in enumerate(paths):
        var = "data" if i == 0 else f"data_{i + 1}"
        load_lines.append(f"{var} = load({json.dumps(str(p))})")
    # Expose the experiment metadata document (datasheet records, Au references,
    # agent notes) that the converter writes next to the converted files, as a
    # ``metadata`` dict in the same cell, when present.
    metadata_doc = Path(paths[0]).expanduser().resolve().parent / "experiment_metadata.json"
    metadata_lines = []
    if metadata_doc.is_file():
        metadata_lines = [
            "import json",
            f"metadata = json.load(open({str(metadata_doc)!r}, encoding='utf-8'))",
        ]
    load_code = "from peaks import load\n" + "\n".join([*load_lines, *metadata_lines])
    slot = f"_peaksMCP_load_{secrets.token_hex(16)}"
    # Each request gets an independent job, captured by the worker closure.
    # The shell must be released so Jupyter can execute the frontend's cell.
    bridge_code = f'''def _peaksMCP_start_load():
    from threading import Thread
    from peaksMCP.server.jupyter_peaks.jupyter_mcp_extension import get_server
    job = {{"status": "pending"}}
    get_ipython().user_ns[{slot!r}] = job
    bridge = get_server().state.bridge
    code = {load_code!r}
    def run():
        try:
            result = bridge.request("execute_code", {{"code": code}}, timeout={timeout!r})
            if (result.get("execution_success") is not True
                    or result.get("cell_type") != "code"
                    or result.get("source") != code or not result.get("id")
                    or result.get("saved") is not True
                    or any(output.get("output_type") == "error" for output in result.get("outputs", []))):
                raise RuntimeError(
                    "Load cell did not confirm successful execution and notebook save"
                    + (": " + str(result.get("save_error")) if result.get("save_error") else "")
                )
            job["status"] = "completed"
        except Exception as exc:
            job.update(status="failed", error=str(exc))
    Thread(target=run, daemon=True).start()
try:
    _peaksMCP_start_load()
finally:
    del _peaksMCP_start_load
'''
    probe_code = f'''if get_ipython().user_ns.get({slot!r}, {{}}).get("status") != "completed":
    raise RuntimeError({slot!r} + ":" + get_ipython().user_ns.get({slot!r}, {{}}).get("status", "missing"))
'''
    deadline = time.monotonic() + timeout
    try:
        started = supervisor.execute_kernel(bridge_code, timeout=min(10, timeout))
        if started.get("status") != "ok":
            return False
        while time.monotonic() < deadline:
            try:
                reply = supervisor.execute_kernel(probe_code, timeout=min(5, deadline - time.monotonic()))
                return reply.get("status") == "ok"
            except RuntimeError as exc:
                if ":failed" in str(exc):
                    # The load cell itself reported a failure: permanent.
                    return False
                # Otherwise the cell is still pending / the kernel is busy.
            except Exception:
                # A busy kernel or channel timeouts surface as various exception
                # types (TimeoutError, queue.Empty, ...); they are transient and
                # must be retried — never report failure while the queued Load
                # may still succeed (a retry would duplicate the execution).
                pass
            time.sleep(min(0.25, max(0, deadline - time.monotonic())))
        return False
    except Exception:
        return False
    finally:
        try:
            supervisor.execute_kernel(f"get_ipython().user_ns.pop({slot!r}, None)", timeout=2)
        except Exception:
            # A crashed/unreachable kernel cannot acknowledge cleanup either.
            pass


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
        # Expose the converted output directory to the notebook kernel so the
        # agent can load the NetCDF files directly without guessing the path,
        # e.g. ``pks.load(f'{CONVERTED_DIR}/BP_0005.nc')``. Best-effort: the
        # conversion result is returned even if the kernel write fails.
        converted = [item.output for item in result.items if item.status == "converted" and item.output]
        if converted:
            try:
                await asyncio.to_thread(
                    supervisor.export_variable, "CONVERTED_DIR", str(Path(converted[0]).parent)
                )
            except Exception:
                pass
        return JSONResponse(payload)

    async def load_notebook(request: Request) -> JSONResponse:
        """Insert and run one ``data = load(...)`` cell for the given NetCDF path(s).

        Accepts a ``paths`` list (multi-load from the dashboard) or a single
        ``path`` (legacy, used by ``peaksMCP load``).  All paths are loaded in a
        single visible cell (variables ``data``, ``data_2``, ...) rather than one
        cell per file; results are reported per path (all-or-nothing).
        """
        require_auth(request, mutation=True)
        body = await request.json()
        raw = body.get("paths")
        if raw is None:
            legacy_path = body.get("path")
            if legacy_path is not None and not isinstance(legacy_path, str):
                return JSONResponse({"error": "path must be a string"}, status_code=400)
            raw = [legacy_path] if legacy_path else []
        elif not isinstance(raw, list):
            return JSONResponse({"error": "paths must be a list of strings"}, status_code=400)
        if len(raw) > 100 or any(not isinstance(path, str) for path in raw):
            return JSONResponse(
                {"error": "paths must contain at most 100 strings"}, status_code=400
            )
        paths = [path for path in raw if path]
        if not paths:
            return JSONResponse({"error": "path or paths is required"}, status_code=400)
        results: dict[str, bool] = {}
        try:
            # All paths go into ONE notebook cell (variables data, data_2, ...);
            # previously each path created its own cell overwriting ``data``.
            ok = await asyncio.to_thread(
                load_into_notebook,
                supervisor,
                paths,
                timeout=min(300, 45 + 10 * len(paths)),
            )
        except Exception:
            ok = False
        for p in paths:
            results[p] = ok
        return JSONResponse({"paths": paths, "results": results, "loaded": ok})

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

    return Starlette(routes=[
        Route("/", index), Route("/open-notebook", open_notebook),
        Route("/assets/{name}", asset), Route("/api/status", status),
        Route("/api/start-mcp", start_mcp, methods=["POST"]),
        Route("/api/mcp/stop", stop_mcp, methods=["POST"]),
        Route("/api/jupyter/{action}", jupyter_control, methods=["POST"]),
        Route("/api/restart/{component}", restart, methods=["POST"]),
        Route("/api/mcp/tool", tool_call, methods=["POST"]),
        Route("/api/metadata/translate", translate, methods=["POST"]),
        Route("/api/convert", convert, methods=["POST"]),
        Route("/api/choose-folder", choose_folder, methods=["POST"]),
        Route("/api/notebook/snapshot", snapshot_notebook, methods=["POST"]),
        Route("/api/notebook/load", load_notebook, methods=["POST"]),
    ])
