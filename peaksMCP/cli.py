"""Command-line entry point for peaksMCP lifecycle and data conversion."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from typing import Any

import httpx

from peaksMCP import __version__


def _json(value: Any) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False, default=str))


def _public_run_state(data: dict[str, Any]) -> dict[str, Any]:
    """Remove local bearer credentials before printing run state."""
    return {key: value for key, value in data.items() if key not in {"token", "dashboard_token"}}


def _terminate_spawned_supervisor(process: subprocess.Popen[Any]) -> None:
    """Stop a supervisor spawned by a launch attempt that did not complete."""
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def _profile(name: str):
    from .app.profiles import load_profile
    return load_profile(name)


def _runfile(required: bool = True) -> dict[str, Any] | None:
    from .observability import read_runfile
    data = read_runfile()
    if required and (not data or data.get("stale")):
        raise SystemExit("peaksMCP supervisor is not running")
    return data


def _dashboard_headers(data: dict[str, Any]) -> dict[str, str]:
    """Return dashboard authentication headers, tolerating pre-auth runfiles."""
    token = data.get("dashboard_token")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _launch_readiness(data: dict[str, Any]) -> tuple[bool, dict[str, Any] | None]:
    """Probe the operator console for the stages required by ``launch``."""
    try:
        response = httpx.get(
            f"{data['dashboard_url']}/api/status",
            headers=_dashboard_headers(data),
            timeout=2,
        )
        response.raise_for_status()
        status = response.json()
    except Exception:
        return False, None
    components = status.get("components") or {}
    required = ["supervisor", "jupyter", "kernel", "extension"]
    if data.get("mcp_autostart", True):
        required.append("mcp")
    ready = all(
        isinstance(components.get(name), dict)
        and components[name].get("state") == "ready"
        for name in required
    )
    return ready, status


def command_launch(args: argparse.Namespace) -> None:
    current = _runfile(False)
    root = Path(os.environ.get("PEAKSMCP_HOME", Path.home() / ".peaksMCP"))
    log = root / "logs" / "supervisor.log"
    process: subprocess.Popen[Any] | None = None
    if not current or current.get("stale"):
        log.parent.mkdir(parents=True, exist_ok=True)
        stream = log.open("a", encoding="utf-8")
        notebook_arg = ["--notebook", args.notebook] if getattr(args, "notebook", None) else []
        try:
            process = subprocess.Popen(
                [sys.executable, "-m", "peaksMCP", "_serve", "--profile", args.profile, *notebook_arg],
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                start_new_session=True, close_fds=True,
            )
        finally:
            stream.close()
    deadline = time.monotonic() + args.timeout
    last_status: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        data = _runfile(False)
        if data and not data.get("stale"):
            ready, last_status = _launch_readiness(data)
            if ready:
                payload = {**_public_run_state(data), "ready": True}
                if last_status:
                    payload["components"] = last_status.get("components", {})
                _json(payload)
                return
        if process is not None and process.poll() is not None:
            raise SystemExit(f"supervisor exited with code {process.returncode}; inspect {log}")
        time.sleep(0.25)
    if process is not None:
        _terminate_spawned_supervisor(process)
    stage = ""
    if last_status:
        pending = [
            name
            for name, component in (last_status.get("components") or {}).items()
            if component.get("state") != "ready" and name != "comm"
        ]
        if pending:
            stage = f"; pending: {', '.join(pending)}"
    raise SystemExit(
        f"supervisor did not become ready within {args.timeout}s{stage}; inspect {log}"
    )


def command_serve(args: argparse.Namespace) -> None:
    from .app.runtime import RuntimeSupervisor
    supervisor = RuntimeSupervisor(_profile(args.profile))
    if getattr(args, "notebook", None):
        supervisor.notebook_path = args.notebook
    supervisor.serve_forever()


def command_status(_args: argparse.Namespace) -> None:
    data = _runfile(False)
    if not data:
        _json({"status": "STOPPED"})
        return
    if data.get("stale"):
        _json(_public_run_state({**data, "status": "STALE"}))
        return
    try:
        status = httpx.get(
            f"{data['dashboard_url']}/api/status",
            headers=_dashboard_headers(data),
            timeout=3,
        ).json()
    except Exception:
        status = _public_run_state(
            {**data, "status": "DEGRADED", "error": "dashboard did not answer"}
        )
    _json(status)


def command_stop(_args: argparse.Namespace) -> None:
    data = _runfile()
    os.kill(int(data["pid"]), signal.SIGTERM)
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        try:
            os.kill(int(data["pid"]), 0)
        except OSError:
            _json({"stopped": True, "pid": data["pid"]})
            return
        time.sleep(0.2)
    raise SystemExit("supervisor did not stop within 20 seconds")


def command_restart(args: argparse.Namespace) -> None:
    """Restart one component, or the whole stack when no component is given.

    ``peaksMCP restart`` (no component) is equivalent to ``stop && launch``:
    it stops the running supervisor and starts a fresh one (JupyterLab + kernel +
    in-kernel MCP + dashboard).  ``restart kernel|mcp|kernel&mcp`` only touches
    the kernel side inside the running supervisor (``kernel&mcp`` = kernel + the
    in-kernel MCP).
    """
    if args.component is None:
        _restart_stack(args)
        return
    component = "all" if args.component == "kernel&mcp" else args.component
    data = _runfile()
    try:
        response = httpx.post(
            f"{data['dashboard_url']}/api/restart/{component}",
            headers=_dashboard_headers(data),
            timeout=120,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise SystemExit(f"restart {component} failed: {exc}") from exc
    payload = response.json()
    _json(payload)
    if payload.get("ready") is False:
        raise SystemExit(f"restart {component} did not become ready")


def _restart_stack(args: argparse.Namespace) -> None:
    """Full-stack restart: stop the current supervisor, then launch a fresh one."""
    from .observability import read_runfile

    data = read_runfile()
    if data and not data.get("stale"):
        try:
            os.kill(int(data["pid"]), signal.SIGTERM)
        except ProcessLookupError:
            pass  # already gone
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            try:
                os.kill(int(data["pid"]), 0)
            except OSError:
                break
            time.sleep(0.2)
        else:
            raise SystemExit("supervisor did not stop within 20 seconds")
    # No supervisor running (or just stopped): launch a fresh stack, exactly
    # like `peaksMCP launch`.
    command_launch(args)


def command_logs(args: argparse.Namespace) -> None:
    root = Path(os.environ.get("PEAKSMCP_HOME", Path.home() / ".peaksMCP"))
    path = root / "logs" / "supervisor.log"
    if not path.exists():
        raise SystemExit("no supervisor log exists")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    print("\n".join(lines[-args.lines:]))
    if args.follow:
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            stream.seek(0, 2)
            try:
                while True:
                    line = stream.readline()
                    if line:
                        print(line, end="")
                    else:
                        time.sleep(0.2)
            except KeyboardInterrupt:
                pass


def command_profiles(args: argparse.Namespace) -> None:
    from .app.profiles import list_profiles, load_profile, profile_path
    if args.action == "list":
        _json({"profiles": list_profiles()})
    elif args.action == "show":
        _json(load_profile(args.name).model_dump(mode="json"))
    else:
        print(profile_path(args.name))


def command_install_extension(args: argparse.Namespace) -> None:
    from .ext_install import install_extension
    print(install_extension(develop=args.develop))


def command_install_kernel(args: argparse.Namespace) -> None:
    from .app.kernel import install_kernel
    print(install_kernel(_profile(args.profile)))


def command_uninstall_kernel(args: argparse.Namespace) -> None:
    from .app.kernel import uninstall_kernel
    _json({"removed": uninstall_kernel(_profile(args.profile).jupyter.kernel_name)})


def command_proxy(args: argparse.Namespace) -> None:
    from .transport.stdio_proxy import run_stdio_proxy
    profile = _profile(args.profile)
    run_stdio_proxy(profile.mcp.host, profile.mcp.port)


def command_ping(args: argparse.Namespace) -> None:
    from .transport import check_http_mcp_server
    profile = _profile(args.profile)
    payload = asyncio.run(check_http_mcp_server(profile.mcp.host, profile.mcp.port))
    _json(payload)
    if not payload.get("ok"):
        raise SystemExit(1)


def command_translate(args: argparse.Namespace) -> None:
    from .pxt_utils import convert_path, translate_datasheet
    output = args.output or str(Path(args.csv).with_name("experiment_metadata.json"))
    result = translate_datasheet(args.csv, output)
    payload = {"output": output, "records": len(result.records), "warnings": result.warnings}
    if args.pxt_dir:
        # --pxt-dir: immediately convert the folder with the freshly translated
        # metadata (default sibling <folder>_netcdf output).
        report = convert_path(args.pxt_dir, metadata_path=output)
        payload["converted"] = {
            "matched": len(report.items),
            "converted": report.converted,
            "failed": report.failed,
        }
    _json(payload)


def command_convert(args: argparse.Namespace) -> None:
    from .pxt_utils import convert_path
    report = convert_path(args.input, args.out, metadata_path=args.metadata, substring=args.filter, force=args.force, cpu_limit_percent=args.cpu_limit)
    _json(report.model_dump(mode="json"))
    if report.failed:
        raise SystemExit(1)


def command_load(args: argparse.Namespace) -> None:
    """Explicitly load a converted NetCDF into the notebook as a visible cell.

    Asks the dashboard to insert and run ``from peaks import load; data = load(...)``
    in the live notebook (requires the supervisor / frontend Comm to be online).
    """
    data = _runfile()
    # Resolve the path against the CLI's cwd so a relative path does not get
    # interpreted inside the dashboard/supervisor process (different cwd).
    path = str(Path(args.input).expanduser().resolve())
    try:
        response = httpx.post(
            f"{data['dashboard_url']}/api/notebook/load",
            headers=_dashboard_headers(data),
            json={"path": path},
            timeout=120,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise SystemExit(f"load failed: {exc}") from exc
    payload = response.json()
    _json(payload)
    if payload.get("loaded") is not True:
        raise SystemExit("load was not confirmed by the notebook")


def command_open(args: argparse.Namespace) -> None:
    """Open the operator-console dashboard (default) or the managed notebook.

    Invoked via ``peaksMCP dash`` (primary name) or the legacy alias
    ``peaksMCP open``.
    """
    data = _runfile()
    target = (
        data["jupyter_url"] + f"/lab/tree/{data.get('notebook_path', 'peaksMCP-runtime.ipynb')}?token={data['token']}"
        if args.jupyter
        else (
            f"{data['dashboard_url']}/?token={data['dashboard_token']}"
            if data.get("dashboard_token")
            else data["dashboard_url"]
        )
    )
    webbrowser.open(target)
    print(target)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="peaksMCP", description="Claude Desktop bridge for Peaks ARPES analysis")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("version").set_defaults(func=lambda _a: print(__version__))
    launch = sub.add_parser("launch", help="start the supervisor; optional notebook path (e.g. a saved snapshot)")
    launch.add_argument("notebook", nargs="?", default=None, help="notebook file to use as the workspace (e.g. peaksMCP-snapshot-xxx.ipynb)")
    launch.add_argument("--profile", default="default")
    launch.add_argument("--timeout", type=float, default=90)
    launch.set_defaults(func=command_launch)
    serve = sub.add_parser("_serve")
    serve.add_argument("--profile", default="default")
    serve.add_argument("--notebook", default=None)
    serve.set_defaults(func=command_serve)
    sub.add_parser("status").set_defaults(func=command_status)
    sub.add_parser("stop").set_defaults(func=command_stop)
    restart = sub.add_parser("restart", help="restart the whole stack, or one component (kernel|mcp|kernel&mcp)")
    restart.add_argument(
        "component", nargs="?",
        choices=("kernel", "mcp", "kernel&mcp"),
        default=None,
        help="component to restart: kernel, mcp, or kernel&mcp (kernel + in-kernel MCP). "
        "Omit to restart the whole stack (like launch). "
        "Note: the & must be quoted in most shells, e.g. peaksMCP restart 'kernel&mcp'.",
    )
    restart.add_argument("--profile", default="default")
    restart.add_argument("--timeout", type=float, default=90)
    restart.set_defaults(func=command_restart)
    logs = sub.add_parser("logs")
    logs.add_argument("-n", "--lines", type=int, default=100)
    logs.add_argument("-f", "--follow", action="store_true")
    logs.set_defaults(func=command_logs)
    profiles = sub.add_parser("profiles")
    profiles.add_argument("action", choices=("list", "show", "path"))
    profiles.add_argument("name", nargs="?", default="default")
    profiles.set_defaults(func=command_profiles)
    ext = sub.add_parser("install-extension")
    ext.add_argument("--develop", action="store_true")
    ext.set_defaults(func=command_install_extension)
    ik = sub.add_parser("install-kernel")
    ik.add_argument("--profile", default="default")
    ik.set_defaults(func=command_install_kernel)
    uk = sub.add_parser("uninstall-kernel")
    uk.add_argument("--profile", default="default")
    uk.set_defaults(func=command_uninstall_kernel)
    proxy = sub.add_parser("stdio-proxy")
    proxy.add_argument("--profile", default="default")
    proxy.set_defaults(func=command_proxy)
    ping = sub.add_parser("mcp-ping")
    ping.add_argument("--profile", default="default")
    ping.set_defaults(func=command_ping)
    metadata = sub.add_parser("metadata")
    metadata_sub = metadata.add_subparsers(dest="metadata_command", required=True)
    translate = metadata_sub.add_parser("translate")
    translate.add_argument("csv")
    translate.add_argument("--pxt-dir")
    translate.add_argument("--output")
    translate.set_defaults(func=command_translate)
    convert = sub.add_parser("convert")
    convert.add_argument("input")
    convert.add_argument("--metadata")
    convert.add_argument("--out")
    convert.add_argument("--filter", default="")
    convert.add_argument("--cpu-limit", type=float, default=60)
    convert.add_argument("--force", action="store_true")
    convert.set_defaults(func=command_convert)
    load_cmd = sub.add_parser("load", help="load a converted NetCDF into the notebook as a visible cell")
    load_cmd.add_argument("input", help="path to the .nc file to load")
    load_cmd.set_defaults(func=command_load)
    dash = sub.add_parser("dash", help="open the operator-console dashboard (or the managed notebook with --jupyter)")
    dash.add_argument("--jupyter", action="store_true")
    dash.set_defaults(func=command_open)
    # Legacy alias for the renamed command.
    opened = sub.add_parser("open", help=argparse.SUPPRESS)
    opened.add_argument("--jupyter", action="store_true")
    opened.set_defaults(func=command_open)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse command-line arguments and dispatch the selected implementation."""
    args = build_parser().parse_args(argv)
    args.func(args)
