"""Command-line entry point for peaksMCP lifecycle and data conversion."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import signal
import socket
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


def _profile(name: str):
    from .app.profiles import load_profile
    return load_profile(name)


def _runfile(required: bool = True) -> dict[str, Any] | None:
    from .observability import read_runfile
    data = read_runfile()
    if required and (not data or data.get("stale")):
        raise SystemExit("peaksMCP supervisor is not running")
    return data


def command_launch(args: argparse.Namespace) -> None:
    current = _runfile(False)
    if current and not current.get("stale"):
        _json(current)
        return
    root = Path(os.environ.get("PEAKSMCP_HOME", Path.home() / ".peaksMCP"))
    log = root / "logs" / "supervisor.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    stream = log.open("a", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, "-m", "peaksMCP", "_serve", "--profile", args.profile],
        stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
        start_new_session=True, close_fds=True,
    )
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        data = _runfile(False)
        if data and not data.get("stale"):
            _json(data)
            return
        if process.poll() is not None:
            raise SystemExit(f"supervisor exited with code {process.returncode}; inspect {log}")
        time.sleep(0.25)
    raise SystemExit(f"supervisor did not start within {args.timeout}s; inspect {log}")


def command_serve(args: argparse.Namespace) -> None:
    from .app.runtime import RuntimeSupervisor
    RuntimeSupervisor(_profile(args.profile)).serve_forever()


def command_status(_args: argparse.Namespace) -> None:
    data = _runfile(False)
    if not data:
        _json({"status": "STOPPED"})
        return
    if data.get("stale"):
        _json({**data, "status": "STALE"})
        return
    try:
        status = httpx.get(f"{data['dashboard_url']}/api/status", timeout=3).json()
    except Exception:
        status = {**data, "status": "DEGRADED", "error": "dashboard did not answer"}
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
    data = _runfile()
    try:
        response = httpx.post(f"{data['dashboard_url']}/api/restart/{args.component}", timeout=120)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise SystemExit(f"restart {args.component} failed: {exc}") from exc
    _json(response.json())


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


def _port_free(host: str, port: int) -> bool:
    with socket.socket() as sock:
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def command_doctor(args: argparse.Namespace) -> None:
    from .app.kernel import kernel_installed
    from .ext_install import extension_source, locate_installed_extension
    profile = _profile(args.profile)
    modules = {name: importlib.util.find_spec(name) is not None for name in ("peaks", "xarray", "fastmcp", "jupyterlab", "igor2", "h5netcdf")}
    report = {
        "ok": all(modules.values()), "python": sys.executable, "modules": modules,
        "kernel_installed": kernel_installed(profile.jupyter.kernel_name),
        "extension_built": (extension_source() / "static").is_dir(),
        "extension_installed": [str(path) for path in locate_installed_extension()],
        "ports_free": {
            "jupyter": _port_free(profile.jupyter.host, profile.jupyter.port),
            "mcp": _port_free(profile.mcp.host, profile.mcp.port),
            "dashboard": _port_free(profile.dashboard.host, profile.dashboard.port),
        },
    }
    report["ok"] = report["ok"] and report["extension_built"]
    _json(report)


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
    _json(asyncio.run(check_http_mcp_server(profile.mcp.host, profile.mcp.port)))


def command_translate(args: argparse.Namespace) -> None:
    from .pxt_utils import translate_datasheet
    output = args.output or str(Path(args.csv).with_name("experiment_metadata.json"))
    result = translate_datasheet(args.csv, output)
    _json({"output": output, "records": len(result.records), "warnings": result.warnings})


def command_convert(args: argparse.Namespace) -> None:
    from .pxt_utils import convert_path
    report = convert_path(args.input, args.out, metadata_path=args.metadata, substring=args.filter, force=args.force, cpu_limit_percent=args.cpu_limit)
    _json(report.model_dump(mode="json"))


def command_open(args: argparse.Namespace) -> None:
    data = _runfile()
    target = (
        data["jupyter_url"] + f"/lab/tree/{data.get('notebook_path', 'peaksMCP-runtime.ipynb')}?token={data['token']}"
        if args.jupyter else data["dashboard_url"]
    )
    webbrowser.open(target)
    print(target)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="peaksMCP", description="Claude Desktop bridge for Peaks ARPES analysis")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("version").set_defaults(func=lambda _a: print(__version__))
    launch = sub.add_parser("launch")
    launch.add_argument("--profile", default="default")
    launch.add_argument("--timeout", type=float, default=90)
    launch.set_defaults(func=command_launch)
    serve = sub.add_parser("_serve")
    serve.add_argument("--profile", default="default")
    serve.set_defaults(func=command_serve)
    sub.add_parser("status").set_defaults(func=command_status)
    sub.add_parser("stop").set_defaults(func=command_stop)
    restart = sub.add_parser("restart")
    restart.add_argument("component", choices=("kernel", "mcp", "all"))
    restart.set_defaults(func=command_restart)
    logs = sub.add_parser("logs")
    logs.add_argument("-n", "--lines", type=int, default=100)
    logs.add_argument("-f", "--follow", action="store_true")
    logs.set_defaults(func=command_logs)
    doctor = sub.add_parser("doctor")
    doctor.add_argument("--profile", default="default")
    doctor.set_defaults(func=command_doctor)
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
    opened = sub.add_parser("open")
    opened.add_argument("--jupyter", action="store_true")
    opened.set_defaults(func=command_open)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse command-line arguments and dispatch the selected implementation."""
    args = build_parser().parse_args(argv)
    args.func(args)
