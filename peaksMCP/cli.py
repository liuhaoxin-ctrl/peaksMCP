"""Command-line entry point for peaksMCP lifecycle and data conversion."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from typing import Any

import httpx
import psutil

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


def _dashboard_port(args: argparse.Namespace) -> int:
    """Configured dashboard port for the active profile (default 8765)."""
    from .app.profiles import load_profile

    name = getattr(args, "profile", "default")
    return load_profile(name).dashboard.port


def _pid_command(pid: int) -> str:
    """Best-effort command line of ``pid``; ``"?"`` when it is unreadable."""
    try:
        return " ".join(psutil.Process(pid).cmdline())
    except Exception:
        try:
            output = subprocess.run(
                ["ps", "-p", str(pid), "-o", "command="],
                capture_output=True, text=True, timeout=5,
                check=False,
            ).stdout.strip()
            return output or "?"
        except Exception:
            return "?"


def _listener_on_port(port: int) -> list[tuple[int, str]]:
    """Return ``(pid, cmdline)`` pairs for TCP listeners bound to ``port``.

    ``lsof`` is used first because it avoids enumerating every process's file
    descriptors (which raises ``AccessDenied`` on macOS for processes the
    caller cannot introspect and is slow); psutil is the fallback for systems
    without lsof.  A listener whose command line cannot be read is reported
    with a ``"?"`` cmdline rather than failing the whole scan.
    """
    found: list[tuple[int, str]] = []
    if shutil.which("lsof"):
        try:
            result = subprocess.run(
                ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
                capture_output=True, text=True, timeout=5,
                check=False,
            )
            for line in result.stdout.split():
                try:
                    pid = int(line)
                except ValueError:
                    continue
                if all(existing_pid != pid for existing_pid, _cmd in found):
                    found.append((pid, _pid_command(pid)))
            if found:
                return found
        except (OSError, subprocess.TimeoutExpired):
            pass
    try:
        for conn in psutil.net_connections(kind="tcp"):
            try:
                if (
                    conn.status == psutil.CONN_LISTEN
                    and conn.laddr
                    and conn.laddr.port == port
                ):
                    found.append((conn.pid, _pid_command(conn.pid)))
            except (psutil.Error, OSError, ValueError):
                continue
    except Exception:
        return found
    return found


def _recover_discovery(args: argparse.Namespace) -> dict[str, Any] | None:
    """Ask a live dashboard host to republish its runfile (via SIGWINCH).

    A dashboard host that is still running after its runfile was removed or
    corrupted cannot be rediscovered from the CLI alone (the dashboard bearer
    token lives only inside the host process), so the CLI signals the host to
    rewrite the runfile itself and returns the fresh run state.  Returns
    ``None`` when no live ``peaksMCP _serve`` host listens on the dashboard
    port, or the host does not answer within 2s.  SIGWINCH is deliberately
    used instead of SIGUSR1: its default action is ``ignore``, so an older
    host that predates runfile republishing is never accidentally killed.
    """
    try:
        port = _dashboard_port(args)
    except Exception:
        return None
    for pid, command in _listener_on_port(port):
        if "peaksMCP" in command and "_serve" in command:
            try:
                os.kill(pid, signal.SIGWINCH)
            except OSError:
                continue
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                data = _runfile(False)
                if data and not data.get("stale"):
                    return data
                time.sleep(0.05)
            return None
    return None


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


def _spawn_host_process(args: argparse.Namespace) -> subprocess.Popen[Any]:
    """Start the detached dashboard host and return its Popen handle."""
    root = Path(os.environ.get("PEAKSMCP_HOME", Path.home() / ".peaksMCP"))
    log = root / "logs" / "supervisor.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    # Supervisor logs carry redacted-but-sensitive lines and dashboard URLs
    # with tokens; create owner-only, like the runfile and audit log.
    descriptor = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    stream = os.fdopen(descriptor, "a", encoding="utf-8")
    notebook_arg = ["--notebook", args.notebook] if getattr(args, "notebook", None) else []
    try:
        # Launch from the package root, NOT the caller's cwd: if the user
        # runs from a directory that contains a sibling ``peaksMCP`` folder
        # (e.g. the project checkout parent), Python treats that folder as a
        # namespace package and shadows the installed package, breaking
        # ``import peaksMCP`` ("unknown location", no __version__).
        return subprocess.Popen(
            [sys.executable, "-m", "peaksMCP", "_serve", "--profile", args.profile, *notebook_arg],
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True, close_fds=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
    finally:
        stream.close()


def _workspace_key(path: str) -> str:
    """Normalize a notebook path the way the host would resolve it.

    The Jupyter host serves from the package root and treats relative
    notebook paths against that directory, so ``x.ipynb``, ``./x.ipynb`` and
    an absolute path to the same checkout file all name one workspace.
    Comparing these keys avoids needless host replacements for equivalent
    spellings of the same notebook.
    """
    root = Path(__file__).resolve().parent.parent
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    return os.path.normpath(str(candidate))


def _host_matches_request(
    args: argparse.Namespace, current: dict[str, Any]
) -> bool:
    """Return whether a live host satisfies the requested workspace."""
    requested_notebook = getattr(args, "notebook", None)
    if requested_notebook is not None:
        active_notebook = current.get("notebook_path")
        if not active_notebook or _workspace_key(active_notebook) != _workspace_key(
            requested_notebook
        ):
            return False
    requested_profile = getattr(args, "profile", None)
    active_profile = current.get("profile")
    # Older runfiles may not include a profile.  Preserve compatibility unless
    # they conflict with an explicitly known active profile.
    return not (
        requested_profile
        and active_profile
        and active_profile != requested_profile
    )


def _replace_host(data: dict[str, Any], args: argparse.Namespace) -> None:
    """Stop a live host that does not satisfy the requested workspace.

    The request (positional notebook / ``--profile``) is authoritative: the
    singleton host is replaced instead of silently opening its old workspace
    under a URL that appears to satisfy the new request.  The reason is
    printed so the replacement is never silent.
    """
    requested_notebook = getattr(args, "notebook", None)
    print(
        f"replacing running host (pid {data['pid']}, profile "
        f"{data.get('profile')!r}, notebook {data.get('notebook_path')!r}) "
        f"to honor requested workspace "
        f"(profile {getattr(args, 'profile', 'default')!r}, "
        f"notebook {requested_notebook!r})",
        file=sys.stderr,
    )
    _terminate_supervisor(int(data["pid"]))


def _ensure_host(args: argparse.Namespace) -> dict[str, Any]:
    """Start the dashboard host if absent, then wait until its dashboard answers.

    Only the co-hosted dashboard (reachable on 8765 within a second) has to be
    ready; JupyterLab + kernel come up inside the host afterwards, so a cold
    ``peaksMCP dash`` returns fast and the UI shows a "starting" state.
    """
    current = _runfile(False)
    root = Path(os.environ.get("PEAKSMCP_HOME", Path.home() / ".peaksMCP"))
    log = root / "logs" / "supervisor.log"
    process: subprocess.Popen[Any] | None = None
    if current and current.get("stale"):
        # A crashed host may have left its JupyterLab tree holding the ports;
        # reap it before the fresh host tries to bind them.
        _cleanup_stale_jupyter_tree(current)
    elif current and not _host_matches_request(args, current):
        # A positional notebook (or a non-default profile) is authoritative.
        # Replace the singleton host instead of silently opening its old
        # workspace under a URL that appears to satisfy the new request.
        _replace_host(current, args)
        current = None
    if not current or current.get("stale"):
        # A live host can outlive its runfile (file removed or corrupted).
        # Ask it to republish before spawning a duplicate that would crash on
        # the already-bound dashboard port.
        recovered = _recover_discovery(args)
        if recovered and not _host_matches_request(args, recovered):
            _replace_host(recovered, args)
            recovered = None
        if recovered:
            current = recovered
        else:
            busy = _listener_on_port(_dashboard_port(args))
            if busy:
                pid, command = busy[0]
                preview = command if len(command) <= 100 else f"{command[:100]}…"
                raise SystemExit(
                    f"dashboard port {_dashboard_port(args)} is already in use by pid {pid} "
                    f"({preview}); no runfile discovery exists to manage that host. "
                    f"Stop it (e.g. `kill {pid}`), then retry `peaksMCP dash`."
                )
            process = _spawn_host_process(args)
    deadline = time.monotonic() + getattr(args, "timeout", 15)
    last_error: str | None = None
    while time.monotonic() < deadline:
        data = _runfile(False)
        if data and not data.get("stale"):
            try:
                response = httpx.get(
                    f"{data['dashboard_url']}/api/status",
                    headers=_dashboard_headers(data),
                    timeout=2,
                )
                if response.status_code == 200:
                    return data
                last_error = f"dashboard answered {response.status_code}"
            except Exception as exc:  # host still starting
                last_error = type(exc).__name__
        if process is not None and process.poll() is not None:
            raise SystemExit(f"dashboard host exited with code {process.returncode}; inspect {log}")
        time.sleep(0.1)
    if process is not None:
        _terminate_spawned_supervisor(process)
    raise SystemExit(
        f"dashboard host did not answer within {getattr(args, 'timeout', 15)}s "
        f"(last error: {last_error}); inspect {log}"
    )


def command_dash(args: argparse.Namespace) -> None:
    """Ensure the dashboard host is running, then open the operator console.

    The managed notebook is opened from the console's "Open Notebook" button,
    which only enables once Jupyter is actually reachable — the console, not
    the CLI, is responsible for deciding when the notebook can be opened.
    """
    data = _ensure_host(args)
    target = (
        f"{data['dashboard_url']}/?token={data['dashboard_token']}"
        if data.get("dashboard_token")
        else data["dashboard_url"]
    )
    webbrowser.open(target)
    print(target)


def command_serve(args: argparse.Namespace) -> None:
    from .app.runtime import RuntimeSupervisor
    supervisor = RuntimeSupervisor(_profile(args.profile))
    if getattr(args, "notebook", None):
        supervisor.notebook_path = args.notebook
    supervisor.serve_forever()


def command_status(args: argparse.Namespace) -> None:
    data = _runfile(False)
    if (not data) or data.get("stale"):
        recovered = _recover_discovery(args)
        if recovered:
            data = recovered
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


def _terminate_supervisor(pid: int) -> None:
    """Stop a supervisor: SIGTERM first, escalate to SIGKILL, wait until gone.

    The supervisor's own ``stop()`` tears down JupyterLab (8s ceiling) and the
    dashboard (3s) before exiting, so a plain 20s SIGTERM wait is enough in the
    normal case; the SIGKILL path only fires for a wedged process.
    """
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return  # already gone
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            return
        time.sleep(0.2)
    os.kill(pid, signal.SIGKILL)
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            return
        time.sleep(0.2)
    raise SystemExit("supervisor did not exit even after SIGKILL; inspect the process tree")


def _jupyter_process_matches(
    pid: int, recorded_create_time: float | str | None
) -> bool:
    """Return whether ``pid`` is the recorded Jupyter process-group leader."""
    if pid <= 0 or recorded_create_time is None:
        return False
    try:
        expected = float(recorded_create_time)
        actual = psutil.Process(pid).create_time()
        # Popen uses start_new_session=True, so the child must remain its own
        # process-group leader.  The small tolerance only covers float transport
        # through JSON; a reused PID with a different start time is rejected.
        return abs(actual - expected) < 0.001 and os.getpgid(pid) == pid
    except (psutil.Error, OSError, AttributeError, TypeError, ValueError):
        return False


def _cleanup_stale_jupyter_tree(data: dict[str, Any]) -> None:
    """Safely reap the recorded Jupyter group left by a crashed supervisor."""
    try:
        jupyter_pid = int(data.get("jupyter_pid") or 0)
    except (TypeError, ValueError):
        return
    recorded_create_time = data.get("jupyter_process_create_time")
    if not _jupyter_process_matches(jupyter_pid, recorded_create_time):
        return
    try:
        os.killpg(jupyter_pid, signal.SIGTERM)
    except OSError:
        return
    time.sleep(0.5)
    # Revalidate immediately before escalation.  If the original leader exited
    # and its PID was reused, never send SIGKILL to the replacement process.
    if not _jupyter_process_matches(jupyter_pid, recorded_create_time):
        return
    try:
        os.killpg(jupyter_pid, signal.SIGKILL)
    except OSError:
        pass


def command_stop(args: argparse.Namespace) -> None:
    data = _runfile(False)
    if (not data) or data.get("stale"):
        recovered = _recover_discovery(args)
        if recovered:
            data = recovered
    if not data or data.get("stale"):
        raise SystemExit("peaksMCP supervisor is not running")
    _terminate_supervisor(int(data["pid"]))
    _json({"stopped": True, "pid": data["pid"]})


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
    data = _runfile(False)
    if (not data) or data.get("stale"):
        recovered = _recover_discovery(args)
        if recovered:
            data = recovered
    if not data or data.get("stale"):
        raise SystemExit("peaksMCP supervisor is not running")
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
    """Full-stack restart: stop the dashboard host, then start a fresh one."""
    from .observability import read_runfile

    data = read_runfile()
    if not data or data.get("stale"):
        recovered = _recover_discovery(args)
        if recovered:
            data = recovered
    if data:
        if data.get("stale"):
            # Previous host crashed: its JupyterLab tree may still own the
            # ports.  Reap it so the fresh host can bind them.
            _cleanup_stale_jupyter_tree(data)
        else:
            _terminate_supervisor(int(data["pid"]))
    # No host running (or just stopped): start a fresh dashboard host and wait
    # for its dashboard to answer.
    fresh = _ensure_host(args)
    _json({"ready": True, "dashboard_url": fresh.get("dashboard_url"), "kernel_id": fresh.get("kernel_id")})


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="peaksMCP", description="Claude Desktop bridge for Peaks ARPES analysis")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("version").set_defaults(func=lambda _a: print(__version__))
    dash = sub.add_parser("dash", help="start the dashboard host if needed, then open the operator console")
    dash.add_argument("notebook", nargs="?", default=None, help="notebook file to use as the workspace (e.g. peaksMCP-snapshot-xxx.ipynb)")
    dash.add_argument("--profile", default="default")
    dash.add_argument("--timeout", type=float, default=15, help="seconds to wait for the dashboard host to answer")
    dash.set_defaults(func=command_dash)
    # Legacy alias for the same entry point.
    opened = sub.add_parser("open", help=argparse.SUPPRESS)
    opened.add_argument("--profile", default="default")
    opened.add_argument("--timeout", type=float, default=15)
    opened.set_defaults(func=command_dash)
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
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse command-line arguments and dispatch the selected implementation."""
    args = build_parser().parse_args(argv)
    args.func(args)
