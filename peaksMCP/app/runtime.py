"""External supervisor for JupyterLab, one kernel and the local dashboard."""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import psutil
import uvicorn
from jupyter_client import BlockingKernelClient
from jupyter_client.connect import find_connection_file

from peaksMCP.observability import remove_runfile, write_runfile
from peaksMCP.transport import check_http_mcp_server

from .kernel import (
    install_kernel,
    kernel_installed,
    kernel_profile_state,
    kernel_spec_state,
)
from .profiles import Profile


def _port_owner(host: str, port: int) -> int | None:
    """Return the pid listening on (host, port), or None when free."""
    try:
        import psutil

        for conn in psutil.net_connections(kind="inet"):
            if conn.laddr and conn.laddr.port == port and conn.laddr.ip in (host, "127.0.0.1", "0.0.0.0", "::"):
                return conn.pid
    except Exception:
        pass
    return None


class RuntimeSupervisor:
    """Launch, monitor and recover the Jupyter/Kernel/MCP process chain."""

    def __init__(self, profile: Profile) -> None:
        self.profile = profile
        self.token = secrets.token_urlsafe(24)
        self.dashboard_token = secrets.token_urlsafe(24)
        self.jupyter: subprocess.Popen[str] | None = None
        self.kernel_id: str | None = None
        self.session_id: str | None = None
        self.notebook_path = "peaksMCP-runtime.ipynb"
        self._reader: threading.Thread | None = None
        self._dashboard: uvicorn.Server | None = None
        self._dashboard_thread: threading.Thread | None = None
        self._dashboard_error: BaseException | None = None
        self._stop = threading.Event()

    @property
    def jupyter_url(self) -> str:
        return f"http://{self.profile.jupyter.host}:{self.profile.jupyter.port}"

    @property
    def dashboard_url(self) -> str:
        return f"http://{self.profile.dashboard.host}:{self.profile.dashboard.port}"

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"token {self.token}"}

    def start(self, timeout: float = 60) -> dict[str, Any]:
        """Start JupyterLab, create the managed kernel and serve the dashboard."""
        if not kernel_installed(self.profile.jupyter.kernel_name):
            install_kernel(self.profile)
        else:
            # Reinstall when any baked endpoint or security setting is stale.
            # The environment below is also authoritative, so even a kernelspec
            # changed concurrently cannot retain an old remote-binding opt-in.
            installed = kernel_spec_state(self.profile.jupyter.kernel_name)
            expected = kernel_profile_state(self.profile)
            if installed != expected:
                install_kernel(self.profile, replace=True)
        # Fail fast when the Jupyter port is already taken instead of waiting for
        # a timeout: JupyterLab would otherwise auto-bind the next free port and
        # the supervisor would keep polling the configured one.
        occupied = _port_owner(self.profile.jupyter.host, self.profile.jupyter.port)
        if occupied is not None:
            raise RuntimeError(
                f"port {self.profile.jupyter.port} is already in use by pid {occupied}; "
                "a previous instance may not have stopped cleanly — run `peaksMCP stop` "
                "or free the port first"
            )
        command = [
            sys.executable, "-m", "jupyterlab", "--no-browser",
            f"--ServerApp.ip={self.profile.jupyter.host}",
            f"--ServerApp.port={self.profile.jupyter.port}",
            "--ServerApp.port_retries=0",
            f"--ServerApp.token={self.token}",
            "--ServerApp.open_browser=False",
        ]
        self.jupyter = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=self._jupyter_environment(),
        )
        self._reader = threading.Thread(target=self._read_logs, name="peaksMCP-jupyter-log", daemon=True)
        self._reader.start()
        self._wait_jupyter(timeout)
        self.session_id, self.kernel_id = self._create_session()
        # The dashboard is the operator console and is co-hosted in this process,
        # so ``launch`` is the single startup entry for everything (JupyterLab +
        # managed kernel + in-kernel MCP + dashboard).
        self._start_dashboard()
        write_runfile({
            "pid": os.getpid(), "profile": self.profile.name, "kernel_id": self.kernel_id,
            "session_id": self.session_id, "notebook_path": self.notebook_path,
            "jupyter_url": self.jupyter_url, "dashboard_url": self.dashboard_url,
            "jupyter_port": self.profile.jupyter.port, "dashboard_port": self.profile.dashboard.port,
            "mcp_port": self.profile.mcp.port, "token": self.token,
            "mcp_autostart": self.profile.mcp.autostart,
            "dashboard_token": self.dashboard_token, "started_at": time.time(),
            "process_create_time": psutil.Process(os.getpid()).create_time(),
        })
        return self.status()

    def _jupyter_environment(self) -> dict[str, str]:
        """Build an isolated Jupyter config that disables competing notebook bridges."""
        environment = os.environ.copy()
        home = Path(environment.get("PEAKSMCP_HOME", Path.home() / ".peaksMCP"))
        config_dir = home / "jupyter"
        page_config = config_dir / "labconfig" / "page_config.json"
        page_config.parent.mkdir(parents=True, exist_ok=True)
        disabled = {name: True for name in self.profile.jupyter.disabled_extensions}
        page_config.write_text(
            json.dumps({"disabledExtensions": disabled}, indent=2) + "\n",
            encoding="utf-8",
        )
        environment["JUPYTER_CONFIG_DIR"] = str(config_dir)
        # Pass the profile's MCP endpoint into the kernel so the auto-loading
        # startup script keeps its baked defaults only when no supervisor env is
        # present (avoids port collisions between profiles/tests).
        environment["PEAKSMCP_HOST"] = self.profile.mcp.host
        environment["PEAKSMCP_PORT"] = str(self.profile.mcp.port)
        environment["PEAKSMCP_MODE"] = self.profile.mcp.mode
        environment["PEAKSMCP_AUTOSTART"] = (
            "true" if self.profile.mcp.autostart else "false"
        )
        environment["PEAKSMCP_ALLOW_REMOTE"] = (
            "true" if self.profile.mcp.allow_remote else "false"
        )
        environment["PEAKSMCP_REQUIRE_CONSENT"] = (
            "true" if self.profile.mcp.require_consent else "false"
        )
        return environment

    def _read_logs(self) -> None:
        """Drain the JupyterLab subprocess stdout to prevent pipe-buffer backpressure."""
        if not self.jupyter or not self.jupyter.stdout:
            return
        for line in self.jupyter.stdout:
            sanitized = line.replace(self.token, "<redacted>").replace(
                self.dashboard_token, "<redacted>"
            )
            print(sanitized, end="", flush=True)

    def _wait_jupyter(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._stop.is_set():
                raise RuntimeError("stopping")
            if self.jupyter and self.jupyter.poll() is not None:
                raise RuntimeError(f"JupyterLab exited with code {self.jupyter.returncode}")
            try:
                response = httpx.get(f"{self.jupyter_url}/api/status", headers=self._headers(), timeout=2)
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.25)
        raise TimeoutError("JupyterLab did not become ready")

    def _create_session(self) -> tuple[str, str]:
        notebook = {
            "type": "notebook", "format": "json",
            "content": {
                "cells": [{"cell_type": "markdown", "metadata": {}, "source": "# peaksMCP ARPES workspace", "id": "peaksmcp-welcome"}],
                "metadata": {"kernelspec": {"name": self.profile.jupyter.kernel_name, "display_name": f"Python (peaksMCP · {self.profile.mcp.mode})", "language": "python"}},
                "nbformat": 4, "nbformat_minor": 5,
            },
        }
        content_url = f"{self.jupyter_url}/api/contents/{self.notebook_path}"
        existing = httpx.get(content_url, headers=self._headers(), timeout=10)
        if existing.status_code == 404:
            if self.notebook_path != "peaksMCP-runtime.ipynb":
                # A user-specified notebook (e.g. a snapshot) must already exist;
                # never fabricate an empty notebook for it.
                raise FileNotFoundError(
                    f"notebook {self.notebook_path!r} does not exist in the Jupyter workspace"
                )
            # Only a definite 404 on the default workspace means the notebook is
            # absent; create it then.  Any other non-200 (e.g. 400 for an
            # unreadable path) must NOT be treated as "missing" and overwritten
            # with a fresh empty notebook.
            created = httpx.put(content_url, headers=self._headers(), json=notebook, timeout=15)
            created.raise_for_status()
        elif existing.status_code != 200:
            raise RuntimeError(
                f"cannot read notebook {self.notebook_path}: GET /api/contents returned "
                f"{existing.status_code} (refusing to overwrite)"
            )
        response = httpx.post(
            f"{self.jupyter_url}/api/sessions",
            headers=self._headers(),
            json={"path": self.notebook_path, "type": "notebook", "name": Path(self.notebook_path).name, "kernel": {"name": self.profile.jupyter.kernel_name}},
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        return str(payload["id"]), str(payload["kernel"]["id"])

    def _start_dashboard(self, timeout: float = 10) -> None:
        from .api import create_app

        host = self.profile.dashboard.host
        loopback = host in ("127.0.0.1", "localhost", "::1")
        if not loopback and not self.profile.dashboard.allow_remote:
            raise ValueError(
                f"dashboard host {host!r} is not loopback; set dashboard.allow_remote=true "
                "explicitly to expose the authenticated operator console"
            )
        if not loopback:
            pass
        config = uvicorn.Config(create_app(self), host=host, port=self.profile.dashboard.port, log_level="warning")
        self._dashboard = uvicorn.Server(config)
        self._dashboard_error = None

        def serve_dashboard() -> None:
            try:
                assert self._dashboard is not None
                self._dashboard.run()
            except BaseException as exc:  # uvicorn uses SystemExit for bind failures
                self._dashboard_error = exc

        self._dashboard_thread = threading.Thread(
            target=serve_dashboard,
            name="peaksMCP-dashboard",
            daemon=True,
        )
        self._dashboard_thread.start()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._dashboard.started:
                return
            if not self._dashboard_thread.is_alive():
                detail = f": {self._dashboard_error}" if self._dashboard_error else ""
                raise RuntimeError(f"dashboard failed to start{detail}")
            time.sleep(0.05)
        self._dashboard.should_exit = True
        self._dashboard_thread.join(timeout=2)
        raise TimeoutError(f"dashboard did not become ready at {self.dashboard_url}")

    def _kernel_client(self, timeout: float = 20) -> BlockingKernelClient:
        if not self.kernel_id:
            raise RuntimeError("no managed kernel")
        # The managed JupyterLab runs with JUPYTER_CONFIG_DIR under PEAKSMCP_HOME;
        # find_connection_file must look there or it may resolve a stale kernel
        # from the default runtime dir (wrong kernel, wrong extension state).
        home = Path(os.environ.get("PEAKSMCP_HOME", Path.home() / ".peaksMCP"))
        os.environ.setdefault("JUPYTER_CONFIG_DIR", str(home / "jupyter"))
        connection = find_connection_file(f"kernel-{self.kernel_id}.json")
        client = BlockingKernelClient(connection_file=connection)
        client.load_connection_file()
        client.start_channels()
        try:
            client.wait_for_ready(timeout=timeout)
        except Exception:
            # Do not leak the shell/iopub channels when the kernel never became
            # ready (start_channels() already opened them).
            client.stop_channels()
            raise
        return client

    def execute_kernel(self, code: str, timeout: float = 30) -> dict[str, Any]:
        """Execute supervisor control code directly on the kernel shell channel."""
        client = self._kernel_client(timeout)
        try:
            message_id = client.execute(code, silent=False, store_history=False)
            deadline = time.monotonic() + timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("kernel execution reply timed out")
                reply = client.get_shell_msg(timeout=remaining)
                if reply.get("parent_header", {}).get("msg_id") == message_id:
                    break
            content = reply.get("content", {})
            if content.get("status") == "error":
                raise RuntimeError("\n".join(content.get("traceback", [])) or content.get("evalue", "kernel execution failed"))
            return content
        finally:
            client.stop_channels()

    def export_variable(self, name: str, value: Any) -> None:
        """Set a Python variable in the notebook kernel namespace (silent, no cell).

        Used after conversions so the agent can reference e.g. ``CONVERTED_DIR``
        without guessing the output path. The variable shows up in
        ``notebook_list_variables`` and can be used in any cell.
        """
        self.execute_kernel(f"{name} = {value!r}", timeout=30)

    def restart_mcp(self, timeout: float = 45) -> dict[str, Any]:
        """Restart only FastMCP inside the existing kernel, preserving variables."""
        previous_instance = self._mcp_instance()
        self.execute_kernel("get_ipython().run_line_magic('peaksMCP_restart', '')", timeout=timeout)
        return self.wait_ready(
            timeout=timeout,
            require_comm=False,
            previous_mcp_instance=previous_instance,
        )

    def start_mcp(self, timeout: float = 45) -> dict[str, Any]:
        """Start the in-kernel MCP server if it is not already serving."""
        self.execute_kernel("get_ipython().run_line_magic('peaksMCP_start', '')", timeout=timeout)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if asyncio.run(
                check_http_mcp_server(self.profile.mcp.host, self.profile.mcp.port)
            ).get("ok"):
                return {"ready": True}
            time.sleep(0.5)
        return {"ready": False, "error": "MCP did not come up in time"}

    def extension_status(self, timeout: float = 3) -> dict[str, Any]:
        """Probe the IPython extension directly when the MCP transport is offline."""
        code = (
            "assert get_ipython().find_line_magic('peaksMCP_start') is not None"
        )
        try:
            self.execute_kernel(code, timeout=timeout)
            return {"loaded": True, "detail": "IPython extension loaded"}
        except Exception as exc:
            return {"loaded": False, "detail": f"extension probe failed: {exc}"}

    def restart_kernel(self, timeout: float = 90, require_comm: bool = False) -> dict[str, Any]:
        """Restart the managed kernel and wait for its auto-loaded extension and MCP.

        With ``require_comm`` the restart is orchestrated through the JupyterLab
        frontend (via the kernel Comm bridge) so the browser session reconnects and
        re-opens the Comm.  When the frontend is not attached (no live Comm), the
        request automatically falls back to a plain REST restart instead of waiting
        for a coordination that can never happen.
        """
        previous_generation = self._mcp_generation()
        if require_comm and previous_generation is None:
            # MCP is offline, so there is no generation to compare AND no Comm
            # coordination possible.  Degrade to a plain REST restart instead of
            # refusing to restart at all (a restart is exactly what can bring
            # the MCP back).
            require_comm = False
        if require_comm and not self._comm_connected():
            # The browser frontend is offline, so Comm-coordinated restart cannot
            # proceed; fall back to a REST restart to avoid hanging for timeout.
            require_comm = False
        if require_comm:
            code = (
                "import threading as _t;"
                "from peaksMCP.server.jupyter_peaks.jupyter_mcp_extension import get_server as _gs;"
                "b = _gs().state.bridge;"
                "_t.Thread(target=lambda: b.request('restart_kernel', timeout=15), daemon=True).start()"
            )
            self.execute_kernel(code, timeout=30)
        else:
            if not self.kernel_id:
                raise RuntimeError("no managed kernel")
            response = httpx.post(
                f"{self.jupyter_url}/api/kernels/{self.kernel_id}/restart",
                headers=self._headers(),
                timeout=20,
            )
            response.raise_for_status()
        return self.wait_ready(
            timeout=timeout,
            require_comm=require_comm,
            previous_generation=previous_generation,
        )

    def _comm_connected(self) -> bool:
        """Return whether the JupyterLab frontend Comm bridge is currently online."""
        try:
            result = asyncio.run(
                check_http_mcp_server(self.profile.mcp.host, self.profile.mcp.port)
            )
            status = result.get("status")
            return bool(isinstance(status, dict) and status.get("comm_connected"))
        except Exception:
            return False

    def _mcp_generation(self) -> str | None:
        """Return the current kernel instance marker reported through MCP."""
        result = asyncio.run(
            check_http_mcp_server(self.profile.mcp.host, self.profile.mcp.port)
        )
        status = result.get("status")
        if not isinstance(status, dict):
            return None
        generation = status.get("kernel_instance_id")
        return str(generation) if generation else None

    def _mcp_instance(self) -> str | None:
        """Return the current in-kernel MCP server instance marker."""
        result = asyncio.run(
            check_http_mcp_server(self.profile.mcp.host, self.profile.mcp.port)
        )
        status = result.get("status")
        if not isinstance(status, dict):
            return None
        instance = status.get("mcp_instance_id")
        return str(instance) if instance else None

    def wait_ready(
        self,
        timeout: float = 90,
        require_comm: bool = False,
        previous_generation: str | None = None,
        previous_mcp_instance: str | None = None,
    ) -> dict[str, Any]:
        """Verify every stage from kernel readiness through a real MCP tool call."""
        deadline = time.monotonic() + timeout
        stages = {
            name: False
            for name in (
                "kernel_restarted", "kernel", "extension", "comm",
                "mcp_restarted", "mcp_initialize", "tools_list", "status_tool",
            )
        }
        stages["kernel_restarted"] = previous_generation is None
        stages["mcp_restarted"] = previous_mcp_instance is None
        diagnostics: list[str] = []
        while time.monotonic() < deadline:
            try:
                if not self.kernel_id:
                    raise RuntimeError("no managed kernel")
                model = httpx.get(
                    f"{self.jupyter_url}/api/kernels/{self.kernel_id}",
                    headers=self._headers(),
                    timeout=3,
                )
                model.raise_for_status()
                stages["kernel"] = model.json().get("execution_state") in {"idle", "busy", "starting"}
            except Exception as exc:
                diagnostics.append(f"kernel: {exc}")
                time.sleep(0.5)
                continue
            result = asyncio.run(check_http_mcp_server(self.profile.mcp.host, self.profile.mcp.port))
            if result.get("ok"):
                stages["mcp_initialize"] = True
                stages["tools_list"] = not any(
                    result.get(key)
                    for key in (
                        "missing_tools",
                        "unexpected_tools",
                        "duplicate_tools",
                    )
                )
                stages["status_tool"] = result.get("status") is not None
                # Structured read of the notebook server status instead of fragile
                # string matching on the serialized status payload.
                status_data = result.get("status")
                stages["extension"] = bool(
                    isinstance(status_data, dict) and status_data.get("extension_loaded")
                )
                current_generation = (
                    str(status_data.get("kernel_instance_id"))
                    if isinstance(status_data, dict) and status_data.get("kernel_instance_id")
                    else None
                )
                stages["kernel_restarted"] = (
                    previous_generation is None
                    or (current_generation is not None and current_generation != previous_generation)
                )
                current_mcp_instance = (
                    str(status_data.get("mcp_instance_id"))
                    if isinstance(status_data, dict) and status_data.get("mcp_instance_id")
                    else None
                )
                stages["mcp_restarted"] = (
                    previous_mcp_instance is None
                    or (
                        current_mcp_instance is not None
                        and current_mcp_instance != previous_mcp_instance
                    )
                )
                stages["comm"] = bool(
                    isinstance(status_data, dict) and status_data.get("comm_connected")
                )
                if all(value for key, value in stages.items() if key != "comm") and (stages["comm"] or not require_comm):
                    return {
                        "ready": True,
                        "stages": stages,
                        "kernel_instance_id": current_generation,
                        "mcp": result,
                        "diagnostics": diagnostics[-5:],
                    }
            else:
                diagnostics.append(str(result.get("error")))
            time.sleep(0.5)
        return {"ready": False, "failed_stage": next((name for name, ok in stages.items() if not ok and (name != "comm" or require_comm)), None), "stages": stages, "diagnostics": diagnostics[-10:]}

    def status(self) -> dict[str, Any]:
        return {
            "status": "RUNNING" if self.jupyter and self.jupyter.poll() is None else "STOPPED",
            "pid": os.getpid(), "profile": self.profile.name, "kernel_id": self.kernel_id,
            "session_id": self.session_id, "notebook_path": self.notebook_path,
            "jupyter_url": self.jupyter_url, "dashboard_url": self.dashboard_url,
            "notebook_url": f"{self.jupyter_url}/lab/tree/{self.notebook_path}",
            "mcp_url": f"http://{self.profile.mcp.host}:{self.profile.mcp.port}/mcp",
        }

    def stop(self) -> None:
        """Gracefully stop dashboard, kernel/Jupyter and remove the runfile."""
        self._stop.set()
        if self._dashboard:
            self._dashboard.should_exit = True
        if self._dashboard_thread and self._dashboard_thread.is_alive():
            self._dashboard_thread.join(timeout=5)
        if self.jupyter and self.jupyter.poll() is None:
            self.jupyter.terminate()
            try:
                self.jupyter.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.jupyter.kill()
        remove_runfile()

    def serve_forever(self) -> None:
        """Start and wait for SIGINT/SIGTERM as the background supervisor process."""
        # Register handlers BEFORE start(): during start() (JupyterLab bring-up
        # can take 30-60s) the default SIGTERM action would kill the process
        # without running the finally block, orphaning the JupyterLab child.
        for name in (signal.SIGINT, signal.SIGTERM):
            signal.signal(name, lambda _signum, _frame: self._stop.set())
        try:
            self.start()
            while not self._stop.wait(0.5):
                if self.jupyter and self.jupyter.poll() is not None:
                    break
        finally:
            self.stop()
