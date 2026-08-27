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
from collections import deque
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from jupyter_client import BlockingKernelClient
from jupyter_client.connect import find_connection_file

from peaksMCP.observability import remove_runfile, write_runfile
from peaksMCP.transport import check_http_mcp_server

from .kernel import install_kernel, kernel_installed
from .profiles import Profile


class RuntimeSupervisor:
    """Launch, monitor and recover the Jupyter/Kernel/MCP process chain."""

    def __init__(self, profile: Profile) -> None:
        self.profile = profile
        self.token = secrets.token_urlsafe(24)
        self.jupyter: subprocess.Popen[str] | None = None
        self.kernel_id: str | None = None
        self.session_id: str | None = None
        self.notebook_path = "peaksMCP-runtime.ipynb"
        self.logs: deque[dict[str, Any]] = deque(maxlen=2000)
        self._reader: threading.Thread | None = None
        self._dashboard: uvicorn.Server | None = None
        self._stop = threading.Event()

    @property
    def jupyter_url(self) -> str:
        return f"http://{self.profile.jupyter.host}:{self.profile.jupyter.port}"

    @property
    def dashboard_url(self) -> str:
        return f"http://{self.profile.dashboard.host}:{self.profile.dashboard.port}"

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"token {self.token}"}

    def _log(self, component: str, message: str) -> None:
        self.logs.append({"timestamp": time.time(), "component": component, "message": message.rstrip()})

    def start(self, timeout: float = 60) -> dict[str, Any]:
        """Start JupyterLab, create the managed kernel and serve the dashboard."""
        if not kernel_installed(self.profile.jupyter.kernel_name):
            install_kernel(self.profile)
        command = [
            sys.executable, "-m", "jupyterlab", "--no-browser",
            f"--ServerApp.ip={self.profile.jupyter.host}",
            f"--ServerApp.port={self.profile.jupyter.port}",
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
        self._start_dashboard()
        write_runfile({
            "pid": os.getpid(), "profile": self.profile.name, "kernel_id": self.kernel_id,
            "session_id": self.session_id, "notebook_path": self.notebook_path,
            "jupyter_url": self.jupyter_url, "dashboard_url": self.dashboard_url,
            "jupyter_port": self.profile.jupyter.port, "dashboard_port": self.profile.dashboard.port,
            "mcp_port": self.profile.mcp.port, "token": self.token, "started_at": time.time(),
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
        return environment

    def _read_logs(self) -> None:
        if not self.jupyter or not self.jupyter.stdout:
            return
        for line in self.jupyter.stdout:
            self._log("jupyter", line)

    def _wait_jupyter(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
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
            created = httpx.put(content_url, headers=self._headers(), json=notebook, timeout=15)
            created.raise_for_status()
        response = httpx.post(
            f"{self.jupyter_url}/api/sessions",
            headers=self._headers(),
            json={"path": self.notebook_path, "type": "notebook", "name": Path(self.notebook_path).name, "kernel": {"name": self.profile.jupyter.kernel_name}},
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        return str(payload["id"]), str(payload["kernel"]["id"])

    def _start_dashboard(self) -> None:
        from .api import create_app

        config = uvicorn.Config(create_app(self), host=self.profile.dashboard.host, port=self.profile.dashboard.port, log_level="warning")
        self._dashboard = uvicorn.Server(config)
        threading.Thread(target=self._dashboard.run, name="peaksMCP-dashboard", daemon=True).start()

    def _kernel_client(self, timeout: float = 20) -> BlockingKernelClient:
        if not self.kernel_id:
            raise RuntimeError("no managed kernel")
        connection = find_connection_file(f"kernel-{self.kernel_id}.json")
        client = BlockingKernelClient(connection_file=connection)
        client.load_connection_file()
        client.start_channels()
        client.wait_for_ready(timeout=timeout)
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

    def restart_mcp(self, timeout: float = 45) -> dict[str, Any]:
        """Restart only FastMCP inside the existing kernel, preserving variables."""
        self.execute_kernel("get_ipython().run_line_magic('peaksMCP_restart', '')", timeout=timeout)
        return self.wait_ready(timeout=timeout, require_comm=False)

    def restart_kernel(self, timeout: float = 90, require_comm: bool = False) -> dict[str, Any]:
        """Restart the managed kernel and wait for its auto-loaded extension and MCP.

        With ``require_comm`` the restart is orchestrated through the JupyterLab
        frontend (via the kernel Comm bridge) so the browser session reconnects and
        re-opens the Comm; a bare REST restart leaves the frontend detached.
        """
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
        return self.wait_ready(timeout=timeout, require_comm=require_comm)

    def wait_ready(self, timeout: float = 90, require_comm: bool = False) -> dict[str, Any]:
        """Verify every stage from kernel readiness through a real MCP tool call."""
        deadline = time.monotonic() + timeout
        stages = {name: False for name in ("kernel", "extension", "comm", "mcp_initialize", "tools_list", "status_tool")}
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
                stages["tools_list"] = result.get("tool_count", 0) >= 12
                stages["status_tool"] = result.get("status") is not None
                stages["extension"] = stages["status_tool"]
                status_text = str(result.get("status", ""))
                stages["comm"] = "'comm_connected': True" in status_text or '"comm_connected":true' in status_text.replace(" ", "").lower()
                if all(value for key, value in stages.items() if key != "comm") and (stages["comm"] or not require_comm):
                    return {"ready": True, "stages": stages, "mcp": result, "diagnostics": diagnostics[-5:]}
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
        if self.jupyter and self.jupyter.poll() is None:
            self.jupyter.terminate()
            try:
                self.jupyter.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.jupyter.kill()
        remove_runfile()

    def serve_forever(self) -> None:
        """Start and wait for SIGINT/SIGTERM as the background supervisor process."""
        try:
            self.start()
            for name in (signal.SIGINT, signal.SIGTERM):
                signal.signal(name, lambda _signum, _frame: self._stop.set())
            while not self._stop.wait(0.5):
                if self.jupyter and self.jupyter.poll() is not None:
                    self._log("supervisor", "JupyterLab exited; stopping supervisor")
                    break
        finally:
            self.stop()
