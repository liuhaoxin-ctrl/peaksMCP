"""Live-kernel E2E for the full peaksMCP chain.

Brings up a real isolated JupyterLab + managed kernel + in-kernel MCP server on
test-only ports, then verifies:

- bring-up reaches the ``restart`` readiness chain (kernel → extension → MCP →
  tools/list → private /healthz);
- the MCP tool surface is live (exactly search/get/inspect_notebook/
  run_cell/save_with_consent) and a real search call ranks results;
- kernel variables are readable through inspect_notebook;
- ``restart mcp`` preserves the kernel namespace;
- ``restart kernel`` rebuilds MCP and clears the namespace;
- with a real browser frontend, the Comm bridge connects and ``restart all``
  (``require_comm=True``) verifies READY through an actual tool call.

Run explicitly with ``pytest -m e2e`` (skipped by the default test run).
"""

from __future__ import annotations

import asyncio
import os
import socket
import time

import httpx
import pytest

pytestmark = [pytest.mark.e2e]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _mcp_endpoint(supervisor) -> str:
    return f"http://{supervisor.profile.mcp.host}:{supervisor.profile.mcp.port}/mcp"


def _tool_call(supervisor, name: str, arguments: dict):
    """Call an MCP tool against the live kernel server (sync wrapper)."""
    from fastmcp import Client

    async def _call():
        async with Client(_mcp_endpoint(supervisor), timeout=60) as client:
            result = await client.call_tool(name, arguments)
            data = getattr(result, "data", None)
            if isinstance(data, (dict, list)):
                return data
            return {"content": [str(item) for item in getattr(result, "content", [])]}

    return asyncio.run(_call())


def _dashboard(supervisor, path: str):
    return httpx.get(
        f"{supervisor.dashboard_url}{path}",
        headers={"Authorization": f"Bearer {supervisor.dashboard_token}"},
        timeout=10,
    ).json()


@pytest.fixture(scope="module")
def supervisor(tmp_path_factory):
    from peaksMCP.app.kernel import uninstall_kernel
    from peaksMCP.app.profiles import Profile
    from peaksMCP.app.runtime import RuntimeSupervisor

    home = tmp_path_factory.mktemp("peaksmcp_e2e_home")
    saved_home = os.environ.get("PEAKSMCP_HOME")
    saved_cwd = os.getcwd()
    os.environ["PEAKSMCP_HOME"] = str(home)
    # Isolate the notebook working directory: JupyterLab runs with the
    # supervisor's cwd, so without this the e2e would mutate the real
    # peaksMCP-runtime.ipynb (test cells leaking into the user notebook).
    os.chdir(str(home))
    kernel_name = f"peaksmcp-e2e-{os.getpid()}"
    profile = Profile(
        name="e2e",
        jupyter={
            "host": "127.0.0.1",
            "port": _free_port(),
            "kernel_name": kernel_name,
        },
        mcp={"host": "127.0.0.1", "port": _free_port()},
        dashboard={"host": "127.0.0.1", "port": _free_port()},
    )
    supervisor = RuntimeSupervisor(profile)
    try:
        supervisor.start(timeout=120)
        yield supervisor
    finally:
        supervisor.stop()
        uninstall_kernel(kernel_name)
        os.chdir(saved_cwd)
        if saved_home is None:
            os.environ.pop("PEAKSMCP_HOME", None)
        else:
            os.environ["PEAKSMCP_HOME"] = saved_home


def test_plot_cell_image_flows_through_comm_to_mcp(supervisor):
    """Real image pipeline: execute a Matplotlib cell -> Jupyter produces a PNG
    output -> run_cell returns the settled normalised summary (figure marker,
    never raw pixels).  Guards the execute -> settle -> normalise path.
    """
    import concurrent.futures

    from playwright.sync_api import sync_playwright

    def _tool_call_thread(name, arguments):
        # asyncio.run cannot run inside sync_playwright's event loop; run the
        # MCP call on a worker thread instead.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(_tool_call, supervisor, name, arguments).result(timeout=90)

    # Wait until the extension has loaded and the in-kernel MCP is serving.
    ready = supervisor.wait_ready(timeout=120, require_comm=False)
    assert ready["ready"], ready
    notebook_url = supervisor.status()["notebook_url"] + f"?token={supervisor.token}"
    with sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch(
                headless=True, channel="chrome",
                args=["--disable-background-timer-throttling", "--disable-backgrounding-occluded-windows", "--disable-gpu"],
            )
        except Exception as exc:  # pragma: no cover - environment dependent
            pytest.skip(f"Chrome not launchable: {exc}")
        page = browser.new_page()
        page.goto(notebook_url, wait_until="domcontentloaded")
        page.wait_for_selector(".jp-Notebook", timeout=90000)
        try:
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                status = _dashboard(supervisor, "/api/status")
                if (status.get("components") or {}).get("comm", {}).get("state") == "ready":
                    break
                time.sleep(1)
            # Inline figure cell: the settled run_cell reply carries the
            # figure marker; raw pixels and print text never return.
            result = _tool_call_thread(
                "run_cell",
                {
                    "code": (
                        "import matplotlib.pyplot as plt\n"
                        "plt.figure(); plt.plot([1, 2, 3])\n"
                        "print('plot rendered')"
                    ),
                },
            )
            assert result is not None
            assert result.get("id")
            assert result.get("execution_success") is True
            assert isinstance(result.get("output"), list)
            assert "outputs" not in result
            assert any("Inline figure rendered" in item for item in result.get("output") or [])
        finally:
            browser.close()


def test_bringup_reaches_ready(supervisor):
    result = supervisor.wait_ready(timeout=120, require_comm=False)
    assert result["ready"], result
    assert result["stages"]["kernel"]
    assert result["stages"]["mcp_initialize"]
    assert result["stages"]["tools_list"]
    assert result["stages"]["health"]
    assert result["stages"]["extension"]


def test_mcp_tool_surface_and_search(supervisor):
    from peaksMCP.transport import check_http_mcp_server

    health = asyncio.run(check_http_mcp_server(supervisor.profile.mcp.host, supervisor.profile.mcp.port))
    assert health["ok"]
    assert health["status"] is not None  # private /healthz payload (no status tool)
    from peaksMCP.config.metadata import tool_names

    assert health["tool_count"] == len(tool_names())
    assert set(health["tools"]) == {
        "search", "get", "inspect_notebook", "run_cell", "save_with_consent",
    }
    assert health["missing_tools"] == []
    assert health["unexpected_tools"] == []
    assert health["duplicate_tools"] == []

    data = _tool_call(supervisor, "search", {"query": "动量转换", "limit": 3})
    assert data["count"] >= 1
    names = [item["name"] for item in data["matches"]]
    assert "k_convert" in names, names

    detail = _tool_call(supervisor, "get", {"canonical_id": "dataarray:peaks.core.process.k_conversion:k_convert"})
    assert "k_convert(" in detail["signature"]
    assert detail["docstring"]


def test_dashboard_status_reports_components(supervisor):
    status = _dashboard(supervisor, "/api/status")
    assert status["aggregate"] in {"ready", "degraded"}
    assert status["components"]["jupyter"]["state"] == "ready"
    assert status["components"]["mcp"]["state"] == "ready"
    assert status["notebook_open_url"]


def test_comm_bridge_connects_and_restart_all(supervisor):
    """Browser-backed: Comm must connect, then ``restart all`` (require_comm) verifies READY.

    Runs before any kernel-restart test so the frontend session is pristine.
    """
    import concurrent.futures

    from playwright.sync_api import sync_playwright

    notebook_url = supervisor.status()["notebook_url"] + f"?token={supervisor.token}"
    with sync_playwright() as playwright:
        try:
                browser = playwright.chromium.launch(
                    headless=True, channel="chrome",
                    args=[
                        "--disable-background-timer-throttling",
                        "--disable-backgrounding-occluded-windows",
                        "--disable-gpu",
                        "--renderer-process-limit=1",
                    ],
            )
        except Exception as exc:  # pragma: no cover - environment dependent
            pytest.skip(f"Chrome not launchable: {exc}")
        page = browser.new_page()
        page.goto(notebook_url, wait_until="domcontentloaded")
        page.wait_for_selector(".jp-Notebook", timeout=90000)
        try:
            deadline = time.monotonic() + 90
            comm_ready = False
            while time.monotonic() < deadline:
                status = _dashboard(supervisor, "/api/status")
                comm_ready = (status.get("components") or {}).get("comm", {}).get("state") == "ready"
                if comm_ready:
                    break
                time.sleep(1)
            assert comm_ready, "JupyterLab Comm bridge did not connect"

            supervisor.execute_kernel("peaksmcp_restart_all_marker = 42", timeout=30)
            previous_generation = _dashboard(supervisor, "/api/status")["mcp"]["status"][
                "kernel_instance_id"
            ]
            response = httpx.post(
                f"{supervisor.dashboard_url}/api/restart/all",
                headers={"Authorization": f"Bearer {supervisor.dashboard_token}"},
                timeout=180,
            )
            result = response.json()
            assert result.get("ready"), result
            assert result["stages"]["comm"], result
            assert result["stages"]["kernel_restarted"], result
            assert result["kernel_instance_id"] != previous_generation
            # The dashboard console has no data endpoints (the Inspector was
            # removed), so read the restarted kernel through the MCP surface.
            # asyncio.run cannot run inside sync_playwright's event loop, so
            # call the MCP tool on a worker thread (same as the plot test).
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                listing = executor.submit(
                    _tool_call, supervisor, "inspect_notebook", {"target": "variables"}
                ).result(timeout=90)
            names = [item["name"] for item in listing["variables"]]
            assert "peaksmcp_restart_all_marker" not in names
        finally:
            browser.close()


def test_kernel_variables_are_listed_and_readable(supervisor):
    supervisor.execute_kernel("peaksmcp_e2e_marker = {'band': 1.0}", timeout=30)
    listing = _tool_call(supervisor, "inspect_notebook", {"target": "variables"})
    names = [item["name"] for item in listing["variables"]]
    assert "peaksmcp_e2e_marker" in names
    read = _tool_call(supervisor, "inspect_notebook", {"target": "variable", "variable_name": "peaksmcp_e2e_marker", "detail": "preview"})
    assert "band" in read.get("repr", "")


def test_restart_mcp_preserves_kernel_state(supervisor):
    supervisor.execute_kernel("peaksmcp_e2e_marker = 42", timeout=30)
    result = supervisor.restart_mcp(timeout=120)
    assert result["ready"], result
    listing = _tool_call(supervisor, "inspect_notebook", {"target": "variables"})
    names = [item["name"] for item in listing["variables"]]
    assert "peaksmcp_e2e_marker" in names, "MCP restart must keep the kernel namespace"


def test_restart_kernel_rebuilds_and_clears_state(supervisor):
    supervisor.execute_kernel("peaksmcp_e2e_marker = 42", timeout=30)
    result = supervisor.restart_kernel(timeout=150)
    assert result["ready"], result
    listing = _tool_call(supervisor, "inspect_notebook", {"target": "variables"})
    names = [item["name"] for item in listing["variables"]]
    assert "peaksmcp_e2e_marker" not in names, "kernel restart must clear the namespace"
