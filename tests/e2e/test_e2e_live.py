"""Live-kernel E2E for the full peaksMCP chain.

Brings up a real isolated JupyterLab + managed kernel + in-kernel MCP server on
test-only ports, then verifies:

- bring-up reaches the ``restart`` readiness chain (kernel → extension → MCP →
  tools/list → notebook_server_status);
- the MCP tool surface is live and a real search call ranks results;
- kernel variables are readable through the notebook tools;
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


def _dashboard_tool(supervisor, name: str, arguments: dict):
    response = httpx.post(
        f"{supervisor.dashboard_url}/api/mcp/tool",
        headers={"Authorization": f"Bearer {supervisor.dashboard_token}"},
        json={"name": name, "arguments": arguments},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["result"]


@pytest.fixture(scope="module")
def supervisor(tmp_path_factory):
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
    profile = Profile(
        name="e2e",
        jupyter={"host": "127.0.0.1", "port": _free_port(), "kernel_name": "peaksmcp"},
        mcp={"host": "127.0.0.1", "port": _free_port(), "mode": "safe"},
        dashboard={"host": "127.0.0.1", "port": _free_port()},
    )
    supervisor = RuntimeSupervisor(profile)
    try:
        supervisor.start(timeout=120)
    except Exception:
        supervisor.stop()
        raise
    yield supervisor
    supervisor.stop()
    os.chdir(saved_cwd)
    if saved_home is None:
        os.environ.pop("PEAKSMCP_HOME", None)
    else:
        os.environ["PEAKSMCP_HOME"] = saved_home


def test_plot_cell_image_flows_through_comm_to_mcp(supervisor):
    """Real image pipeline: execute a Matplotlib cell -> Jupyter produces a PNG
    output -> frontend Comm pushes it -> MCP ImageContent is served.

    Guards against stale ``active_cell_output`` (previously only published on
    activeCellChanged, so a freshly inserted cell's late-arriving image was
    never synced).
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
    # Dangerous mode: execution/editing tools auto-approve, so no consent dialog.
    supervisor.execute_kernel(
        "from peaksMCP.server.jupyter_peaks.jupyter_mcp_extension import get_server as _g; _g().set_mode('dangerous')",
        timeout=30,
    )
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
            # Execute a plotting cell through the MCP tool (frontend runs it).
            # New kernels have no matplotlib backend configured; enable inline so
            # the PNG lands in the cell output (matches the user notebooks that
            # call matplotlib.use(inline) explicitly).
            # Execute a plotting cell through the MCP tool (frontend runs it).
            # matplotlib inline display needs IPython integration; to keep the
            # executed code plain-Python (scanner-parsable) we render the PNG via
            # canvas.print_png and display it as an IPython Image — this exercises
            # the exact pipeline: cell produces an image/png output -> frontend
            # Comm push -> MCP ImageContent.
            result = _tool_call_thread("notebook_execute_code", {
                "code": (
                    "import io\n"
                    "import matplotlib.pyplot as plt\n"
                    "from IPython.display import Image, display\n"
                    "fig = plt.figure(); plt.plot([1, 2, 3])\n"
                    "buf = io.BytesIO()\n"
                    "fig.canvas.print_png(buf)\n"
                    "display(Image(data=buf.getvalue(), format='png'))\n"
                    "print('png bytes:', len(buf.getvalue()))"
                ),
            })
            assert result is not None
            # Wait for the PNG output to arrive and be served as ImageContent.
            deadline = time.monotonic() + 60
            image_seen = False
            while time.monotonic() < deadline:
                output = _tool_call_thread("notebook_read_active_cell_output", {})
                # With image data the tool returns a list of content objects;
                # without any output it returns {"content": [text]}.
                blocks = (
                    output
                    if isinstance(output, list)
                    else (output.get("content", []) if isinstance(output, dict) else [])
                )
                if any(
                    (isinstance(b, dict) and b.get("type") == "image")
                    or (hasattr(b, "type") and b.type == "image")
                    or (isinstance(b, str) and "type='image'" in b)
                    for b in blocks
                ):
                    image_seen = True
                    break
                time.sleep(1)
            assert image_seen, f"no image served; output={output!r}"
        finally:
            browser.close()


def test_load_cell_appears_runs_and_persists(supervisor):
    """Real Load chain: a NetCDF on disk -> /api/notebook/load -> a visible
    ``data = load(...)`` cell is inserted and executed -> the ``data`` variable
    exists in the kernel -> the notebook file is saved with that cell."""
    import concurrent.futures
    import json
    import os

    import numpy as np
    import xarray as xr
    from playwright.sync_api import sync_playwright

    def _tool_call_thread(name, arguments):
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(_tool_call, supervisor, name, arguments).result(timeout=90)

    ready = supervisor.wait_ready(timeout=120, require_comm=False)
    assert ready["ready"], ready
    supervisor.execute_kernel(
        "from peaksMCP.server.jupyter_peaks.jupyter_mcp_extension import get_server as _g; _g().set_mode('dangerous')",
        timeout=30,
    )

    # A real NetCDF to load (the e2e cwd is the isolated notebook directory).
    nc_path = os.path.join(os.getcwd(), "test_load.nc")
    xr.DataArray(
        np.ones((3, 4)),
        dims=("eV", "theta_par"),
        coords={"eV": [0, 1, 2], "theta_par": [0, 1, 2, 3]},
        attrs={"units": "counts"},
    ).to_netcdf(nc_path)
    assert os.path.exists(nc_path)

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
            # Call the Load endpoint (dashboard API) with the real NetCDF path.
            response = httpx.post(
                f"{supervisor.dashboard_url}/api/notebook/load",
                headers={"Authorization": f"Bearer {supervisor.dashboard_token}"},
                json={"path": nc_path},
                timeout=120,
            )
            assert response.status_code == 200, response.text
            assert response.json().get("loaded") is True, response.text
            # The data variable must appear in the kernel namespace.
            deadline = time.monotonic() + 30
            names: list[str] = []
            while time.monotonic() < deadline:
                listing = _tool_call_thread("notebook_list_variables", {})
                names = [item["name"] for item in listing.get("variables", [])]
                if "data" in names:
                    break
                time.sleep(1)
            assert "data" in names, f"data variable not found; vars={names}"
            # The notebook file must have been saved with the load cell.
            deadline = time.monotonic() + 15
            sources: list[str] = []
            while time.monotonic() < deadline:
                nb = json.loads(open("peaksMCP-runtime.ipynb", encoding="utf-8").read())
                sources = ["".join(c.get("source", [])) for c in nb.get("cells", [])]
                if any("data = load(" in s for s in sources):
                    break
                time.sleep(1)
            assert any("data = load(" in s for s in sources), f"load cell not persisted; sources={sources}"
        finally:
            browser.close()


def test_bringup_reaches_ready(supervisor):
    result = supervisor.wait_ready(timeout=120, require_comm=False)
    assert result["ready"], result
    assert result["stages"]["kernel"]
    assert result["stages"]["mcp_initialize"]
    assert result["stages"]["tools_list"]
    assert result["stages"]["status_tool"]
    assert result["stages"]["extension"]


def test_mcp_tool_surface_and_search(supervisor):
    from peaksMCP.transport import check_http_mcp_server

    health = asyncio.run(check_http_mcp_server(supervisor.profile.mcp.host, supervisor.profile.mcp.port))
    assert health["ok"]
    assert health["tool_count"] >= 12
    safe_tools = {
        "peaks_search_api", "peaks_get_api", "askuserquestion",
        "notebook_list_variables", "notebook_read_variable", "notebook_read_active_cell",
        "notebook_read_active_cell_output", "notebook_read_content", "notebook_move_cursor",
        "notebook_server_status", "notebook_kernel_status", "notebook_wait_for_kernel",
    }
    assert safe_tools <= set(health["tools"])

    data = _tool_call(supervisor, "peaks_search_api", {"query": "动量转换", "limit": 3})
    assert data["count"] >= 1
    names = [item["name"] for item in data["matches"]]
    assert "k_convert" in names, names

    detail = _tool_call(supervisor, "peaks_get_api", {"canonical_id": "dataarray:peaks.core.process.k_conversion:k_convert"})
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
            listing = _dashboard_tool(supervisor, "notebook_list_variables", {})
            names = [item["name"] for item in listing["variables"]]
            assert "peaksmcp_restart_all_marker" not in names
        finally:
            browser.close()


def test_kernel_variables_are_listed_and_readable(supervisor):
    supervisor.execute_kernel("peaksmcp_e2e_marker = {'band': 1.0}", timeout=30)
    listing = _tool_call(supervisor, "notebook_list_variables", {})
    names = [item["name"] for item in listing["variables"]]
    assert "peaksmcp_e2e_marker" in names
    read = _tool_call(supervisor, "notebook_read_variable", {"name": "peaksmcp_e2e_marker"})
    assert "band" in read.get("repr", "")


def test_restart_mcp_preserves_kernel_state(supervisor):
    supervisor.execute_kernel("peaksmcp_e2e_marker = 42", timeout=30)
    result = supervisor.restart_mcp(timeout=120)
    assert result["ready"], result
    listing = _tool_call(supervisor, "notebook_list_variables", {})
    names = [item["name"] for item in listing["variables"]]
    assert "peaksmcp_e2e_marker" in names, "MCP restart must keep the kernel namespace"


def test_restart_kernel_rebuilds_and_clears_state(supervisor):
    supervisor.execute_kernel("peaksmcp_e2e_marker = 42", timeout=30)
    result = supervisor.restart_kernel(timeout=150)
    assert result["ready"], result
    listing = _tool_call(supervisor, "notebook_list_variables", {})
    names = [item["name"] for item in listing["variables"]]
    assert "peaksmcp_e2e_marker" not in names, "kernel restart must clear the namespace"
