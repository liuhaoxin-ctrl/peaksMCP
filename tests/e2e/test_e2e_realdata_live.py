"""Real-data E2E: a human at JupyterLab driving peaksMCP over its five tools.

The suite replaces the earlier mechanism-only live tests.  It simulates the
real usage story end to end on the machine that holds the beamtime data:

1. **cut preprocessing** — index the converted folder, fit the gold reference
   once through native ``fit_gold``, flatten the Fermi edge, zero the
   high-symmetry angle and convert one cut to k-space, with the validation
   figure rendered inline;
2. **mapping preprocessing** — convert a full 3-D cube (never a centre slice)
   and render the mapping's binding-energy slices;
3. **dashboard** — the operator console state (component readiness, the
   five-tool surface, a console-driven MCP restart) while a real browser keeps
   the JupyterLab Comm bridge alive.

Everything after loading goes through the model-facing tool surface
(``search`` / ``get`` / ``run_cell``) instead of the supervisor REST shortcuts,
so the suite exercises the same API-proof gate a real agent meets: canonical
ids are proven with ``get`` in ONE persistent MCP session, and each native
reference is declared per cell through ``api_ids``.

Data-privacy contract: the raw folder is only read.  The fixture copies the
three scans it needs into a temporary home and converts them into the sibling
``data_netcdf/`` layout the real dataset uses (``convert_pxt``), so nothing is
written next to the source; the dashboard scenario re-checks that.

Requirements: the raw beamtime folder (``PEAKSMCP_REALDATA_PXT``, defaulting
to the L112 BP260623 dataset), Google Chrome (Playwright ``channel="chrome"``)
and the ``peaks`` conda environment.  Without the raw folder the module skips.

Run explicitly::

    pytest -m e2e -v
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import shutil
import socket
import threading
import time
from pathlib import Path

import httpx
import pytest

pytestmark = [pytest.mark.e2e]

#: Raw beamtime folder (same convention as the real-data integration tests).
RAW_PXT_DIR = Path(
    os.environ.get("PEAKSMCP_REALDATA_PXT")
    or "/Users/haoxin/Documents/实验数据/BP260623/data"
)
_CONVERTED_HINT = Path("/Users/haoxin/Documents/实验数据/BP260623/data_netcdf")

#: The three scans the story needs: gold reference, one cut, one mapping cube.
GOLD_STEM = "BP_0020"
CUT_STEM = "BP_0015"
MAPPING_STEM = "BP_0001"
THETA_OFFSET_DEG = 1.5
SLICE_ENERGIES = (0.0, -0.3, -0.6)

#: Canonical ids the notebook cells rely on (native + facade).
API_FIT_GOLD = "dataarray:peaks.core.fitting.fit:fit_gold"
API_K_CONVERT = "dataarray:peaks.core.process.k_conversion:k_convert"
API_SET_EF = "metadata:peaks.core.metadata.metadata_methods:set_EF_correction"
API_LOAD_DATA = "module:peaksMCP.overrides:load_data"
API_INSPECT = "module:peaksMCP.overrides:inspect_experiment"
API_VALIDATION = "module:peaksMCP.overrides:plot_validation_pair"
API_SLICE = "module:peaksMCP.overrides:show_mapping_slice"

FIVE_TOOLS = {"search", "get", "inspect_notebook", "run_cell", "save_with_consent"}

def _missing_raw() -> str | None:
    """Return a skip reason when the raw beamtime folder is unavailable."""
    if not all(
        (RAW_PXT_DIR / f"{stem}.pxt").is_file() for stem in (GOLD_STEM, CUT_STEM, MAPPING_STEM)
    ):
        return (
            f"raw L112 PXT data not found under {RAW_PXT_DIR} "
            f"({GOLD_STEM}/{CUT_STEM}/{MAPPING_STEM}.pxt); set PEAKSMCP_REALDATA_PXT"
        )
    return None


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class McpSession:
    """One persistent MCP client session (the API-proof ledger lives here).

    A fresh client per call would reset the ledger, so the "human's agent"
    keeps a single long-lived connection exactly like Claude Desktop does
    through the STDIO proxy.
    """

    def __init__(self, url: str) -> None:
        self.url = url
        self.loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._stack: contextlib.AsyncExitStack | None = None
        self._client = None
        self._thread = threading.Thread(target=self._run, name="peaksmcp-e2e-session", daemon=True)
        self._thread.start()
        if not self._ready.wait(60):
            raise RuntimeError("MCP session did not open within 60s")

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._open())
        self.loop.run_forever()

    async def _open(self) -> None:
        from fastmcp import Client

        self._stack = contextlib.AsyncExitStack()
        self._client = await self._stack.enter_async_context(Client(self.url, timeout=900))
        self._ready.set()

    def call(self, name: str, arguments: dict, timeout: float = 900.0) -> dict:
        """Call one MCP tool and return its structured payload."""
        future = asyncio.run_coroutine_threadsafe(
            self._client.call_tool(name, arguments), self.loop
        )
        result = future.result(timeout=timeout)
        data = getattr(result, "data", None)
        if isinstance(data, dict):
            return data
        return {"content": [str(item) for item in getattr(result, "content", [])]}

    def reopen(self) -> None:
        """Open a fresh session after the in-kernel MCP server restarted."""
        self.close()
        self.loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="peaksmcp-e2e-session", daemon=True
        )
        self._thread.start()
        if not self._ready.wait(60):
            raise RuntimeError("MCP session did not reopen within 60s")

    def close(self) -> None:
        async def _shutdown() -> None:
            if self._stack is not None:
                await self._stack.aclose()

        with contextlib.suppress(Exception):
            asyncio.run_coroutine_threadsafe(_shutdown(), self.loop).result(20)
        self.loop.call_soon_threadsafe(self.loop.stop)


class Live:
    """Live peaksMCP stack + browser + one MCP session, shared by the scenarios."""

    def __init__(self, supervisor, session: McpSession, browser, page) -> None:
        self.supervisor = supervisor
        self.session = session
        self.browser = browser
        self.page = page

    # -- MCP surface -----------------------------------------------------
    def tool(self, name: str, arguments: dict) -> dict:
        return self.session.call(name, arguments)

    def prove(self, canonical_id: str) -> dict:
        """Prove one canonical id with ``get``; returns the contract payload."""
        data = self.tool("get", {"canonical_id": canonical_id})
        served = data.get("id") or data.get("canonical_id")
        assert served == canonical_id, data
        return data

    def cell(self, label: str, code: str, *, api_ids=None, timeout: float = 240.0) -> dict:
        """Run one notebook cell through run_cell and fail loudly on blocks."""
        arguments: dict = {"code": code, "timeout": timeout}
        if api_ids:
            arguments["api_ids"] = list(api_ids)
        data = self.tool("run_cell", arguments)
        assert data.get("blocked") is not True, (
            f"{label}: cell was blocked by the API gate: {data.get('api_check')}"
        )
        assert data.get("execution_success") is True, f"{label}: {data}"
        return data

    def figure_markers(self, cell_result: dict) -> int:
        """Number of normalised figure-marker lines in one cell reply."""
        return sum("Inline figure rendered" in str(item) for item in cell_result.get("output") or [])

    def figure_images(self, cell_result: dict) -> int:
        """How many inline images the settled reply reports for one cell."""
        total = 0
        for item in cell_result.get("output") or []:
            text = str(item)
            match = re.search(r"\((\d+) image", text)
            if match:
                total += int(match.group(1))
        return total

    # -- dashboard surface ----------------------------------------------
    def dashboard(self, path: str, *, token: str | None = "default", method: str = "GET", **kw):
        headers = {}
        if token == "default":
            headers["Authorization"] = f"Bearer {self.supervisor.dashboard_token}"
        elif token:
            headers["Authorization"] = f"Bearer {token}"
        url = f"{self.supervisor.dashboard_url}{path}"
        if method == "POST":
            return httpx.post(url, headers=headers, timeout=kw.pop("timeout", 240), **kw)
        return httpx.get(url, headers=headers, timeout=kw.pop("timeout", 15), **kw)

    def status(self) -> dict:
        response = self.dashboard("/api/status")
        assert response.status_code == 200, response.text
        return response.json()

    def variable(self, name: str) -> dict:
        """Preview one notebook variable (JSON-safe values only)."""
        return self.tool(
            "inspect_notebook",
            {"target": "variable", "variable_name": name, "detail": "preview"},
        )

    def health(self) -> dict:
        """Private in-kernel /healthz payload: readiness + the exact tool surface."""
        from peaksMCP.transport import check_http_mcp_server

        async def _check() -> dict:
            return await check_http_mcp_server(
                self.supervisor.profile.mcp.host, self.supervisor.profile.mcp.port
            )

        return asyncio.run_coroutine_threadsafe(_check(), self.session.loop).result(timeout=60)


def _prepare_home(home: Path) -> None:
    """Copy the three raw scans and convert them into the sibling layout.

    Mirrors the real dataset (raw in ``data/``, converted NetCDF in
    ``data_netcdf/``) without ever writing inside the source folder.
    """
    data = home / "data"
    converted = home / "data_netcdf"
    data.mkdir(parents=True)
    converted.mkdir(parents=True)
    for stem in (GOLD_STEM, CUT_STEM, MAPPING_STEM):
        shutil.copy2(RAW_PXT_DIR / f"{stem}.pxt", data / f"{stem}.pxt")
    if (RAW_PXT_DIR / "datasheet.csv").is_file():
        shutil.copy2(RAW_PXT_DIR / "datasheet.csv", data / "datasheet.csv")
    metadata = _CONVERTED_HINT / "experiment_metadata.json"
    if metadata.is_file():
        shutil.copy2(metadata, converted / "experiment_metadata.json")

    from peaksMCP.pxt_utils.converter import convert_pxt

    for stem in (GOLD_STEM, CUT_STEM, MAPPING_STEM):
        report = convert_pxt(data / f"{stem}.pxt", converted / f"{stem}.nc")
        assert report.output is not None, f"conversion failed for {stem}: {report}"


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    """Bring up the isolated stack, open the notebook in real Chrome, prove the API ids."""
    reason = _missing_raw()
    if reason is not None:
        pytest.skip(reason)
    from peaksMCP.app.kernel import uninstall_kernel
    from peaksMCP.app.profiles import Profile
    from peaksMCP.app.runtime import RuntimeSupervisor

    home = tmp_path_factory.mktemp("peaksmcp_realdata_home")
    saved_home = os.environ.get("PEAKSMCP_HOME")
    saved_cwd = os.getcwd()
    source_before = sorted(p.name for p in RAW_PXT_DIR.iterdir())
    _prepare_home(home)
    os.environ["PEAKSMCP_HOME"] = str(home)
    os.chdir(str(home))

    kernel_name = f"peaksmcp-e2e-{os.getpid()}"
    profile = Profile(
        name="e2e-realdata",
        jupyter={"host": "127.0.0.1", "port": _free_port(), "kernel_name": kernel_name},
        mcp={"host": "127.0.0.1", "port": _free_port()},
        dashboard={"host": "127.0.0.1", "port": _free_port()},
    )
    supervisor = RuntimeSupervisor(profile)
    playwright = browser = page = session = None
    try:
        supervisor.start(timeout=180)
        ready = supervisor.wait_ready(timeout=180, require_comm=False)
        assert ready["ready"], ready

        # The human opens the notebook: run_cell refuses without a live Comm
        # bridge, which only the real JupyterLab frontend provides.
        from playwright.sync_api import sync_playwright

        playwright = sync_playwright().start()
        browser = playwright.chromium.launch(
            headless=True,
            channel="chrome",
            args=[
                "--disable-background-timer-throttling",
                "--disable-backgrounding-occluded-windows",
                "--disable-gpu",
                "--renderer-process-limit=1",
            ],
        )
        page = browser.new_page()
        page.goto(
            supervisor.status()["notebook_url"] + f"?token={supervisor.token}",
            wait_until="domcontentloaded",
        )
        page.wait_for_selector(".jp-Notebook", timeout=90000)

        session = McpSession(
            f"http://{profile.mcp.host}:{profile.mcp.port}/mcp"
        )

        def _comm_ready() -> bool:
            return (
                (httpx.get(
                    f"{supervisor.dashboard_url}/api/status",
                    headers={"Authorization": f"Bearer {supervisor.dashboard_token}"},
                    timeout=10,
                ).json().get("components") or {})
                .get("comm", {})
                .get("state")
                == "ready"
            )

        deadline = time.monotonic() + 120
        while time.monotonic() < deadline and not _comm_ready():
            time.sleep(1)

        stack = Live(supervisor, session, browser, page)
        assert _comm_ready(), "JupyterLab Comm bridge did not connect"

        # The agent's discovery loop: search, then prove every id it will use.
        for query, expected in (
            ("fit_gold", API_FIT_GOLD),
            ("k_convert", API_K_CONVERT),
            ("set_EF_correction", API_SET_EF),
        ):
            found = stack.tool("search", {"query": query, "limit": 5})
            ids = [match.get("id") for match in found.get("matches") or []]
            assert expected in ids, f"search({query!r}) did not surface {expected}: {ids}"
            stack.prove(expected)
        for facade in (API_LOAD_DATA, API_INSPECT, API_VALIDATION, API_SLICE):
            stack.prove(facade)

        yield stack, source_before
    finally:
        if session is not None:
            session.close()
        if page is not None:
            with contextlib.suppress(Exception):
                page.close()
        if browser is not None:
            with contextlib.suppress(Exception):
                browser.close()
        if playwright is not None:
            with contextlib.suppress(Exception):
                playwright.stop()
        with contextlib.suppress(Exception):
            supervisor.stop()
        uninstall_kernel(kernel_name)
        os.chdir(saved_cwd)
        if saved_home is None:
            os.environ.pop("PEAKSMCP_HOME", None)
        else:
            os.environ["PEAKSMCP_HOME"] = saved_home


def test_cut_preprocessing_on_real_data(live):
    """Human story 1: index -> gold fit -> Fermi leveling -> k-space cut + figure."""
    stack, _ = live

    index_cell = stack.cell(
        "index the converted folder",
        "from peaksMCP.overrides import load_data, inspect_experiment\n"
        "scans = load_data('data_netcdf')\n"
        "summary = inspect_experiment(scans)",
    )
    assert index_cell["stdout_lines"] >= 1
    assert "load_data:" in str(index_cell.get("stdout_head"))

    stack.cell(
        "bind the three scans",
        f"gold = scans['{GOLD_STEM}']\n"
        f"cut = scans['{CUT_STEM}']\n"
        f"mp = scans['{MAPPING_STEM}']",
    )

    stack.cell(
        "fit the gold reference once",
        "fit = gold.fit_gold(plot=False, show=False)\n"
        "ef = dict(fit.attrs['EF_correction'])\n"
        "assert 'c0' in ef, ef",
        api_ids=[API_FIT_GOLD],
        timeout=300.0,
    )
    ef_preview = stack.variable("ef")["repr"]
    assert "c0" in ef_preview and "2.6" in ef_preview, ef_preview

    stack.cell(
        "flatten EF and zero the high-symmetry angle",
        "cut.metadata.set_EF_correction(ef)\n"
        f"shifted = cut.assign_coords(theta_par=cut.theta_par - {THETA_OFFSET_DEG})",
        api_ids=[API_SET_EF],
    )

    stack.cell(
        "convert the cut to k-space",
        "kcut = shifted.k_convert(quiet=True)\n"
        "assert kcut.dims == ('eV', 'kx'), kcut.dims\n"
        "assert float(kcut.kx.min()) < 0.0 < float(kcut.kx.max()), kcut.kx.values[[0, -1]]",
        api_ids=[API_K_CONVERT],
    )
    kcut_preview = stack.variable("kcut")
    assert kcut_preview["dims"] == ["eV", "kx"], kcut_preview
    assert kcut_preview["dtype"] == "float64", kcut_preview
    assert kcut_preview["units"] is None or isinstance(kcut_preview["units"], str)

    figure_cell = stack.cell(
        "render the raw vs k-space validation figure",
        "from peaksMCP.overrides import plot_validation_pair\n"
        "fig = plot_validation_pair(cut, kcut, shared_scale='auto')",
        api_ids=[API_VALIDATION],
    )
    assert stack.figure_markers(figure_cell) >= 1, figure_cell.get("output")


def test_mapping_preprocessing_and_binding_energy_slices(live):
    """Human story 2: full-cube k-conversion, then the mapping's eV slices."""
    stack, _ = live

    stack.cell(
        "convert the full mapping cube",
        "assert 'ef' in dir(), 'run the cut scenario first: the gold calibration is shared'\n"
        "mp.metadata.set_EF_correction(ef)\n"
        "kmap = mp.k_convert(quiet=True)\n"
        "assert {'eV', 'kx', 'ky'} <= set(kmap.dims), kmap.dims\n"
        "assert kmap.sizes['kx'] > 100 and kmap.sizes['ky'] > 100, kmap.sizes",
        api_ids=[API_SET_EF, API_K_CONVERT],
        timeout=300.0,
    )

    kmap_preview = stack.variable("kmap")
    assert set(kmap_preview["dims"]) == {"eV", "kx", "ky"}, kmap_preview

    slice_cell = stack.cell(
        "render the mapping's binding-energy slices",
        "import numpy as np\n"
        "from peaksMCP.overrides import show_mapping_slice\n"
        f"idx = [int(np.argmin(np.abs(kmap.eV.values - e))) for e in {SLICE_ENERGIES!r}]\n"
        "figs = [show_mapping_slice(kmap, dim='eV', index=i) for i in idx]\n"
        "assert len(figs) == 3",
        api_ids=[API_SLICE],
    )
    assert stack.figure_images(slice_cell) >= 3, slice_cell.get("output")

    stack.cell(
        "sanity-check the processed cube",
        "import numpy as np\n"
        "assert float(np.isfinite(kmap.values).mean()) > 0.9\n"
        "assert float(kmap.eV.min()) < -0.5 < float(kmap.eV.max()), float(kmap.eV.min())",
    )


def test_dashboard_reflects_the_live_session(live):
    """Human story 3: operator console state, a console-driven MCP restart, Comm."""
    stack, source_before = live

    status = stack.status()
    assert status["aggregate"] in {"ready", "degraded"}
    assert status["components"]["jupyter"]["state"] == "ready"
    assert status["components"]["mcp"]["state"] == "ready"
    assert status["components"]["comm"]["state"] == "ready"
    assert status["notebook_open_url"]

    health = stack.health()
    assert health["ok"], health
    assert set(health["tools"]) == FIVE_TOOLS
    assert health["missing_tools"] == [] and health["unexpected_tools"] == []

    unauthenticated = stack.dashboard("/api/status", token=None)
    assert unauthenticated.status_code in {401, 403}, unauthenticated.status_code

    restarted = stack.dashboard("/api/restart/mcp", method="POST")
    assert restarted.status_code == 200, restarted.text
    assert restarted.json().get("ready"), restarted.json()
    # The in-kernel server instance is new, so the client session is too.
    stack.session.reopen()

    after = stack.health()
    assert after["ok"], after
    assert set(after["tools"]) == FIVE_TOOLS
    assert (after.get("status") or {}).get("comm_connected") is True

    # The MCP restart must not touch the analysis state the human built up.
    stack.cell(
        "analysis state survives the console MCP restart",
        "assert {'kcut', 'kmap'} <= set(dir()), sorted(n for n in dir() if not n.startswith('_'))[:12]",
    )

    # A human reopening the notebook re-establishes the Comm bridge.
    stack.page.reload(wait_until="domcontentloaded")
    stack.page.wait_for_selector(".jp-Notebook", timeout=90000)
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if stack.status()["components"]["comm"]["state"] == "ready":
            break
        time.sleep(1)
    assert stack.status()["components"]["comm"]["state"] == "ready"

    # The raw beamtime folder was only read: no conversion output landed there.
    assert sorted(p.name for p in RAW_PXT_DIR.iterdir()) == source_before
    assert not list(RAW_PXT_DIR.glob("*.nc")), "conversion must never write into the raw folder"
