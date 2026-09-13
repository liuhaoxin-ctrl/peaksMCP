"""Real-data E2E: a human at JupyterLab driving peaksMCP over its five tools.

Scope: this file is the **product-path acceptance** for the MCP/Jupyter/Comm
chain — real data, real browser, real kernel, all driven
through the model-facing tools exactly as an agent would.  It is *not* the
autonomous-agent benchmark: whether a model can interpret a task prompt,
schedule tools and recover from its own mistakes is measured by
``benchmark/run_campaign.py`` (campaign trials with P1/P2 conditions), which
needs a model provider and therefore never runs in CI.

The suite replaces the earlier mechanism-only live tests.  It simulates the
real usage story end to end on the machine that holds the beamtime data:

1. **conversion and cut preprocessing** — convert raw PXT twice to prove cache
   reuse, load/classify the experiment, fit the gold reference
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

Data-privacy contract: the raw folder is only read. The fixture copies three
scans into a temporary home; ``peaks.pxt2nc`` creates the sibling cache there,
then a second call must reuse it. No processed NetCDF or image is persisted.

Requirements: the raw beamtime folder (``PEAKSMCP_REALDATA_PXT``, defaulting
to the L112 BP260623 dataset), Google Chrome (Playwright ``channel="chrome"``)
and the ``peaks`` conda environment.  Without the raw folder the module skips.

Run explicitly::

    python tools/test.py e2e
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import re
import shutil
import socket
import threading
import time
from pathlib import Path

import httpx
import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.realdata, pytest.mark.browser, pytest.mark.slow]

#: Raw beamtime folder (same convention as the real-data integration tests).
RAW_PXT_DIR = Path(
    os.environ.get("PEAKSMCP_REALDATA_PXT")
    or "/Users/haoxin/Documents/实验数据/BP260623/data"
)
_CONVERTED_HINT = Path(
    os.environ.get("PEAKSMCP_BENCH_REFERENCE")
    or "/Users/haoxin/Documents/实验数据/BP260623/data_netcdf"
)

#: Experiment metadata document.  Overridable together with the raw folder so
#: pointing PEAKSMCP_REALDATA_PXT at another dataset never silently pairs that
#: data with BP260623 metadata (the identity check below enforces it).
METADATA_JSON = Path(
    os.environ.get("PEAKSMCP_REALDATA_METADATA")
    or (_CONVERTED_HINT / "experiment_metadata.json")
)

#: The three scans the story needs: gold reference, one cut, one mapping cube.
GOLD_STEM = "BP_0020"
CUT_STEM = "BP_0015"
MAPPING_STEM = "BP_0001"
THETA_OFFSET_DEG = 1.5
SLICE_ENERGIES = (0.0, -0.3, -0.6)

#: Canonical ids the notebook cells rely on.
API_PXT2NC = "top_level:peaks.core.fileIO.experiment:pxt2nc"
API_LOAD_EXPERIMENT = "top_level:peaks.core.fileIO.experiment:load_experiment"
API_FIT_GOLD = "dataarray:peaks.core.fitting.fit:fit_gold"
API_K_CONVERT = "dataarray:peaks.core.process.k_conversion:k_convert"
API_ASSIGN_NORMAL = "metadata:peaks.core.metadata.metadata_methods:assign_normal_emission"

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
        self._error: BaseException | None = None
        self._start(what="open")

    def _start(self, *, what: str) -> None:
        """Start the session thread and fail with the real cause, never a bare timeout."""
        self._error = None
        self._ready = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="peaksmcp-e2e-session", daemon=True
        )
        self._thread.start()
        if not self._ready.wait(60):
            raise RuntimeError(f"MCP session did not {what} within 60s")
        if self._error is not None:
            raise RuntimeError(f"MCP session {what} failed: {self._error!r}") from self._error

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._open())
        except BaseException as exc:  # noqa: BLE001 - reported to the caller
            self._error = exc
            self._ready.set()  # a failed open must not wait out the 60s budget
            return
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
        self._start(what="reopen")

    def close(self) -> None:
        """Close the client, join the thread and close the loop (no races)."""

        async def _shutdown() -> None:
            if self._stack is not None:
                await self._stack.aclose()

        with contextlib.suppress(Exception):
            asyncio.run_coroutine_threadsafe(_shutdown(), self.loop).result(20)
        with contextlib.suppress(RuntimeError):
            self.loop.call_soon_threadsafe(self.loop.stop)
        thread = getattr(self, "_thread", None)
        if thread is not None and thread.is_alive():
            thread.join(timeout=10)
        with contextlib.suppress(Exception):
            if not self.loop.is_running() and not self.loop.is_closed():
                self.loop.close()


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


def _stem_index(stem: str) -> int:
    """Experiment index encoded in a scan stem (``BP_0015`` -> 15)."""
    return int(stem.rsplit("_", 1)[-1])


def _raw_fingerprint() -> dict[str, tuple[int, str]]:
    """Size + sha256 per raw input file we actually read.

    A filename listing cannot detect an edited source file, so the integrity
    check hashes the inputs themselves (the suite is read-only towards them).
    """
    wanted = [f"{stem}.pxt" for stem in (GOLD_STEM, CUT_STEM, MAPPING_STEM)]
    wanted.append("datasheet.csv")
    fingerprint: dict[str, tuple[int, str]] = {}
    for name in wanted:
        path = RAW_PXT_DIR / name
        if path.is_file():
            fingerprint[name] = (path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest())
    return fingerprint


def _prepare_home(home: Path) -> None:
    """Copy the three raw scans; conversion is exercised through peaksMCP.

    Mirrors the real dataset (raw in ``data/``, converted NetCDF in
    ``data_netcdf/``) without ever writing inside the source folder.  The
    metadata document must describe the data it is paired with, so its record
    indexes are checked against the stems before anything is copied.
    """
    data = home / "data"
    data.mkdir(parents=True)
    for stem in (GOLD_STEM, CUT_STEM, MAPPING_STEM):
        shutil.copy2(RAW_PXT_DIR / f"{stem}.pxt", data / f"{stem}.pxt")
    if (RAW_PXT_DIR / "datasheet.csv").is_file():
        shutil.copy2(RAW_PXT_DIR / "datasheet.csv", data / "datasheet.csv")
    if METADATA_JSON.is_file():
        document = json.loads(METADATA_JSON.read_text(encoding="utf-8"))
        records = document.get("records") or {}
        known = {str(key) for key in records}
        # Identity first: the document embeds the hash of the datasheet it was
        # translated from, so pairing data with another experiment's metadata is
        # detected instead of silently mixing two datasets.
        datasheet = RAW_PXT_DIR / "datasheet.csv"
        source_hash = document.get("source_sha256")
        if datasheet.is_file() and source_hash:
            observed = hashlib.sha256(datasheet.read_bytes()).hexdigest()
            assert observed == str(source_hash), (
                f"{METADATA_JSON} was translated from a different datasheet "
                f"(source_sha256={source_hash}, observed={observed}); set "
                "PEAKSMCP_REALDATA_METADATA to the document of this dataset"
            )
        # Coverage: the workflow's decisions come from the gold and cut records.
        missing = [
            str(_stem_index(stem))
            for stem in (GOLD_STEM, CUT_STEM)
            if str(_stem_index(stem)) not in known
        ]
        assert not missing, (
            f"{METADATA_JSON} has no record for index(es) {missing}; set "
            "PEAKSMCP_REALDATA_METADATA to the matching document"
        )
        # Keep the translated document next to the raw datasheet. pxt2nc owns
        # copying/refreshing it into the cache directory.
        shutil.copy2(METADATA_JSON, data / "experiment_metadata.json")


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
    source_before = _raw_fingerprint()
    source_listing = sorted(p.name for p in RAW_PXT_DIR.iterdir())
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
        dashboard_page = browser.new_page()
        dashboard_page.goto(
            supervisor.dashboard_url + f"/?token={supervisor.dashboard_token}",
            wait_until="domcontentloaded",
        )
        dashboard_page.wait_for_selector("#open-lab", timeout=30000)
        dashboard_page.wait_for_function(
            "document.querySelector('#open-lab')?.getAttribute('href')?.length > 0"
        )
        with dashboard_page.expect_popup(timeout=30000) as opened:
            dashboard_page.locator("#open-lab").click()
        page = opened.value
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
            ("pxt to netcdf", API_PXT2NC),
            ("load experiment", API_LOAD_EXPERIMENT),
            ("fit_gold", API_FIT_GOLD),
            ("k_convert", API_K_CONVERT),
            ("assign normal emission", API_ASSIGN_NORMAL),
        ):
            found = stack.tool("search", {"query": query, "limit": 5})
            ids = [match.get("canonical_id") for match in found.get("matches") or []]
            assert expected in ids, f"search({query!r}) did not surface {expected}: {ids}"
            stack.prove(expected)
        yield stack, (source_before, source_listing)
    finally:
        # Teardown order matters: the supervisor removes its runfile from
        # whatever PEAKSMCP_HOME is set at that moment, so the environment must
        # only be restored AFTER the host is stopped - otherwise the test would
        # delete the developer's real runfile.
        try:
            with contextlib.suppress(Exception):
                if session is not None:
                    session.close()
            with contextlib.suppress(Exception):
                if page is not None:
                    page.close()
            with contextlib.suppress(Exception):
                if browser is not None:
                    browser.close()
            with contextlib.suppress(Exception):
                if playwright is not None:
                    playwright.stop()
            with contextlib.suppress(Exception):
                supervisor.stop()
            with contextlib.suppress(Exception):
                uninstall_kernel(kernel_name)
        finally:
            with contextlib.suppress(Exception):
                os.chdir(saved_cwd)
            if saved_home is None:
                os.environ.pop("PEAKSMCP_HOME", None)
            else:
                os.environ["PEAKSMCP_HOME"] = saved_home
        # Source integrity is verified unconditionally, not inside one test:
        # an edited or added file in the raw folder must always fail loudly.
        after = _raw_fingerprint()
        assert after == source_before, (
            "raw beamtime inputs were modified by this suite: "
            f"{sorted(set(source_before) ^ set(after)) or [n for n in after if source_before.get(n) != after[n]]}"
        )
        assert not list(RAW_PXT_DIR.glob("*.nc")), "conversion must never write into the raw folder"
        assert sorted(p.name for p in RAW_PXT_DIR.iterdir()) == source_listing, (
            "the suite added or removed entries in the raw beamtime folder"
        )


def test_acceptance_realdata_workflow(live):
    """The acceptance scenario: an explicitly sequential, discovery-driven chain.

    One test owns the whole ordered story (classification → gold fit → Fermi
    leveling → angular zeroing → k-space cut → figure → mapping cube → slices →
    console/MCP restart → Comm reconnect) because later steps reuse variables
    the earlier steps created: splitting it into independent tests would either
    duplicate the expensive gold fit or hide the dependency behind collection
    order.  Everything the chain processes is *derived* from
    ``peaks.load_experiment`` classification and experiment metadata, never
    hard-coded, so a broken classifier or offset source fails here.
    """
    stack, _ = live

    first_conversion = stack.cell(
        "convert raw PXT into the validated cache",
        "import peaks\n"
        "conversion_first = peaks.pxt2nc('data')\n"
        "print(f'converted={conversion_first.converted} cached={conversion_first.cached} failed={conversion_first.failed}')",
        api_ids=[API_PXT2NC],
        timeout=300.0,
    )
    assert "failed=0" in str(first_conversion.get("stdout_head"))
    assert "converted=3" in str(first_conversion.get("stdout_head"))

    second_conversion = stack.cell(
        "prove a second conversion reuses the cache",
        "conversion_recheck = peaks.pxt2nc('data')\n"
        "print(f'converted={conversion_recheck.converted} cached={conversion_recheck.cached} failed={conversion_recheck.failed}')",
        api_ids=[API_PXT2NC],
        timeout=300.0,
    )
    assert "cached=3" in str(second_conversion.get("stdout_head"))
    assert "converted=0" in str(second_conversion.get("stdout_head"))

    # The classification is the agent's entry point into the chain: it must be
    # readable from the run_cell reply, must be exactly right for this frozen
    # dataset, and must drive every later choice (stems and theta offset).
    classify_cell = stack.cell(
        "load and classify the experiment",
        "experiment = peaks.load_experiment('data_netcdf')\n"
        # Exact classification for the frozen dataset: the metadata document
        # classifies every record it lists, and the fixture's mapping (no
        # record) is classified from its shape.  The conflict path needs a 3-D
        # record labelled "sweep", which this subset does not load - it is
        # unit-tested instead.
        "assert experiment.gold == [20], experiment.gold\n"
        "assert len(experiment.cuts) == 14 and 15 in experiment.cuts, experiment.cuts\n"
        "assert 1 in experiment.mappings, experiment.mappings\n"
        "present = {int(s[-4:]) for s in experiment.stems}\n"
        "gold_index = next(i for i in experiment.gold if i in present)\n"
        "cut_index = next(i for i in experiment.cuts if i in present)\n"
        "mapping_index = next(i for i in experiment.mappings if i in present)\n"
        "gold_stem = f'BP_{gold_index:04d}'\n"
        "cut_stem = f'BP_{cut_index:04d}'\n"
        "mapping_stem = f'BP_{mapping_index:04d}'\n"
        "theta_offset = next(r.theta_offset_deg for r in experiment.records if r.index == cut_index)\n"
        "assert theta_offset, 'the angular offset must come from the metadata'",
        api_ids=[API_LOAD_EXPERIMENT],
    )
    classification = str(classify_cell.get("stdout_head"))
    assert "load_experiment:" in classification, classify_cell
    assert "gold=[20]" in classification and "cuts=14" in classification, classify_cell

    stack.cell(
        "bind the three scans chosen by the classification",
        "gold = experiment[gold_stem]\n"
        "cut = experiment[cut_stem]\n"
        "mp = experiment[mapping_stem]",
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
        "assign the high-symmetry angle from metadata",
        # The offset must be the experiment's own value, not merely non-zero:
        # the contract value comes from the metadata document.
        "assert abs(theta_offset - " + str(THETA_OFFSET_DEG) + ") < 1e-9, theta_offset\n"
        "shifted = cut.metadata.assign_normal_emission(theta_par=theta_offset)",
        api_ids=[API_ASSIGN_NORMAL],
    )

    stack.cell(
        "convert the cut to k-space and verify the alignment",
        "kcut = shifted.k_convert(EF_correction=fit, quiet=True)\n"
        "assert kcut.dims == ('eV', 'kx'), kcut.dims\n"
        # Fermi leveling: the axis crosses E_F = 0 *and* carries the fitted
        # shift - a wrong constant (2.4 eV, say) also crosses zero, so the
        # crossing alone proves nothing.
        "assert float(kcut.eV.min()) <= 0.0 <= float(kcut.eV.max()), float(kcut.eV.min())\n"
        # A wrong constant would still cross zero, so this only guards the
        # scale; the frozen human reference comparison below is the real gate.
        "assert abs(float(kcut.eV.mean())) <= 0.5, float(kcut.eV.mean())\n"
        # Angular zeroing: kx is odd about the high-symmetry angle.  On this data
        # the symmetry test passes with AND without the offset (0.0204 vs
        # 0.0212), so it guards the shape only - the frozen human reference
        # comparison below is what actually pins the 1.5 deg.
        "assert abs(float(kcut.kx.min()) + float(kcut.kx.max())) <= 0.05, "
        "(float(kcut.kx.min()), float(kcut.kx.max()))",
        api_ids=[API_K_CONVERT],
    )
    kcut_preview = stack.variable("kcut")
    assert kcut_preview["dims"] == ["eV", "kx"], kcut_preview
    assert kcut_preview["dtype"] == "float64", kcut_preview
    assert kcut_preview["units"] is None or isinstance(kcut_preview["units"], str)

    figure_cell = stack.cell(
        "render the raw vs k-space validation figure",
        "import matplotlib.pyplot as plt\n"
        "fig, axes = plt.subplots(1, 2, figsize=(10, 4))\n"
        "cut.plot(ax=axes[0], add_colorbar=False)\n"
        "kcut.plot(ax=axes[1], add_colorbar=False)\n"
        "axes[0].set_title('raw detector coordinates')\n"
        "axes[1].set_title('EF-corrected momentum space')\n"
        "fig.tight_layout()",
    )
    assert stack.figure_markers(figure_cell) >= 1, figure_cell.get("output")

    # Frozen reference grading runs through the test-only kernel channel. It is
    # deliberately absent from the notebook and from the model's tool surface.
    reference_product = _CONVERTED_HINT / f"{CUT_STEM}_processed.nc"
    assert reference_product.is_file(), (
        f"the reference product {reference_product} is required for the "
        "coordinate/numeric comparison; point PEAKSMCP_BENCH_REFERENCE at the "
        "converted reference folder"
    )
    stack.supervisor.execute_kernel(
        "import numpy as np\n"
        "import xarray as xr\n"
        f"with xr.open_dataset({str(reference_product)!r}) as _ref:\n"
        "    _ref_da = _ref[list(_ref.data_vars)[0]]\n"
        "    assert list(kcut.dims) == list(_ref_da.dims), (kcut.dims, _ref_da.dims)\n"
        # Native conversion can produce one more interpolation point than the
        # historical human product. Compare physical extents, then interpolate
        # the reference onto today's grid before comparing masks and intensity.
        "    _coord_delta = max(max(abs(float(kcut[d].min()) - float(_ref_da[d].min())), "
        "abs(float(kcut[d].max()) - float(_ref_da[d].max())), "
        "abs(float(kcut[d].mean()) - float(_ref_da[d].mean()))) for d in kcut.dims)\n"
        "    assert _coord_delta <= 0.003, _coord_delta\n"
        "    _aligned = _ref_da.interp_like(kcut, method='linear')\n"
        "    _a = kcut.values.astype(float)\n"
        "    _b = _aligned.values.astype(float)\n"
        "    _ma, _mb = np.isfinite(_a), np.isfinite(_b)\n"
        "    _union, _both = _ma | _mb, _ma & _mb\n"
        "    _overlap = float(_both.sum() / _union.sum())\n"
        "    assert _overlap >= 0.97, _overlap\n"
        "    _corr = float(np.corrcoef(_a[_both], _b[_both])[0, 1])\n"
        "    assert _corr >= 0.98, _corr\n"
        "    _efficiency = float(np.mean(np.abs(_a[_both])) / np.mean(np.abs(_b[_both])))\n"
        "    assert 0.5 <= _efficiency <= 2.0, _efficiency\n"
        "    assert float(np.min(np.abs(kcut.eV.values))) <= 0.15\n"
        "    assert float(np.min(np.abs(kcut.kx.values))) <= 0.05\n",
        timeout=180,
    )

    stack.cell(
        "convert the full mapping cube",
        "kmap = mp.k_convert(EF_correction=fit, quiet=True)\n"
        "assert {'eV', 'kx', 'ky'} <= set(kmap.dims), kmap.dims\n"
        "assert kmap.sizes['kx'] > 100 and kmap.sizes['ky'] > 100, kmap.sizes\n"
        # The mapping keeps its own geometry (no angular shift is applied), so
        # only the converted ranges are asserted here.
        "assert float(kmap.kx.max()) > 0.1 and float(kmap.ky.max()) > 0.1, "
        "(float(kmap.kx.max()), float(kmap.ky.max()))",
        api_ids=[API_K_CONVERT],
        timeout=300.0,
    )
    kmap_preview = stack.variable("kmap")
    assert set(kmap_preview["dims"]) == {"eV", "kx", "ky"}, kmap_preview

    slice_cell = stack.cell(
        "render the mapping's binding-energy slices",
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        f"idx = [int(np.argmin(np.abs(kmap.eV.values - e))) for e in {SLICE_ENERGIES!r}]\n"
        "mapping_fig, mapping_axes = plt.subplots(1, 3, figsize=(12, 4))\n"
        "for axis, index, energy in zip(mapping_axes, idx, " + repr(SLICE_ENERGIES) + "):\n"
        "    kmap.isel(eV=index).plot(ax=axis, add_colorbar=False)\n"
        "    axis.set_title(f'{energy:.1f} eV')\n"
        "mapping_fig.tight_layout()",
    )
    assert stack.figure_markers(slice_cell) >= 1, slice_cell.get("output")

    stack.cell(
        "sanity-check the processed cube",
        "import numpy as np\n"
        "assert float(np.isfinite(kmap.values).mean()) > 0.9\n"
        "assert float(kmap.eV.min()) < -0.5 < float(kmap.eV.max()), float(kmap.eV.min())",
    )

    # --- the operator console drives the running session ---
    status = stack.status()
    assert status["aggregate"] in {"ready", "degraded"}
    assert status["components"]["comm"]["state"] == "ready"
    assert status["notebook_open_url"]

    console = stack.browser.new_page()
    try:
        console.goto(
            stack.supervisor.dashboard_url + f"/?token={stack.supervisor.dashboard_token}",
            wait_until="domcontentloaded",
        )
        console.wait_for_selector("#restart-mcp", timeout=30000)
        with console.expect_response(
            lambda response: response.url.endswith("/api/restart/mcp")
            and response.request.method == "POST",
            timeout=90000,
        ) as pending_restart:
            console.locator("#restart-mcp").click()
        restarted = pending_restart.value
        assert restarted.ok, restarted.status
        assert restarted.json().get("ready"), restarted.json()
    finally:
        console.close()
    # The in-kernel server instance is new, so the client session is too.
    stack.session.reopen()

    after = stack.health()
    assert after["ok"], after
    assert set(after["tools"]) == FIVE_TOOLS
    assert (after.get("status") or {}).get("comm_connected") is True

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
    # "ready" is a status field, and a stale bridge can keep reporting it (the
    # frontend Comm lingers up to 600 s), so issue a real request through the
    # reloaded page: appending and executing a cell only works if the new Comm
    # is actually carrying traffic.
    probe = stack.cell(
        "the reloaded page still serves execution",
        "reload_probe = True\nassert reload_probe",
    )
    assert probe.get("execution_success") is True, probe

    # The human sees real images in JupyterLab and the durable notebook keeps
    # their static MIME payloads. No processed array or image is a final file.
    stack.page.wait_for_selector(".jp-OutputArea-output img", timeout=90000)
    notebook_path = Path(os.environ["PEAKSMCP_HOME"]) / stack.supervisor.notebook_path
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    image_outputs = [
        output
        for cell in notebook.get("cells", [])
        for output in cell.get("outputs", [])
        if {"image/png", "image/jpeg", "image/svg+xml"} & set(output.get("data") or {})
    ]
    assert len(image_outputs) >= 2
    home = Path(os.environ["PEAKSMCP_HOME"])
    assert not list(home.rglob("*_processed.nc"))
    assert not [
        path for path in home.rglob("*")
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg"}
    ]


def test_dashboard_reports_components_and_auth(live):
    """Order-independent console smoke: component state, tool surface, auth gate.

    Deliberately touches no notebook variable, so it can run alone, first, or
    under test distribution without the acceptance scenario having executed.
    """
    stack, _ = live

    status = stack.status()
    assert status["components"]["jupyter"]["state"] == "ready"
    assert status["components"]["mcp"]["state"] == "ready"
    assert status["notebook_open_url"]

    health = stack.health()
    assert health["ok"], health
    assert set(health["tools"]) == FIVE_TOOLS
    assert health["missing_tools"] == [] and health["unexpected_tools"] == []

    unauthenticated = stack.dashboard("/api/status", token=None)
    assert unauthenticated.status_code in {401, 403}, unauthenticated.status_code
