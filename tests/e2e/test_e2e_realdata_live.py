"""Real-data E2E: a human at JupyterLab driving peaksMCP over its five tools.

Scope: this file is the **product-path acceptance** for the MCP/Jupyter/Comm
chain — real data, real browser, real kernel, real consent cards, all driven
through the model-facing tools exactly as an agent would.  It is *not* the
autonomous-agent benchmark: whether a model can interpret a task prompt,
schedule tools and recover from its own mistakes is measured by
``benchmark/run_campaign.py`` (campaign trials with P1/P2 conditions), which
needs a model provider and therefore never runs in CI.

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

pytestmark = [pytest.mark.e2e]

#: Raw beamtime folder (same convention as the real-data integration tests).
RAW_PXT_DIR = Path(
    os.environ.get("PEAKSMCP_REALDATA_PXT")
    or "/Users/haoxin/Documents/实验数据/BP260623/data"
)
_CONVERTED_HINT = Path("/Users/haoxin/Documents/实验数据/BP260623/data_netcdf")

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
    """Copy the three raw scans and convert them into the sibling layout.

    Mirrors the real dataset (raw in ``data/``, converted NetCDF in
    ``data_netcdf/``) without ever writing inside the source folder.  The
    metadata document must describe the data it is paired with, so its record
    indexes are checked against the stems before anything is copied.
    """
    data = home / "data"
    converted = home / "data_netcdf"
    data.mkdir(parents=True)
    converted.mkdir(parents=True)
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
        shutil.copy2(METADATA_JSON, converted / "experiment_metadata.json")

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


def _answer_save_card(
    stack: Live, *, approve: bool, timeout: float = 120.0, pending=None
) -> int:
    """Answer staged-save consent cards in the real notebook frontend.

    Mirrors the benchmark approval harness: look at every visible JupyterLab
    dialog, act only on the one carrying the save-card marker, and keep
    answering until the pending tool call settles (a staged save blocks on the
    card, so the call runs on a worker thread while this loop clicks).
    """
    page = stack.page
    selector = "button.jp-mod-accept" if approve else "button.jp-mod-reject"
    deadline = time.monotonic() + timeout
    clicks = 0
    while time.monotonic() < deadline:
        if pending is not None and pending.done():
            return clicks
        dialogs = page.locator(".jp-Dialog")
        for index in range(dialogs.count()):
            dialog = dialogs.nth(index)
            if not dialog.is_visible():
                continue
            if not dialog.locator('[data-peaks-mcp-dialog="save-consent"]').count():
                continue  # not a save card: leave other dialogs alone
            button = dialog.locator(selector)
            if button.count():
                button.first.click()
                clicks += 1
                page.wait_for_timeout(300)
        time.sleep(0.2)
    if pending is not None and not pending.done():
        raise AssertionError(f"the save consent card was never answered ({clicks} click(s))")
    return clicks


def test_acceptance_realdata_workflow(live):
    """The acceptance scenario: an explicitly sequential, discovery-driven chain.

    One test owns the whole ordered story (classification → gold fit → Fermi
    leveling → angular zeroing → k-space cut → figure → mapping cube → slices →
    console/MCP restart → Comm reconnect) because later steps reuse variables
    the earlier steps created: splitting it into independent tests would either
    duplicate the expensive gold fit or hide the dependency behind collection
    order.  Everything the chain processes is *derived* from
    ``inspect_experiment``\'s classification and the experiment metadata, never
    hard-coded, so a broken classifier or offset source fails here.
    """
    stack, _ = live

    index_cell = stack.cell(
        "index the converted folder",
        "from peaksMCP.overrides import load_data, inspect_experiment\n"
        "scans = load_data('data_netcdf')",
    )
    assert index_cell["stdout_lines"] >= 1
    assert "load_data:" in str(index_cell.get("stdout_head"))

    # The classification is the agent's entry point into the chain: it must be
    # readable from the run_cell reply, must be exactly right for this frozen
    # dataset, and must drive every later choice (stems and theta offset).
    classify_cell = stack.cell(
        "classify the experiment",
        "summary = inspect_experiment(scans)\n"
        # Exact classification for the frozen dataset: the metadata document
        # classifies every record it lists, and the fixture's mapping (no
        # record) is classified from its shape.  The conflict path needs a 3-D
        # record labelled "sweep", which this subset does not load - it is
        # unit-tested instead.
        "assert summary.gold == [20], summary.gold\n"
        "assert len(summary.cuts) == 14 and 15 in summary.cuts, summary.cuts\n"
        "assert 1 in summary.mappings, summary.mappings\n"
        "assert summary.conflicts == [], summary.conflicts\n"
        "present = {int(s[-4:]) for s in scans.stems}\n"
        "gold_index = next(i for i in summary.gold if i in present)\n"
        "cut_index = next(i for i in summary.cuts if i in present)\n"
        "mapping_index = next(i for i in summary.mappings if i in present)\n"
        "gold_stem = f'BP_{gold_index:04d}'\n"
        "cut_stem = f'BP_{cut_index:04d}'\n"
        "mapping_stem = f'BP_{mapping_index:04d}'\n"
        "theta_offset = next(r.theta_offset_deg for r in summary.records if r.index == cut_index)\n"
        "assert theta_offset, 'the angular offset must come from the metadata'",
    )
    classification = str(classify_cell.get("stdout_head"))
    assert "inspect_experiment:" in classification, classify_cell
    assert "gold=[20]" in classification and "cuts=14" in classification, classify_cell

    stack.cell(
        "bind the three scans chosen by the classification",
        "gold = scans[gold_stem]\n"
        "cut = scans[cut_stem]\n"
        "mp = scans[mapping_stem]",
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
        "flatten EF and zero the high-symmetry angle from metadata",
        "cut.metadata.set_EF_correction(ef)\n"
        "shifted = cut.assign_coords(theta_par=cut.theta_par - theta_offset)",
        api_ids=[API_SET_EF],
    )

    stack.cell(
        "convert the cut to k-space and verify the alignment",
        "kcut = shifted.k_convert(quiet=True)\n"
        "assert kcut.dims == ('eV', 'kx'), kcut.dims\n"
        # Fermi leveling: the binding-energy axis crosses E_F = 0.
        "assert float(kcut.eV.min()) <= 0.0 <= float(kcut.eV.max()), float(kcut.eV.min())\n"
        # Angular zeroing: kx is odd about the high-symmetry angle, so the axis
        # comes out symmetric about 0 (measured 0.021 on this reference data).
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
        "from peaksMCP.overrides import plot_validation_pair\n"
        "fig = plot_validation_pair(cut, kcut, shared_scale='auto')",
        api_ids=[API_VALIDATION],
    )
    assert stack.figure_markers(figure_cell) >= 1, figure_cell.get("output")

    # --- persistence is part of the product path (finding: it was never exercised) ---
    import concurrent.futures
    import hashlib

    saved_path = Path(os.environ["PEAKSMCP_HOME"]) / "saved" / f"{CUT_STEM}_processed.nc"
    saved_path.parent.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        # save_with_consent stages the bytes and blocks on the card, so the
        # tool call runs on a worker thread while the page answers it.
        pending = executor.submit(
            stack.session.call,
            "save_with_consent",
            {"variable_name": "kcut", "path": str(saved_path)},
        )
        _answer_save_card(stack, approve=True, pending=pending)
        receipt = pending.result(timeout=240)
    assert receipt.get("status") == "saved", receipt
    assert Path(str(receipt.get("path"))) == saved_path, receipt
    assert saved_path.is_file(), receipt
    digest = hashlib.sha256(saved_path.read_bytes()).hexdigest()
    assert receipt.get("sha256") == digest, (receipt.get("sha256"), digest)

    denied_path = Path(os.environ["PEAKSMCP_HOME"]) / "denied" / f"{CUT_STEM}_processed.nc"
    denied_path.parent.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(
            stack.session.call,
            "save_with_consent",
            {"variable_name": "kcut", "path": str(denied_path)},
        )
        _answer_save_card(stack, approve=False, pending=pending)
        denied = pending.result(timeout=240)
    assert denied.get("status") == "denied", denied
    assert not denied_path.exists(), "a denied save must write nothing"

    stack.cell(
        "convert the full mapping cube",
        "mp.metadata.set_EF_correction(ef)\n"
        "kmap = mp.k_convert(quiet=True)\n"
        "assert {'eV', 'kx', 'ky'} <= set(kmap.dims), kmap.dims\n"
        "assert kmap.sizes['kx'] > 100 and kmap.sizes['ky'] > 100, kmap.sizes\n"
        # The mapping keeps its own geometry (no angular shift is applied), so
        # only the converted ranges are asserted here.
        "assert float(kmap.kx.max()) > 0.1 and float(kmap.ky.max()) > 0.1, "
        "(float(kmap.kx.max()), float(kmap.ky.max()))",
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

    # --- the operator console drives the running session ---
    status = stack.status()
    assert status["aggregate"] in {"ready", "degraded"}
    assert status["components"]["comm"]["state"] == "ready"
    assert status["notebook_open_url"]

    restarted = stack.dashboard("/api/restart/mcp", method="POST")
    assert restarted.status_code == 200, restarted.text
    assert restarted.json().get("ready"), restarted.json()
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
