# Agent instructions — peaksMCP

peaksMCP connects an agent to a live Jupyter kernel so ARPES data can be explored,
processed, converted and plotted using natural language, while every executable analysis
remains visible in the notebook. The **Pi** coding agent is the recommended one: live trials
run through `tools/trial.py` and paired campaigns through
`benchmark/run_campaign.py --runner pi-tui`. Claude Desktop is supported through the bundled
STDIO proxy and plugin.

This file is the single agent-guidance document for this repository. Agents (Claude,
Codex, etc.) must follow it when working here. It is modelled on the development-guide
conventions used by instrMCP (Claude Desktop ↔ Jupyter MCP bridges).

---

## 1. Boundaries

- Work only inside this repository unless the user explicitly requests otherwise.
- Never edit site-packages. The editable upstream checkout lives at
  `/Users/haoxin/peaks_dev`; public workflow fixes belong there when the user has
  placed both repositories in scope.
- Keep the **MCP server inside the Jupyter kernel** and the **supervisor outside it**
  (see Architecture). The kernel is the only place that executes user code.
- Preserve raw PXT inputs. Conversion writes new NetCDF files **atomically**
  (`.part` + rename) and never modifies the source.
- Do not add Qt/GUI plotting paths; use Jupyter inline Matplotlib output.
- Public Python APIs use **NumPy-style docstrings** (`Parameters` / `Returns` /
  `Raises` sections).
- No secrets in code. Jupyter and dashboard tokens are generated with
  `secrets.token_urlsafe` and stored only in `PEAKSMCP_HOME` files written with `0o600`
  permissions. Never return the raw Jupyter token from the dashboard status API.
- The dashboard binds `127.0.0.1` by default; never weaken that default silently.

---

## 2. Development environment

Use the `peaks` conda environment (Python 3.12):

```bash
# Always activate the environment first (or use the absolute interpreter)
conda activate peaks
# If "Run 'conda init' before 'conda activate'": source ~/.zshrc, then verify `type conda`
# prints "conda is a shell function".

pip install -e '.[dev]'        # install with dev dependencies
peaksMCP version               # verify installation
```

Absolute-interpreter form (works without an active shell hook):

```bash
/opt/homebrew/Caskroom/miniforge/base/envs/peaks/bin/python -m pip install -e '.[dev]'
/opt/homebrew/Caskroom/miniforge/base/envs/peaks/bin/peaksMCP version
```

### Code quality

```bash
ruff check peaksMCP tests tools    # CI gate (see .github/workflows/ci.yml)
```

Formatting/typing tooling may be added as dev dependencies; keep `ruff check` clean.

---

## 3. Architecture

### Communication flow

```text
Agent (Pi directly, or Claude Desktop through the STDIO proxy)
   -> kernel HTTP MCP server (127.0.0.1:8123/mcp) -> notebook backend <-> JupyterLab Comm
```

- `peaksMCP dash` starts the external **dashboard host** (`python -m peaksMCP _serve`,
  previously called the supervisor) if it is not already running and opens the
  operator console.  The host owns JupyterLab + one managed kernel + the dashboard
  and manages their start/stop/restart (the dashboard exposes Jupyter-group and
  MCP-group controls).  `peaksMCP stop`/`restart` manage only the host; stopping it
  also tears down the tree it manages.
- The **in-kernel MCP server** (`FastMCP`, HTTP on `127.0.0.1:8123/mcp`) only exists
  while the kernel runs; the supervisor is *outside* the kernel and never executes
  analysis code.
- Every agent talks to the same in-kernel MCP: the recommended **Pi** coding agent
  connects directly over HTTP MCP (its MCP config points at `127.0.0.1:8123/mcp`),
  while Claude Desktop goes through the **STDIO proxy** (`peaksMCP stdio-proxy`), so
  no URL/https configuration is needed.

### Key directories

- `peaksMCP/server/jupyter_peaks/` — in-kernel MCP: `mcp_server.py` (FastMCP HTTP),
  `backend/` (notebook read/write), `core/tools.py` (tool registration),
  `security/` (code scanner / consent / audit), `active_cell_bridge.py` (Comm),
  `jupyter_mcp_extension.py` (IPython extension + magics)
- `peaksMCP/app/` — external supervisor: `runtime.py` (process chain + dashboard),
  `kernel.py` (kernelspec), `profiles.py` (pydantic config), `api.py` + `webapp/`
  (dashboard)
- `peaksMCP/discovery/` — API index built from the unified
  `config/api_catalog.yaml`. Canonical IDs identify public Peaks callables;
  hidden implementation helpers never surface in search/get/run_cell.
- `peaksMCP/overrides/` — internal compatibility and persistence machinery,
  not a model-facing facade package. The initial public facade count is zero.
  Models use `peaks.pxt2nc`, `peaks.load_experiment`, then compose the remaining
  documented Peaks APIs in the notebook.
- `peaksMCP/pxt_utils/` — PXT→NetCDF conversion: `loader.py`, `converter.py`
  (atomic batch), `csv_translator.py`, `models.py`
- `peaksMCP/batch/` — CPU-budgeted process pool (`executor.py`, `resource_budget.py`)
- `peaksMCP/plotting/`, `peaksMCP/workflows/` — publication plotting and validation
- `peaksMCP/transport/` — `stdio_proxy.py` (Claude Desktop proxy)
- `peaksMCP/config/metadata_baseline.yaml` — tool presentation metadata (single source)
- `peaksMCP/config/prompts.yaml` — runtime prompt text (interactive/guidance copy,
  `notebook_unsafe` hard-block replies, code/ipython scanner issue descriptions)
- `peaksMCP/config/api_catalog.yaml` — one strict catalog for public native APIs
  and any evidence-backed future facade. Entries carry exposure
  (`core`/`advanced`/`hidden`), aliases and the callable contract; signature
  drift fails validation and `get`.
- `peaksMCP/config/schema.py` — strict catalog validation (duplicate keys,
  unknown fields, enum values, ghost references); a damaged default catalog
  fails index construction loudly.
- `claude_plugin/` — Claude Desktop plugin (`.mcp.json`, skills)

---

## 4. Testing

Use the named test runner so agents do not conflate deterministic tests, live
product-path checks, and autonomous-model experiments:

```bash
PY=/opt/homebrew/Caskroom/miniforge/base/envs/peaks/bin/python
$PY tools/test.py --list
$PY tools/test.py quick       # default offline gate
$PY tools/test.py benchmark   # grader regressions + golden/poison selftest
$PY tools/test.py e2e         # real Dashboard/Jupyter/MCP/Chrome path
```

- `quick` contains unit tests and mocked component integration. Plain `pytest`
  uses the same `not e2e and not slow` selection.
- Tests under `tests/integration` must carry `integration`; local-data tests
  additionally carry `realdata` and `slow`, and a real kernel adds
  `live_kernel`.
- E2E is the **real-data human-simulation suite**
  (`tests/e2e/test_e2e_realdata_live.py`, `pytest.mark.e2e`), run in the live
  environment only, never in CI. It launches a real JupyterLab + kernel, opens
  the notebook in real Google Chrome (Playwright `channel="chrome"`) so the
  Comm bridge is alive, and drives the notebook exclusively through the
  model-facing tools (`search` / `get` / `run_cell`) in ONE persistent MCP
  session (canonical ids proven with `get`, declared per cell through
  `api_ids`). Its three scenarios: cut preprocessing (gold `fit_gold` → Fermi
  leveling → high-symmetry zeroing → k-space + validation figure), mapping
  preprocessing (full cube, never a centre slice, then the binding-energy
  slices) and the dashboard (component state, five-tool surface via `/healthz`,
  console MCP restart, Comm reconnect). The raw beamtime folder is read-only
  (`PEAKSMCP_REALDATA_PXT`, default the L112 BP260623 dataset) and the suite
  skips without it; copies are converted inside the test home, never beside
  the source.
- `tests/e2e/test_campaign_acceptance.py` is a separate deterministic scripted-
  agent campaign (`acceptance` suite). No pytest layer drives a model: both e2e
  and acceptance are deterministic drivers of the product path.
- Live model trials (the `pi` TUI on `deepseek-v4-flash`, a fresh managed kernel
  and browser-backed JupyterLab per trial, fixed prompt, locked figure format,
  deterministic verification) run through `tools/trial.py`
  (`status` / `run` / `verify` / `restore`) and keep their sessions under
  `.pi_sessions/`. Use it instead of hand-orchestrating stack switches, pi
  invocations, or figure checks.
- `benchmark/run_campaign.py --runner pi-tui` is the campaign-level path: it
  drives the same pi TUI but adds the paired-replicate design, validity gates,
  grading and promotion comparison. Use it to compare configurations; use
  `tools/trial.py` for a single interactive or repeated trial run.
- CI runs the `quick` selection. The complete suite map and change-to-test
  routing table live in `tests/README.md`.
- When changing API discovery, run the full name-coverage and natural-language ranking
  tests (`tests/unit/test_discovery.py`).
- When adding a tool, the exact tool-list tests must be updated
  (`tests/unit/test_tools.py`, `tests/unit/test_extension_and_package.py`).

### CPU budget for tests and agent runs

- **Never let test/agent processes hold CPU above 60% sustained.** Keep compute light,
  prefer single-threaded workloads, and cap native thread pools (BLAS/OpenMP) when
  tests exercise numeric code.
- Batch conversion uses best-effort progressive throttling: it starts with one
  in-flight worker, increases concurrency only below the resume threshold, and
  stops new submissions while CPU is high. It is not an operating-system CPU quota;
  reports expose the requested budget, strategy, peak and whether it was exceeded.
- Check CPU before/after a heavy run (`ps aux -r | head`); if anything lingers above
  60%, stop and investigate rather than piling on more work.

---

## 5. Tool-change checklist

When adding, removing or renaming an MCP tool, update **all** of these:

- [ ] Tool implementation in `peaksMCP/server/jupyter_peaks/core/tools.py`
      (`register_safe_tools` / `register_unsafe_tools`)
- [ ] Presentation metadata in `peaksMCP/config/metadata_baseline.yaml`
- [ ] Exact tool-list tests: `tests/unit/test_tools.py`,
      `tests/unit/test_extension_and_package.py`
- [ ] Claude plugin skills under `claude_plugin/skills/` if user-facing
- [ ] `README.md` CLI/tool reference and `docs/ARCHITECTURE.md`
- [ ] Run `python tools/test.py check`

Tool metadata lives in YAML, not hardcoded in Python. `config/metadata.py` loads
`metadata_baseline.yaml` (exactly 5 tools: search / get / inspect_notebook /
run_cell / save_with_consent) as the single source of truth for titles and
descriptions, and `prompts.yaml` for the runtime prompt text that tools,
`notebook_unsafe.py` and the code scanners show the model/user.

---

## 6. Security and consent

The model surface is exactly FIVE tools, always exposed: `search` / `get` /
`inspect_notebook` / `run_cell` / `save_with_consent`.  Consent has three
independent layers:

- **Plain execution** follows the single `require_consent` master switch: when
  it is off (default) ordinary analysis cells run without a prompt (the AST
  scanner still hard-blocks dangerous code and every call is audit-logged);
- **run_cell is never an analysis-result persistence path**: file-write intents (scanner
  findings `SAVE001` savefig, `SAVE002` file writers, `FILE002` unclear file
  mode) are hard-blocked with a pointer to `save_with_consent` /
  `peaks.pxt2nc` — there is no in-cell result write that a prompt could unlock;
- **staged persistence**: `save_with_consent` stages the exact
  bytes in a server-owned gateway (strict TTL) and only an affirmative
  decision on the frontend save card publishes them; network egress
  (`NET001`) keeps the explicit-consent gate. `peaks.pxt2nc` is the sole
  automatic-persistence exception: it writes only an atomic, fingerprinted
  conversion cache and never modifies raw PXT.

There is no security mode.

The notebook is a **strictly append-only log**: `run_cell` only appends a new
cell at the END (internal record cells append the same way) and nothing can
edit, delete or reorder an existing cell, so the agent's full work history is
preserved top-to-bottom. `notebook_delete_cell`, the frontend
`delete_cell`/`apply_patch` handlers and the `mcp_list_resources` resources
system were all removed for exactly this reason (see changelog).

The consent master switch is `mcp.require_consent` in the active profile
(default **false**; the supervisor also exposes `PEAKSMCP_REQUIRE_CONSENT`).
With consent **disabled** (the default) plain execution shows no frontend
prompt: the AST code scanner (always-on hard block) and the audit log are the
only guards for ordinary cells. Analysis-result persistence is never unlocked
by a prompt on a write cell (run_cell blocks in-cell writes outright): it
happens only through staged `save_with_consent`. `pxt2nc` may maintain its
conversion cache without consent. There is no mode that
relaxes this policy.

The scanner (`security/code_scanner.py`) is AST-semantic (alias-aware,
attribute-chain matching) and an early rejection layer, **not a complete
security boundary**: it blocks exec/eval/compile, indirect fetches,
`os.environ` mutation, `open(mode=w/a/+)`, destructive `pathlib` methods,
subprocess and dynamic imports, but cannot bound arbitrary Python reflection or
import side effects. It hard-blocks regardless of `require_consent`; do not
weaken it; add bypass cases to `tests/unit/test_security.py` when extending it.

Consent decisions and every tool call are written to the audit log
(`PEAKSMCP_HOME/audit/tool_audit.log`, `0o600`).

---

## 7. Magic commands (in the managed kernel)

```python
%load_ext peaksMCP.server.jupyter_peaks.jupyter_mcp_extension
%peaksMCP_start          # start the in-kernel MCP server
%peaksMCP_stop           # stop it
%peaksMCP_restart        # restart (rebuild MCP)
%peaksMCP_status         # show status
```

There is **no security mode**. Plain-execution consent follows the single
`require_consent` master switch (profile `mcp.require_consent`, default
**false**): when off, ordinary cells run without a prompt (the scanner still
hard-blocks dangerous code and every call is audit-logged). run_cell hard-
blocks in-cell file writes (never a persistence path); persistence happens
only through `save_with_consent` / conversion consent cards; network egress
(`NET001`) keeps its explicit-approval gate.

---

## 8. Data conversion

**Conversion runs in the notebook, not from a shortcut.** There is deliberately
no `peaksMCP convert` / `metadata translate` / `load` CLI command and no
dashboard conversion endpoint: every data operation runs as a notebook cell
written through `run_cell`, so the code scanner, the API proof check and the
persistence policy always apply.

- `peaks.pxt2nc` is the public single-file/folder conversion entry; the CLI and
  console equivalents are deliberately absent.
- A **folder** conversion defaults to a **sibling `<folder>_netcdf/`** directory
  (created on demand); an explicit output path is honoured as-is.
- A **single file** defaults to `<stem>.nc` next to the source.
- `translate_datasheet(datasheet.csv)` produces `experiment_metadata.json`
  (records keyed by `Index`, source `sha256` embedded).
- Conversion never modifies the source and never aborts a batch on one bad file.
- The operator console only controls processes (Jupyter, kernel, MCP) and
  snapshots; it has no data endpoints.

---

## 9. CI/CD

- `.github/workflows/ci.yml` — `ruff check` + the `quick` selection on Python 3.11/3.12/3.13
  (ubuntu-latest).
- `.github/workflows/release.yml` — build `python -m build` and upload `dist/`.
- Version is managed in `peaksMCP/__init__.py` (`__version__`); keep it in sync with
  `pyproject.toml` and `claude_plugin/.claude-plugin/plugin.json`.

---

## 10. Lifecycle & troubleshooting

```bash
peaksMCP dash            # start the dashboard host if needed and open the console (single entry; idempotent)
peaksMCP status          # host / jupyter / kernel state
peaksMCP mcp-ping        # verify in-kernel MCP endpoint + private /healthz (expect ok: true, 5 tools)
peaksMCP stop            # stop the dashboard host (and the Jupyter/MCP tree it manages)
peaksMCP restart         # stop the host and start a fresh one
peaksMCP restart kernel  # kernel-side reload (the API index now hot-rebuilds itself when the source changes, so no restart is needed for index freshness)
peaksMCP logs -f         # follow host logs
```

Notes:

- The in-kernel MCP only listens while the dashboard host / kernel are running; the
  STDIO proxy forwards to it, so a running host (`peaksMCP dash` / `status`) must
  precede any agent usage - Pi connects to the same endpoint directly.
- Running instances load code at process start — after editing source, restart the
  kernel (`peaksMCP restart kernel` or `peaksMCP restart 'kernel&mcp'`) for changes
  to take effect.
- A host whose `run.json` discovery file was removed/corrupted while it kept
  running is re-adopted automatically: `status`/`dash`/`stop`/`restart` detect the
  live `_serve` host on the dashboard port and signal it (SIGWINCH, default-ignored
  on hosts that predate the handler) to republish its own runfile — never spawning
  a duplicate that would crash on the already-bound port.
- Do not configure peaksMCP as a *remote* MCP server in Claude Desktop: remote URLs
  must be `https://`, but the in-kernel MCP is plain HTTP on localhost. Use the STDIO
  proxy (`claude_plugin/.mcp.json`) instead.

---

## 11. Core design (derived from requirements, not from the file tree)

The system is designed from the core requirements below; every module exists to
serve one of them, and anything that duplicates or bypasses them is a defect.

Requirements:
1. The model understands user intent and schedules work; it composes functions
   through `search`/`get`, and only falls back to native `peaks` calls when the
   curated surface is insufficient. It never re-implements an existing function.
2. All execution happens in the managed Jupyter notebook (append-only cells).
3. Output is normalized: calling an existing function shows that function's own
   standard output; model-generated results use the canonical minimal summary.
   Nothing is added to reduce human review cost — noise is removed instead.
4. Analysis results are not persisted unless the user explicitly consents, and
   consent is only asked after the exact result is shown. The only automatic
   persistence exception is the fingerprinted PXT→NC conversion cache.

Minimal subsystem map (single responsibility each):

| Subsystem | Responsibility | Model sees |
|---|---|---|
| Contract (catalog) | One registry for public native APIs and evidence-backed facade entries | curated contract only |
| Access (search/get) | Searchable index built only from the unified catalog | contracts, never implementation |
| Run (notebook) | Single append-and-run entry with AST/API/consent gates; functions compose here | executes in notebook |
| Show (output normalization) | By task id: existing function -> its standard output verbatim; model-authored result -> canonical terse summary; strip progress/duplicate reprs | normalized text + rendered figures |
| Save (persist) | One write primitive: render full preview of what will be written -> user consent -> atomic write; default off | preview + consent |
| Observability | Minimal host/kernel/audit state for the human | dashboard/logs |

House rules while developing:
- Prefer the existing Peaks API. A facade may be added only when the same glue
  failure recurs in at least 2 of 3 valid TUI trials and a named rubric check
  improves in a paired rerun without regressions; otherwise keep zero facades.
- Single canonical source per contract/prompt/output format: catalog, prompts
  YAML and the Show formatter respectively — no duplicated instruction text.
- Any new analysis-result write path must route through the Save gateway
  (preview -> consent). Do not broaden the one pxt2nc cache exception.
- Results are typed models (ExperimentIndex / ConversionReport /
  SaveReceipt); there is deliberately no generic Report layer.
