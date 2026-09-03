# Agent instructions — peaksMCP

peaksMCP connects Claude Desktop to a live Jupyter kernel so ARPES data can be explored,
processed, converted and plotted using natural language, while every executable analysis
remains visible in the notebook.

This file is the single agent-guidance document for this repository. Agents (Claude,
Codex, etc.) must follow it when working here. It is modelled on the development-guide
conventions used by instrMCP (Claude Desktop ↔ Jupyter MCP bridges).

---

## 1. Boundaries

- Work only inside this repository unless the user explicitly requests otherwise.
- Treat the installed `peaks` package as an **external dependency**; do not edit
  site-packages. The editable checkout lives at `/Users/haoxin/peaks_dev` (0.5.3) and is
  consumed read-only.
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
Claude Desktop <-> STDIO proxy <-> HTTP MCP <-> Jupyter kernel <-> JupyterLab Comm
                   (cli stdio-proxy)   (127.0.0.1:8123/mcp)   (in-kernel)   (frontend)
```

- `peaksMCP launch` starts an external **supervisor** (`python -m peaksMCP _serve`) that
  owns JupyterLab + one managed kernel + the local dashboard.
- The **in-kernel MCP server** (`FastMCP`, HTTP on `127.0.0.1:8123/mcp`) only exists
  while the kernel runs; the supervisor is *outside* the kernel and never executes
  analysis code.
- Claude Desktop talks to the kernel MCP through the **STDIO proxy**
  (`peaksMCP stdio-proxy`), so no URL/https configuration is needed.

### Key directories

- `peaksMCP/server/jupyter_peaks/` — in-kernel MCP: `mcp_server.py` (FastMCP HTTP),
  `backend/` (notebook read/write), `core/tools.py` (tool registration),
  `security/` (code scanner / consent / audit), `active_cell_bridge.py` (Comm),
  `jupyter_mcp_extension.py` (IPython extension + magics)
- `peaksMCP/app/` — external supervisor: `runtime.py` (process chain + dashboard),
  `kernel.py` (kernelspec), `profiles.py` (pydantic config), `api.py` + `webapp/`
  (dashboard)
- `peaksMCP/discovery/` — API index: `index.py` (AST scan + fingerprint + search),
  `signatures.py` (signature/docstring extraction)
- `peaksMCP/pxt_utils/` — PXT→NetCDF conversion: `loader.py`, `converter.py`
  (atomic batch), `csv_translator.py`, `models.py`
- `peaksMCP/batch/` — CPU-budgeted process pool (`executor.py`, `resource_budget.py`)
- `peaksMCP/plotting/`, `peaksMCP/workflows/` — publication plotting and validation
- `peaksMCP/transport/` — `stdio_proxy.py` (Claude Desktop proxy)
- `peaksMCP/config/metadata_baseline.yaml` — tool presentation metadata (single source)
- `claude_plugin/` — Claude Desktop plugin (`.mcp.json`, skills)

---

## 4. Testing

Run fast by default (e2e excluded); only bring up live kernels/browsers for the
explicit e2e acceptance:

```bash
/opt/homebrew/Caskroom/miniforge/base/envs/peaks/bin/python -m pytest            # unit + integration
/opt/homebrew/Caskroom/miniforge/base/envs/peaks/bin/python -m pytest -m e2e     # live-kernel + browser
```

- Unit/integration tests are mocked and must stay fast and hardware-free.
- E2E tests launch a real JupyterLab + kernel + browser (Playwright chromium) and are
  marked `pytest.mark.e2e`; CI runs `pytest -m 'not e2e'`.
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
- [ ] Run `ruff check peaksMCP tests tools` and the unit tests

Tool metadata lives in YAML, not hardcoded in Python. `config/metadata.py` loads
`metadata_baseline.yaml` (15 tools) as the single source of truth for titles and
descriptions.

---

## 6. Security modes and consent

All 15 tools (13 read-only/guidance + 2 mutation) are **always exposed** in every
mode; the mode only changes how strictly the 2 mutation tools
(`notebook_write_with_api_check`, `notebook_add_cell`) ask for frontend consent
**when consent is enabled**.

The notebook is a **strictly append-only log**: both mutation tools only append
a new cell at the END and can never edit, delete or reorder an existing cell, so
the agent's full work history is preserved top-to-bottom. `notebook_delete_cell`
was removed for exactly this reason.

The consent master switch is `mcp.require_consent` in the active profile
(default **false**; the supervisor also exposes `PEAKSMCP_REQUIRE_CONSENT`).
With consent **disabled** (the default) no frontend prompt is shown in any mode:
the AST code scanner (always-on hard block) and the audit log are the only
guards. With consent **enabled**:

- **safe** / **unsafe** (identical tool surface): every mutation tool asks for
  explicit frontend consent shown in the notebook.
- **dangerous**: Python execution and destructive edits still require explicit
  frontend consent; only non-executing, append-only mutations (adding a cell)
  may be auto-approved.

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
%peaksMCP_safe           # switch to safe mode
%peaksMCP_unsafe         # switch to unsafe mode
%peaksMCP_dangerous      # relax only non-executing append operations
```

---

## 8. Data conversion

- `peaksMCP convert <pxt-file-or-folder> [--out OUT] [--metadata meta.json]
  [--filter SUBSTRING] [--cpu-limit PERCENT] [--force]`
- A **folder** conversion defaults to a **sibling `<folder>_netcdf/`** directory
  (created on demand); an explicit `--out` is honoured as-is.
- A **single file** defaults to `<stem>.nc` next to the source; an `--out` directory
  receives `<stem>.nc` inside it.
- `peaksMCP metadata translate datasheet.csv` produces `experiment_metadata.json`
  (records keyed by `Index`, source `sha256` embedded).
- Conversion never modifies the source and never aborts a batch on one bad file.

---

## 9. CI/CD

- `.github/workflows/ci.yml` — `ruff check` + `pytest -m 'not e2e'` on Python 3.11/3.12/3.13
  (ubuntu-latest).
- `.github/workflows/release.yml` — build `python -m build` and upload `dist/`.
- Version is managed in `peaksMCP/__init__.py` (`__version__`); keep it in sync with
  `pyproject.toml` and `claude_plugin/.claude-plugin/plugin.json`.

---

## 10. Lifecycle & troubleshooting

```bash
peaksMCP launch          # start supervisor (idempotent)
peaksMCP status          # supervisor + kernel state
peaksMCP mcp-ping        # verify in-kernel MCP endpoint (expect ok: true, 15 tools)
peaksMCP dash            # dashboard (127.0.0.1:8765)
peaksMCP stop            # stop supervisor
peaksMCP restart kernel  # reload running code (the API index now hot-rebuilds itself when the source changes, so no restart is needed for index freshness)
peaksMCP logs -f         # follow supervisor logs
```

Notes:

- The in-kernel MCP only listens while the supervisor/kernel are running; the STDIO
  proxy forwards to it, so `launch` must precede Claude Desktop usage.
- Running instances load code at process start — after editing source, restart the
  kernel (`peaksMCP restart kernel` or `peaksMCP restart 'kernel&mcp'`) for changes
  to take effect.
- Do not configure peaksMCP as a *remote* MCP server in Claude Desktop: remote URLs
  must be `https://`, but the in-kernel MCP is plain HTTP on localhost. Use the STDIO
  proxy (`claude_plugin/.mcp.json`) instead.
