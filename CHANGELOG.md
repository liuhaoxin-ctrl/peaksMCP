# Changelog

## [Unreleased] — 2026-09-10

### Fixed: P0 review findings (classification visibility, NET001 consent path)

Two defects confirmed by the 2026-09-10 requirement review, both reproduced
live before the fix:

- **`inspect_experiment` results are visible now.** Classification is the entry
  point of the preprocessing chain, but the function returned a bare
  `ExperimentSummary`: the cell produced a `text/plain` repr, which the
  `run_cell` normaliser deliberately drops, so the agent got an empty reply
  right after `load_data` told it to "classify with inspect_experiment(scans)".
  `ExperimentSummary.summary_line()` (gold index in full, cut/mapping/conflict
  counts) is printed by the facade, exactly like `load_data`'s report line; the
  real-data E2E asserts the line arrives in the cell's `stdout_head`.
- **NET001 consent no longer crashes.** `backend/notebook_unsafe.py` tested
  `issue.id`, but `SecurityIssue` is a slots dataclass whose field is
  `rule_id`; with the default `require_consent=False` the `or` did not
  short-circuit, so any cell matching an explicit-consent rule (typically
  network egress) raised `AttributeError` — no consent card, no actionable
  message, and the documented "network egress always needs approval" guarantee
  was dead code. Fixed to `rule_id`, with a regression test that asserts the
  card is requested (and declined) with the master switch off.

## [Unreleased] — 2026-09-10 (earlier)

### E2E is now the real-data human-simulation suite (BREAKING for the suite)

`tests/e2e/test_e2e_live.py` (8 mechanism-only cases) is replaced by
`tests/e2e/test_e2e_realdata_live.py`: a real JupyterLab + kernel + Google
Chrome session driving the notebook exclusively through the model-facing tools
in ONE persistent MCP session (ids proven with `get`, declared per cell via
`api_ids`). Three scenarios on the L112 BP260623 data:

- cut preprocessing: index → gold `fit_gold` → Fermi leveling → high-symmetry
  zeroing → `k_convert` → validation figure;
- mapping preprocessing: full 3-D cube (never a centre slice) → `k_convert` →
  the mapping's binding-energy slices;
- dashboard: component state, five-tool surface via `/healthz`, console MCP
  restart (namespace survives), Comm reconnect after a notebook reload.

Raw data stays read-only (`PEAKSMCP_REALDATA_PXT`, default the local BP260623
folder); the suite skips without it and never writes beside the source.

### Managed JupyterLab pins comms to the main shell

`app/runtime.py` writes `kernels-settings.jupyterlab-settings`
(`commsOverSubshells=disabled`) next to `page_config.json` in the isolated
Jupyter config: JupyterLab 4.6+ routes Comm traffic over ipykernel subshells by
default, which races around kernel restarts (shell-channel drops flood the
kernel log and stalled `restart all` past its readiness window).

### Fixed: three defects the real-data E2E surfaced

- **`load_data` now loads eagerly by default** (`lazy: bool = False`). The old
  default returned dask-backed values, and native `fit_gold` cannot fit a
  chunked array (`'float' object has no attribute 'astype'`) — the canonical
  workflow broke one step after loading. `lazy=True` stays available for
  header-only inspection and is documented as needing `da.load()` before
  native numerics.
- **`inspect_notebook` handles Peaks data again**: a `Dataset` variable (what
  `fit_gold` returns) no longer crashes the listing (`'Dataset' object has no
  attribute 'dtype'` — the summary line reports `n_vars` instead), and pint
  units (`Unit`/`Quantity`) are rendered as text, so the structured reply is
  JSON-serialisable and strict MCP clients stop rejecting it with
  `outputSchema defined but no structured output returned`. The `lazy` flag is
  also accurate now: a pint-wrapped numpy array is in memory (xarray reports
  duck arrays as not-in-memory), a pint-wrapped dask array stays lazy.
- **A stem present as both `.pxt` and `.nc` in one folder is resolved, not
  gambled**: the converted NetCDF is indexed (the raw PXT loads without
  instrument geometry, so a later `k_convert` failed with a confusing
  metadata error) and the collision is reported in the `load_data` summary
  (`N stem(s) had both .pxt and .nc (NetCDF indexed)`), with the stems on
  `LoadedScans.duplicate_stems`.

Tests: `tests/unit/test_notebook.py` (pint payload, Dataset listing/summary),
`tests/unit/test_overrides.py` (NetCDF preference + eager default), and the
real-data E2E now asserts the structured DataArray preview and the eager
default end to end.

## [Unreleased] — 2026-09-08

### API surface: catalog keeps only model-facing verbs (BREAKING)

The search/get catalog and index now expose verbs the model can act on;
implementation parts were made private and disappear from the index:

- `pxt_utils/converter.py`: `default_output_dir` / `index_from_path` are now
  `_default_output_dir` / `_index_from_path` (conversion internals).
- `pxt_utils/loader.py`: `register_l112_loader` is now `_register_l112_loader`;
  the in-kernel extension still registers the L112 loader at startup, but the
  model no longer needs to call it.
- `pxt_utils/metadata.py`: `load_metadata` is now `_load_metadata` — the raw
  document parser is an internal detail of the single `read_meta` verb (its
  aliases were merged onto `read_meta`).
- `batch/resource_budget.py`: `batch_execution_lock` is now `_batch_execution_lock`
  (used only by `BatchExecutor`).
- `workflows/publication.py`: `publication_grid` is now `_publication_grid`;
  publication intent ("论文配图" etc.) resolves to `plot_batch`, whose alias
  set absorbed the publication terms. `validate_arpes_metadata` stays public.
- `discovery/api_overrides.yaml`: six `project: true` rows removed (18 -> 12);
  `index.py`'s unused `IndexStaleError` deleted.

### Performance and dead code

- `discovery/index.py`: `ApiIndex.is_stale()` previously re-walked every Peaks +
  peaksMCP source file on every search/write. The fingerprint result is now
  cached for `STALE_REFRESH_INTERVAL_S` (5 s); `ensure_fresh_index` still
  hot-rebuilds once a stale fingerprint is observed.
- `security/code_scanner.py`: removed `call_names` (no production caller since
  the savefig special case folded into the scanner in the previous change); its
  export and unit test were deleted.
### Output normalisation lands on the write tool; save gate is consent-first (BREAKING)

The model no longer reads outputs back: `notebook_write_with_api_check` is the
single execution channel and returns the **normalised** output summary (errors,
the "Inline figure rendered ..." line, interactive markers) instead of the raw
Jupyter outputs (text echoes + base64 images). Text-only outputs are suppressed
by design — they are displayed in the notebook for the user.

- `core/tools.py`: extracted `_normalize_outputs` / `_text_blocks`; the write
  tool now returns `{id, index, cell_type, source, execution_success, saved,
  save_error, output, api_check}` and never the raw `outputs`. Outputs settle
  against the Comm push cache for a short bounded window so trailing inline
  images are not lost (`_settle_executed_outputs`).
- Tool surface 14 -> 9: removed `notebook_read_active_cell_output` (superseded
  by the write-tool summary), `notebook_read_content`, `notebook_move_cursor`,
  `notebook_kernel_status`, `notebook_wait_for_kernel`. `notebook_read_active_cell`
  now returns a normalised `output` list too.
- Output state dedup: `SharedState.active_cell_output` deleted; cell outputs
  live only in the bounded per-cell `cell_outputs` settle buffer. Frontend
  `active_cell` notifications carry cursor metadata only (no outputs).
- Save gate (requirement-first): scanner `SAVE001` savefig is no longer a hard
  block — it joins `SAVE002` file writers / `NET001` egress as an explicit
  consent finding. `UnsafeNotebookBackend._authorize` now requires user approval
  whenever the scanner reports a write/network intent, **regardless of the
  `require_consent` switch**: a result is never persisted unless the user sees
  the exact cell and approves it. Figures can be saved again after approval.
- `config/prompts.yaml`: `savefig_forbidden` removed, `savefig_consent` added,
  server instructions state the show-first / approve-to-save rule.
- Frontend: removed the dead `read_notebook` / `read_cell_at` / `move_cursor`
  handlers and the active-cell outputs push; the executed-cell push
  (`cell_output`) stays for the settle buffer.


### Security mode removed; single `require_consent` switch (BREAKING)

The `ExecutionMode` enum (`safe` / `unsafe` / `dangerous`) and the `mode` /
`PEAKSMCP_MODE` plumbing are gone. All 14 tools are always exposed; consent for
the two mutation tools is now governed solely by the `mcp.require_consent`
master switch (profile, default **false**). There is no mode that relaxes it.

- `backend/base.py`: deleted `ExecutionMode` and the `SharedState.mode` field.
- `backend/notebook_unsafe.py`: `_authorize` gate collapsed to a single
  `if self.state.require_consent:` check; removed the now-dead
  `_PYTHON_EXECUTION_OPERATIONS` set.
- `mcp_server.py`: removed `set_mode`; `jupyter_mcp_extension.py`: removed the
  `%peaksMCP_safe` / `%peaksMCP_unsafe` / `%peaksMCP_dangerous` magics and the
  `mode` field from every status dict.
- `app/profiles.py`, `app/kernel.py`, `app/runtime.py`, `app/defaults/default.yaml`:
  dropped the `mode` field, the `PEAKSMCP_MODE` env var and the `· safe`
  kernelspec display suffix.
- Behaviour change: with `require_consent: true`, append-only `notebook_add_cell`
  now also prompts (previously `dangerous` auto-approved non-executing appends).
  With `require_consent: false` (default) nothing changes.

### Leftover dead parameters and stale comments removed

Follow-up convergence after the `ExecutionMode` collapse:

- `backend/notebook_unsafe.py`: `_authorize` dropped the dead `force_consent`
  parameter (no caller ever passed it) and the dead `cell` parameter plus its
  `if cell is not None: details["cell"] = cell` branch (never triggered).
- `security/code_scanner.py`, `core/tools.py`, `AGENTS.md`: corrected comments
  still referencing the removed "dangerous mode" / "active mode" consent policy
  to describe the single `require_consent` switch.

### Tool-name inventory now derived from one source (BREAKING-adjacent)

`transport/stdio_proxy.py` no longer hand-lists the expected tool names.
`config/metadata.py` exposes `tool_names()` (the `tools:` keys of
`metadata_baseline.yaml`); both tool registration and the STDIO-proxy inventory
guard read from it, so a renamed/added/removed tool needs a single edit. The
`functions` dict in `core/tools.py` remains the registration map (name →
callable) and is now the only other place a name is written.

### Output-state cache field removed

`SharedState.last_execution_cell_id` was write-only (never read); removed the
field and its sole write in `active_cell_bridge.py`.

Removed/updated tests: the `set_mode` / `ExecutionMode.DANGEROUS` security tests,
the `mcp={"mode": ...}` profile fixtures, the `PEAKSMCP_MODE` env assertions and
the `last_execution_cell_id` assertion. `test_tools.py` now asserts the exposed
surface equals `metadata.tool_names()`.

### Data operations now only run inside the notebook (BREAKING)

Conversion / datasheet translation / loading a scan used to have three entry
points (CLI, operator console, notebook). Only the notebook path applies the code
scanner, the API check and the consent gate, so the other two are removed:

- CLI: dropped `peaksMCP convert`, `peaksMCP metadata translate`, `peaksMCP load`.
- Dashboard: dropped `POST /api/convert`, `/api/metadata/translate`,
  `/api/notebook/load`, `/api/choose-folder` and the 270-line
  `load_into_notebook` helper (it injected a `data = load(...)` cell through the
  Comm bridge, bypassing every check). The console keeps process control and
  snapshots only.
- The webapp's conversion form (and its `pretty` / `showResultPanel` /
  `hideResultPanel` helpers) is gone.
- Tests for the removed paths deleted (`test_notebook_load.py`, 7 load tests in
  `test_app_api.py`, 1 e2e load test): −902 lines of source, −24 tests.
- `README.md` / `AGENTS.md` now state that data operations are notebook-only.

Use `convert_pxt` / `convert_path` / `translate_datasheet` / `load_pxt` in a
notebook cell instead.

### Resources system, Inspector, frontend delete/patch removed and output normalised (BREAKING)

Per the project intent (AI drives black-box functions through `search`/`get`, the
notebook is the only execution surface, results are shown before any save), the
following were removed:

- **`mcp_list_resources` tool and the whole `resources:` block** (incl. the 6
  inline matplotlib plot templates) deleted. The templates taught the model to
  hand-write `pcolormesh` and bypass the `plot_batch` / `plot_validation_pair`
  façades — the single biggest leak against the black-box intent. Figure styling
  conventions now live in the always-on server instructions, and the plotting
  façades (`plot_batch`, `plot_validation_pair`, `show_mapping_slice`,
  `publication_grid`) are the documented path. Tool count: 15 → 14.
- **Inspector removed**: `POST /api/mcp/tool` + `_INSPECTOR_ALLOWED` (8 read-only
  tools) and the `Client` import gone from `app/api.py`; the dashboard is once
  again console-only (process control + snapshots).
- **Frontend dead handlers removed**: `delete_cell` / `apply_patch` comm handlers
  in the JupyterLab extension (the notebook is strictly append-only anyway).
- **Output normalisation (#4)**: `_output_content` now returns a summary *only*
  when a figure or an error is present; plain text / `text/plain` / markdown
  analysis boxes are no longer echoed to the model (re-stating them in the
  notebook is review noise). A text-only cell returns an empty payload.
- **Conversion is automatic and idempotent (#3)**: `convert_pxt` / `convert_path`
  were clarified as the auto-prerequisite transform (not a saved result — no
  consent needed) and already skip any `.nc` that already exists; the `load_pxt`
  / `convert_*` override APIs now carry this guidance so the model converts as
  the first step of any raw-scan workflow.

Removed/updated tests: `test_plot_resources_exposed_and_templates_compile`,
`test_inspector_whitelist_and_call`, two `read_plot_resources` gate tests, the
frontend `delete_cell` tests, plus the tool-count assertions (15 → 14).

Curated-presentation, prompt-governance & override-tier pass (`a58da6f` →
`HEAD`).

### Override tier: black-box, override-first discovery
- Every index entry is tagged `tier` (`override` = the 18 peaksMCP project
  APIs, `native` = peaks).
- `peaks_search_api` is **two-tier override-first**: when a query exactly
  matches an override name or alias (score ≥ 900) it returns override matches
  alone (`searched_tier: override`); otherwise it falls back over the full
  index (`searched_tier: native`). Partial names like `plot` are never
  hijacked by `plot_batch`.
- `peaks_get_api` renders overrides as **black-box interfaces**: `describe_api`
  no longer returns `source_path` for override entries; native peaks APIs keep
  source-backed docs.
- The override → native priority is restated in the L1 tool copy
  (`metadata_baseline.yaml`) and in a new `api_check_rule` runtime prompt
  (moved into `config/prompts.yaml`; 46 curated strings now).

### API discovery / curated presentation (L5)
- `api_overrides.yaml` rewritten to **one entry per API** (v2): each API's
  `aliases`, `docstring_note`, optional `module` and `project: true` read
  together; the audited project set is derived from the flag
  (`load_project_added()`), so there is no second list to keep in sync.
- All 18 project functions now carry search aliases (≥ 3 English each),
  closing the natural-language coverage gap (`convert_pxt` / `load_pxt` /
  `load_metadata` were previously exact-name-only).
- `build_index()` merges aliases/docstring notes per API and flags
  `project_added=True`; bidirectional tests turn silent exposure drift into a
  CI failure (nothing stale declared, nothing exposed undeclared).

### Runtime prompt text (L3) now YAML-managed
- New `config/prompts.yaml` (now 46 strings): always-on FastMCP server
  instructions, interactive-output note, `mcp_list_resources` guidance (6),
  `notebook_unsafe` hard-block replies (5) + the `api_check_rule` priority
  prompt, code-scanner (30) + IPython (2) issue templates.
- `mcp_server.py`, `core/tools.py`, `backend/notebook_unsafe.py`,
  `security/code_scanner.py`, `security/ipython_scanner.py` read the wording
  once at import and only format `{placeholders}`; copy no longer lives in
  Python (rule ids / risk levels / audit reasons stay in code).

### Skills / plot templates
- `cut-preprocessing` skill: two-verb gold-fit flow (`fit_gold` once →
  `metadata.set_EF_correction` → per-cut `k_convert()`), `da.save(path)`
  accessor, figure-debug protocol; removed dangling `preprocess_cut` /
  `save_processed` references.
- `metadata_baseline.yaml`: fixed block-scalar comment indentation in five
  plot templates (12 → 6 spaces).

### Docs
- Added `docs/CODE_REVIEW_2026-09-08.md`, `docs/PROJECT_ADDITIONS.md` and the
  English rewrite of `docs/PROMPT_AND_API_INVENTORY.md` (synced to the current
  state, incl. the L1–L6 layer cheat-sheet, full L6 copy lines, declared
  non-goals, override-tier discovery and fix log #1–#9).
- `AGENTS.md` documents `config/prompts.yaml` as the runtime-prompt single
  source.

### Tests & gates
- New regression tests: bidirectional project-exposure audit, alias/orphan
  guards, EF-handoff signature contract, prompts-config rendering incl. the
  server-instructions guard, override-tier / black-box behaviour
  (`tests/unit/test_override_tier.py`), an end-to-end override workflow test
  (`tests/unit/test_override_workflow.py`) that loads a real synthetic PXT
  fixture via `load_pxt`, stages the preprocessed (eV, kx) cut in memory and
  renders the before/after `plot_validation_pair` figure plus a `plot_batch`
  grid, then verifies the black-box docs match usage.
- Real-data acceptance (`tests/integration/
  test_override_realdata_cut_preprocessing.py`, runs against the raw L112
  `BP260623/data` PXT folder): mirrors a Jupyter session — starts from the
  raw `.pxt` cut, converts via the override `convert_pxt` (the only on-disk
  artifact), keeps EF/offset/`k_convert` and the figures in memory (figures
  are inline `Figure` objects, no image files), asserts a second NetCDF is
  written only when `kd.save(...)` is explicitly requested, and keeps
  `experiment_metadata.json` consumable by the override `load_metadata`.
- Live-kernel (no browser) acceptance (`tests/integration/
  test_kernel_inline_cut_figure.py`): boots a real IPython kernel and runs the
  same raw-PXT -> convert -> EF/offset -> `k_convert` -> figure session inside
  it with the inline backend, asserting an `image/png` `display_data` arrives
  (the skill's "figure really rendered" criterion) and that nothing but the
  final `.nc` is written. Enabled with `PEAKSMCP_LIVE_KERNEL=1`; skipped in
  the default suite so no kernel is ever booted there.
- Non-e2e suite: **405 passed locally** (401 + 4 real-data tests; CI without
  the data folder skips those four), 9 deselected ·
  `ruff check peaksMCP tests tools`: all checks passed.

## 0.1.0

- Initial kernel-hosted MCP server, dynamic Peaks API discovery, notebook tools, PXT conversion,
  batch plotting, CPU budgeting, dashboard, Claude Desktop plugin and test helper.

