# peaksMCP Prompt & API Injection Inventory

- **Snapshot**: 2026-09-08 · committed on top of `main` @ `a58da6f` (covers the v2 rewrite of the curated API-presentation document **and** the move of L3 runtime prompt text into `config/prompts.yaml`) · all counts below verified by actually running `build_index()` / `load_api_overrides()` / `load_project_added()` / `prompts()` plus `ruff` and `pytest`.
- **Totals**: 15 MCP tool descriptions · 7 plot-resource templates · **44 curated runtime prompt strings in `config/prompts.yaml`** (1 interactive note + 6 `mcp_list_resources` guidance strings + 5 `notebook_unsafe` hard-block replies + 30 code-scanner + 2 IPython-scanner issue templates) · 2 skill/command files · 50 curated API entries (262 alias terms — 145 pure-ASCII, 117 containing CJK — and 5 `docstring_note`s) · a few frontend UI strings.
- **One-line conclusion**: every curated prompt the model sees now lives in YAML — L1/L2 in `config/metadata_baseline.yaml`, **L3 in `config/prompts.yaml`** (new; wording no longer requires editing Python), L5 in `discovery/api_overrides.yaml`. Only mechanisms (docstring-note prepend, search ranking, audit keys) and small operational strings remain in code.
- This English edition **replaces the earlier Chinese-language edition** of the same date. Figures in that edition described the pre-rewrite v1 state (e.g. “13 of 18 project functions have no aliases”, a 91-line YAML) and are now stale: the current tree consolidated `api_overrides.yaml` into one entry per API, gave every project function aliases, and moved the L3 copy out of Python.

---

# Part I · Prompt surfaces (what the model sees)

| Layer | Content | Delivery | Curated in |
|---|---|---|---|
| L1 | 15 MCP tool descriptions | every tool-list broadcast | `metadata_baseline.yaml` |
| L2 | 7 plotting-format templates | each `mcp_list_resources` call | `metadata_baseline.yaml` |
| L3 | 44 runtime strings (guidance, block replies, scanner issue text) | inline payloads / errors / consent | **`prompts.yaml`** (Python only formats) |
| L4 | skill + slash command | loaded on demand | Markdown files |
| L5 | per-API search aliases + docstring notes | injected into the search index / docstrings | `api_overrides.yaml` |
| L6 | frontend consent copy | JupyterLab extension UI | `index.ts` |

## L1 — MCP tool descriptions (15)

**Where**: `peaksMCP/config/metadata_baseline.yaml:2-78` · **Injected at**: `core/tools.py:129` (`mcp.tool(name=…, title=metadata["title"], description=metadata["description"])`). Full text lives in the YAML (single source); one-line intents:

| Tool | Intent |
|---|---|
| `peaks_search_api` | MUST precede any Peaks analysis code; search the live index for the user's intent and write code from the returned canonical entries. |
| `peaks_get_api` | After searching, fetch exact signature / parameters / source-backed docs for one canonical API ID before writing code. |
| `mcp_list_resources` | Call FIRST when a figure is needed; returns every plotting format with uri, when-to-use, figure contract and the full template inline. |
| `askuserquestion` | Report missing experimental information; return `{"status":"needs_input",…}` and let the caller ask. `options` must be plain strings; never invent Fermi level / angle / polarization / temperature. |
| `notebook_list_variables` | List useful kernel variables without reading array values. |
| `notebook_read_variable` | Summarize a variable (xarray-aware: dims, coords, units, attrs, chunks, Peaks accessors). |
| `notebook_read_active_cell` | Read source + identity of the currently selected cell. |
| `notebook_read_active_cell_output` | Read text + inline-image outputs of the current cell. |
| `notebook_read_content` | Read the notebook cell structure over the JupyterLab Comm bridge. |
| `notebook_move_cursor` | Select previous / next / numbered cell. |
| `notebook_server_status` | Mode, API-index and Comm-bridge readiness of this kernel MCP server. |
| `notebook_kernel_status` | Kernel busy/idle state + busy-start time. |
| `notebook_wait_for_kernel` | Wait for idle up to a bounded timeout. |
| `notebook_write_with_api_check` | Append + execute model code and verify every Peaks reference against the live index; **all model-generated Peaks analysis code must go through this tool**. |
| `notebook_add_cell` | Append a code/Markdown/raw cell at the END (append-only log). Mutation tool: always scanner-checked + audit-logged, consent-gated when `mcp.require_consent` is on. |

## L2 — Plot-resource templates (7)

**Where**: `metadata_baseline.yaml:80-252` · **Delivered**: whole payload of `mcp_list_resources` (`core/tools.py:245-289`), which embeds each template inline because some clients reject custom-scheme URIs. Templates carry behavioral comments the model is told to obey (`# Run this cell verbatim — the final plt.show() renders exactly ONE figure.`; `before_after` adds “renders inline exactly ONCE”).

| Resource | when_to_use (condensed) |
|---|---|
| `figure_conventions` | Read FIRST whenever a figure is drawn; every template below assumes it. |
| `fermi_surface` | 2D (kx, ky) slice at E_F — symmetric square axes, one shared colorbar. |
| `dispersion_single` | One 2D cut in binding energy after k-conversion; E_F must sit at 0 with a dashed guide. |
| `dispersion_grid` | Many 2D cuts to compare — shared color scale, per-panel index + polarization titles. |
| `waterfall` | Stacked 1D MDC/EDC curves — offset traces, color by position. |
| `before_after` | Validate one cut: raw (angle domain) vs k-converted, EF-corrected result, shared colorbar. |
| `kz_map` | Photon-energy scan converted to (kz, k∥) — vertical axis kz. |

**`figure_conventions` style contract** (the base for all figures): constrained_layout only · DejaVu Sans + mathtext, `unicode_minus` off, CJK fallback PingFang SC / Microsoft YaHei · English labels, symbols in mathtext · 150 dpi screen / 300 save, vector PDF for papers.

## L3 — Runtime prompt text (now YAML-curated in `config/prompts.yaml`)

This used to be the hidden hardcoded layer. After the 2026-09-08 refactor the wording lives in **`peaksMCP/config/prompts.yaml`** (44 strings), read once at import by `config/metadata.py` (`prompts()`); Python only formats the `{placeholders}`. Editing copy no longer touches Python. Rule ids, risk levels and audit `reason` keys intentionally stay in code.

### 3.1 Interactive-output note — `prompts.yaml` → `interactive_omitted_note`
Formatted at `core/tools.py:102` (`_interactive_omitted_content`, function at `:89`).
```text
Interactive panel/widget rendered in the notebook for the user; it cannot be
embedded here. Consider it done and describe the figure to the user.
```

### 3.2 `mcp_list_resources` guidance block (6 strings) — `prompts.yaml` → `list_resources_guidance`
Merged into the resources payload at `core/tools.py:272-285`. Keys: `resources_vs_tools.resources` (templates are reference data — run verbatim), `resources_vs_tools.tools` (search APIs first, write cells via `notebook_write_with_api_check`), `when_to_use_resources[0..2]` (pick a format before drawing; verbatim run; many cuts → `dispersion_grid`, one → `dispersion_single`, EF slice → `fermi_surface`), `first_use` (“Call mcp_list_resources() BEFORE plotting…”).

### 3.3 Hard-block replies (5) — `prompts.yaml` → `notebook_unsafe`
Formatted in `backend/notebook_unsafe.py` (`_PROMPTS`); these receipts steer the model's next action.

| Key | Used at | Message (verbatim) |
|---|---|---|
| `index_build_failed` | `:186` | “The Peaks API index could not be built; restart the kernel.” |
| `plot_templates_not_read` | `:193` | “Plotting code detected, but the canonical plot templates have not been read yet this session. Call mcp_list_resources() first — it returns every plotting format's template inline. After that single call this guard stays satisfied and plotting code runs freely.” |
| `savefig_forbidden` | `:199` | “This cell saves a figure to disk (savefig), which is permanently disabled: figures are rendered inline in the notebook. Remove the savefig call — there is no user-confirmation path for saving figures.” |
| `unknown_api_first` | `:327` | “Execution blocked: unverifiable API reference(s) {names}. None of these resolve to a Peaks API by exact name. Use peaks_search_api to find the correct API (candidates: {suggestions}) or fix the typo. If the receiver is complex (e.g. a function return value), assign it to an intermediate variable first. This name must be fetched with peaks_get_api before it can be used.” |
| `unknown_api_retry` | `:310` | “Execution blocked: these names were already reported as unverifiable this session ({names}) and no successful peaks_get_api has followed. Call peaks_search_api, then peaks_get_api(<canonical id>) once for each name, then resubmit the cell.” |

### 3.4 Scanner & IPython issue templates (32) — `prompts.yaml` → `scanner` + `ipython`
Shown to the model on a block and surfaced as `issue.description` in the consent dialog. Formatted via `_desc()` in `security/code_scanner.py:33` (`_SCANNER_TEXT` at `:30`) and `_IPYTHON_TEXT` in `security/ipython_scanner.py:17`.

Rule codes mapped to the wording at each call site (placeholders `{name}`, `{target}`, `{value_name}`, `{module}`):

| Rule(s) | prompts.yaml key(s) | Wording (representative) |
|---|---|---|
| `EXEC001` | `exec_dynamic` / `exec_indirect` | dynamic code execution via `{name}` / … via indirect fetch of `{target}` |
| `SYS001` | `sys_destructive` / `sys_destructive_indirect` / `path_destructive` | system or destructive operation via `{name}` / destructive path operation via `{name}` |
| `IMPORT001` | `import_dynamic` | dynamic import via `{name}` |
| `FILE001` | `file_modifying` / `file_modifying_path` / `file_modifying_indirect` | file / path opened in a modifying mode (also via indirect fetch) |
| `FILE002` | `file_mode_unclear` / `file_mode_unclear_indirect` | `open()` with a non-constant mode; read-only-ness cannot be confirmed — approve only if safe (consent-only) |
| `ENV001` | `env_mutation` / `env_mutation_aliased` | process environment modification (also through an aliased `os.environ`) |
| `SAVE001` | `savefig_disabled` | figure save to disk (savefig) is disabled; figures are rendered inline |
| `SAVE002` | `file_write_consent` / `file_write_consent_indirect` | file write via `{name}`; approve only to write to disk |
| `REF001` / `REF003` | `ref_chain_call` / `ref_getattr_dynamic` / `ref_dynamic_attr` | reflection attribute chain / dynamic attribute name cannot be verified statically |
| `SMUG001` / `SMUG002` | `smug_attribute_delete` / `smug_attr_mutation` / `smug_setattr_dangerous` / `smug_hidden_callable` | attribute mutation/deletion/smuggling of dangerous callables |
| `CAP001` / `CAP002` / `CAP003` | `cap_sandbox_via` / `cap_reflection_via` / `cap_star_import` | operation outside the analysis sandbox / reflection / star import via `{name}` / `{module}` |
| `IND002` | `ind_subscript_call` | calling a value fetched through a subscript cannot be tracked statically |
| `FILE003` | `file_raw_write` | raw-descriptor file write via `{name}` |
| `DES001` | `deserialize_unsafe` | unsafe deserialization via `{name}` |
| `NET001` | `network_consent` | network request via `{name}`; approve only to send data externally |
| `IPY001` / `IPY002` | `ipy_shell` / `ipy_env_magic` | shell escape / environment-or-extension modifying IPython magic |

### 3.5 docstring-note injection (mechanism) — `discovery/signatures.py:161-164`
Notes come from `api_overrides.yaml` (L5) and are **prepended**, not appended: `details["docstring"] = f"{note}\n\n{doc}".strip()`.

### 3.6 Search ranking (mechanism) — `discovery/index.py`
`_search` at `index.py:525-569` scores hits in tiers before falling back to token overlap: exact name 1000 → alias exact 900 → name-prefix 800 → name-substring 700 → alias-substring 650 → else the fuzzy formula at `index.py:566`:
```python
score = name_overlap*100 + alias_overlap*80 + summary_overlap*30 + docstring_overlap*20 + module_overlap*15
```
Aliases therefore outrank summary/docstring text and are the main lever for natural-language recall.

> **Deliberately left in code** (operational, not curated prompts): output-reporting lines in `tools.py` (“Inline figure rendered in the notebook …”, “No active-cell output.”), `ScanResult` syntax-error reasons and the permission/error fallbacks in `backend/notebook_unsafe.py`. The frontend consent copy (L6) is a separate JupyterLab build target and was not moved.

## L4 — Skill & slash command

- **`claude_plugin/skills/cut-preprocessing/SKILL.md`** — loaded when the user asks for cut preprocessing/leveling/k-space conversion. Key rules (current, post-fix): read `experiment_metadata.json` and pick the cut file from the datasheet; fit the gold (`Au`) reference for the Fermi level; **one gold fit, then per-cut conversion** (`fit_gold` once → `da.metadata.set_EF_correction(EF_correction)` → `da.k_convert()` per cut — no single-call shortcut); drop outliers before `poly4`; `theta_par_offset_deg` must come from the user; save via `da.save(path)` (not `peaks.save(da, path)` / `save_processed`, which do not exist); review `mcp_list_resources` templates and follow their intent; figure-debug protocol — check `notebook_read_active_cell_output` for the `inline_image_rendered` marker before changing any code (marker present → fix content; bare `<Figure>` repr → fix display).
- **`.claude/commands/test_helper.md`** — `/test_helper [connection|api|notebook|images|pxt|restart|all]`; check server/kernel state first, then exercise representative tools with small read-only inputs, announce expected consent dialogs before unsafe calls, record tool/params/result/timing/errors, and summarize passed/failed/skipped with reproducible steps.

## L5 — Curated per-API presentation (`api_overrides.yaml`)

**Where**: `peaksMCP/discovery/api_overrides.yaml` (v2, 50 entries, 350 lines) — the consolidated single source. **Consumed by**: `discovery/index.py` — `load_overrides()` / `load_api_overrides()` return per-API entries; `load_project_added()` derives the audited project set from the `project: true` flag (no second list to keep in sync); `build_index()` merges each entry's `aliases` into the item, attaches `docstring_note`, and flags `project_added=True`; `signatures.py:161-164` prepends the note to the live docstring.

The whole curated record therefore reads as **one table** (full alias text lives in the YAML; counts verified live):

### Upstream peaks entries (32) — no `module`, `project` absent

| API | EN aliases | total | API | EN aliases | total |
|---|---|---|---|---|---|
| `load` | 2 | 4 | `drop_nan_borders` | 2 | 4 |
| `save` | 2 | 4 | `drift_correction` | 2 | 4 |
| `History` | 2 | 3 | `correct_isolated_bad_pixels` | 2 | 4 |
| `EDC` | 1 | 3 | `sum_data` | 2 | 4 |
| `MDC` | 1 | 3 | `subtract_data` | 2 | 4 |
| `DOS` | 1 | 3 | `merge_data` | 3 | 5 |
| `tot` | 3 | 5 | `estimate_EF` | 2 | 3 |
| `extract_cut` | 3 | 6 | `estimate_sym_point` | 2 | 4 |
| `radial_cuts` | 2 | 4 | `fit_gold` ★ | 4 | 9 |
| `deriv` | 2 | 5 | `k_convert` ★ | 2 | 4 |
| `curvature` | 1 | 2 | `disp_from_hv` | 2 | 3 |
| `smooth` | 2 | 4 | `plot_grid` | 3 | 6 |
| `norm` | 2 | 4 | `disp` ★ | 8 | 13 |
| `bgs` | 2 | 3 | `bin_data` | 2 | 4 |
| `sym` | 2 | 4 | `rotate` | 2 | 4 |
| `degrid` | 2 | 4 | `mask_data` | 2 | 4 |

★ carries a `docstring_note` (see below).

### Project-added entries (18) — `module` + `project: true`

| # | module | API | EN aliases | total | note |
|---|---|---|---|---|---|
| 1 | `batch.resource_budget` | `batch_execution_lock` | 4 | 7 | — |
| 2 | `plotting.layout` | `plot_batch` ★ | 4 | 8 | ✅ |
| 3 | `plotting.validation` | `plot_validation_pair` ★ | 6 | 9 | ✅ |
| 4 | `pxt_utils.converter` | `convert_path` | 4 | 7 | — |
| 5 | `pxt_utils.converter` | `convert_pxt` | 5 | 8 | — |
| 6 | `pxt_utils.converter` | `default_output_dir` | 3 | 5 | — |
| 7 | `pxt_utils.converter` | `index_from_path` | 4 | 6 | — |
| 8 | `pxt_utils.csv_translator` | `translate_datasheet` | 4 | 6 | — |
| 9 | `pxt_utils.loader` | `load_pxt` | 5 | 8 | — |
| 10 | `pxt_utils.loader` | `register_l112_loader` | 3 | 5 | — |
| 11 | `pxt_utils.metadata` | `load_metadata` | 4 | 7 | — |
| 12 | `pxt_utils.metadata` | `read_meta` | 6 | 10 | — |
| 13 | `pxt_utils.metadata` | `classify_data_format` | 4 | 6 | — |
| 14 | `pxt_utils.metadata` | `is_gold_format` | 3 | 5 | — |
| 15 | `pxt_utils.metadata` | `theta_offset_deg` | 3 | 5 | — |
| 16 | `workflows.publication` | `publication_grid` | 3 | 5 | — |
| 17 | `workflows.publication` | `validate_arpes_metadata` | 3 | 5 | — |
| 18 | `workflows.slice_view` | `show_mapping_slice` | 5 | 10 | — |

**Coverage guarantees** (all verified live and enforced by tests):
- Every one of the 50 documented APIs has search aliases — no documented API is reachable only by its exact name. Every **project** API has ≥ 3 pure-ASCII (English) aliases.
- No orphan records: `test_alias_and_override_keys_resolve_to_real_apis` asserts every key in the YAML resolves to a real indexed API.
- Project exposure cannot silently drift: `test_project_added_declaration_matches_the_live_index` is bidirectional (nothing declared-but-gone, nothing exposed-but-undeclared) and `test_project_added_entries_are_flagged_in_the_index` asserts flagging equals exposure; `test_every_documented_api_has_search_aliases` blocks alias-less entries.

**The 5 `docstring_note`s** (prepended to the live docstring; the API contract the model is told to follow):
- `k_convert` — converts a cut to (eV, kx) and flattens the Fermi edge in the same call; input raw (kinetic, theta_par) + `EF_correction` from `fit_gold` passed as `k_convert(EF_correction=…)`; hierarchy `fit_gold` once → `k_convert` per cut; `return_kz_scan_in_hv` applies to hv (kz) scans.
- `fit_gold` — fits the Fermi edge of a gold reference; output `EF_correction` (poly dict), `EF_quality`, inline figure; apply the correction to each cut via `metadata.set_EF_correction` before `k_convert`.
- `plot_batch` — inline grid of matplotlib figures for a list of DataArrays.
- `plot_validation_pair` — inline before/after figure for one cut: raw angle-space vs processed k-space.
- `disp` — peaks' native Qt interactive viewer; use `data.disp()` or `disp([da,…], primary_dim=…)`, import from `peaks.core.GUI.disp_panels`.

## L6 — Frontend consent copy (UI, not model-facing)

**Where**: `peaksMCP/extensions/jupyterlab/src/index.ts` — Chinese labels (`请求的操作: …` :81, delete/overwrite dialog text :94-95, `需要您确认的操作` fallback :146, `拒绝` / `允许` :155-156, plus one English code comment :218). Consent dialogs must stay in sync with the model-facing wording if the scanner issues ever change. This is the only remaining copy layer not yet YAML-driven (it ships inside the JupyterLab build).

---

# Part II · Extra APIs injected into the index

## 2.1 Index composition (measured)

`build_index()` produces **251 entries = 233 upstream `peaks` + 18 peaksMCP project functions**; all 18 are flagged `project_added`. Module split of the project block: `pxt_utils.metadata` 5 · `pxt_utils.converter` 4 · `pxt_utils.loader` 2 · `workflows.publication` 2 · one each in `batch.resource_budget`, `plotting.layout`, `plotting.validation`, `pxt_utils.csv_translator`, `workflows.slice_view`.

Mechanism to remember: project functions are **auto-discovered by the AST scan**, not hand-registered. Deleting one removes it from the index (and now fails the bidirectional test if the YAML still declares it); adding a new public function adds it to the index (and now fails the same test until it is declared). “Silently widening the exposure surface” became “CI failure” — but declaration alone still does not limit what is exposed.

## 2.2 `peaks` library edits made outside this repo

`api_overrides.yaml` (tail note) records that in-repo additions to the `peaks` package itself (e.g. two-pass gold-fit helpers) are private/underscore and live outside the discovery file. Verified against the local editable checkout `/Users/haoxin/peaks_dev` (no git remote; latest commit `ca965e8`, 2026-09-08):

| Commit | Change (affects the docstring the model sees) |
|---|---|
| `ca965e8` | `k_convert`: accept `EF_correction` and flatten in the same call |
| `d8a80d8` | `k_convert`: refuse uncalibrated (EF_correction missing) cuts |
| `12d2598` | `fit_gold`: two-pass semantics; delete progress; drop the need-review alarm |
| `e27c9c6` | fit: dynamic EF bounds strictly symmetric around a robust I50 estimate |
| `e94f5c5` | fit: robust I50 estimate as the Fermi-level seed (drop `estimate_EF`) |
| `bbd442d` | `k_convert`: remove the photon-energy dependency (Fermi level only) |
| `2504156` | metadata: split `EF_correction` and `V0` out of the calibration container |
| `c55c419` | remove beamline-specific loaders and their base classes |

New private helpers (underscore, not indexed): `_fit_gold_2d_from_center`, `_fit_gold_2d_seeded_from`, `_gold_fit_diagnostic_warnings`, `_plot_gold_2d_diagnostics` in `peaks/core/fitting/fit.py`. **Drift guard**: because the index reads signatures at runtime, `test_ef_correction_handoff_notes_match_real_signatures` fails if `k_convert` loses `EF_correction=` or `fit_gold` loses `EF_correction_type` — the hand-written notes then misrepresent the real API.

---

# Part III · Findings & fix status

| # | Severity | Finding | Where | Status |
|---|---|---|---|---|
| 1 | high | `preprocess_cut` referenced by SKILL.md but nonexistent | `SKILL.md` | ✅ fixed (`f72b45d` + rewrite) |
| 2 | high | `save_processed` nonexistent; `peaks.save(da, path)` wrong (save/fit_gold/k_convert/set_EF_correction are xarray accessors) | `SKILL.md` | ✅ fixed (`da.save(path)`) |
| 3 | medium | 13/18 project functions had no aliases → `convert_pxt` / `load_pxt` / `load_metadata` effectively unsearchable | `api_overrides.yaml` | ✅ fixed in the v2 rewrite (all 18 now carry aliases; ≥ 3 English each) |
| 4 | medium | `project_added` list was a dead record nothing consumed → could not audit the real exposure | `api_overrides.yaml` / `index.py` | ✅ fixed (`project: true` flag drives `load_project_added()`; bidirectional tests) |
| 5 | medium | five templates' second-line comments had 6 stray leading spaces (misaligned when delivered) | `metadata_baseline.yaml` | ✅ fixed |
| 6 | low | hand-written `docstring_note`s had no automatic drift protection vs the `peaks` side | `api_overrides.yaml` | ✅ fixed (signature-contract test) |
| 7 | medium | L3 runtime prompt text (guidance, block replies, scanner issue copy) was hardcoded across four Python files — the layer most likely to drift silently | `tools.py`, `notebook_unsafe.py`, `code_scanner.py`, `ipython_scanner.py` | ✅ fixed (moved to `config/prompts.yaml`; code only formats placeholders) |

**Gate (clean interpreter, current tree)**:
```
ruff check peaksMCP tests tools        → All checks passed
pytest -m 'not e2e'                    → 391 passed, 9 deselected
```
(391 includes the five discovery regression tests from the v2 rewrite and seven new prompts-config tests in `tests/unit/test_prompts_config.py`; discovery-only run: 14 passed, security+prompts run: 163 passed.)

**Fix log — 2026-09-08** (see also the repo memory file): #1/#2 removed dangling SKILL references and corrected the accessor wording after confirming `hasattr(peaks, "save") is False`; #3 closed the alias gap by consolidating the curated record to one entry per API, so all 18 project functions became searchable; #4 made the exposure audit bidirectional via the `project: true` flag driving `load_project_added()`; #5 re-indented the block-scalar comments 12→6 spaces; #6 added the EF-handoff signature-contract test; **#7 moved the L3 copy into `config/prompts.yaml`** — `prompts()` in `config/metadata.py` loads it, `tools.py`/`notebook_unsafe.py`/`code_scanner.py`/`ipython_scanner.py` read it once at import and only format `{placeholders}`. Wording was preserved byte-for-byte (regression tests in `test_prompts_config.py` tie scanner output back to the YAML templates); rule ids, risk levels and audit `reason` keys stayed in code. Negative checks were run (e.g. injecting a ghost project entry turns the tests red) so the assertions are not vacuous.

---

## Open recommendations

1. **Remaining copy in code is small and intentional** — the output-reporting lines in `tools.py`, scanner syntax-error reasons and permission fallbacks are operational text, not curated prompts. If you want *everything* data-driven, they can move to `prompts.yaml` too.
2. **L6 frontend copy (`index.ts`) is the last prompt layer not in YAML** — it is Chinese UI text that ships inside the JupyterLab extension build; consolidating it would need a frontend-side strings module or JSON, not the config YAML.
3. **This file will go stale with every prompt change** — treat it as a living snapshot: refresh the header counts and anchors whenever L3/L5 YAML text changes, and re-run the gates before claiming any number here.
4. **`api_overrides.yaml` is now the only hand-maintained exposure record** — it still does not *limit* exposure (AST auto-discovery does), so keep the bidirectional test as the enforcement point.
