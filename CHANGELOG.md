# Changelog

## [Unreleased] — 2026-09-08

Curated-presentation & prompt-governance pass (`a58da6f` → `e97729e`).

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
- New `config/prompts.yaml` (45 strings): always-on FastMCP server
  instructions, interactive-output note, `mcp_list_resources` guidance (6),
  `notebook_unsafe` hard-block replies (5), code-scanner (30) + IPython (2)
  issue templates.
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
  non-goals and fix log #1–#8).
- `AGENTS.md` documents `config/prompts.yaml` as the runtime-prompt single
  source.

### Tests & gates
- New regression tests: bidirectional project-exposure audit, alias/orphan
  guards, EF-handoff signature contract, prompts-config rendering incl. the
  server-instructions guard.
- Non-e2e suite: **392 passed, 9 deselected** · `ruff check peaksMCP tests
  tools`: all checks passed.

## 0.1.0

- Initial kernel-hosted MCP server, dynamic Peaks API discovery, notebook tools, PXT conversion,
  batch plotting, CPU budgeting, dashboard, Claude Desktop plugin and test helper.

