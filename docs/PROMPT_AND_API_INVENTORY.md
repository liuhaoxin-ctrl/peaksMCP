# peaksMCP Prompt And API Inventory

Current as of 2026-09-10.

This document records prompt ownership, not a frozen copy of every prompt. The executable sources
below are authoritative; generated trial prompts and historical campaign artifacts are immutable.

## Model-Facing Layers

| Layer | Owner | Purpose |
|---|---|---|
| Server instructions and refusal text | `peaksMCP/config/prompts.yaml` | Global operating, safety, recovery, and persistence rules |
| Tool titles and descriptions | `peaksMCP/config/metadata_baseline.yaml` | Exact five-tool schemas and per-tool semantics |
| Curated project API contracts | `peaksMCP/config/override_manifest.yaml` | Six black-box peaksMCP adapters |
| Native API presentation | `peaksMCP/config/native_catalog.yaml` | Search aliases and usage notes for upstream Peaks APIs |
| Domain workflow skill | `claude_plugin/skills/cut-preprocessing/SKILL.md` | Cut-specific scientific invariants and completion checks |
| Benchmark conditions | `benchmark/prompts/` | Immutable P1/P2 experiment prompts rendered per trial |

Human-facing JupyterLab consent copy lives in `peaksMCP/extensions/jupyterlab/src/index.ts`. It is UI
text, not a model prompt, and must remain consistent with save-card and execution-consent behavior.

## Ownership Rules

1. `prompts.yaml` is global and task-agnostic. It explains how to discover, execute, inspect,
   validate, recover, and persist; it does not encode a particular beamtime or answer key.
2. `metadata_baseline.yaml` owns tool mechanics. Do not repeat full schemas in server instructions or
   skills. The model surface is exactly `search`, `get`, `inspect_notebook`, `run_cell`, and
   `save_with_consent`.
3. `override_manifest.yaml` owns the complete public contract for each peaksMCP adapter. The six
   adapters are `load_data`, `convert_experiment`, `inspect_experiment`, `plot_batch`,
   `plot_validation_pair`, and `show_mapping_slice`.
4. `native_catalog.yaml` describes upstream Peaks APIs but does not reimplement them. The installed
   `peaks` package remains an external dependency.
5. Skills contain domain-specific sequencing and scientific invariants only. Generic API proof,
   append-only, timeout, and save policy remains in the runtime prompt and tool contracts.
6. Benchmark P1 and P2 share the same scientific task body. P1 adds no tool map; P2 adds a semantic
   map and execution checkpoints. Never hand-edit a rendered `workspace/prompt.txt`.

## Current Runtime Contract

- Start experiment workflows with `load_data`, then `inspect_experiment`; use the structured summary
  instead of manually parsing or dumping metadata files.
- Resolve APIs through `search` then `get`. Exact-name Peaks calls in code cells require canonical
  proof with declared `api_ids`; scope matching is enforced where the call site has a scope of its
  own (a live DataArray/Dataset/accessor receiver accepts only a proof in that scope). Bare calls and
  receivers whose type cannot be inferred accept any proven id for the name, because `search` exposes
  one row per name, so the index's module-scope twin of a DataArray method cannot be discovered.
- `run_cell(cell_type="code")` appends and executes an API-checked code cell.
  `run_cell(cell_type="markdown")` appends a non-executed narrative cell. Both are append-only.
- Inspect live variables, bounded notebook history, and kernel state through `inspect_notebook`.
- Direct file writes in code cells are blocked. Save live variables and figures through
  `save_with_consent`; PXT conversion persists through `convert_experiment`.
- A timeout is not cancellation, and a requested save is not success. Check kernel state before a
  retry and require receipt status `saved` before reporting a product as persisted.

## Prompt Regression Gates

Tests must fail when any of these drift:

- the model tool list is not exactly five;
- runtime instructions omit `search -> get -> run_cell`, `load_data -> inspect_experiment`, timeout
  recovery, Markdown append support, or consent-gated persistence;
- runtime or benchmark prompts mention retired model tools;
- P1 accidentally receives the P2 tool map;
- P1 and P2 no longer share the same common task body;
- the cut skill teaches direct writers, guessed offsets, repeated gold fitting, or obsolete facades.

Run the relevant gates with:

```bash
/opt/homebrew/Caskroom/miniforge/base/envs/peaks/bin/python -m pytest \
  tests/unit/test_prompts_config.py \
  tests/unit/test_benchmark_experiment.py \
  tests/integration/test_mcp_protocol.py -q
/opt/homebrew/Caskroom/miniforge/base/envs/peaks/bin/python -m ruff check \
  peaksMCP tests tools benchmark
```

Historical prompt counts, old tool names, removed resource APIs, deleted workflow facades, and source
commit snapshots are intentionally not duplicated here. Git history and frozen campaign manifests
are the audit record for those versions.
