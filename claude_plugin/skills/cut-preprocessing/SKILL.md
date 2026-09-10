---
name: cut-preprocessing
description: Use peaksMCP for ARPES cut preprocessing: classify an experiment, fit one gold reference, apply Fermi and angular corrections, convert every cut to momentum space, validate, and persist approved outputs.
---

# Cut Preprocessing

The server instructions and live tool contracts own the generic five-tool protocol. This skill
contains only the cut-specific scientific workflow and completion checks.

## Entry

1. Resolve and fetch the contracts for `load_data` and `inspect_experiment`, then run them in that
   order. Use `ExperimentSummary.gold`, `.cuts`, `.mappings`, `.records`, and `.conflicts` as the
   classification authority; do not manually parse the datasheet when the summary is available.
2. Build the full cut inventory before processing. Bind records by the indices returned from the
   summary, not by guessed filenames or scan order.
3. Resolve every native API needed for the batch before the first expensive operation. The usual
   chain is `fit_gold`, `metadata.set_EF_correction`, and `k_convert`; trust the fetched signatures
   over examples in this file.

## Scientific Invariants

- Select the gold record from `summary.gold`. Fit it exactly once during the task and reuse the
  resulting `EF_correction` for every cut. Never hard-code a correction or refit per scan.
- When the fetched gold-fit contract offers polynomial order and outlier handling, use the requested
  fourth-order fit with outlier exclusion. Keep the fit result and correction in named variables so
  they can be inspected.
- Read `theta_offset_deg` from the `ExperimentSummary.records` entry for the cut being processed.
  If the required offset is absent from both data and metadata, ask the user; never infer or guess it.
- For each cut: apply the shared Fermi correction, shift `theta_par` by that record's metadata offset,
  then convert the complete cut with `k_convert`. Do not report a slice or preview as the final result.
- Do not process gold, mappings, or unsupported records as cuts. A conflict is diagnostic, not an
  automatic exclusion: follow the resolved `summary.cuts` / record kind, retain records that the
  classifier keeps, and report the metadata-versus-shape disagreement.

## Validation And Persistence

1. Inspect every processed variable. A completed cut must have momentum-space coordinates and an
   energy axis aligned to the Fermi level; dimensions, coordinate ranges, and NaN fraction must be
   scientifically plausible.
2. Render at least one raw-versus-processed comparison with `plot_validation_pair`. Use `plot_batch`
   for multi-cut review. A render marker proves only that an image exists; inspect the values and
   coordinates before accepting its content.
3. Persist each final cut from its live variable with `save_with_consent`, one expected NetCDF path
   per call. Count only receipts whose status is `saved`. Persist a figure only when requested, using
   its figure variable through the same tool.
4. Append a final Markdown cell through `run_cell(cell_type="markdown")`. Record the gold evidence,
   Fermi correction, per-record angular offset source, processed count, output filenames, failures,
   validation result, and save-receipt outcome.
