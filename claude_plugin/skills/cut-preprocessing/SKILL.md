---
name: cut-preprocessing
description: Use peaksMCP for ARPES cut preprocessing: cache raw PXT, classify an experiment, fit one gold reference, apply Fermi and angular corrections, convert every cut to momentum space, and validate all results inline.
---

# Cut Preprocessing

The server instructions and live tool contracts own the generic five-tool protocol. This skill
contains only the cut-specific scientific workflow and completion checks.

## Entry

1. For raw PXT input, resolve and fetch `peaks.pxt2nc`, create the conversion cache, then repeat the
   identical call and confirm that valid entries report cache hits. Preserve both reports in named
   variables. This cache and its metadata sidecar are the only automatic disk artifacts.
2. Resolve and fetch the combined `peaks.load_experiment` contract, then use its returned
   `ExperimentIndex.gold`, `.cuts`, `.mappings`, `.records`, `.conflicts`, `.unsupported`, and
   `.metadata_provenance` as the classification authority. Do not manually parse the datasheet.
3. Build the full cut inventory before processing. Bind records by the indices returned from the
   experiment index, not by guessed filenames or scan order.
4. Resolve every native API needed for the batch before the first expensive operation. The usual
   chain is `fit_gold`, `metadata.assign_normal_emission`, and `k_convert`; pass the fitted result
   through the fetched `EF_correction` parameter. Trust fetched signatures over examples here.

## Scientific Invariants

- Select the gold record from `experiment.gold`. Fit it exactly once during the task and reuse the
  resulting `EF_correction` for every cut. Never hard-code a correction or refit per scan.
- When the fetched gold-fit contract offers polynomial order and outlier handling, use the requested
  fourth-order fit with outlier exclusion. Keep the fit result and correction in named variables so
  they can be inspected.
- Read `theta_offset_deg` from the corresponding `ExperimentIndex.records` entry.
  If the required offset is absent from both data and metadata, ask the user; never infer or guess it.
- For each cut: create the angular-zeroed copy with `metadata.assign_normal_emission`, then convert
  the complete cut with `k_convert(EF_correction=fit_result)`. Do not report a slice or preview as
  the final result.
- End the single batch with exactly one live-key receipt using `print("processed_stems=" +
  ",".join(sorted(result_dict)))`, substituting the actual result dictionary. Keep total batch
  stdout within three lines of at most 200 characters each. This returned list, not the count, is
  the authority for the closing summary; never infer consecutive scan numbers.
- Do not process gold, mappings, or unsupported records as cuts. A conflict is diagnostic, not an
  automatic exclusion: follow the resolved `experiment.cuts` / record kind, retain records that the
  classifier keeps, and report the metadata-versus-shape disagreement.

## Validation And Completion

1. Inspect every processed variable. A completed cut must have momentum-space coordinates and an
   energy axis aligned to the Fermi level; dimensions, coordinate ranges, and NaN fraction must be
   scientifically plausible.
2. Render every final cut and the necessary raw-versus-processed comparisons inline with existing
   Peaks plotting APIs or small Matplotlib/xarray glue. Render binding-energy slices for mappings.
   A render marker proves only that an image exists; inspect values and coordinates before accepting
   its content.
3. Keep final cuts, mapping views, fit results, and validation figures in clearly named live
   variables. Do not call `save_with_consent` for this workflow, write `*_processed.nc`, call
   `savefig`, or otherwise persist analysis results or images.
4. Append a final Markdown cell through `run_cell(cell_type="markdown")`. Record the gold evidence,
   Fermi correction, processed count, variable/cell references, conversion/cache-reuse counts,
   failures, and validation result. Identify the selected gold as `gold=SELECTED_STEM (index
   SELECTED_INDEX)`, replacing both placeholders with observed values; "gold scan fitted once"
   alone is incomplete. Copy the complete returned `processed_stems=...` token verbatim. State each observed numeric angular correction together with its
   exact source field, for example `theta_offset=VALUE deg from record.theta_offset_deg`, replacing
   VALUE with the observed number; the field name alone is not a complete result.
