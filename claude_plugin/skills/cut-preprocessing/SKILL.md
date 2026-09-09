---
name: cut-preprocessing
description: Use `peaksMCP` to call the `peaks` package and perform preprocessing on cut (sweep/fix) data in a notebook (Fermi surface leveling & zeroing $\rightarrow$ high-symmetry point zeroing $\rightarrow$ k-space conversion). Trigger this workflow when the user requests cut data processing, preprocessing, leveling, or k-space conversion.
---

# Cut Data Preprocessing

## Data Acquisition

**Read metadata to extract input information**: There is an experiment record table `experiment_metadata.json` adjacent to the data directory. Extract the following:
* The input file (`BP_XXXX`) and type for each experiment (Index).
* Select the cut (`sweep`/`cut`) file to be processed based on the datasheet; use the gold data (labeled `Au`) for Fermi energy fitting.

## Workflow

1. Use the fitted gold curve to level the Fermi edge of the $E_k\text{--}k$ data and set the Fermi energy to zero $\rightarrow$ set the high-symmetry point in angle space to zero $\rightarrow$ convert to k-space.
2. Compose the curated peaksMCP facades (`peaksMCP.overrides`) found via
   `peaks_search_api` / `peaks_get_api`; fall back to native `peaks` only when no
   facade fits. Never reimplement an existing function.
3. When performing 4th-order polynomial fitting (`poly4`) on gold data, outliers must be excluded prior to fitting (`outlier_exclusion=True`).
4. **One gold fit, then per-scan conversion**: run `fit_gold_reference` once on
   the Au reference; then preprocess each scan with
   `preprocess_cut(cut, calibration=cal, theta_par_offset_deg=...)` — the facade
   applies the EF correction and the theta offset and converts to k-space in one
   call. There is no shortcut that skips the calibration: never re-fit per cut.
5. **3-D data must go through `preprocess_mapping`** with the full cube and its
   normal-emission reference angles — never extract a centre slice and report it
   as a complete conversion (a sliced result is only a partial preview).

## Required Parameters

* `theta_par_offset_deg` (High-symmetry point position): Provided either in the agent note inside `experiment_metadata.json` or explicitly specified by the user via the `askuserquestion` tool. This parameter cannot be deduced from the data.
* **Fermi Energy**: Priority sequence is Gold data fitting (`fit_gold_reference`, files labeled `Au` in the datasheet) $>$ Ask the user via the `askuserquestion` tool. Inspect the metadata document first with `inspect_experiment` — it reports classification conflicts (e.g. 3-D cubes labelled `sweep`) before any preprocessing starts.
* Other metadata (such as polarization, photon energy, etc.) is used strictly for judgment/verification and does not participate in calculations.

## Deliverables & Output Protocol

Use `plot` to convey **key** information to the user.
Save processed cuts with `da.save(path)` (peaks' own writer sanitises metadata
attrs; raw `da.to_netcdf` can fail on unsanitized attrs). For results the user
explicitly asks to keep, use the `save_result` facade: it stages the result,
shows the user a consent card with the real content summary (path, size,
sha256, structure) and writes the file only when the user approves on that
card. There is no code-level approval flag.

**Use the peaksMCP plotting façades (`plot_batch`, `plot_validation_pair`, `show_mapping_slice`) for figures; when raw matplotlib is unavoidable, follow the figure conventions in the server instructions (constrained_layout, DejaVu Sans with mathtext symbols, English labels, 150/300 dpi).**

## Figure debugging protocol

When a figure output is questioned, diagnose **whether an image rendered** before changing any code:

1. Call `notebook_read_active_cell_output` and look for the `inline_image_rendered` marker (or `image/png` in the mime list).
2. **Marker present** → the figure WAS rendered. The problem is its *content*: inspect the data/values used.
3. **Bare `<Figure ...>` repr with no image marker** → the figure was NOT displayed. Fix the display, not the fit.
