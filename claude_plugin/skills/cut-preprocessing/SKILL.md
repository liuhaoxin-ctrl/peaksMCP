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
2. Compose the adapter surface (`peaksMCP.overrides`: `load_data`,
   `inspect_experiment`, `convert_experiment`) with native
   `peaks` steps obtained via `peaks_search_api` / `peaks_get_api`
   (`da.fit_gold`, `da.metadata.set_EF_correction`, coordinate shifts,
   `da.k_convert`). Never reimplement an existing function.
3. When performing 4th-order polynomial fitting (`poly4`) on gold data, outliers must be excluded prior to fitting (`outlier_exclusion=True`).
4. **One gold fit, then per-scan conversion**: run the native
   `da.fit_gold` once on the Au reference, apply its `EF_correction` with
   `da.metadata.set_EF_correction(...)`, shift the high-symmetry angle
   (`da.assign_coords(theta_par=da.theta_par - theta_par_offset_deg)`) and call
   `da.k_convert()` per scan — compose these in the notebook; there is no
   shortcut that skips the calibration: never re-fit per cut.
5. **3-D cubes**: convert the FULL cube (never extract a centre slice and
   report it as a complete conversion — a sliced result is only a partial
   preview); set the normal-emission reference angles on the metadata before
   `k_convert` when the geometry requires it.

## Required Parameters

* `theta_par_offset_deg` (High-symmetry point position): Provided either in the agent note inside `experiment_metadata.json` or explicitly specified by the user via the `askuserquestion` tool. This parameter cannot be deduced from the data.
* **Fermi Energy**: Priority sequence is Gold data fitting (`fit_gold_reference`, files labeled `Au` in the datasheet) $>$ Ask the user via the `askuserquestion` tool. Inspect the metadata document first with `inspect_experiment` — it reports classification conflicts (e.g. 3-D cubes labelled `sweep`) before any preprocessing starts.
* Other metadata (such as polarization, photon energy, etc.) is used strictly for judgment/verification and does not participate in calculations.

## Deliverables & Output Protocol

Use `plot` to convey **key** information to the user.
Save processed cuts with `da.save(path)` (peaks' own writer sanitises metadata
attrs; raw `da.to_netcdf` can fail on unsanitized attrs). For results the user
explicitly asks to keep, call the `save_with_consent` MCP tool: it appends a
preview record cell, stages the variable's exact bytes in a unified temp area,
shows the user a consent card with the real content summary (path, size,
sha256, structure) and writes the file only when the user approves on that
card. There is no code-level approval flag and no save function in the model
Python surface.

**Use the peaksMCP plotting façades (`plot_batch`, `plot_validation_pair`, `show_mapping_slice`) for figures; when raw matplotlib is unavoidable, follow the figure conventions in the server instructions (constrained_layout, DejaVu Sans with mathtext symbols, English labels, 150/300 dpi).**

## Figure debugging protocol

When a figure output is questioned, diagnose **whether an image rendered** before changing any code:

1. Call `notebook_read_active_cell_output` and look for the `inline_image_rendered` marker (or `image/png` in the mime list).
2. **Marker present** → the figure WAS rendered. The problem is its *content*: inspect the data/values used.
3. **Bare `<Figure ...>` repr with no image marker** → the figure was NOT displayed. Fix the display, not the fit.
