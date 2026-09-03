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
2. Prioritize using the public `peaks` APIs (which contain complete docstrings); do not reimplement internal `peaks` functions.
3. When performing 4th-order polynomial fitting (`poly4`) on gold data, outliers must be excluded prior to fitting.
4. **Use preprocess_cut for fast process**

## Required Parameters

* `theta_par_offset_deg` (High-symmetry point position): Provided either in the agent note inside `experiment_metadata.json` or explicitly specified by the user via the `askuserquestion` tool. This parameter cannot be deduced from the data.
* **Fermi Energy**: Priority sequence is Gold data fitting (`fit_gold`, files labeled `Au` in the datasheet) $>$ Ask the user via the `askuserquestion` tool.
* Other metadata (such as polarization, photon energy, etc.) is used strictly for judgment/verification and does not participate in calculations.

## Deliverables & Output Protocol

Use `plot` to convey **key** information to the user.
Save processed cuts with `peaks.save(da, path)` (or `save_processed`), not raw `da.to_netcdf` — it fails on unsanitized metadata attrs.

**Carefully review the template code returned by mcp_list_resources and follow its intent.**
