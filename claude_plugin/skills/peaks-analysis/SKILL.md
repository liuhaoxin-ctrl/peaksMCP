---
name: peaks-analysis
description: Use for ARPES loading, cuts, fitting, momentum conversion, batch processing, and publication-quality plotting with a live peaksMCP Jupyter notebook.
version: 0.1.0
---

# Peaks ARPES analysis

Keep the notebook as the visible record of analysis. Inspect variables before transforming them.
For every unfamiliar Peaks operation, call `peaks_search_api` with the user's intent and then
`peaks_get_api` for the selected canonical ID before writing code. Prefer deterministic functions
from `peaksMCP.workflows` and `peaksMCP.plotting.plot_batch` for repeated work.

Ask the user through `askuserquestion` when a physical input such as Fermi level, polarization
geometry, temperature or angle convention cannot be derived from notebook variables or experiment
metadata. Never invent missing experimental values. Ordinary angle-to-momentum conversion
(`k_convert`) does NOT need photon energy — photon energy (hv) is only required when converting
photon-energy (hv) scans to out-of-plane momentum (kz, `return_kz_scan_in_hv=True`).

Use Matplotlib inline output. Never save figures to disk (`plt.savefig` / `fig.savefig`)
unless the user explicitly asks for a saved file — saving triggers an extra consent prompt.
Verify axis names, units, color normalization, labels, panel order and
DPI before presenting a figure as publication-ready. Read `cut-preprocessing.md` when the task
involves cut preprocessing, background subtraction, normalization or momentum conversion.
