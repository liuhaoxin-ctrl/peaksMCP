# ARPES workflows

## Load and inspect

Translate the experiment datasheet, load the PXT or NetCDF data, inspect xarray structure and only
then choose processing functions.

## Cuts and fitting

Preserve the raw variable. Create named intermediate variables for ROI selection, normalization,
EDC/MDC extraction and fitting. Include the chosen range and model in the notebook cell.

## Momentum conversion

Require valid energy calibration, angle coordinates with degree units and all physical metadata
reported by the selected Peaks API. If any field is missing, ask rather than guess.

## Publication plotting

Use `plot_batch` for comparable panels. Share units and colorbars only when quantities and
normalization agree. Include sample/condition titles, physical axis labels and a documented color
scale.

