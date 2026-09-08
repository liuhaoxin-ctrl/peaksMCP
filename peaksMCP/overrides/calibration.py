"""Gold-reference calibration facade.

``fit_gold_reference`` fits the Fermi edge of a gold scan through peaks'
own fitting machinery (``da.fit_gold``) exactly once and returns a JSON-safe
:class:`GoldCalibration` that other facades accept — the fit is never
re-run per cut.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr
from pydantic import BaseModel, Field


class GoldCalibration(BaseModel):
    """JSON-safe outcome of one gold-reference fit.

    Carries the correction to apply (float or ``c0..cN`` dict), the fit
    quality summary and a source summary.  Deliberately holds no DataArray
    and no Figure: the rendered diagnostic figure stays in the notebook.
    """

    operation: str = "fit_gold_reference"
    status: str = "ok"
    warnings: list[str] = Field(default_factory=list)
    correction: Any = None
    correction_type: str = "poly4"
    ef_at_normal_emission_eV: float | None = None
    quality: dict[str, Any] = Field(default_factory=dict)
    average_resolution_eV: float | None = None
    accuracy_by_2nd_fitting_eV: float | None = None
    source: dict[str, Any] = Field(default_factory=dict)
    rendered: bool = False
    provenance: list[str] = Field(default_factory=list)


def _validate_gold(data: Any) -> xr.DataArray:
    if not isinstance(data, xr.DataArray):
        raise TypeError(
            "fit_gold_reference: gold must be an xarray.DataArray, got "
            f"{type(data).__name__}."
        )
    if "eV" not in data.dims:
        raise ValueError(
            f"fit_gold_reference: expected an 'eV' dimension, got {data.dims}."
        )
    if data.ndim not in (1, 2):
        raise ValueError(
            f"fit_gold_reference: expected a 1D or 2D gold scan, got "
            f"ndim={data.ndim} ({data.dims})."
        )
    values = np.asarray(data.values)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError(
            "fit_gold_reference: gold scan must be finite everywhere (no "
            "NaN/inf values and no empty arrays)."
        )
    return data


def _fit_gold(
    data: xr.DataArray,
    *,
    correction: str,
    outlier_exclusion: bool,
    outlier_sigma: float,
    plot: bool,
    show: bool,
) -> Any:
    """Run peaks' own Fermi-edge fit once (delegation point for tests)."""
    return data.fit_gold(
        EF_correction_type=correction,
        outlier_exclusion=outlier_exclusion,
        outlier_sigma=outlier_sigma,
        plot=plot,
        show=show,
    )


def fit_gold_reference(
    gold: Any,
    *,
    correction: str = "poly4",
    outlier_exclusion: bool = True,
    outlier_sigma: float = 3.0,
    plot: bool = True,
) -> GoldCalibration:
    """Fit the Fermi edge of one gold reference scan.

    Parameters
    ----------
    gold : xarray.DataArray
        Finite gold scan with an ``eV`` dimension (typically
        ``(eV, theta_par)``).  Load it with :func:`load_data` first.
    correction : str, default "poly4"
        EF-correction model passed to peaks: one of ``poly4``, ``poly3``,
        ``quadratic``, ``linear``, ``average``.
    outlier_exclusion : bool, default True
        Exclude angular outliers from the poly fit of EF.
    outlier_sigma : float, default 3.0
        MAD-scaled outlier threshold used by the fit.
    plot : bool, default True
        Render the diagnostic figure inline in the notebook for the user.

    Returns
    -------
    GoldCalibration
        JSON-safe calibration: ``correction`` (float for ``average``, else a
        ``c0..cN`` dict), ``quality``, ``average_resolution_eV``,
        ``accuracy_by_2nd_fitting_eV``, source summary and ``rendered``.
        Apply it once with ``da.metadata.set_EF_correction(calibration.correction)``
        or pass it straight to :func:`preprocess_cut` /
        :func:`preprocess_mapping`.

    Raises
    ------
    ValueError
        For non-finite, empty, or >2D input, or when peaks reports an
        unusable fit.
    """
    data = _validate_gold(gold)
    result = _fit_gold(
        data,
        correction=correction,
        outlier_exclusion=outlier_exclusion,
        outlier_sigma=outlier_sigma,
        plot=plot,
        show=plot,
    )
    attrs = dict(getattr(result, "attrs", {}) or {})
    ef_correction = attrs.get("EF_correction")
    if ef_correction is None:
        raise ValueError(
            "fit_gold_reference: peaks returned no EF_correction; the fit did "
            "not converge on a usable edge."
        )
    rendered = bool(plot) and getattr(result, "attrs", {}).get("figure") is not None
    provenance = [f"fit_gold(correction={correction}, sigma={outlier_sigma})"]
    return GoldCalibration(
        correction=ef_correction,
        correction_type=correction,
        ef_at_normal_emission_eV=attrs.get("EF_poly4"),
        quality=dict(attrs.get("EF_quality") or {}),
        average_resolution_eV=attrs.get("average_resolution_eV"),
        accuracy_by_2nd_fitting_eV=attrs.get("accuracy_by_2nd_fitting_eV"),
        source={
            "dims": list(result.dims),
            "shape": list(result.sizes.values()),
            "name": getattr(data, "name", None),
        },
        rendered=rendered,
        provenance=provenance,
    )
