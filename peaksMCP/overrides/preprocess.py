"""Cut and mapping preprocessing facades (EF flattening -> k-space).

Both facades operate on copies: the input DataArray (values, coordinates,
attrs and peaks metadata) is never mutated.  They accept a
:class:`GoldCalibration` (or the raw float/dict EF correction it carries)
and return a :class:`ProcessingResult` whose ``report`` is JSON-safe while
the converted DataArray stays in the notebook.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr
from pydantic import BaseModel, Field

from .calibration import GoldCalibration


class ProcessingReport(BaseModel):
    """JSON-safe report of one preprocessing run."""

    operation: str
    status: str = "ok"
    warnings: list[str] = Field(default_factory=list)
    dims_in: list[str] = Field(default_factory=list)
    dims_out: list[str] = Field(default_factory=list)
    complete: bool = True
    selection: dict[str, Any] = Field(default_factory=dict)
    provenance: list[str] = Field(default_factory=list)


class ProcessingResult:
    """Notebook-side result of one preprocessing run.

    ``data`` is the converted DataArray (kept out of the JSON report by
    design); ``report`` carries the JSON-safe summary.  ``summary()``
    renders the Show-convention one-liner.
    """

    def __init__(self, data: xr.DataArray, report: ProcessingReport) -> None:
        self.data = data
        self.report = report

    def summary(self) -> str:
        report = self.report
        return (
            f"{report.operation}: dims {tuple(report.dims_in)} -> "
            f"{tuple(report.dims_out)} ({report.status})"
        )

    def to_dict(self) -> dict[str, Any]:
        payload = self.report.model_dump(mode="json")
        payload["summary"] = self.summary()
        payload["shape_out"] = list(self.data.shape)
        return payload


def _correction_value(calibration: Any) -> Any:
    """Accept a GoldCalibration or the raw EF correction it wraps."""
    if calibration is None:
        raise ValueError("preprocessing: a calibration is required (fit_gold_reference output)")
    if isinstance(calibration, GoldCalibration):
        return calibration.correction
    return calibration


def _require_eV_axis(data: xr.DataArray) -> None:
    if "eV" not in data.dims:
        raise ValueError(
            f"expected an 'eV' dimension, got {data.dims}. Load data with "
            "load_data (converted NetCDF) before preprocessing."
        )


def _require_finite(data: xr.DataArray) -> None:
    values = np.asarray(data.values)
    if values.size == 0:
        raise ValueError("preprocessing: data has no values (empty array).")
    finite = np.isfinite(values)
    if not finite.all():
        raise ValueError(
            "preprocessing: data contains non-finite values "
            f"({int((~finite).sum())} NaN/inf); clean the data first."
        )


def _shift_theta(data: xr.DataArray, offset_deg: float) -> xr.DataArray:
    """Copy with the high-symmetry angle shift applied to the coordinate."""
    if not isinstance(offset_deg, (int, float)):
        raise TypeError(
            f"theta_par_offset_deg must be a number, got {type(offset_deg).__name__}."
        )
    if "theta_par" not in data.coords:
        raise ValueError(
            "preprocess_cut: theta_par_offset_deg requires a 'theta_par' "
            f"coordinate; got dims {data.dims}."
        )
    return data.assign_coords(theta_par=data.theta_par - offset_deg)


def _k_convert(
    data: xr.DataArray,
    *,
    ef: Any,
    eV: Any,
    kx: Any,
    ky: Any,
    quiet: bool,
) -> xr.DataArray:
    """Run peaks' k_convert once (delegation point for tests)."""
    return data.k_convert(
        eV=eV,
        kx=kx,
        ky=ky,
        quiet=quiet,
        EF_correction=ef,
    )


def _set_normal_emission(data: xr.DataArray, normal_emission: dict[str, float]) -> xr.DataArray:
    """Apply the normal-emission reference angles on a copy's metadata."""
    try:
        data.metadata.set_normal_emission(**normal_emission)
    except Exception as exc:  # peaks may lack loader support or geometry
        raise ValueError(
            "preprocess_mapping: could not set the normal-emission reference "
            f"angles {normal_emission!r}: {exc}. Load the scan with load_data "
            "(converted NetCDF with full geometry) before preprocessing."
        ) from exc
    return data


def _preprocess_cut_impl(
    cut: xr.DataArray,
    *,
    calibration: GoldCalibration | float | dict[str, Any],
    theta_par_offset_deg: float,
    eV: Any = None,
    kx: Any = None,
    quiet: bool = True,
) -> ProcessingResult:
    """Flatten the Fermi edge of one cut and convert it to k-space.

    Accepts exactly a 2-D ``(eV, theta_par)`` cut.  The returned DataArray
    carries binding-energy ``(eV, kx)`` coordinates; the input array is not
    modified.

    Parameters
    ----------
    cut : xarray.DataArray
        Finite 2-D ``(eV, theta_par)`` cut (converted NetCDF loaded with
        :func:`load_data`).
    calibration : GoldCalibration, float or dict
        Output of :func:`fit_gold_reference` (or the raw EF correction it
        wraps).  Applied on the copy through ``k_convert``.
    theta_par_offset_deg : float
        High-symmetry angle offset; applied by shifting the theta_par
        coordinate on the copy (the repo's established convention).
    eV, kx : slice, optional
        Passed through to ``k_convert``.
    quiet : bool, default True
        Suppress k_convert chatter.

    Returns
    -------
    ProcessingResult
        ``data`` (binding-energy ``(eV, kx)`` DataArray in the notebook) and
        a JSON-safe ``report``.

    Raises
    ------
    ValueError
        For 3-D/1-D input, missing eV/theta_par, non-finite data, missing
        calibration or a missing angle offset.
    """
    if not isinstance(cut, xr.DataArray):
        raise TypeError(
            f"preprocess_cut: cut must be an xarray.DataArray, got {type(cut).__name__}."
        )
    _require_eV_axis(cut)
    if cut.ndim != 2 or "theta_par" not in cut.dims:
        raise ValueError(
            f"preprocess_cut: expected a 2-D (eV, theta_par) cut, got ndim="
            f"{cut.ndim} ({cut.dims}). 3-D data belongs to preprocess_mapping."
        )
    _require_finite(cut)
    ef = _correction_value(calibration)
    shifted = _shift_theta(cut, theta_par_offset_deg)
    converted = _k_convert(shifted, ef=ef, eV=eV, kx=kx, ky=None, quiet=quiet)
    report = ProcessingReport(
        operation="preprocess_cut",
        dims_in=list(cut.dims),
        dims_out=list(converted.dims),
        selection={"theta_par_offset_deg": float(theta_par_offset_deg)},
        provenance=[
            "shift theta_par by theta_par_offset_deg",
            "k_convert with EF_correction",
        ],
    )
    return ProcessingResult(converted, report)


def _preprocess_mapping_impl(
    mapping: xr.DataArray,
    *,
    calibration: GoldCalibration | float | dict[str, Any],
    normal_emission: dict[str, float],
    eV: Any = None,
    kx: Any = None,
    ky: Any = None,
    quiet: bool = True,
) -> ProcessingResult:
    """Flatten the Fermi edge of a full 3-D mapping cube and convert it.

    Accepts exactly a 3-D ``(eV, theta_par, deflector_perp)`` (or other
    3-D geometry) mapping; the full cube is converted — no internal centre
    slice is ever selected.  ``normal_emission`` supplies the reference
    angles the loader needs (e.g. ``{"theta_par": 12.0, "polar": 5.0}``).

    Returns
    -------
    ProcessingResult
        ``data`` with ``eV``, ``kx`` and ``ky`` dimensions (order varies by
        geometry — select by dimension name, never by position) and a
        JSON-safe ``report`` with an empty ``selection``.

    Raises
    ------
    ValueError
        For non-3-D input, missing eV dimension, non-finite data, missing
        calibration or normal-emission reference.
    """
    if not isinstance(mapping, xr.DataArray):
        raise TypeError(
            f"preprocess_mapping: mapping must be an xarray.DataArray, got "
            f"{type(mapping).__name__}."
        )
    _require_eV_axis(mapping)
    if mapping.ndim != 3:
        raise ValueError(
            f"preprocess_mapping: expected a full 3-D mapping cube, got "
            f"ndim={mapping.ndim} ({mapping.dims}). Centre slices are never "
            "extracted here; pass the whole cube."
        )
    _require_finite(mapping)
    if not normal_emission or not isinstance(normal_emission, dict):
        raise ValueError(
            "preprocess_mapping: normal_emission reference angles are "
            f"required (e.g. {{'theta_par': 0.0, 'polar': 0.0}}), got {normal_emission!r}."
        )
    ef = _correction_value(calibration)
    work = mapping.copy(deep=False)  # values shared; metadata copied on write
    _set_normal_emission(work, dict(normal_emission))
    converted = _k_convert(work, ef=ef, eV=eV, kx=kx, ky=ky, quiet=quiet)
    missing = {"eV", "kx", "ky"} - set(converted.dims)
    if missing:
        raise ValueError(
            "preprocess_mapping: k_convert output lacks expected dims "
            f"{sorted(missing)}; got {list(converted.dims)}."
        )
    report = ProcessingReport(
        operation="preprocess_mapping",
        dims_in=list(mapping.dims),
        dims_out=list(converted.dims),
        selection={},  # the full cube: no slice was made
        provenance=[
            "set normal-emission reference",
            "k_convert(EF_correction) over the full cube",
        ],
    )
    return ProcessingResult(converted, report)


def preprocess_cut(
    cut: xr.DataArray,
    *,
    calibration: GoldCalibration | float | dict[str, Any],
    theta_par_offset_deg: float,
    eV: Any = None,
    kx: Any = None,
    quiet: bool = True,
) -> ProcessingResult:
    """Flatten the Fermi edge of one cut and convert it to k-space.

    Accepts exactly a 2-D ``(eV, theta_par)`` cut; the input array is never
    modified.  See :func:`_preprocess_cut_impl` for the full contract.
    """
    result = _preprocess_cut_impl(
        cut,
        calibration=calibration,
        theta_par_offset_deg=theta_par_offset_deg,
        eV=eV,
        kx=kx,
        quiet=quiet,
    )
    print(result.summary())
    return result


def preprocess_mapping(
    mapping: xr.DataArray,
    *,
    calibration: GoldCalibration | float | dict[str, Any],
    normal_emission: dict[str, float],
    eV: Any = None,
    kx: Any = None,
    ky: Any = None,
    quiet: bool = True,
) -> ProcessingResult:
    """Flatten the Fermi edge of a full 3-D mapping cube and convert it.

    The full cube is converted — no internal centre slice is ever selected.
    See :func:`_preprocess_mapping_impl` for the full contract.
    """
    result = _preprocess_mapping_impl(
        mapping,
        calibration=calibration,
        normal_emission=normal_emission,
        eV=eV,
        kx=kx,
        ky=ky,
        quiet=quiet,
    )
    print(result.summary())
    return result
