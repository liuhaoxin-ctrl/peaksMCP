"""Deterministic ARPES cut preprocessing for k-space analysis.

``process_cut`` performs the full angle-domain workflow (method A):

1. ``fit_gold`` (poly4, outlier exclusion, dynamic EF bounds) -> ``EF_correction``
   plus an ``EF_quality`` report (angular uniformity = convergence benchmark);
2. ``metadata.set_EF_correction`` (kinetic -> binding, Fermi level -> 0);
3. ``metadata.set_normal_emission(theta_par=theta_offset)`` (angle-domain shift
   of the high-symmetry point to zero — NEVER a fixed kx shift after conversion);
4. ``k_convert`` -> ``(eV, kx)`` in binding energy with kx = 0 at the
   high-symmetry point.

Instrument geometry for the DA30L data converted from PXT: the slit axis is
``tilt`` (theta_par maps into the tilt group) and the mapping axis is ``polar``;
kx (along the slit) therefore corresponds to tilt, ky to polar.  This matches
the base-ARPES-loader default geometry (slit azi = 0), so no sign-convention
overrides are needed for loc-less data.
"""

from __future__ import annotations

from typing import Any

import xarray as xr

GEOMETRY_NOTE = (
    "Instrument geometry: slit axis = tilt (theta_par), mapping axis = polar; "
    "kx (along slit) = tilt direction, ky = polar direction."
)


def process_cut(
    da: xr.DataArray,
    theta_offset: float | None = None,
    ef_correction: float | int | dict[str, Any] | None = None,
    ef_bounds: tuple[float, float] | None = None,
    fit_kwargs: dict[str, Any] | None = None,
    quiet: bool = True,
) -> dict[str, Any]:
    """Preprocess one ARPES cut (2D sweep) into k-space.

    Parameters
    ----------
    da : xarray.DataArray
        2D data with ``eV`` and ``theta_par`` dimensions (no beamline ``loc``
        metadata required — the base ARPES geometry is used).
    theta_offset : float, optional
        High-symmetry-point offset in degrees: ``theta_par`` is shifted by this
        amount so that the high-symmetry point maps to kx = 0 after conversion.
        MUST come from experiment metadata (e.g. ``notes``), never invented.
        When ``None``, the function looks it up in ``da.attrs``
        ``experiment_metadata_json`` notes; if not found the offset is treated
        as 0 and a warning is attached.
    ef_correction : float, int, dict, optional
        Fermi-level correction to APPLY. When provided (e.g. from a gold
        reference fitted once with ``fit_gold``), no re-fitting is performed on
        this cut — use this mode for ordinary sweep data. When ``None``
        (default), the Fermi edge is fitted ON THIS data (Au-reference mode),
        which also produces the ``EF_quality`` convergence report.
    ef_bounds : tuple of float, optional
        Bounds for the fitted Fermi level (Au-reference mode only). ``None``
        uses dynamic bounds around the derivative-peak estimate (both
        Fermi-Dirac plateaus stay inside the window), which prevents the EF
        from drifting onto the window edge.
    fit_kwargs : dict, optional
        Extra keyword arguments for ``fit_gold`` (Au-reference mode only).
    quiet : bool, optional
        Suppress progress output. Defaults to True.

    Returns
    -------
    dict
        ``data`` : converted DataArray ``(eV, kx)`` in binding energy;
        ``EF_correction`` : Fermi-level correction applied;
        ``EF_quality`` : angular-uniformity report from the gold fit (None in
        apply mode) — a convergence benchmark; a drifted (non-uniform) fit must
        not be used.
        ``theta_offset_deg`` / ``geometry`` : provenance of the shift.
    """
    if ef_correction is None:
        # Au-reference mode: fit the Fermi edge on this data (produces the
        # EF_quality convergence report).
        fit_kwargs = dict(fit_kwargs or {})
        if "EF_correction_type" not in fit_kwargs:
            fit_kwargs["EF_correction_type"] = "poly4"
        fit_kwargs.setdefault("outlier_exclusion", True)
        fit_kwargs["EF_bounds"] = ef_bounds

        fit = da.fit_gold(**fit_kwargs)
        ef_correction = fit.attrs.get("EF_correction")
        ef_quality = fit.attrs.get("EF_quality")
    else:
        # Apply mode: reuse a gold-reference correction; no refit on this cut.
        ef_quality = None

    offset = theta_offset
    if offset is None:
        offset = _theta_offset_from_metadata(da)
        if offset is None:
            offset = 0.0

    da.metadata.set_EF_correction(ef_correction)
    da.metadata.set_normal_emission(theta_par=offset)
    da_k = da.k_convert(quiet=quiet)

    return {
        "data": da_k,
        "EF_correction": ef_correction,
        "EF_quality": ef_quality,
        "theta_offset_deg": float(offset),
        "geometry": GEOMETRY_NOTE,
    }


def _theta_offset_from_metadata(da: xr.DataArray) -> float | None:
    """Try to extract the high-symmetry offset from the experiment notes.

    Looks for the note pattern ``差了差不多0.5度`` / ``0.5度`` (or an explicit
    ``theta_par_offset`` field) inside ``attrs["experiment_metadata_json"]``.
    Returns None when nothing usable is found.
    """
    import json
    import re

    raw = da.attrs.get("experiment_metadata_json")
    if not raw:
        return None
    try:
        meta = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        return None

    if isinstance(meta, dict) and meta.get("theta_par_offset_deg") is not None:
        try:
            return float(meta["theta_par_offset_deg"])
        except (TypeError, ValueError):
            pass

    notes = meta.get("notes", []) if isinstance(meta, dict) else []
    if isinstance(meta, dict) and "notes" not in meta and "title" in meta:
        # The top-level metadata JSON (per record) has no notes; fall back to
        # searching every string field for the offset pattern.
        notes = []
    for note in notes:
        if isinstance(note, str):
            m = re.search(r"([0-9]*\.?[0-9]+)\s*度", note)
            if m:
                return float(m.group(1))
    return None
