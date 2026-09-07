"""Unified experiment-metadata reading for peaksMCP.

Single home for the field matchers that previously lived separately in the
datasheet translator (``csv_translator``) and the cut workflow
(``process_cut``): loading the metadata document (path / parsed dict /
``DataArray`` attrs), extracting the high-symmetry offset from notes, and
classifying a ``Data format`` value as gold / sweep / mapping.  ``read_meta``
builds the structured per-index digest used before preprocessing, so agents
call one function instead of hand-parsing the JSON repeatedly.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

#: Number following the ``theta_offset`` token in a note, e.g.
#: ``theta_offset：1.5`` / ``theta_offset= -0.5度``.
_THETA_OFFSET_RE = re.compile(
    r"theta[_ ]?offset\s*[:：=]?\s*([+-]?\d+(?:\.\d+)?)", re.IGNORECASE
)


def load_metadata(source: str | os.PathLike[str] | dict[str, Any] | Any) -> dict[str, Any]:
    """Load the experiment metadata document from a path, dict or DataArray.

    A ``DataArray`` contributes ``attrs["experiment_metadata_json"]`` (the
    converter embeds it on load); a missing value yields an empty dict.
    """
    if isinstance(source, dict):
        return source
    if isinstance(source, (str, os.PathLike)):
        with open(os.fspath(source), encoding="utf-8") as f:
            return json.load(f)
    raw = getattr(source, "attrs", {}).get("experiment_metadata_json")
    if not raw:
        return {}
    return json.loads(raw) if isinstance(raw, str) else raw


def theta_offset_deg(text: str) -> float | None:
    """Return the first ``theta_offset``-prefixed number in a note, or None."""
    match = _THETA_OFFSET_RE.search(text or "")
    if match is None:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def is_gold_format(data_format: str) -> bool:
    """True when ``Data format`` marks this index as a gold (Au) reference.

    The datasheet tags gold data (used for Fermi-edge fitting) with ``Au`` /
    ``gold`` / ``金`` in the ``Data format`` column, either alone or combined
    with the scan type in a comma/space separated list, e.g. ``Au``,
    ``Au sweep`` or ``sweep,Au``.
    """
    raw = (data_format or "").strip()
    lowered = (
        raw.lower()
        .replace("_", " ")
        .replace("-", " ")
        .replace(",", " ")
        .replace(";", " ")
        .split()
    )
    return "au" in lowered or "gold" in lowered or "金" in raw


def classify_data_format(data_format: str) -> str | None:
    """Classify a ``Data format`` value as ``"gold"``, ``"sweep"``, ``"mapping"`` or None."""
    if is_gold_format(data_format):
        return "gold"
    lowered = (data_format or "").lower()
    if "sweep" in lowered:
        return "sweep"
    if "mapping" in lowered:
        return "mapping"
    return None


def read_meta(
    metadata: str | os.PathLike[str] | dict[str, Any] | Any,
    data: dict[int | str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize the experiment record table for cut preprocessing.

    Returns a structured digest of ``experiment_metadata.json`` (or an
    already-parsed dict): which indices are sweeps / mappings / gold, plus each
    record's kind, polarization, energy window and theta offset — and, when the
    loaded DataArrays are supplied, the data dimensionality.  Call this once
    instead of hand-parsing the JSON repeatedly.

    Parameters
    ----------
    metadata : str, os.PathLike, dict or xarray.DataArray
        Path to ``experiment_metadata.json``, its parsed dict, or a loaded
        scan carrying ``attrs["experiment_metadata_json"]``.
    data : dict of int|str to xarray.DataArray, optional
        Loaded scans keyed by record index; when given, each digest reports
        ``ndim`` and ``dims`` so 3D records (e.g. a ``deflector_perp`` sweep)
        are visible before processing.

    Returns
    -------
    dict
        ``records`` (per-index digest with a ``kind`` of gold/sweep/mapping),
        ``sweeps`` / ``mappings`` / ``gold`` index lists, the free-text
        ``notes`` and the distinct ``energy_windows_eV``.

    Examples
    --------
    >>> summary = read_meta("experiment_metadata.json", data=scans)
    >>> summary["sweeps"]
    [5, 6, 9]
    """
    meta = load_metadata(metadata)
    records_in = meta.get("records") or {}
    records_out: list[dict[str, Any]] = []
    sweeps: list[Any] = []
    mappings: list[Any] = []
    gold: list[Any] = []
    windows: set[tuple[float, float]] = set()

    for index_key, rec in records_in.items():
        try:
            index: Any = int(index_key)
        except (TypeError, ValueError):
            index = index_key
        exp = rec.get("experiment") or {}
        data_format = str(exp.get("data_format") or "").strip()
        is_gold = bool(rec.get("is_gold_reference")) or is_gold_format(data_format)
        kind = classify_data_format(data_format)
        start = exp.get("energy_start_eV")
        stop = exp.get("energy_stop_eV")
        window: tuple[float, float] | None = None
        if start is not None and stop is not None:
            window = (float(start), float(stop))
            windows.add(window)
        digest: dict[str, Any] = {
            "index": index,
            "data_format": data_format,
            "kind": kind,
            "is_gold": is_gold,
            "polarisation": (rec.get("photon") or {}).get("polarisation"),
            "energy_window_eV": window,
            "theta_offset_deg": rec.get("theta_offset_deg"),
        }
        if data is not None and index in data:
            array = data[index]
            digest["ndim"] = int(array.ndim)
            digest["dims"] = list(array.dims)
        records_out.append(digest)
        if is_gold:
            gold.append(index)
        elif kind == "sweep":
            sweeps.append(index)
        elif kind == "mapping":
            mappings.append(index)

    records_out.sort(key=lambda r: r["index"])
    return {
        "records": records_out,
        "sweeps": sorted(sweeps),
        "mappings": sorted(mappings),
        "gold": sorted(gold),
        "notes": meta.get("notes") if isinstance(meta, dict) else None,
        "energy_windows_eV": sorted(windows),
    }