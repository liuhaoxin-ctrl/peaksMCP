"""Experiment inspection facade: one structured summary of a datasheet.

``inspect_experiment`` reads the translated metadata document (or a loaded
scan carrying it) through the single metadata implementation in
``peaksMCP.pxt_utils.metadata`` and produces a JSON-safe
:class:`ExperimentSummary`: one :class:`ScanSummary` per record index with a
:class:`ScanKind`, the energy window, theta offset, polarisation and — when
the loaded arrays are supplied — the actual dimensions.  Records whose
declared ``Data format`` disagrees with their shape (e.g. a 3-D cube labelled
``sweep``) are reported as classification conflicts so the caller never feeds
a mapping into a cut-only workflow.
"""

from __future__ import annotations

import os
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ScanKind(StrEnum):
    """What one record actually is, judging from format and shape."""

    GOLD = "gold"
    CUT = "cut"
    MAPPING = "mapping"
    HV_SCAN = "hv_scan"
    SPATIAL_MAP = "spatial_map"
    SPECTRUM = "spectrum"
    UNKNOWN = "unknown"


class ExperimentConflict(BaseModel):
    """One record whose declared format does not match its data shape."""

    index: int | str
    data_format: str = ""
    dims: list[str] = Field(default_factory=list)
    issue: str


class ScanSummary(BaseModel):
    """Normalized digest of one experiment record."""

    index: int | str
    data_format: str = ""
    kind: ScanKind = ScanKind.UNKNOWN
    is_gold: bool = False
    ndim: int | None = None
    dims: list[str] = Field(default_factory=list)
    energy_window_eV: tuple[float, float] | None = None
    theta_offset_deg: float | None = None
    polarisation: str | None = None


class ExperimentSummary(BaseModel):
    """JSON-safe structured view of one experiment metadata document."""

    records: list[ScanSummary] = Field(default_factory=list)
    gold: list[int | str] = Field(default_factory=list)
    cuts: list[int | str] = Field(default_factory=list)
    mappings: list[int | str] = Field(default_factory=list)
    conflicts: list[ExperimentConflict] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    energy_windows_eV: list[tuple[float, float]] = Field(default_factory=list)


def _scan_kind(
    format_kind: str | None,
    is_gold: bool,
    dims: list[str] | None,
) -> ScanKind:
    """Infer the executable ScanKind from the datasheet format + real shape.

    Rules (heuristic, conservative — anything unverifiable is ``unknown``):
    - gold records are always ``gold``;
    - a 3-D cube labelled ``sweep`` is really a (low-energy) mapping — the
      caller is expected to confirm with ``preprocess_mapping``;
    - 1-D -> spectrum, 2-D -> cut, 3-D -> mapping, unless dimension names
      clearly say ``hv`` (hv scan) or spatial ``x``/``y`` (spatial map).
    """
    if is_gold or format_kind == "gold":
        return ScanKind.GOLD
    if not dims:
        # No loaded shape: only the datasheet format is available.
        return {
            "sweep": ScanKind.CUT,
            "mapping": ScanKind.MAPPING,
            None: ScanKind.UNKNOWN,
        }.get(format_kind, ScanKind.UNKNOWN)
    if len(dims) == 1:
        return ScanKind.SPECTRUM
    names = {str(dim).lower() for dim in dims}
    if "hv" in names:
        return ScanKind.HV_SCAN
    if {"x", "y"} <= names:
        return ScanKind.SPATIAL_MAP
    if len(dims) == 2:
        return ScanKind.CUT if format_kind != "mapping" else ScanKind.MAPPING
    # 3+ dims: mapping shapes unless declared as a sweep (conflict raised
    # separately by the caller).
    return ScanKind.MAPPING


def _conflict_for(
    index: int | str,
    data_format: str,
    format_kind: str | None,
    dims: list[str] | None,
    kind: ScanKind,
) -> ExperimentConflict | None:
    if not dims or format_kind is None:
        return None
    names = {str(dim).lower() for dim in dims}
    if len(dims) >= 3 and format_kind == "sweep":
        return ExperimentConflict(
            index=index,
            data_format=data_format,
            dims=dims,
            issue=(
                f"3-D record labelled 'sweep' (dims {dims}) is a mapping-shaped "
                "cube; use preprocess_mapping, never preprocess_cut"
            ),
        )
    if len(dims) == 2 and format_kind == "mapping" and not ({"x", "y"} <= names):
        return ExperimentConflict(
            index=index,
            data_format=data_format,
            dims=dims,
            issue=(
                f"2-D record labelled 'mapping' (dims {dims}) has no spatial "
                "x/y axes; confirm the intended geometry"
            ),
        )
    if kind == ScanKind.UNKNOWN and format_kind is None and len(dims) >= 3:
        return ExperimentConflict(
            index=index,
            data_format=data_format,
            dims=dims,
            issue=(
                f"{len(dims)}-D record without a usable 'Data format' value; "
                "classification is unknown"
            ),
        )
    return None


def _digests(metadata: Any, scans: dict[int | str, Any] | None) -> dict[str, Any]:
    """Run the single metadata digest implementation (read_meta).

    A ``datasheet.csv`` path is translated first through the datasheet
    translator, so the standard experiment folder (raw PXT + sibling
    ``datasheet.csv``) can be inspected before any conversion ran.
    """
    from peaksMCP.pxt_utils.metadata import read_meta

    if isinstance(metadata, (str, os.PathLike)) and str(metadata).lower().endswith(".csv"):
        from peaksMCP.pxt_utils.csv_translator import translate_datasheet

        translated = translate_datasheet(Path(metadata))
        metadata = translated.model_dump(mode="python")
    return read_meta(metadata, data=scans)


def inspect_experiment(
    metadata: str | os.PathLike[str] | dict[str, Any] | Any,
    *,
    scans: dict[int | str, Any] | None = None,
) -> ExperimentSummary:
    """Summarize one experiment metadata document for preprocessing.

    Parameters
    ----------
    metadata : str, os.PathLike, dict or xarray.DataArray
        Path to ``experiment_metadata.json``, its parsed dict, or a loaded
        scan carrying ``attrs["experiment_metadata_json"]``.
    scans : dict of int|str to xarray.DataArray, optional
        Loaded scans keyed by record index; supplying them lets the summary
        verify the declared ``Data format`` against the real dimensions and
        report classification conflicts (3-D cubes labelled ``sweep``, 2-D
        records labelled ``mapping`` without spatial axes, unverifiable 3-D
        records).

    Returns
    -------
    ExperimentSummary
        Per-index ScanSummary rows (kind, dims, energy window, theta offset,
        polarisation), gold/cut/mapping index lists and structured
        ``conflicts``.  JSON-safe via ``model_dump(mode="json")``.

    Examples
    --------
    >>> summary = inspect_experiment("experiment_metadata.json", scans=loaded)
    >>> [c.issue for c in summary.conflicts]
    []
    """
    document = _digests(metadata, scans)
    summaries: list[ScanSummary] = []
    gold: list[int | str] = []
    cuts: list[int | str] = []
    mappings: list[int | str] = []
    conflicts: list[ExperimentConflict] = []
    for record in document.get("records") or []:
        index: int | str = record["index"]
        data_format = str(record.get("data_format") or "")
        format_kind = record.get("kind")
        is_gold = bool(record.get("is_gold"))
        dims = record.get("dims")
        kind = _scan_kind(format_kind, is_gold, dims)
        conflict = _conflict_for(index, data_format, format_kind, dims, kind)
        if conflict is not None:
            conflicts.append(conflict)
        summaries.append(
            ScanSummary(
                index=index,
                data_format=data_format,
                kind=kind,
                is_gold=is_gold,
                ndim=record.get("ndim"),
                dims=list(dims or []),
                energy_window_eV=record.get("energy_window_eV"),
                theta_offset_deg=record.get("theta_offset_deg"),
                polarisation=record.get("polarisation"),
            )
        )
        if kind == ScanKind.GOLD:
            gold.append(index)
        elif kind in {ScanKind.CUT, ScanKind.SPECTRUM, ScanKind.SPATIAL_MAP, ScanKind.HV_SCAN}:
            cuts.append(index)
        elif kind == ScanKind.MAPPING:
            mappings.append(index)
    summaries.sort(key=lambda row: str(row.index))
    return ExperimentSummary(
        records=summaries,
        gold=gold,
        cuts=cuts,
        mappings=mappings,
        conflicts=conflicts,
        notes=list(document.get("notes") or []),
        energy_windows_eV=list(document.get("energy_windows_eV") or []),
    )
