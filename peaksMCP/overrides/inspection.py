"""Experiment inspection facade: the single classification owner.

``inspect_experiment(scans)`` reads a :class:`LoadedScans` index (from
``load_data``) and produces the structured, JSON-safe
:class:`ExperimentSummary`: one :class:`ScanSummary` per experiment record
with a :class:`ScanKind`, the energy window, theta offset, polarisation and
the actual dimensions (from the header sizes each entry carries — no data is
materialised).  Records whose declared ``Data format`` disagrees with their
shape (e.g. a 3-D cube labelled ``sweep``) are reported as classification
conflicts so the caller never feeds a mapping into a cut-only workflow.

This facade is the ONLY classification owner: ``load_data`` deliberately
indexes identity only (representation + sizes + provenance), and nothing
else in the package decides gold/cut/mapping kinds or decision lists.  The
low-level format-string rules live privately in ``pxt_utils.metadata``;
``inspect_experiment`` combines those with the real shape information.

The document-only forms (a path to ``experiment_metadata.json`` / parsed
dict / a scan carrying it, with an optional ``scans`` dict of loaded arrays)
remain supported for standalone inspection.
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

    def summary_line(self) -> str:
        """One bounded line for the notebook: the decision lists at a glance.

        The gold reference index is printed in full (usually one scan) because
        everything downstream depends on it; cut/mapping lists are summarised by
        count — the full lists stay in the returned variable.
        """
        head = (
            f"inspect_experiment: {len(self.records)} record(s); "
            f"gold={self.gold}; cuts={len(self.cuts)}; mappings={len(self.mappings)}"
        )
        tail = ""
        if self.conflicts:
            tail += f"; conflicts={len(self.conflicts)}"
        if self.notes:
            tail += f"; notes={len(self.notes)}"
        return (head + tail)[:200]


def _scan_kind(
    format_kind: str | None,
    is_gold: bool,
    dims: list[str] | None,
) -> ScanKind:
    """Infer the executable ScanKind from the datasheet format + real shape.

    Rules (heuristic, conservative — anything unverifiable is ``unknown``):
    - gold records are always ``gold``;
    - a 3-D cube labelled ``sweep`` is really a (low-energy) mapping — the
      caller is expected to treat it as a mapping scan;
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
                "cube; treat it as a mapping scan, never as a cut"
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


def _loaded_scans_index(experiment: Any) -> Any | None:
    """Detect a LoadedScans index by its protocol shape (no import cycle).

    Anything exposing ``entries`` with ``representation``/``experiment_index``
    plus a ``metadata_document`` is treated as a load_data index.
    """
    entries = getattr(experiment, "entries", None)
    if not entries:
        return None
    sample = entries[0]
    if not (
        hasattr(sample, "stem")
        and hasattr(sample, "representation")
        and hasattr(sample, "experiment_index")
        and hasattr(sample, "sizes")
    ):
        return None
    if not (hasattr(experiment, "stems") and hasattr(experiment, "metadata_document")):
        return None
    return experiment


def _record_like(
    index: int | str,
    record: dict[str, Any] | None,
    dims: list[str] | None,
) -> tuple[dict[str, Any], ScanKind]:
    """Normalize one metadata record into ScanSummary inputs.

    Returns ``(fields, kind)`` where ``fields`` mirrors what the scan layer
    knows (data_format, is_gold, windows, offset, polarisation) and ``kind``
    is the executable ScanKind from format + shape.
    """
    if record is None:
        record = {}
    experiment = (record.get("experiment") or {}) if isinstance(record, dict) else {}
    data_format = str(experiment.get("data_format") or "")
    is_gold = bool(
        (record.get("is_gold_reference") if isinstance(record, dict) else False)
        or _format_is_gold(data_format)
    )
    format_kind = _classify_format(data_format)
    kind = _scan_kind(format_kind, is_gold, dims)
    start = experiment.get("energy_start_eV")
    stop = experiment.get("energy_stop_eV")
    window: tuple[float, float] | None = None
    if isinstance(start, (int, float)) and isinstance(stop, (int, float)):
        window = (float(start), float(stop))
    photon = record.get("photon") or {}
    fields: dict[str, Any] = {
        "data_format": data_format,
        "is_gold": is_gold,
        "energy_window_eV": window,
        "theta_offset_deg": record.get("theta_offset_deg"),
        "polarisation": photon.get("polarisation") if isinstance(photon, dict) else None,
    }
    return fields, kind


def _format_is_gold(data_format: str) -> bool:
    from peaksMCP.pxt_utils.metadata import _is_gold_format

    return _is_gold_format(data_format)


def _classify_format(data_format: str) -> str | None:
    from peaksMCP.pxt_utils.metadata import _classify_data_format

    return _classify_data_format(data_format)


def _summarize_rows(
    rows: list[tuple[int | str, dict[str, Any], ScanKind, list[str] | None]],
) -> ExperimentSummary:
    """Build an ExperimentSummary from normalized (index, fields, kind, dims)."""
    summaries: list[ScanSummary] = []
    gold: list[int | str] = []
    cuts: list[int | str] = []
    mappings: list[int | str] = []
    conflicts: list[ExperimentConflict] = []
    windows: set[tuple[float, float]] = set()
    for index, fields, kind, dims in rows:
        data_format = str(fields["data_format"])
        conflict = _conflict_for(
            index,
            data_format,
            _classify_format(data_format),
            dims,
            kind,
        )
        if conflict is not None:
            conflicts.append(conflict)
        if fields["energy_window_eV"] is not None:
            windows.add(fields["energy_window_eV"])  # type: ignore[arg-type]
        summaries.append(
            ScanSummary(
                index=index,
                data_format=data_format,
                kind=kind,
                is_gold=bool(fields["is_gold"]),
                ndim=len(dims) if dims is not None else None,
                dims=list(dims or []),
                energy_window_eV=fields["energy_window_eV"],
                theta_offset_deg=fields["theta_offset_deg"],
                polarisation=fields["polarisation"],
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
        energy_windows_eV=sorted(windows),
    )


def _as_key(index_key: Any) -> int | str:
    try:
        return int(index_key)
    except (TypeError, ValueError):
        return str(index_key)


def inspect_experiment(
    experiment: Any,
    *,
    scans: dict[int | str, Any] | None = None,
) -> ExperimentSummary:
    """Classify one experiment into a structured summary (the only owner).

    Parameters
    ----------
    experiment : LoadedScans, str, os.PathLike, dict or xarray.DataArray
        A :class:`LoadedScans` index from ``load_data`` (primary form:
        entries carry the header sizes; the index carries the metadata
        document provenance).  Standalone documents are also accepted: a
        path to ``experiment_metadata.json`` / ``datasheet.csv``, its parsed
        dict, or a loaded scan carrying ``attrs["experiment_metadata_json"]``.
    scans : dict of int|str to xarray.DataArray, optional
        Loaded arrays keyed by record index (document-only form): supplying
        them lets the summary verify the declared ``Data format`` against the
        real dimensions and report classification conflicts.

    Returns
    -------
    ExperimentSummary
        Per-index ScanSummary rows (kind, dims, energy window, theta offset,
        polarisation), gold/cut/mapping decision lists and structured
        ``conflicts``.  JSON-safe via ``model_dump(mode="json")``.

    Examples
    --------
    >>> scans = load_data("data/")
    >>> summary = inspect_experiment(scans)
    >>> summary.gold
    [20]
    """
    loaded = _loaded_scans_index(experiment)
    summary = (
        _inspect_loaded(loaded)
        if loaded is not None
        else _inspect_document(experiment, scans=scans)
    )
    # Classification is the entry point of the preprocessing chain: the agent
    # must see which scan is gold and how many cuts/mappings exist.  A bare
    # returned object would land as a text/plain repr, which the run_cell
    # normaliser deliberately drops, leaving the agent with an empty reply and
    # the load_data hint "classify with inspect_experiment(scans)" unanswered.
    print(summary.summary_line())
    return summary


def _inspect_loaded(loaded: Any) -> ExperimentSummary:
    """Classify a LoadedScans index: document records x loaded entries."""
    document = loaded.metadata_document or {}
    records_in = document.get("records") or {}
    rows: list[tuple[int | str, dict[str, Any], ScanKind, list[str] | None]] = []

    # One entry per experiment index for real-shape info: prefer the raw
    # converted NetCDF, then the raw PXT, then the processed product.
    def _rank(entry: Any) -> int:
        representation = str(entry.representation)
        if representation == "netcdf":
            return 2
        if representation == "raw_pxt":
            return 1
        return 0

    best_entry: dict[int | str, Any] = {}
    for entry in loaded.entries:
        if entry.experiment_index is None:
            continue
        key = entry.experiment_index
        prior = best_entry.get(key)
        if prior is None or _rank(entry) > _rank(prior):
            best_entry[key] = entry

    seen: set[int | str] = set()
    # Loaded entries first: their shapes drive classification even when no
    # metadata document exists (dimension-shape classification).
    for key, entry in sorted(best_entry.items(), key=lambda pair: str(pair[0])):
        dims = list((entry.sizes or {}).keys()) or None
        record = records_in.get(str(key))
        fields, kind = _record_like(key, record, dims)
        rows.append((key, fields, kind, dims))
        seen.add(key)
    # Document records without a loaded entry: format-only classification.
    for index_key, record in records_in.items():
        key = _as_key(index_key)
        if key in seen:
            continue
        if not isinstance(record, dict):
            continue
        fields, kind = _record_like(key, record, None)
        rows.append((key, fields, kind, None))

    summary = _summarize_rows(rows)
    notes = document.get("notes")
    if isinstance(notes, list):
        summary.notes = [str(note) for note in notes]
    return summary


def _inspect_document(metadata: Any, *, scans: dict[int | str, Any] | None) -> ExperimentSummary:
    """Classify a standalone metadata document (path / dict / DataArray).

    A ``datasheet.csv`` path is translated first through the datasheet
    translator, so the standard experiment folder (raw PXT + sibling
    ``datasheet.csv``) can be inspected before any conversion ran.
    """
    from peaksMCP.pxt_utils.metadata import _load_metadata, _read_meta

    if isinstance(metadata, (str, os.PathLike)) and str(metadata).lower().endswith(".csv"):
        from peaksMCP.pxt_utils.csv_translator import translate_datasheet

        translated = translate_datasheet(Path(metadata))
        metadata = translated.model_dump(mode="python")
    document = _load_metadata(metadata)
    digest = _read_meta(document, data=scans)
    rows: list[tuple[int | str, dict[str, Any], ScanKind, list[str] | None]] = []
    for record in digest.get("records") or []:
        index: int | str = record["index"]
        dims = record.get("dims")
        fields, kind = _record_like(
            index,
            {
                "experiment": {"data_format": record.get("data_format")},
                "is_gold_reference": record.get("is_gold"),
                "theta_offset_deg": record.get("theta_offset_deg"),
                "photon": {"polarisation": record.get("polarisation")},
            },
            dims,
        )
        # _read_meta already computed the window from start/stop; keep it.
        rows.append(
            (index, {**fields, "energy_window_eV": record.get("energy_window_eV")}, kind, dims)
        )
    summary = _summarize_rows(rows)
    notes = document.get("notes")
    if isinstance(notes, list):
        summary.notes = [str(note) for note in notes]
    return summary
