"""Unified data loading facade: metadata first, data as the lazy layer.

One entry for the agent to load ANY data — any number of files, any
supported type — from a notebook:

    data       = load_data("BP_0020.nc")            # one NetCDF -> DataArray
    raw        = load_data("raw/BP_0020.pxt")       # one PXT    -> DataArray
    exp        = load_data("data/")                 # folder     -> LoadedScans
    exp        = load_data("data_netcdf/")          # already-converted folder
    subset     = load_data(["a.nc", "b.nc"])        # explicit list -> LoadedScans

Folder / list loads return a lightweight :class:`LoadedScans` index — the
metadata the agent needs to DECIDE (which scans are gold/cut/mapping, which
still need conversion) — with the data itself as one extra layer: access
``exp[stem]`` to load that single file (NetCDF lazily).  Whatever the number
of files, loading never materialises the data; the printed summary is one
line and the index object is the shared record in the notebook.

Rules:
- Single file returns a DataArray; a folder or a list returns LoadedScans.
- Metadata source order: explicit ``metadata`` argument > sibling
  ``datasheet.csv`` in the folder (translated once) > each NetCDF's embedded
  ``experiment_metadata_json`` (header read only).  PXT files without a
  datasheet carry path info only.
- Data layer: ``exp[stem]`` reads NetCDF lazily (default) or the raw PXT via
  the internal PXT reader; arrays that already carry converted metadata are
  left untouched; raw PXT gets the translated document attached.
- The original files are never modified.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import Report

_SUPPORTED_SUFFIXES = {".pxt", ".nc"}


class LoadReport(Report):
    """Result of load_data for the single-file form: which file, how read."""

    operation = "load_data"
    status = "ok"
    path = ""
    kind = ""
    dims: dict[str, int] | None = None

    def summary_line(self) -> str:
        dims = self.dims or {}
        size = ", ".join(f"{k}={v}" for k, v in dims.items())
        return f"load_data: {self.path} ({self.kind}) dims [{size}]; result in the cell variable"


def _register_l112_once() -> None:
    try:
        from peaksMCP.pxt_utils.loader import _register_l112_loader

        _register_l112_loader()
    except Exception:
        pass  # native peaks already knows the location or registration is a no-op


def _attach_metadata(data: Any, metadata: Any) -> None:
    if metadata is None:
        return
    # The metadata reader is the internal _load_metadata helper (the public
    # digest entry is read_meta); import errors are NOT swallowed here so a
    # future rename fails loudly instead of silently dropping metadata.
    from peaksMCP.pxt_utils.metadata import _load_metadata

    payload = _load_metadata(metadata)
    data.attrs["experiment_metadata_json"] = payload


def _single(path: Path, lazy: bool) -> tuple[Any, str]:
    """Load one supported file into a DataArray (delegation seam for tests).

    peaks.load prints a per-file Markdown line on the lazy path; it is
    swallowed here so a load stays quiet (load_data prints its own summary).
    Loading itself is unaffected.
    """
    suffix = path.suffix.lower()
    if suffix == ".pxt":
        from peaksMCP.pxt_utils.loader import load_pxt

        return load_pxt(str(path)), "PXT"
    _register_l112_once()
    from peaks import load

    if lazy:
        import contextlib
        import io

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            data = load(str(path), lazy=True, quiet=True)
        return data, "NetCDF"
    return load(str(path), lazy=False, quiet=True), "NetCDF"


def _index_from_stem(stem: str) -> int | None:
    """Parse a trailing index out of a filename stem (``BP_0020`` -> 20)."""
    match = re.search(r"(?:^|_|-)(\d+)\s*$", stem)
    return int(match.group(1)) if match else None


def _scan_kind_of(is_gold: bool, data_format: str) -> str:
    """Datasheet-based kind (no data read): gold/cut/mapping/unknown."""
    if is_gold or "au" in (data_format or "").lower() or "金" in (data_format or ""):
        return "gold"
    lowered = (data_format or "").lower()
    if "sweep" in lowered:
        return "cut"
    if "mapping" in lowered:
        return "mapping"
    return "unknown"


@dataclass
class ScanEntry:
    """One file in a LoadedScans index: identity + decision metadata."""

    stem: str
    path: str
    file_kind: str  # "pxt" | "netcdf"
    index: int | None = None
    data_format: str = ""
    is_gold: bool = False
    scan_kind: str = "unknown"
    theta_offset_deg: float | None = None
    energy_window_eV: tuple[float, float] | None = None
    #: Dimensions read from the file header (PXT wave header / NetCDF
    #: metadata) without materialising any data block; None when unreadable.
    sizes: dict[str, int] | None = None
    #: True for preprocessing products (``<stem>_processed.nc``, the single
    #: preprocess_batch output naming): they index as their own entry, tagged
    #: processed, and inherit the datasheet identity of the raw stem.
    processed: bool = False
    #: None = unknown (e.g. explicit list); True = NetCDF present; False =
    #: raw PXT without its converted sibling (still needs conversion).
    converted: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "stem": self.stem,
            "path": self.path,
            "file_kind": self.file_kind,
            "index": self.index,
            "data_format": self.data_format,
            "is_gold": self.is_gold,
            "scan_kind": self.scan_kind,
            "theta_offset_deg": self.theta_offset_deg,
            "energy_window_eV": self.energy_window_eV,
            "sizes": self.sizes,
            "processed": self.processed,
            "converted": self.converted,
        }


def _fill_from_record(entry: ScanEntry, index: int | None, record: Any) -> ScanEntry:
    """Fill decision metadata from one datasheet/embedded record."""
    if index is not None:
        entry.index = index
    experiment = (record or {}).get("experiment") or {}
    data_format = str(experiment.get("data_format") or "")
    entry.data_format = data_format
    entry.is_gold = bool(record.get("is_gold_reference"))
    entry.scan_kind = _scan_kind_of(entry.is_gold, data_format)
    offset = record.get("theta_offset_deg")
    entry.theta_offset_deg = float(offset) if isinstance(offset, (int, float)) else None
    start, stop = experiment.get("energy_start_eV"), experiment.get("energy_stop_eV")
    if isinstance(start, (int, float)) and isinstance(stop, (int, float)):
        entry.energy_window_eV = (float(start), float(stop))
    return entry


def _datasheet_candidates(directory: Path) -> list[Path]:
    """Where a datasheet may live for a folder being indexed.

    The standard experiment layout keeps ``datasheet.csv`` inside the raw
    ``data/`` folder; the converted sibling is ``<folder>_netcdf/`` (or the
    raw sibling when indexing ``<folder>_netcdf/``).  Candidates: inside the
    folder, in its parent, and in the raw sibling of a ``*_netcdf`` folder.
    """
    candidates = [directory / name for name in ("datasheet.csv", "Datasheet.csv")]
    candidates.append(directory.parent / "datasheet.csv")
    if directory.name.endswith("_netcdf"):
        raw_sibling = directory.parent / directory.name[: -len("_netcdf")]
        candidates.extend(
            raw_sibling / name for name in ("datasheet.csv", "Datasheet.csv")
        )
    return candidates


def _auto_datasheet_payload(directory: Path) -> dict[str, Any] | None:
    """Translate the experiment's ``datasheet.csv`` once for a directory load."""
    for datasheet in _datasheet_candidates(directory):
        if not datasheet.exists():
            continue
        try:
            from peaksMCP.pxt_utils.csv_translator import translate_datasheet

            return translate_datasheet(datasheet).model_dump(mode="python")
        except Exception:
            return None
    return None


def _scan_nc_header(path: Path) -> tuple[int | None, Any, dict[str, int] | None]:
    """Read a converted NetCDF's header: sizes, index and embedded record.

    NetCDF arrays carry ``sizes`` plus the converted ``experiment_index`` /
    ``experiment_metadata_json`` attributes; reading the header is cheap and
    never loads the data blocks.
    """
    try:
        data, _ = _single(path, lazy=True)
    except Exception:
        return None, None, None
    try:
        sizes = dict(data.sizes) if data.sizes else None
        index = data.attrs.get("experiment_index")
        raw = data.attrs.get("experiment_metadata_json")
        payload = json.loads(raw) if isinstance(raw, str) else (raw or {})
        records = payload.get("records") or {}
        return index, records.get(str(index)), sizes
    except Exception:
        return None, None, None


def _scan_pxt_header_sizes(path: Path) -> dict[str, int] | None:
    """Read a raw PXT's wave header: dimension sizes without materialising
    the data payload (matches load_pxt dims exactly)."""
    try:
        from peaksMCP.pxt_utils.loader import _scan_pxt_header

        sizes, _units = _scan_pxt_header(path)
        return sizes
    except Exception:
        return None


def _record_of(payload: dict[str, Any] | None, index: int | None) -> Any:
    if payload is None or index is None:
        return None
    try:
        return (payload.get("records") or {}).get(str(index))
    except Exception:
        return None


def _conversion_sibling_dir(folder: Path) -> Path | None:
    """The conversion target for a raw data folder: sibling <name>_netcdf."""
    return folder.parent / f"{folder.name}_netcdf"


class LoadedScans:
    """Lightweight experiment index: decision metadata + lazy data layer.

    Whatever the file count, constructing the index reads no data blocks:
    entries carry the datasheet/embedded metadata (kind, gold, offsets,
    windows, conversion state) and ``scans[stem]`` loads that one file on
    demand (NetCDF lazily by default).  The index object itself is the
    shared record left in the notebook.
    """

    def __init__(
        self,
        entries: list[ScanEntry],
        *,
        source: str,
        lazy: bool = True,
        payload: dict[str, Any] | None = None,
    ) -> None:
        entries = sorted(entries, key=lambda entry: entry.stem)
        self.entries = entries
        self._by_stem = {entry.stem: entry for entry in entries}
        self.source = source
        self.lazy = lazy
        self.payload = payload
        self._cache: dict[str, Any] = {}

    # -- mapping-style access ------------------------------------------------
    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    def __contains__(self, stem: object) -> bool:
        return stem in self._by_stem

    @property
    def stems(self) -> list[str]:
        return [entry.stem for entry in self.entries]

    def keys(self) -> list[str]:
        return self.stems

    # -- decision helpers ----------------------------------------------------
    @property
    def raw_entries(self) -> list[ScanEntry]:
        """Original scans only (preprocessing products excluded)."""
        return [entry for entry in self.entries if not entry.processed]

    @property
    def gold(self) -> list[str]:
        return [entry.stem for entry in self.raw_entries if entry.scan_kind == "gold"]

    @property
    def cuts(self) -> list[str]:
        return [entry.stem for entry in self.raw_entries if entry.scan_kind == "cut"]

    @property
    def mappings(self) -> list[str]:
        return [entry.stem for entry in self.raw_entries if entry.scan_kind == "mapping"]

    @property
    def processed(self) -> list[str]:
        """Preprocessing products in this folder (``*_processed.nc``)."""
        return [entry.stem for entry in self.entries if entry.processed]

    @property
    def needs_conversion(self) -> list[str]:
        """Raw PXT entries whose converted NetCDF is missing."""
        return [entry.stem for entry in self.entries if entry.converted is False]

    # -- data layer (the extra dimension) ------------------------------------
    def __getitem__(self, stem: str) -> Any:
        """Load ONE file on demand; NetCDF stays lazy, PXT is read eagerly."""
        if stem in self._cache:
            return self._cache[stem]
        entry = self._by_stem.get(stem)
        if entry is None:
            raise KeyError(
                f"load_data: {stem!r} is not in this index. Stems: "
                + ", ".join(self.stems[:10])
                + ("..." if len(self.stems) > 10 else "")
            )
        path = Path(entry.path)
        data, _kind = _single(path, lazy=self.lazy)
        if "experiment_metadata_json" not in data.attrs and self.payload is not None:
            index = _index_from_stem(stem)
            record = _record_of(self.payload, index)
            if record is not None:
                _attach_to(data, self.payload, index)
        self._cache[stem] = data
        return data

    def load(self, stem: str) -> Any:
        """Alias of ``scans[stem]`` for readability."""
        return self[stem]

    def summary_line(self) -> str:
        raw = self.raw_entries
        counts = {
            kind: sum(entry.scan_kind == kind for entry in raw)
            for kind in ("gold", "cut", "mapping", "unknown")
        }
        to_convert = len(self.needs_conversion)
        n_processed = len(self.entries) - len(raw)
        where = self.source
        text = (
            f"load_data: {len(self.entries)} file(s) indexed from {where} "
            f"(gold={counts['gold']}, cuts={counts['cut']}, "
            f"mappings={counts['mapping']}, unknown={counts['unknown']}"
        )
        if n_processed:
            text += f", processed={n_processed}"
        text += ")"
        if self.payload is not None:
            text += " metadata=datasheet"
        elif any(entry.file_kind == "netcdf" for entry in raw):
            text += " metadata=embedded"
        else:
            text += " metadata=none"
        if to_convert:
            text += f"; {to_convert} still need(s) conversion"
        gold_names = self.gold[:3]
        if gold_names and counts["gold"] <= 3:
            text += "; gold=" + ",".join(gold_names)
        text += "; data layer via scans[stem]"
        return text[:200]

    def __repr__(self) -> str:
        """print(exp) shows the decision summary (the agent's first read)."""
        return self.summary_line()

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "lazy": self.lazy,
            "summary": self.summary_line(),
            "entries": [entry.to_dict() for entry in self.entries],
        }


def _attach_to(data: Any, payload: dict[str, Any] | None, index: int | None) -> None:
    """Embed the experiment metadata document onto a loaded DataArray."""
    if payload is None:
        return
    try:
        if "experiment_metadata_json" in data.attrs:
            return
        data.attrs["experiment_metadata_json"] = payload
        if index is not None:
            data.attrs["experiment_index"] = index
    except Exception:
        pass  # object without attrs: metadata simply not embedded


def load_data(
    source: str | Path | list[str | Path] | tuple[str | Path, ...],
    *,
    lazy: bool = True,
    metadata: str | Path | dict[str, Any] | None = None,
) -> Any:
    """Load data for the agent: metadata first, data as the lazy layer.

    - one ``.pxt`` / ``.nc`` file  -> a single peaks DataArray;
    - a folder or a path list      -> a :class:`LoadedScans` index: every
      file's decision metadata (index, data format, gold/cut/mapping kind,
      theta offset, energy window, conversion state) without reading any
      data block, plus the data layer through ``scans[stem]`` (NetCDF lazy).

    Metadata for the index comes from, in order: the explicit ``metadata``
    argument, a sibling ``datasheet.csv`` in the folder, or each NetCDF's
    embedded ``experiment_metadata_json`` (header only).  Raw PXT files
    without a datasheet carry path info only.

    Parameters
    ----------
    source : str, Path, list or tuple
        One file, one directory, or an explicit sequence of files.
    lazy : bool, default True
        NetCDF data stays chunked until accessed.
    metadata : str, Path or dict, optional
        Explicit experiment metadata used for the index and attached to raw
        PXT arrays on load.

    Returns
    -------
    peaks DataArray or LoadedScans
        A DataArray for one file; a LoadedScans index for a folder/list.

    Raises
    ------
    ValueError
        For missing paths, empty sequences, folders without supported files,
        or an unsupported file type.
    """
    if isinstance(source, (list, tuple)):
        paths = [Path(item).expanduser() for item in source]
        if not paths:
            raise ValueError("load_data: no paths given (empty sequence).")
        return _index_paths(paths, lazy=lazy, metadata=metadata)

    path = Path(source).expanduser()
    if not path.exists():
        raise ValueError(
            f"load_data: path not found: {path}. Check the path before retrying."
        )
    if path.is_dir():
        groups = _data_groups(path)
        flat = [file for _dir, files in groups for file in files]
        if not flat:
            found = [d.name for d in path.iterdir() if d.is_dir()]
            hint = (
                f" Data subfolders found: {sorted(found)[:8]} - point "
                "load_data at a data/ or data_netcdf/ folder, or the whole "
                "experiment root."
                if found
                else ""
            )
            raise ValueError(
                f"load_data: no supported data files (.pxt, .nc) in {path}.{hint}"
            )
        if len(groups) == 1:
            folder, files = groups[0]
            data_dir = folder if folder is not None else path
            return _index_paths(
                files, lazy=lazy, metadata=metadata, directory=data_dir
            )
        # Experiment root: index each data subfolder with its own context
        # (sibling datasheet + conversion state), then merge.
        first_dir = groups[0][0] if groups[0][0] is not None else path
        scans = _index_paths(
            groups[0][1], lazy=lazy, metadata=metadata, directory=first_dir
        )
        merged_entries = list(scans.entries)
        payload = scans.payload
        for folder, files in groups[1:]:
            part = _index_paths(
                files, lazy=lazy, metadata=metadata, directory=folder
            )
            merged_entries.extend(part.entries)
            if payload is None:
                payload = part.payload
        # Deduplicate by stem across subfolders: the converted NetCDF wins
        # over the raw PXT (same scan, full geometry) - unless no NetCDF
        # exists, in which case the raw PXT entry stays (needs conversion).
        by_stem: dict[str, ScanEntry] = {}
        for entry in sorted(merged_entries, key=lambda e: e.file_kind != "netcdf"):
            prior = by_stem.get(entry.stem)
            if prior is None or entry.file_kind == "netcdf":
                by_stem[entry.stem] = entry
        merged_entries = list(by_stem.values())
        subdirs = sorted({str(Path(e.path).parent.name) for e in merged_entries})
        merged = LoadedScans(
            merged_entries,
            source=f"{path.name}/{{{', '.join(subdirs)}}}",
            lazy=lazy,
            payload=payload,
        )
        print(merged.summary_line())
        return merged
    suffix = path.suffix.lower()
    if suffix not in _SUPPORTED_SUFFIXES:
        raise ValueError(
            f"load_data: unsupported file type {suffix!r}; supported: .pxt, .nc"
        )
    data, kind = _single(path, lazy)
    _attach_metadata(data, metadata)
    report = LoadReport(path=str(path), kind=kind, dims=dict(data.sizes))
    print(report.summary_line())
    return data


def _data_groups(directory: Path) -> list[tuple[Path | None, list[Path]]]:
    """Supported files grouped by their data folder.

    Files directly inside ``directory`` form one group (folder=None); each
    direct subfolder that contains supported files forms its own group, so an
    experiment root (data/ + data_netcdf/) indexes every subfolder with its
    own sibling-datasheet and conversion-state context.
    """
    own = sorted(
        item
        for item in directory.iterdir()
        if item.is_file() and item.suffix.lower() in _SUPPORTED_SUFFIXES
    )
    groups: list[tuple[Path | None, list[Path]]] = []
    if own:
        groups.append((None, own))
    for item in sorted(
        child for child in directory.iterdir()
        if child.is_dir() and not child.name.startswith(".")
    ):
        files = sorted(
            f
            for f in item.iterdir()
            if f.is_file() and f.suffix.lower() in _SUPPORTED_SUFFIXES
        )
        if files:
            groups.append((item, files))
    return groups


def _index_paths(
    paths: list[Path],
    *,
    lazy: bool,
    metadata: Any,
    directory: Path | None = None,
) -> LoadedScans:
    """Build a LoadedScans index for a set of paths (no data read)."""
    if not paths:
        where = f" in {directory}" if directory is not None else ""
        hint = ""
        if directory is not None:
            found = [d.name for d in directory.iterdir() if d.is_dir()]
            hint = (
                f" Data subfolders found: {sorted(found)[:8]} - point "
                "load_data at a data/ or data_netcdf/ folder, or the whole "
                "experiment root."
                if found
                else ""
            )
        raise ValueError(
            f"load_data: no supported data files (.pxt, .nc){where}.{hint}"
        )
    payload: dict[str, Any] | None = None
    if metadata is not None:
        if isinstance(metadata, dict):
            payload = metadata
        else:
            try:
                from peaksMCP.pxt_utils.metadata import _load_metadata

                payload = _load_metadata(metadata)
            except Exception:
                payload = None
    elif directory is not None:
        payload = _auto_datasheet_payload(directory)

    sibling = _conversion_sibling_dir(directory) if directory is not None else None
    entries: list[ScanEntry] = []
    for path in paths:
        stem = path.stem
        file_kind = "netcdf" if path.suffix.lower() == ".nc" else "pxt"
        processed = bool(
            file_kind == "netcdf" and stem.endswith("_processed")
        )
        identity_stem = stem[: -len("_processed")] if processed else stem
        index = _index_from_stem(identity_stem)
        record = _record_of(payload, index)
        converted: bool | None
        if file_kind == "netcdf":
            converted = True
        elif sibling is not None:
            converted = (sibling / f"{stem}.nc").exists()
        else:
            converted = None
        sizes: dict[str, int] | None = None
        if file_kind == "netcdf":
            header_index, embedded, header_sizes = _scan_nc_header(path)
            sizes = header_sizes
            if record is None and embedded is not None:
                index = header_index if header_index is not None else index
                record = embedded
        else:
            sizes = _scan_pxt_header_sizes(path)
        entry = ScanEntry(
            stem=stem,
            path=str(path),
            file_kind=file_kind,
            index=index,
            sizes=sizes,
            processed=processed,
            converted=converted,
        )
        if record is not None:
            entry = _fill_from_record(entry, index, record)
        entries.append(entry)
    shown_source = directory.name if directory is not None else "sequence"
    scans = LoadedScans(entries, source=shown_source, lazy=lazy, payload=payload)
    print(scans.summary_line())
    return scans
