"""Unified data loading facade: identity index first, then the data layer.

One entry for the agent to load ANY data — any number of files, any
supported type — from a notebook:

    data       = load_data("BP_0020.nc")            # one NetCDF -> DataArray
    raw        = load_data("raw/BP_0020.pxt")       # one PXT    -> DataArray
    exp        = load_data("data/")                 # folder     -> LoadedScans
    exp        = load_data("data_netcdf/")          # already-converted folder
    subset     = load_data(["a.nc", "b.nc"])        # explicit list -> LoadedScans

Folder / list loads return a lightweight :class:`LoadedScans` index: per-file
identity only — stem, path, ``representation`` (``raw_pxt`` / ``netcdf`` /
``processed_netcdf``), the experiment index parsed from the name/header, and
the dimension sizes read from the file header — plus the metadata provenance
(where the experiment metadata document came from: datasheet / embedded /
explicit / none).  What each scan IS (gold/cut/mapping, offsets, windows,
shape conflicts) is deliberately NOT decided here: classification is the job
of ``inspect_experiment`` (the single classification owner), which reads this
index.  The data itself is one extra layer: access ``exp[stem]`` to load that
single file (NetCDF lazily).  Whatever the number of files, loading never
materialises the data; the printed summary is one line and the index object
is the shared record in the notebook.

Rules:
- Single file returns a DataArray; a folder or a list returns LoadedScans.
- Metadata document source order: explicit ``metadata`` argument > sibling
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

_SUPPORTED_SUFFIXES = {".pxt", ".nc"}

#: Representation of one indexed file: what it IS on disk, nothing more.
#: ``converted``/``processed``/``needs_conversion`` derive from this value.
REPRESENTATION_RAW_PXT = "raw_pxt"
REPRESENTATION_NETCDF = "netcdf"
REPRESENTATION_PROCESSED_NETCDF = "processed_netcdf"

#: Where the LoadedScans metadata document came from (provenance only; the
#: document itself is used by inspect_experiment, never classified here).
METADATA_SOURCE_EXPLICIT = "explicit"
METADATA_SOURCE_DATASHEET = "datasheet"
METADATA_SOURCE_EMBEDDED = "embedded"
METADATA_SOURCE_NONE = "none"


@dataclass
class LoadReport:
    """Result of load_data for the single-file form: which file, how read."""

    operation: str = "load_data"
    status: str = "ok"
    path: str = ""
    kind: str = ""
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
    # The metadata reader is the internal _load_metadata helper; import errors
    # are NOT swallowed here so a future rename fails loudly instead of
    # silently dropping metadata.
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


@dataclass
class ScanEntry:
    """One file in a LoadedScans index: identity + representation only.

    Decision metadata is deliberately absent: what a scan IS (gold/cut/
    mapping kind, offsets, windows, shape conflicts) is classified once by
    ``inspect_experiment`` from the experiment metadata document combined
    with this entry's ``sizes`` — never here.
    """

    stem: str
    path: str
    #: raw_pxt | netcdf | processed_netcdf — what the file on disk is.
    representation: str
    #: Experiment record index parsed from the stem or the file header
    #: (``BP_0020`` -> 20); None when unparseable.
    experiment_index: int | None = None
    #: Dimension sizes read from the file header (PXT wave header / NetCDF
    #: metadata) without materialising any data block; None when unreadable.
    sizes: dict[str, int] | None = None

    @property
    def file_kind(self) -> str:
        """``"pxt"`` or ``"netcdf"``, derived from the representation."""
        return "pxt" if self.representation == REPRESENTATION_RAW_PXT else "netcdf"

    @property
    def processed(self) -> bool:
        """True for preprocessing products (``<stem>_processed.nc``)."""
        return self.representation == REPRESENTATION_PROCESSED_NETCDF

    @property
    def converted(self) -> bool:
        """True when a NetCDF exists for this entry (raw PXT needs conversion)."""
        return self.representation != REPRESENTATION_RAW_PXT

    def to_dict(self) -> dict[str, Any]:
        return {
            "stem": self.stem,
            "path": self.path,
            "representation": self.representation,
            "experiment_index": self.experiment_index,
            "sizes": self.sizes,
        }


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


def _auto_datasheet_payload(directory: Path) -> tuple[dict[str, Any] | None, Path | None]:
    """Translate the experiment's ``datasheet.csv`` once for a directory load.

    Returns ``(document, datasheet_path)`` or ``(None, None)`` when no
    readable datasheet exists.
    """
    for datasheet in _datasheet_candidates(directory):
        if not datasheet.exists():
            continue
        try:
            from peaksMCP.pxt_utils.csv_translator import translate_datasheet

            return translate_datasheet(datasheet).model_dump(mode="python"), datasheet
        except Exception:
            return None, None
    return None, None


def _scan_nc_header(
    path: Path,
) -> tuple[int | None, dict[str, Any] | None, dict[str, int] | None]:
    """Read a converted NetCDF's header: sizes, index and embedded document.

    NetCDF arrays carry ``sizes`` plus the ``experiment_index`` /
    ``experiment_metadata_json`` attributes; reading the header is cheap and
    never loads the data blocks.  Returns ``(experiment_index, document,
    sizes)`` where ``document`` is the full embedded metadata payload (all
    records) or None.
    """
    try:
        data, _ = _single(path, lazy=True)
    except Exception:
        return None, None, None
    try:
        sizes = dict(data.sizes) if data.sizes else None
        index = data.attrs.get("experiment_index")
        index = int(index) if isinstance(index, (int, float)) else index
        raw = data.attrs.get("experiment_metadata_json")
        document = json.loads(raw) if isinstance(raw, str) else (raw or {})
        if not isinstance(document, dict):
            document = {}
        return index, document, sizes
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


def _conversion_sibling_dir(folder: Path) -> Path | None:
    """The conversion target for a raw data folder: sibling <name>_netcdf."""
    return folder.parent / f"{folder.name}_netcdf"


def _merge_records_into(document: dict[str, Any], extra: dict[str, Any]) -> None:
    """Fill per-index records from ``extra`` into ``document`` in place.

    Existing records win (the earlier source — explicit metadata or the
    datasheet — is authoritative); records absent from ``document`` are
    filled from the embedded document.
    """
    if not isinstance(extra, dict):
        return
    records = document.setdefault("records", {})
    if not isinstance(records, dict):
        records = {}
        document["records"] = records
    for index, record in (extra.get("records") or {}).items():
        if index not in records:
            records[index] = record


class LoadedScans:
    """Lightweight experiment index: identity + metadata provenance.

    Whatever the file count, constructing the index reads no data blocks:
    entries carry stem / path / representation / experiment index / header
    sizes, plus the provenance of the experiment metadata document
    (datasheet / embedded / explicit / none — the document itself rides
    along for ``inspect_experiment``, never for classification here).
    ``scans[stem]`` loads that one file on demand (NetCDF lazily by default).
    The index object itself is the shared record left in the notebook.
    """

    def __init__(
        self,
        entries: list[ScanEntry],
        *,
        source: str,
        lazy: bool = False,
        metadata_document: dict[str, Any] | None = None,
        metadata_source: str = METADATA_SOURCE_NONE,
        metadata_path: str | None = None,
        duplicate_stems: list[str] | None = None,
    ) -> None:
        entries = sorted(entries, key=lambda entry: entry.stem)
        self.entries = entries
        self._by_stem = {entry.stem: entry for entry in entries}
        self.source = source
        self.lazy = lazy
        #: Stems that appeared as both raw ``.pxt`` and converted ``.nc`` in the
        #: SAME folder: the NetCDF entry is indexed (full geometry) and the
        #: collision is reported, never silently resolved.
        self.duplicate_stems = sorted(duplicate_stems or [])
        #: Translated/parsed experiment metadata document (all records) when
        #: one was found; the single classification owner (inspect_experiment)
        #: reads it together with entry sizes.  Not part of the model-facing
        #: index record (to_dict/summary) — provenance fields are.
        self.metadata_document = metadata_document
        #: "explicit" | "datasheet" | "embedded" | "none"
        self.metadata_source = metadata_source
        #: Path of the metadata source (datasheet.csv / explicit path); None
        #: for embedded or none.
        self.metadata_path = metadata_path
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

    # -- representation-derived views ---------------------------------------
    @property
    def raw_entries(self) -> list[ScanEntry]:
        """Original scans only (preprocessing products excluded)."""
        return [entry for entry in self.entries if not entry.processed]

    @property
    def processed(self) -> list[str]:
        """Preprocessing products in this folder (``*_processed.nc``)."""
        return [entry.stem for entry in self.entries if entry.processed]

    @property
    def converted(self) -> list[str]:
        """Entries already converted to NetCDF (incl. processed products)."""
        return [entry.stem for entry in self.entries if entry.converted]

    @property
    def needs_conversion(self) -> list[str]:
        """Raw PXT entries whose converted NetCDF is missing."""
        return [entry.stem for entry in self.entries if not entry.converted]

    @property
    def representation_counts(self) -> dict[str, int]:
        """Per-representation file counts: raw_pxt / netcdf / processed_netcdf."""
        counts = {REPRESENTATION_RAW_PXT: 0, REPRESENTATION_NETCDF: 0, REPRESENTATION_PROCESSED_NETCDF: 0}
        for entry in self.entries:
            counts[entry.representation] = counts.get(entry.representation, 0) + 1
        return counts

    # -- data layer (the extra dimension) ------------------------------------
    def __getitem__(self, stem: str) -> Any:
        """Load ONE file on demand (eager by default; PXT always eager)."""
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
        if "experiment_metadata_json" not in data.attrs and self.metadata_document:
            _attach_to(data, self.metadata_document, entry.experiment_index)
        self._cache[stem] = data
        return data

    def load(self, stem: str) -> Any:
        """Alias of ``scans[stem]`` for readability."""
        return self[stem]

    def summary_line(self) -> str:
        counts = self.representation_counts
        head = (
            f"load_data: {len(self.entries)} file(s) indexed from {self.source} "
            f"(raw_pxt={counts[REPRESENTATION_RAW_PXT]}, "
            f"netcdf={counts[REPRESENTATION_NETCDF]}, "
            f"processed={counts[REPRESENTATION_PROCESSED_NETCDF]})"
        )
        tail = "; classify with inspect_experiment(scans); data layer via scans[stem]"
        need = ""
        if self.needs_conversion:
            need = f"; {len(self.needs_conversion)} still need(s) conversion"
        dupes = ""
        if self.duplicate_stems:
            dupes = (
                f"; {len(self.duplicate_stems)} stem(s) had both .pxt and .nc "
                "(NetCDF indexed)"
            )
        meta = f"; metadata={self.metadata_source}"
        if self.metadata_path:
            meta += f" at {self.metadata_path}"
        reserved = len(need) + len(dupes) + len(tail)
        # Keep the line within 200 chars end to end: the metadata path is only
        # included when the whole line still fits (truncation must never cut
        # the actionable conversion/classification tail first).
        if self.metadata_path and len(head) + len(meta) + reserved > 200:
            meta = f"; metadata={self.metadata_source}"
        return (head + meta + need + dupes + tail)[:200]

    def __repr__(self) -> str:
        """print(exp) shows the identity summary (the agent's first read)."""
        return self.summary_line()

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "metadata_source": self.metadata_source,
            "metadata_path": self.metadata_path,
            "summary": self.summary_line(),
            "entries": [entry.to_dict() for entry in self.entries],
        }


def _attach_to(data: Any, document: dict[str, Any] | None, index: int | None) -> None:
    """Embed the experiment metadata document onto a loaded DataArray."""
    if document is None:
        return
    try:
        if "experiment_metadata_json" in data.attrs:
            return
        data.attrs["experiment_metadata_json"] = document
        if index is not None:
            data.attrs["experiment_index"] = index
    except Exception:
        pass  # object without attrs: metadata simply not embedded


def load_data(
    source: str | Path | list[str | Path] | tuple[str | Path, ...],
    *,
    lazy: bool = False,
    metadata: str | Path | dict[str, Any] | None = None,
) -> Any:
    """Load data for the agent: identity index first, then the data layer.

    - one ``.pxt`` / ``.nc`` file  -> a single peaks DataArray;
    - a folder or a path list      -> a :class:`LoadedScans` index: every
      file's identity (stem, path, representation raw_pxt/netcdf/
      processed_netcdf, experiment index, header sizes) plus metadata
      provenance, without reading any data block, and the data layer
      through ``scans[stem]`` (eager NetCDF by default).

    What each scan IS (gold/cut/mapping, decision lists, offsets, windows,
    shape conflicts) is classified by ``inspect_experiment(scans)`` — the
    single classification owner.  This facade never classifies.

    Metadata for the index comes from, in order: the explicit ``metadata``
    argument, a sibling ``datasheet.csv`` in the folder, or each NetCDF's
    embedded ``experiment_metadata_json`` (header only).  Raw PXT files
    without a datasheet carry path info only.

    Parameters
    ----------
    source : str, Path, list or tuple
        One file, one directory, or an explicit sequence of files.
    lazy : bool, default False
        ``True`` keeps NetCDF values dask-backed until accessed, for
        header-only inspection of many scans.  Analysis is eager by default:
        dask-backed values must be materialised (``da.load()``) before native
        Peaks numerics — ``fit_gold`` cannot fit a chunked array.
    metadata : str, Path or dict, optional
        Explicit experiment metadata document used for the index and attached
        to raw PXT arrays on load.

    Returns
    -------
    peaks DataArray or LoadedScans
        A DataArray for one file; a LoadedScans index for a folder/list.

    Raises
    ------
    ValueError
        For missing paths, empty sequences, folders without supported files,
        or an unsupported file type.

    Examples
    --------
    >>> scans = load_data("data/")            # index: identity + sizes only
    >>> summary = inspect_experiment(scans)   # classification lives here
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
        document = scans.metadata_document
        metadata_source = scans.metadata_source
        metadata_path = scans.metadata_path
        for folder, files in groups[1:]:
            part = _index_paths(
                files, lazy=lazy, metadata=metadata, directory=folder
            )
            merged_entries.extend(part.entries)
            if part.metadata_document is not None:
                if document is None:
                    document = part.metadata_document
                    metadata_source = part.metadata_source
                    metadata_path = part.metadata_path
                else:
                    # Fill per-index record gaps with later sources (e.g. the
                    # datasheet covers the raw folder; the converted sibling's
                    # embedded records fill indexes the datasheet lacks).
                    _merge_records_into(document, part.metadata_document)
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
            metadata_document=document,
            metadata_source=metadata_source,
            metadata_path=metadata_path,
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
    document: dict[str, Any] | None = None
    metadata_source = METADATA_SOURCE_NONE
    metadata_path: str | None = None
    if metadata is not None:
        if isinstance(metadata, dict):
            document = metadata
        else:
            try:
                from peaksMCP.pxt_utils.metadata import _load_metadata

                document = _load_metadata(metadata)
            except Exception:
                document = None
        if document is not None:
            metadata_source = METADATA_SOURCE_EXPLICIT
            metadata_path = (
                None if isinstance(metadata, dict) else str(Path(metadata).expanduser())
            )
    elif directory is not None:
        translated, datasheet = _auto_datasheet_payload(directory)
        if translated is not None:
            document = translated
            metadata_source = METADATA_SOURCE_DATASHEET
            metadata_path = str(datasheet)

    entries: list[ScanEntry] = []
    for path in paths:
        stem = path.stem
        file_kind = "netcdf" if path.suffix.lower() == ".nc" else "pxt"
        processed = bool(file_kind == "netcdf" and stem.endswith("_processed"))
        identity_stem = stem[: -len("_processed")] if processed else stem
        representation = (
            REPRESENTATION_PROCESSED_NETCDF
            if processed
            else (REPRESENTATION_NETCDF if file_kind == "netcdf" else REPRESENTATION_RAW_PXT)
        )
        experiment_index = _index_from_stem(identity_stem)
        sizes: dict[str, int] | None = None
        if file_kind == "netcdf":
            header_index, embedded_document, header_sizes = _scan_nc_header(path)
            sizes = header_sizes
            if header_index is not None:
                experiment_index = header_index
            if embedded_document:
                # The embedded document fills per-index records the earlier
                # source (explicit metadata / datasheet) did not provide;
                # provenance stays with the earlier source when it exists.
                if document is None:
                    document = embedded_document
                    metadata_source = METADATA_SOURCE_EMBEDDED
                    metadata_path = None
                else:
                    _merge_records_into(document, embedded_document)
        else:
            sizes = _scan_pxt_header_sizes(path)
        entries.append(
            ScanEntry(
                stem=stem,
                path=str(path),
                representation=representation,
                experiment_index=experiment_index,
                sizes=sizes,
            )
        )
    # A folder may hold both the raw scan and its conversion under the same
    # stem (``BP_0015.pxt`` next to ``BP_0015.nc``).  Only one entry can be
    # indexed per stem: the converted NetCDF wins, because the raw PXT loads
    # without the instrument geometry and a later ``k_convert`` would fail with
    # a confusing metadata error.  The collision is reported, never silent.
    by_stem: dict[str, ScanEntry] = {}
    duplicate_stems: list[str] = []
    for entry in sorted(entries, key=lambda item: (item.file_kind != "netcdf", item.path)):
        prior = by_stem.get(entry.stem)
        if prior is None:
            by_stem[entry.stem] = entry
            continue
        duplicate_stems.append(entry.stem)
        if entry.file_kind == "netcdf" and prior.file_kind != "netcdf":
            by_stem[entry.stem] = entry
    entries = list(by_stem.values())

    shown_source = directory.name if directory is not None else "sequence"
    scans = LoadedScans(
        entries,
        source=shown_source,
        lazy=lazy,
        metadata_document=document,
        metadata_source=metadata_source,
        metadata_path=metadata_path,
        duplicate_stems=duplicate_stems,
    )
    print(scans.summary_line())
    return scans
