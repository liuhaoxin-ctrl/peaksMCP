"""Unified data loading facade.

One entry for the agent to load ANY data — any number of files, any
supported type — from a notebook:

    data       = load_data("BP_0020.nc")            # one NetCDF -> DataArray
    raw        = load_data("raw/BP_0020.pxt")       # one PXT    -> DataArray
    scans      = load_data("data/")                 # folder     -> {stem: DataArray}
    scans      = load_data("data_netcdf/")          # already-converted folder
    subset     = load_data(["a.nc", "b.nc"])        # explicit list -> same mapping

Rules:
- Single file returns a DataArray; a folder or a list returns a
  ``{stem: DataArray}`` mapping (sorted, so data/ and data_netcdf/ load in
  one call). NetCDF stays lazy by default.
- A sibling ``datasheet.csv`` in a folder is translated automatically and
  attached to raw PXT arrays; arrays that already carry converted metadata
  are left untouched.
- PXT input is read through the internal PXT reader; NetCDF through
  ``peaks.load`` (L112 geometry registered).
- Optional explicit ``metadata`` (path or dict) is embedded into
  ``attrs["experiment_metadata_json"]``.
- The original files are never modified.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .models import Report


class LoadReport(Report):
    """Result of load_data: which file, how it was read, dims."""

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


_SUPPORTED_SUFFIXES = {".pxt", ".nc"}


def _index_from_stem(stem: str) -> int | None:
    """Parse a trailing index out of a filename stem (``BP_0020`` -> 20)."""
    match = re.search(r"(?:^|_|-)(\d+)\s*$", stem)
    return int(match.group(1)) if match else None


def _single(path: Path, lazy: bool) -> tuple[Any, str]:
    """Load one supported file into a DataArray (delegation seam for tests).

    peaks.load prints a per-file Markdown line on the lazy path; it is
    swallowed here so a folder load stays quiet (load_data prints its own
    one-line summary). Loading itself is unaffected.
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


def _auto_datasheet_payload(directory: Path) -> dict[str, Any] | None:
    """Translate a sibling ``datasheet.csv`` once for a directory load.

    Mirrors the converter's sibling-datasheet discovery so a raw experiment
    folder loads with its record table attached before any conversion ran.
    """
    for name in ("datasheet.csv", "Datasheet.csv"):
        datasheet = directory / name
        if not datasheet.exists():
            continue
        try:
            from peaksMCP.pxt_utils.csv_translator import translate_datasheet

            return translate_datasheet(datasheet).model_dump(mode="python")
        except Exception:
            return None
    return None


def _attach_to(data: Any, payload: dict[str, Any] | None, index: int | None) -> None:
    """Embed the experiment metadata document onto a loaded DataArray.

    Arrays that already carry converted metadata (NetCDF loaded from a
    conversion) are left untouched; raw PXT arrays get the document and,
    when the index is known, an ``experiment_index`` attribute.
    """
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
    """Load data for the agent: one file, many files, or a whole folder.

    Unified loading entry that accepts any number of supported inputs and
    returns what matches the shape of the request:

    - one ``.pxt`` / ``.nc`` file  -> a single peaks DataArray;
    - a folder                    -> ``{stem: DataArray, ...}`` for every
      ``.pxt`` / ``.nc`` inside it (sorted, so a directory like ``data/`` or
      ``data_netcdf/`` loads in one call); a sibling ``datasheet.csv`` is
      translated automatically and attached to raw PXT arrays (arrays that
      already carry converted metadata are left untouched);
    - a list/tuple of paths       -> same mapping as the folder case.

    Parameters
    ----------
    source : str, Path, list or tuple
        One file, one directory, or an explicit sequence of files.
    lazy : bool, default True
        Passed to the underlying reader (NetCDF keeps chunks lazy).
    metadata : str, Path or dict, optional
        Explicit experiment metadata (``experiment_metadata.json``) embedded
        into ``attrs["experiment_metadata_json"]`` for every loaded array.

    Returns
    -------
    peaks DataArray or dict of DataArray
        A single DataArray for one file; a ``{stem: DataArray}`` mapping for
        a folder or a sequence of files.

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
        return _load_many(paths, lazy=lazy, metadata=metadata)

    path = Path(source).expanduser()
    if not path.exists():
        raise ValueError(
            f"load_data: path not found: {path}. Check the path before retrying."
        )
    if path.is_dir():
        return _load_many(
            sorted(
                item for item in path.iterdir()
                if item.is_file() and item.suffix.lower() in _SUPPORTED_SUFFIXES
            ),
            lazy=lazy,
            metadata=metadata,
            directory=path,
        )
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


def _load_many(
    paths: list[Path],
    *,
    lazy: bool,
    metadata: Any,
    directory: Path | None = None,
) -> dict[str, Any]:
    """Load a set of paths into a ``{stem: DataArray}`` mapping."""
    if not paths:
        where = f" in {directory}" if directory is not None else ""
        raise ValueError(
            f"load_data: no supported data files (.pxt, .nc){where}."
        )
    payload = metadata
    if payload is not None and not isinstance(payload, dict):
        try:
            from peaksMCP.pxt_utils.metadata import _load_metadata

            payload = _load_metadata(payload)
        except Exception:
            payload = None
    elif payload is None and directory is not None:
        payload = _auto_datasheet_payload(directory)

    loaded: dict[str, Any] = {}
    pxt_count = nc_count = 0
    failed: list[str] = []
    for path in paths:
        try:
            data, kind = _single(path, lazy)
        except Exception as exc:
            failed.append(f"{path.name} ({type(exc).__name__}: {exc})")
            continue
        if kind == "PXT":
            pxt_count += 1
            index = _index_from_stem(path.stem)
            _attach_to(data, payload, index)
        else:
            nc_count += 1
            _attach_to(data, payload, _index_from_stem(path.stem))
        loaded[path.stem] = data
    if not loaded:
        raise ValueError(
            "load_data: none of the requested files could be loaded: "
            + "; ".join(failed[:5])
        )
    where = directory or ("sequence" if len(paths) > 1 else str(paths[0]))
    names = sorted(loaded)
    if len(names) <= 12:
        keys_text = "{" + ", ".join(names) + "}"
    else:
        keys_text = (
            "{" + ", ".join(names[:5]) + ", ..., " + names[-1]
            + f"}} ({len(names)} keys)"
        )
    # First line = the agent's cognition channel (short, echoed to the model):
    # what was loaded, where, and where the mapping lives.
    print(
        f"load_data: {len(loaded)} file(s) loaded from {where} "
        f"({pxt_count} pxt, {nc_count} netcdf); returned dict {keys_text}"
    )
    # Following rows = the notebook archive for the user (per-file identities).
    for path in paths:
        data = loaded.get(path.stem)
        if data is None:
            continue
        kind = "pxt" if path.suffix.lower() == ".pxt" else "netcdf"
        shape = ", ".join(f"{dim}×{size}" for dim, size in data.sizes.items())
        note = _identity_note(payload, _index_from_stem(path.stem))
        print(f"  {path.stem:<16s} {kind:<7s} ({shape}){note}")
    if failed:
        print(f"load_data: skipped {len(failed)} file(s): {'; '.join(failed[:3])}")
    return loaded


def _identity_note(payload: dict[str, Any] | None, index: int | None) -> str:
    """Datasheet identity for one file's archive row (index, data format)."""
    if payload is None or index is None:
        return ""
    try:
        record = (payload.get("records") or {}).get(str(index))
        data_format = str(((record or {}).get("experiment") or {}).get("data_format") or "")
    except Exception:
        return ""
    parts = [f"index={index}"]
    if data_format:
        parts.append(data_format)
    return "  " + "  ".join(parts)
