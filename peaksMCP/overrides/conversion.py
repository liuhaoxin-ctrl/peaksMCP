"""Thin PXT-to-NetCDF compatibility boundary: pure conversion + consented
publication.  No analysis freedom: one file is converted by repeating the
single conversion verb per file; nothing else happens here.

``convert_experiment`` never writes by itself and never prints: it computes
the converted arrays (reading raw PXT, embedding the datasheet/metadata
record), stages every output in the unified staging area (strict TTL; the
target directory is created only at publish time, never early), shows ONE
consent card whose manifest lists every item as source -> target with
dims/dtype/units/warnings, and only a human approval atomically publishes
the staged bytes.  The returned ``ConversionReport`` is the outcome; nothing
is echoed to stdout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import save as save_module


def _load_document(metadata: Any) -> Any | None:
    """Read an explicit experiment-metadata JSON path into its model."""
    if metadata is None:
        return None
    from peaksMCP.pxt_utils.models import ExperimentMetadata

    target = Path(metadata).expanduser()
    if not target.is_file():
        raise ValueError(
            f"convert_experiment: metadata file not found: {metadata}."
        )
    return ExperimentMetadata.model_validate_json(target.read_text(encoding="utf-8"))


def _auto_document(source: Path) -> Any | None:
    """Translate the sibling datasheet once (pure; no file is written)."""
    from peaksMCP.pxt_utils.converter import _find_datasheet
    from peaksMCP.pxt_utils.csv_translator import translate_datasheet

    datasheet = _find_datasheet(source)
    if datasheet is None:
        return None
    try:
        return translate_datasheet(datasheet)
    except Exception:
        return None


def _plan_targets(
    source: Path,
    output_dir: Path | None,
) -> tuple[list[Path], Path | None]:
    """Resolve the source files and their output directory.

    A file input converts just it; a directory input converts every matching
    ``.pxt`` inside.  Default outputs follow the converter conventions:
    single file -> ``<stem>.nc`` next to the source (or in ``output_dir``);
    folder -> sibling ``<folder>_netcdf/`` (or ``output_dir``).
    """
    files: list[Path] = []
    destination: Path | None
    if source.is_file():
        files = [source]
        if output_dir is None:
            destination = source.parent
        else:
            destination = output_dir
    else:
        files = sorted(
            item
            for item in source.iterdir()
            if item.is_file()
            and item.suffix.lower() == ".pxt"
            and not item.name.startswith(".")
        )
        destination = output_dir or source.parent / f"{source.name}_netcdf"
    return files, destination


def convert_experiment(
    source: str | Path,
    *,
    output_dir: str | Path | None = None,
    metadata: str | Path | None = None,
    match: str = "",
    overwrite: bool = False,
):
    """Convert one PXT file or a whole folder to NetCDF under human consent.

    Thin compatibility boundary: pure PXT -> NetCDF conversion (one verb,
    repeated per file - no analysis freedom) plus consented publication.
    Each scan is read and prepared (metadata record embedded); every output
    - including the translated ``experiment_metadata.json`` when a datasheet
    was found - is staged in the unified staging area (strict TTL) and ONE
    consent card shows the conversion manifest (source -> target, dims,
    dtype, units, warnings).  Nothing is written unless the user approves
    the card; existing targets are skipped idempotently unless
    ``overwrite=True``.  Prints nothing; the ``ConversionReport`` is the
    outcome.

    Parameters
    ----------
    source : str or Path
        One ``.pxt`` file, or a directory containing PXT files.
    output_dir : str or Path, optional
        Destination directory for the NetCDF outputs (and the metadata JSON).
        Never created early - only at publish time, after approval.
    metadata : str or Path, optional
        Explicit ``experiment_metadata.json`` document (otherwise a sibling
        ``datasheet.csv`` is translated automatically).
    match : str, default ""
        Filename substring used to filter a directory batch.
    overwrite : bool, default False
        Permit replacing existing outputs (gateway overwrite policy, only
        effective after approval).

    Returns
    -------
    peaksMCP.pxt_utils.models.ConversionReport
        Per-item outcomes.  ``status`` is ``converted`` only after the human
        approved and the gateway published the staged bytes; ``skipped`` for
        idempotent skips; ``awaiting_consent``/``denied`` when no approval
        happened.  JSON-safe via ``model_dump(mode="json")``.

    Raises
    ------
    ValueError
        When the source does not exist.
    """
    from peaksMCP.pxt_utils.converter import (
        _converted_array,
        _index_from_path,
    )
    from peaksMCP.pxt_utils.models import ConversionItem, ConversionReport

    source_path = Path(source).expanduser()
    if not source_path.exists():
        raise ValueError(
            f"convert_experiment: source not found: {source_path}. "
            "Check the path before retrying."
        )
    explicit = _load_document(metadata)
    document = explicit if explicit is not None else _auto_document(source_path)
    files, destination = _plan_targets(source_path, Path(output_dir).expanduser() if output_dir else None)
    if match:
        files = [file for file in files if match in file.name]

    def target_for(file: Path) -> Path:
        if source_path.is_file():
            if output_dir is None:
                return file.with_suffix(".nc")
            return destination / f"{file.stem}.nc"
        return destination / f"{file.stem}.nc"

    items: list[ConversionItem] = []
    staged: list[tuple[ConversionItem, Path]] = []
    requests: list[tuple[Any, Path, bool]] = []
    planned = 0
    for file in files:
        target = target_for(file)
        if target.exists() and not overwrite:
            items.append(
                ConversionItem(
                    input=str(file), output=str(target), status="skipped",
                    output_exists=True, warnings=["output exists"],
                )
            )
            continue
        planned += 1
        try:
            index = _index_from_path(file)
            data, warnings = _converted_array(file, index, document)
        except Exception as exc:
            items.append(
                ConversionItem(
                    input=str(file), output=str(target), status="failed",
                    error_type=type(exc).__name__, error=str(exc),
                )
            )
            continue
        item = ConversionItem(
            input=str(file), output=str(target), index=index,
            status="awaiting_consent", warnings=warnings,
        )
        staged.append((item, target))
        requests.append(
            (data, target, overwrite, {"input": str(file), "warnings": warnings})
        )
        items.append(item)
    # The translated metadata JSON joins the SAME consented batch (no report
    # row of its own - it is conversion plumbing).
    metadata_target: Path | None = None
    if document is not None and planned and destination is not None:
        metadata_target = destination / "experiment_metadata.json"
        if not metadata_target.exists() or overwrite:
            requests.append(
                (document.model_dump(mode="python"), metadata_target, overwrite)
            )

    report = ConversionReport(items=items, cpu={}, warnings=[])
    if not staged:
        return report

    summary = f"convert_experiment: publish {len(requests)} file(s) to {destination}"
    outcome = save_module._run_staged("convert_experiment", requests, summary)
    status = outcome["status"]
    if status == "saved":
        published = {p["path"] for p in outcome.get("published", [])}
        skipped_now = {s["path"] for s in outcome.get("skipped", [])}
        for item, target in staged:
            key = str(target)
            if key in skipped_now:
                item.status = "skipped"
                item.output_exists = True
                item.warnings = item.warnings + ["output exists at publish"]
            elif key in published:
                item.status = "converted"
                item.output_exists = True
            else:
                item.status = "converted"
                item.output_exists = True
    elif status == "pending_consent":
        for item, _target in staged:
            item.status = "awaiting_consent"
            item.output_exists = False
    else:
        for item, _target in staged:
            item.status = "denied"
            item.output_exists = False
    return report
