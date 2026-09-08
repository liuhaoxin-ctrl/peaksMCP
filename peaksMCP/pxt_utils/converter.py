"""Atomic PXT-to-NetCDF conversion with experiment metadata embedding."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import xarray as xr

from peaksMCP.batch import BatchExecutor, ResourceBudget

from .csv_translator import translate_datasheet
from .loader import load_pxt
from .models import (
    ConversionItem,
    ConversionReport,
    ConversionTask,
    ExperimentMetadata,
)

_INDEX_RE = re.compile(r"(?:^|_)(\d+)$")


def _default_output_dir(source: Path) -> Path:
    """Default destination for a folder conversion: a sibling ``<name>_netcdf/`` folder.

    Parameters
    ----------
    source : pathlib.Path
        The source PXT folder.

    Returns
    -------
    pathlib.Path
        ``source.parent / f"{source.name}_netcdf"``.
    """
    return source.parent / f"{source.name}_netcdf"


def _discover_pxt_files(source: Path, substring: str = "") -> list[Path]:
    """Return top-level PXT files with case-insensitive extension matching."""
    return sorted(
        (
            path
            for path in source.iterdir()
            if path.is_file()
            and path.suffix.casefold() == ".pxt"
            and (not substring or substring in path.name)
        ),
        key=lambda path: (path.name.casefold(), path.name),
    )


def _index_from_path(path: str | Path) -> int | None:
    """Extract the trailing integer index from a PXT filename stem."""
    match = _INDEX_RE.search(Path(path).stem)
    return int(match.group(1)) if match else None


def _load_metadata(path: str | Path | None) -> ExperimentMetadata | None:
    if path is None:
        return None
    target = Path(path).expanduser().resolve()
    if not target.is_file():
        raise FileNotFoundError(f"experiment metadata file not found: {target}")
    return ExperimentMetadata.model_validate_json(target.read_text(encoding="utf-8"))


def _resolve_metadata_path(metadata_path: str | Path | None) -> str | None:
    """Validate and normalise an optional experiment-metadata path.

    Expands ``~``, resolves symlinks and fails fast with one clear error when the
    file is missing or is a directory — so a bad metadata path cannot surface as
    an identical per-file failure for every item in a batch.
    """
    if metadata_path is None or str(metadata_path) == "":
        return None
    target = Path(metadata_path).expanduser().resolve()
    if not target.is_file():
        raise FileNotFoundError(f"experiment metadata file not found: {target}")
    return str(target)


def _find_datasheet(source: Path) -> Path | None:
    """Locate a ``datasheet.csv`` living next to the data.

    The user keeps the datasheet and the raw data in the same folder, so for a
    folder input we check the folder itself and then its parent; for a single
    file we check the file's folder and then its parent.
    """
    roots = (
        [source.parent, source.parent.parent]
        if source.is_file()
        else [source, source.parent]
    )
    for root in roots:
        for name in ("datasheet.csv", "Datasheet.csv"):
            candidate = root / name
            if candidate.is_file():
                return candidate
    return None


def _auto_metadata(
    metadata_path: str | None,
    source: Path,
    destination: Path,
    report_warnings: list[str] | None = None,
) -> str | None:
    """Translate the sibling ``datasheet.csv`` when no explicit metadata was given.

    The translated document is written to ``destination/experiment_metadata.json``
    so it is reused across runs. A malformed/irrelevant CSV is ignored and the
    conversion proceeds without metadata rather than aborting.
    """
    if metadata_path is not None:
        return metadata_path
    datasheet = _find_datasheet(source)
    if datasheet is None:
        return None
    try:
        translated = translate_datasheet(datasheet)
        target = destination / "experiment_metadata.json"
        translated.write(target)
        return str(target)
    except Exception as exc:
        # Never silently drop translated metadata: duplicate Index, parse or
        # write failures must be visible so the user knows the conversion ran
        # without metadata and can fix the datasheet.
        import warnings

        message = (
            f"datasheet translation failed for {datasheet.name}: "
            f"{type(exc).__name__}: {exc}"
        )
        if report_warnings is not None:
            report_warnings.append(message)
        warnings.warn(
            message,
            UserWarning,
            stacklevel=2,
        )
        return None





def _record_comment(data: xr.DataArray) -> str:
    """Return the datasheet Comment text for this record, if any."""
    raw = data.attrs.get("experiment_metadata_json")
    if not raw:
        return ""
    try:
        record = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        return ""
    comment = record.get("experiment", {}).get("comment")
    return str(comment) if comment else ""


def _manipulator_json(angles: dict[str, float]) -> str:
    """Serialise the L112 manipulator model (polar/tilt/azi, deg) to JSON.

    Matches what peaks' NetCDF loader rebuilds for loc='L112'
    (``_manipulator_axes`` = polar/tilt/azi).
    """
    import pint
    from peaks.core.metadata.base_metadata_models import (
        AxisMetadataModelWithReference,
    )
    from pydantic import create_model

    ureg = pint.get_application_registry()
    fields = {
        axis: (AxisMetadataModelWithReference | None, None)
        for axis in ("polar", "tilt", "azi")
    }
    ManipulatorModel = create_model("ManipulatorMetadataModel", **fields)
    manip = ManipulatorModel(
        **{
            axis: AxisMetadataModelWithReference(
                value=0.0 * ureg.deg,
                reference_value=angles[axis] * ureg.deg,
            )
            for axis in ("polar", "tilt", "azi")
        }
    )
    return manip.model_dump_json(by_alias=True)

def _attach_peaks_metadata(data: xr.DataArray, source: Path) -> None:
    """Write the minimal peaks metadata models onto a converted DataArray.

    L112 PXT data are always analysed as NetCDF, so every converted file carries
    ``loc='L112'`` (registered DA30L geometry: slit axis = tilt, mapping axis =
    polar) and an analyser block with zero installation angles.  Loading through
    peaks then has full ``metadata.scan.loc`` / ``metadata.analyser`` and the
    normal loader path applies — no fallbacks needed.
    """
    from datetime import datetime

    from peaks.core.metadata.base_metadata_models import (
        ARPESMetadataModel,
        BaseScanMetadataModel,
    )

    # Avoid re-attaching on repeated runs of the same source path.
    if data.attrs.get("_scan") or data.attrs.get("metadata_models"):
        return

    scan_model = BaseScanMetadataModel(
        name=Path(source).stem,
        filepath=str(source),
        loc="L112",
        timestamp=datetime.now().astimezone().isoformat(),
    )
    data.attrs["_scan"] = scan_model.model_dump_json(by_alias=True)

    # _manipulator: three rotation angles default to 0, unless the record
    # comment carries explicit angles ("polar=12 tilt=-3 ..."); other comment
    # text is ignored.
    comment = _record_comment(data)
    angles = {}
    for axis in ("polar", "tilt", "azi"):
        match = re.search(rf"\b{axis}\s*=\s*(-?\d+(?:\.\d+)?)", comment)
        angles[axis] = float(match.group(1)) if match else 0.0
    data.attrs["_manipulator"] = _manipulator_json(angles)

    analyser = ARPESMetadataModel()
    # Zero installation angles (numeric: the Quantity validator rejects strings).
    # azi = 0 keeps slit along tilt -> theta_par in the tilt group.
    analyser.angles.polar = 0
    analyser.angles.tilt = 0
    analyser.angles.azi = 0
    data.attrs["_analyser"] = analyser.model_dump_json(by_alias=True)

    data.attrs["metadata_models"] = json.dumps(
        {
            "_scan": (
                "peaks.core.metadata.base_metadata_models.BaseScanMetadataModel"
            ),
            "_analyser": "peaks.core.metadata.base_metadata_models.ARPESMetadataModel",
            "_manipulator": (
                "peaks.core.fileIO.base_data_classes.base_manipulator_class."
                "ManipulatorMetadataModel"
            ),
        }
    )

def _safe_attrs(attributes: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in attributes.items():
        if isinstance(value, (str, int, float)) or value is None:
            safe[key] = "" if value is None else value
        else:
            safe[key] = json.dumps(value, ensure_ascii=False, default=str)
    return safe


def _temporary_output(target: Path) -> Path:
    """Reserve a unique temporary file in the destination filesystem."""
    descriptor, name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".part",
        dir=target.parent,
    )
    os.close(descriptor)
    return Path(name)


def _validate_output_target(source: Path, target: Path) -> None:
    """Reject raw-data destinations, including symlink and hard-link aliases."""
    resolved = target.resolve()
    if resolved == source or (source.exists() and resolved.exists() and source.samefile(resolved)):
        raise ValueError("NetCDF output must not refer to the input PXT file")
    if target.suffix.lower() != ".nc" or resolved.suffix.lower() != ".nc":
        raise ValueError("NetCDF output must use the .nc extension; raw PXT files cannot be overwritten")
    if resolved.is_dir():
        raise ValueError("NetCDF output must be a file, not a directory")


def _publish_output(temporary: Path, target: Path, *, force: bool) -> bool:
    """Atomically publish a completed NetCDF without violating overwrite policy.

    ``force=True`` uses ``os.replace`` so an existing valid output remains in
    place until the new file is complete. Without force, a same-filesystem hard
    link provides an atomic create-if-absent operation; it returns ``False`` if
    another conversion published the target while this worker was running.
    """
    if force:
        os.replace(temporary, target)
        return True
    try:
        os.link(temporary, target)
    except FileExistsError:
        return False
    try:
        temporary.unlink()
    except OSError:
        # The target already references the complete inode; failure to remove the
        # private staging name must not turn a successful conversion into failure.
        pass
    return True


def convert_pxt(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    metadata_path: str | Path | None = None,
    force: bool = False,
) -> ConversionItem:
    """Convert one PXT file to NetCDF and embed its matching metadata record.

    Parameters
    ----------
    input_path : path-like
        Source PXT file, which is never modified.
    output_path : path-like, optional
        Destination NetCDF file ending in ``.nc``. Defaults to the source stem
        with ``.nc``. It must not alias the input, including through links.
    metadata_path : path-like, optional
        Translated ``experiment_metadata.json`` document.
    force : bool, default False
        Replace an existing NetCDF destination only when explicitly enabled.
        Input-file protection cannot be overridden.

    Returns
    -------
    ConversionItem
        Structured success or failure information.

    Examples
    --------
    >>> result = convert_pxt("BP_0005.pxt", "BP_0005.nc", metadata_path="experiment_metadata.json")
    >>> result.status in {"converted", "skipped", "failed"}
    True
    """
    source = Path(input_path).expanduser().resolve()
    requested_target = Path(output_path).expanduser() if output_path else source.with_suffix(".nc")
    target = requested_target.resolve()
    index = _index_from_path(source)
    temporary: Path | None = None
    try:
        # Check safety before both loading data and the existing-output shortcut.
        _validate_output_target(source, requested_target)
        if target.exists() and not force:
            return ConversionItem(
                input=str(source),
                output=str(target),
                index=index,
                status="skipped",
                warnings=["output exists"],
            )
        data = load_pxt(source)
        document = _load_metadata(metadata_path)
        warnings: list[str] = []
        if document is not None:
            record = document.records.get(str(index)) if index is not None else None
            if record is None:
                warnings.append(f"no metadata record for Index {index}")
            else:
                payload = record.model_dump(mode="json")
                data.attrs["experiment_metadata_json"] = json.dumps(
                    payload, ensure_ascii=False
                )
                data.attrs["experiment_index"] = record.index
                data.attrs["experiment_title"] = document.title
                data.attrs["experiment_source_sha256"] = document.source_sha256
        _attach_peaks_metadata(data, source)
        data.attrs = _safe_attrs(dict(data.attrs))
        for coordinate in data.coords.values():
            coordinate.attrs = _safe_attrs(dict(coordinate.attrs))
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = _temporary_output(target)
        data.to_netcdf(temporary, engine="h5netcdf")
        # A target may have changed while the data was being serialized.
        _validate_output_target(source, target)
        if not _publish_output(temporary, target, force=force):
            temporary.unlink()
            return ConversionItem(
                input=str(source),
                output=str(target),
                index=index,
                status="skipped",
                warnings=["output was created by another conversion"],
            )
        temporary = None
        return ConversionItem(
            input=str(source),
            output=str(target),
            index=index,
            status="converted",
            warnings=warnings,
        )
    except Exception as exc:
        if temporary is not None and temporary.exists():
            temporary.unlink()
        return ConversionItem(
            input=str(source),
            output=str(target),
            index=index,
            status="failed",
            error_type=type(exc).__name__,
            error=str(exc),
        )


def _convert_task(task: ConversionTask) -> dict[str, Any]:
    return convert_pxt(
        task.input_path,
        task.output_path,
        metadata_path=task.metadata_path,
        force=task.force,
    ).model_dump(mode="json")


def convert_path(
    input_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    metadata_path: str | Path | None = None,
    substring: str = "",
    force: bool = False,
    cpu_limit_percent: float = 60.0,
) -> ConversionReport:
    """Convert one PXT file or a filtered folder using bounded parallelism.

    Parameters
    ----------
    input_path : path-like
        Source PXT file or directory containing PXT files.
    output_dir : path-like, optional
        Destination file for a single input or destination directory for a batch.
    metadata_path : path-like, optional
        Translated experiment metadata JSON document.
    substring : str, default ""
        Filename substring used to filter a directory batch.
    force : bool, default False
        Permit replacing existing NetCDF outputs.
    cpu_limit_percent : float, default 60
        System CPU threshold above which new work is not submitted.

    Returns
    -------
    ConversionReport
        Per-item outcomes plus aggregate CPU and duration statistics.

    Examples
    --------
    >>> report = convert_path("raw/", "netcdf/", substring="BP_", cpu_limit_percent=60)
    >>> isinstance(report.items, list)
    True
    """
    source = Path(input_path).expanduser().resolve()
    report_warnings: list[str] = []
    # Validate the optional metadata file once, before any conversion runs, so a
    # missing metadata document aborts with a single clear error instead of a
    # per-file failure for the whole batch.
    metadata_path = _resolve_metadata_path(metadata_path)
    if source.is_file():
        destination = Path(output_dir).expanduser().resolve() if output_dir else source.parent
        # The auto-translated metadata lives next to the data (never derived from
        # an explicit single-file output path, which could be a .nc file).
        metadata_path = _auto_metadata(
            metadata_path, source, source.parent, report_warnings
        )
        target = (
            destination / f"{source.stem}.nc"
            if destination.is_dir() or not destination.suffix
            else destination
        )
        return ConversionReport(
            items=[
                convert_pxt(
                    source,
                    target,
                    metadata_path=metadata_path,
                    force=force,
                )
            ],
            warnings=report_warnings,
        )
    if not source.is_dir():
        raise FileNotFoundError(source)
    # Default output is a sibling folder named after the source folder
    # (e.g. raw/ -> raw_netcdf/), created on demand.  An explicit output_dir
    # is honoured as-is.
    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir
        else _default_output_dir(source)
    )
    metadata_path = _auto_metadata(
        metadata_path, source, destination, report_warnings
    )
    files = _discover_pxt_files(source, substring)
    tasks = [
        ConversionTask(
            input_path=str(path),
            output_path=str(destination / f"{path.stem}.nc"),
            metadata_path=metadata_path,
            force=force,
        )
        for path in files
    ]
    budget = ResourceBudget(
        cpu_limit_percent=cpu_limit_percent,
        resume_percent=max(0, cpu_limit_percent - 10),
    )
    batch = BatchExecutor(budget).run(_convert_task, tasks)
    items: list[ConversionItem] = []
    for result in batch.items:
        if result.status == "completed":
            item = ConversionItem.model_validate(result.output)
            item.output_exists = bool(item.output and os.path.exists(item.output))
            items.append(item)
        else:
            task = result.input
            items.append(
                ConversionItem(
                    input=task.input_path,
                    output=task.output_path,
                    index=_index_from_path(task.input_path),
                    status="failed" if result.status == "failed" else result.status,
                    error_type=result.error_type,
                    error=result.error,
                    output_exists=os.path.exists(task.output_path),
                )
            )
    return ConversionReport(
        items=items,
        warnings=report_warnings,
        cpu={
            "average_percent": batch.average_cpu_percent,
            "peak_percent": batch.peak_cpu_percent,
            "peak_moving_average_percent": batch.peak_moving_average_cpu_percent,
            "budget_percent": batch.cpu_budget_percent,
            "budget_strategy": batch.cpu_budget_strategy,
            "budget_exceeded": batch.cpu_budget_exceeded,
            "duration_s": batch.duration_s,
        },
    )
