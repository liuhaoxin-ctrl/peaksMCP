"""Atomic PXT-to-NetCDF conversion with experiment metadata embedding."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from peaksMCP.batch import BatchExecutor, ResourceBudget

from .loader import load_pxt
from .models import (
    ConversionItem,
    ConversionReport,
    ConversionTask,
    ExperimentMetadata,
)

_INDEX_RE = re.compile(r"(?:^|_)(\d+)$")


def index_from_path(path: str | Path) -> int | None:
    """Extract the trailing integer index from a PXT filename stem."""
    match = _INDEX_RE.search(Path(path).stem)
    return int(match.group(1)) if match else None


def _load_metadata(path: str | Path | None) -> ExperimentMetadata | None:
    if path is None:
        return None
    return ExperimentMetadata.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _safe_attrs(attributes: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in attributes.items():
        if isinstance(value, (str, int, float)) or value is None:
            safe[key] = "" if value is None else value
        else:
            safe[key] = json.dumps(value, ensure_ascii=False, default=str)
    return safe


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
        Destination NetCDF file. Defaults to the source stem with ``.nc``.
    metadata_path : path-like, optional
        Translated ``experiment_metadata.json`` document.
    force : bool, default False
        Replace an existing destination only when explicitly enabled.

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
    target = (
        Path(output_path).expanduser().resolve()
        if output_path
        else source.with_suffix(".nc")
    )
    index = index_from_path(source)
    if target.exists() and not force:
        return ConversionItem(
            input=str(source),
            output=str(target),
            index=index,
            status="skipped",
            warnings=["output exists"],
        )
    temporary = target.with_suffix(target.suffix + ".part")
    try:
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
        data.attrs = _safe_attrs(dict(data.attrs))
        for coordinate in data.coords.values():
            coordinate.attrs = _safe_attrs(dict(coordinate.attrs))
        target.parent.mkdir(parents=True, exist_ok=True)
        if temporary.exists():
            temporary.unlink()
        data.to_netcdf(temporary, engine="h5netcdf")
        if target.exists() and force:
            target.unlink()
        temporary.replace(target)
        return ConversionItem(
            input=str(source),
            output=str(target),
            index=index,
            status="converted",
            warnings=warnings,
        )
    except Exception as exc:
        if temporary.exists():
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
    if source.is_file():
        if output_dir:
            destination = Path(output_dir).expanduser().resolve()
            # A directory destination (existing, or a path with no extension)
            # receives the source stem; an explicit file path is used as-is.
            target = (
                destination / f"{source.stem}.nc"
                if destination.is_dir() or not destination.suffix
                else destination
            )
        else:
            target = source.with_suffix(".nc")
        return ConversionReport(
            items=[
                convert_pxt(
                    source,
                    target,
                    metadata_path=metadata_path,
                    force=force,
                )
            ]
        )
    if not source.is_dir():
        raise FileNotFoundError(source)
    destination = Path(output_dir).expanduser().resolve() if output_dir else source
    files = [
        path
        for path in sorted(source.glob("*.pxt"))
        if not substring or substring in path.name
    ]
    tasks = [
        ConversionTask(
            input_path=str(path),
            output_path=str(destination / f"{path.stem}.nc"),
            metadata_path=str(Path(metadata_path).resolve()) if metadata_path else None,
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
            items.append(ConversionItem.model_validate(result.output))
        else:
            task = result.input
            items.append(
                ConversionItem(
                    input=task.input_path,
                    output=task.output_path,
                    index=index_from_path(task.input_path),
                    status="failed" if result.status == "failed" else result.status,
                    error_type=result.error_type,
                    error=result.error,
                )
            )
    return ConversionReport(
        items=items,
        cpu={
            "average_percent": batch.average_cpu_percent,
            "peak_percent": batch.peak_cpu_percent,
            "duration_s": batch.duration_s,
        },
    )
