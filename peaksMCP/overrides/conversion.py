"""Unified conversion facade: one entry for a single file or a folder.

``convert_experiment`` routes every conversion through the same CPU-budgeted,
failure-isolated path and returns a :class:`ConversionReport` (JSON-safe via
``model_dump(mode="json")``) with per-item status, ``output_exists``, errors
and aggregate CPU statistics.  It is the model-facing conversion verb; the
lower-level ``convert_pxt`` / ``convert_path`` stay importable (advanced
tier) but carry no separate search aliases.
"""

from __future__ import annotations

from pathlib import Path


def convert_experiment(
    source: str | Path,
    *,
    output_dir: str | Path | None = None,
    metadata: str | Path | None = None,
    match: str = "",
    force: bool = False,
    cpu_limit_percent: float = 60.0,
):
    """Convert one PXT file or a whole folder to NetCDF.

    Unified entry: a file input converts that one scan (idempotently), a
    directory input converts every matching PXT file with bounded CPU
    parallelism.  Outputs are written atomically next to the source (or into
    ``output_dir``) and the source is never modified.

    Parameters
    ----------
    source : str or Path
        One ``.pxt`` file, or a directory containing PXT files.
    output_dir : str or Path, optional
        Destination directory (single-file and batch inputs share the rule).
    metadata : str or Path, optional
        Translated ``experiment_metadata.json`` document (the converter
        embeds each index's record into the matching NetCDF).
    match : str, default ""
        Filename substring used to filter a directory batch.
    force : bool, default False
        Replace an existing NetCDF output only when explicitly enabled.
    cpu_limit_percent : float, default 60
        System CPU threshold above which no new work is submitted.

    Returns
    -------
    peaksMCP.pxt_utils.models.ConversionReport
        Per-item outcomes (status, ``output_exists``, errors) plus aggregate
        CPU and duration statistics.  JSON-safe: ``report.model_dump(mode="json")``.

    Raises
    ------
    ValueError
        When the source does not exist or the metadata document is missing.
    """
    from peaksMCP.pxt_utils.converter import convert_path

    source_path = Path(source).expanduser()
    if not source_path.exists():
        raise ValueError(
            f"convert_experiment: source not found: {source_path}. "
            "Check the path before retrying."
        )
    if metadata is not None and not Path(metadata).expanduser().exists():
        raise ValueError(
            f"convert_experiment: metadata file not found: {metadata}."
        )
    report = convert_path(
        source_path,
        output_dir,
        metadata_path=metadata,
        substring=match,
        force=force,
        cpu_limit_percent=cpu_limit_percent,
    )
    converted = sum(item.status == "converted" for item in report.items)
    skipped = sum(item.status == "skipped" for item in report.items)
    failed = sum(item.status == "failed" for item in report.items)
    print(
        f"convert_experiment: {len(report.items)} input(s) - "
        f"{converted} converted, {skipped} skipped, {failed} failed"
    )
    return report
