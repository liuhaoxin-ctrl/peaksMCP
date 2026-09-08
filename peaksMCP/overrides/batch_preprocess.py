"""Path-driven batch preprocessing with per-item isolation.

``preprocess_batch`` processes scans one at a time from their paths — load,
preprocess (cut or mapping), atomic NetCDF save, release — under the shared
CPU budget, so the kernel never holds every experiment in memory.  Each item
succeeds or fails independently; worker crashes and CPU-wait timeouts never
terminate the other items.

The per-item writes are batch-task outputs of a user-requested processing
run (like the conversion facade), not model-initiated result saves: the
destination directory is caller-supplied and every file is written
atomically.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from .calibration import GoldCalibration


class BatchPreprocessItem(BaseModel):
    """One batch input: a path to process plus its geometry parameters."""

    index: int | str
    source: str
    kind: Literal["cut", "mapping"]
    output: str = ""
    # cut: high-symmetry offset in degrees.
    theta_par_offset_deg: float | None = None
    # mapping: normal-emission reference angles, e.g. {"theta_par": 0.0, "polar": 0.0}.
    normal_emission: dict[str, float] | None = None
    # Optional (start, stop, step) slices forwarded to k_convert.
    eV: tuple | None = None
    kx: tuple | None = None
    ky: tuple | None = None
    quiet: bool = True


class BatchPreprocessItemResult(BaseModel):
    """Outcome of one batch item (JSON-safe; no DataArray/Figure)."""

    index: int | str
    source: str
    status: str
    output: str | None = None
    output_exists: bool = False
    error_type: str | None = None
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)
    duration_s: float = 0.0


class BatchProcessingReport(BaseModel):
    """Aggregate batch outcome plus per-item statuses."""

    items: list[BatchPreprocessItemResult] = Field(default_factory=list)
    cpu: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    @property
    def completed(self) -> int:
        return sum(item.status == "completed" for item in self.items)

    @property
    def failed(self) -> int:
        return sum(item.status == "failed" for item in self.items)

    @property
    def skipped(self) -> int:
        return sum(item.status in {"skipped", "cancelled"} for item in self.items)


def _slice_of(value: tuple | None) -> Any:
    """Convert a JSON-safe (start, stop, step) tuple to a python slice."""
    if value is None:
        return None
    return slice(*value)


def _load_array(source: str) -> Any:
    """Load one converted NetCDF scan with the L112 geometry registered."""
    from peaksMCP.pxt_utils.loader import _register_l112_loader

    try:
        _register_l112_loader()
    except Exception:
        pass  # native peaks may already know the location
    from peaks import load

    return load(source)


def _atomic_netcdf(data: Any, output: str) -> None:
    """Serialize one DataArray to NetCDF and publish it atomically."""
    from .save import _atomic_write_bytes

    buffer = io.BytesIO()
    data.to_netcdf(buffer)
    _atomic_write_bytes(Path(output), buffer.getvalue())


def _run_item(
    item: BatchPreprocessItem,
    calibration: Any,
    force: bool = False,
) -> BatchPreprocessItemResult:
    """Load, preprocess and save one item.  Runs inside a worker process, so
    every dependency is imported locally; exceptions become ``failed`` items."""
    started = __import__("time").monotonic()

    def failed(error_type: str, error: str) -> BatchPreprocessItemResult:
        return BatchPreprocessItemResult(
            index=item.index,
            source=item.source,
            status="failed",
            error_type=error_type,
            error=error,
        )

    output_path = Path(item.output)
    if output_path.exists() and not force:
        return BatchPreprocessItemResult(
            index=item.index,
            source=item.source,
            status="skipped",
            output=str(output_path),
            output_exists=True,
        )
    try:
        if not output_path.parent.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)
        data = _load_array(item.source)
    except Exception as exc:
        return failed(type(exc).__name__, f"load failed: {exc}")
    try:
        if item.kind == "cut":
            if item.theta_par_offset_deg is None:
                return failed("ValueError", "cut items require theta_par_offset_deg")
            from .preprocess import _preprocess_cut_impl

            result = _preprocess_cut_impl(
                data,
                calibration=calibration,
                theta_par_offset_deg=item.theta_par_offset_deg,
                eV=_slice_of(item.eV),
                kx=_slice_of(item.kx),
                quiet=item.quiet,
            )
        else:
            if not item.normal_emission:
                return failed("ValueError", "mapping items require normal_emission")
            from .preprocess import _preprocess_mapping_impl

            result = _preprocess_mapping_impl(
                data,
                calibration=calibration,
                normal_emission=item.normal_emission,
                eV=_slice_of(item.eV),
                kx=_slice_of(item.kx),
                ky=_slice_of(item.ky),
                quiet=item.quiet,
            )
        _atomic_netcdf(result.data, str(output_path))
    except Exception as exc:
        return failed(type(exc).__name__, f"preprocess/save failed: {exc}")
    return BatchPreprocessItemResult(
        index=item.index,
        source=item.source,
        status="completed",
        output=str(output_path),
        output_exists=output_path.exists(),
        duration_s=__import__("time").monotonic() - started,
    )


def _default_output_name(item: BatchPreprocessItem, output_dir: Path) -> str:
    stem = Path(item.source).stem
    return str(output_dir / f"{stem}_processed.nc")


def preprocess_batch(
    items: list[BatchPreprocessItem],
    *,
    calibration: GoldCalibration | float | dict[str, Any],
    output_dir: str | Path,
    cpu_limit_percent: float = 60.0,
    force: bool = False,
) -> BatchProcessingReport:
    """Preprocess a list of scan paths, one at a time, under a CPU budget.

    Parameters
    ----------
    items : list of BatchPreprocessItem
        Per-path processing requests (index, source, kind=cut|mapping, and
        the matching geometry parameters).
    calibration : GoldCalibration, float or dict
        Output of :func:`fit_gold_reference` (or its raw EF correction),
        applied to every item.
    output_dir : str or Path
        Destination directory for the processed NetCDF files
        (``<stem>_processed.nc`` per item; override per item via ``output``).
    cpu_limit_percent : float, default 60
        System CPU threshold above which no new work is submitted.
    force : bool, default False
        Replace existing outputs only when explicitly enabled (existing
        outputs are otherwise reported ``skipped``).

    Returns
    -------
    BatchProcessingReport
        Per-item statuses (``completed``/``skipped``/``failed``/``cancelled``)
        with output paths and errors, plus aggregate CPU statistics.
        JSON-safe via ``model_dump(mode="json")``; holds no DataArray.

    Raises
    ------
    ValueError
        For an empty item list, an invalid calibration or an invalid item.
    """
    from peaksMCP.batch import BatchExecutor, ResourceBudget

    output_root = Path(output_dir).expanduser()
    if not items:
        raise ValueError("preprocess_batch: items must not be empty.")
    if calibration is None:
        raise ValueError(
            "preprocess_batch: a calibration is required (fit_gold_reference output)."
        )
    if isinstance(calibration, GoldCalibration):
        calibration = calibration.correction
    resolved: list[BatchPreprocessItem] = []
    for item in items:
        if not item.output:
            item = item.model_copy(update={"output": _default_output_name(item, output_root)})
        resolved.append(item)
    if any(Path(item.source).expanduser().exists() is False for item in resolved):
        missing = [item.source for item in resolved if not Path(item.source).expanduser().exists()]
        raise ValueError(f"preprocess_batch: source not found: {missing[0]}.")

    def worker(item: BatchPreprocessItem) -> BatchPreprocessItemResult:
        return _run_item(item, calibration, force=force)

    budget = ResourceBudget(
        cpu_limit_percent=cpu_limit_percent,
        resume_percent=max(0, cpu_limit_percent - 10),
    )
    batch = BatchExecutor(budget).run(worker, resolved)
    results: list[BatchPreprocessItemResult] = []
    for batch_item, request in zip(batch.items, resolved, strict=False):
        if batch_item.status == "failed":
            results.append(
                BatchPreprocessItemResult(
                    index=request.index,
                    source=request.source,
                    status="failed",
                    error_type=batch_item.error_type,
                    error=batch_item.error,
                )
            )
        elif batch_item.status != "completed":
            results.append(
                BatchPreprocessItemResult(
                    index=request.index,
                    source=request.source,
                    status=batch_item.status,
                    output=request.output,
                    error_type=batch_item.error_type,
                    error=batch_item.error,
                )
            )
        else:
            results.append(
                BatchPreprocessItemResult(
                    index=request.index,
                    source=request.source,
                    status="completed",
                    output=str(request.output),
                    output_exists=Path(request.output).exists(),
                )
            )
    report = BatchProcessingReport(items=results)
    print(
        f"preprocess_batch: {len(results)} item(s) - "
        f"{report.completed} completed, {report.skipped} skipped, "
        f"{report.failed} failed"
    )
    return report
