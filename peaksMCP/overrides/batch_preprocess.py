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


def _run_item(
    item: BatchPreprocessItem,
    calibration: Any,
    force: bool = False,
) -> dict[str, Any]:
    """Load, preprocess and STAGE one item (never publishes).

    Runs inside a worker process; dependencies import locally.  Returns
    ``{"pending": <PendingItem>}`` with the staged bytes when processing
    succeeded, ``{"skipped_existing": ...}`` for idempotent skips, and
    raises on failure (the executor maps it to a failed item).
    """
    from . import save as save_module

    output_path = Path(item.output)
    if output_path.exists() and not force:
        return {"skipped_existing": str(output_path)}
    data = _load_array(item.source)
    if item.kind == "cut":
        if item.theta_par_offset_deg is None:
            raise ValueError("cut items require theta_par_offset_deg")
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
            raise ValueError("mapping items require normal_emission")
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
    pending = save_module._stage_item(result.data, output_path, force)
    return {"pending": pending}


def _mark(
    results: list[BatchPreprocessItemResult],
    output: Path,
    status: str,
    output_exists: bool | None,
) -> None:
    """Set the final status on the report row owning ``output``."""
    for row in results:
        if row.output == str(output):
            row.status = status
            if output_exists is not None:
                row.output_exists = output_exists
            return


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

    def worker(item: BatchPreprocessItem) -> dict[str, Any]:
        return _run_item(item, calibration, force=force)

    budget = ResourceBudget(
        cpu_limit_percent=cpu_limit_percent,
        resume_percent=max(0, cpu_limit_percent - 10),
    )
    batch = BatchExecutor(budget).run(worker, resolved)

    from . import save as save_module

    staged_requests: list[tuple[Any, Any]] = []  # (pending, request)
    results: list[BatchPreprocessItemResult] = []
    for batch_item, request in zip(batch.items, resolved, strict=False):
        if batch_item.status == "failed":
            results.append(
                BatchPreprocessItemResult(
                    index=request.index, source=request.source,
                    status="failed", error_type=batch_item.error_type,
                    error=batch_item.error,
                )
            )
            continue
        if batch_item.status != "completed":
            results.append(
                BatchPreprocessItemResult(
                    index=request.index, source=request.source,
                    status=batch_item.status, output=request.output,
                    error_type=batch_item.error_type, error=batch_item.error,
                )
            )
            continue
        outcome = batch_item.output
        if isinstance(outcome, dict) and "skipped_existing" in outcome:
            results.append(
                BatchPreprocessItemResult(
                    index=request.index, source=request.source,
                    status="skipped", output=str(outcome["skipped_existing"]),
                    output_exists=True, warnings=["output exists"],
                )
            )
            continue
        pending = outcome.get("pending") if isinstance(outcome, dict) else None
        if pending is None:
            results.append(
                BatchPreprocessItemResult(
                    index=request.index, source=request.source,
                    status="failed", error="worker returned no staged output",
                )
            )
            continue
        staged_requests.append((pending, request))
        results.append(
            BatchPreprocessItemResult(
                index=request.index, source=request.source,
                status="awaiting_consent", output=str(pending.path),
            )
        )

    report = BatchProcessingReport(items=results)
    if not staged_requests:
        print(
            f"preprocess_batch: {len(results)} item(s) - "
            f"{report.completed} completed, {report.skipped} skipped, "
            f"{report.failed} failed (nothing to publish)"
        )
        return report

    summary = (
        f"preprocess_batch: publish {len(staged_requests)} processed "
        f"file(s) to {output_root}"
    )
    ticket = save_module._create_ticket(
        "preprocess_batch",
        [pending for pending, _request in staged_requests],
        summary,
    )
    approved = save_module._request_consent(ticket)
    status: str
    if approved is None:
        status = "pending_consent"
        for pending, _request in staged_requests:
            _mark(results, pending.path, "pending_consent", None)
    elif approved:
        ticket.authorized = True
        outcome = save_module._publish_batch(ticket.ticket_id)
        status = "saved"
        published = {p["path"] for p in outcome["published"]}
        skipped_now = {s["path"] for s in outcome["skipped"]}
        for pending, _request in staged_requests:
            key = str(pending.path)
            if key in published:
                _mark(results, pending.path, "completed", True)
            elif key in skipped_now:
                _mark(results, pending.path, "skipped", True)
            else:
                _mark(results, pending.path, "completed", True)
    else:
        status = "denied"
        save_module._discard_ticket(ticket.ticket_id)
        for pending, _request in staged_requests:
            _mark(results, pending.path, "denied", None)
    print(
        f"preprocess_batch: {len(results)} item(s) - "
        f"{sum(r.status == 'completed' for r in results)} completed, "
        f"{sum(r.status == 'skipped' for r in results)} skipped, "
        f"{sum(r.status == 'failed' for r in results)} failed "
        f"({status})"
    )
    return report
