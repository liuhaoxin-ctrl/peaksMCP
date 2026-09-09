"""Curated project surface (facades) on top of peaks.

The model composes these functions instead of re-implementing analysis; native
peaks functions stay reachable through search/get.  Every facade:

- returns a small JSON-safe dict or a DataArray/Dataset (never prints noise);
- renders at most one canonical summary line (Show convention);
- never writes to disk by itself — persistence goes through ``save_result``,
  which stages the result and publishes it only after a human approves the
  real-content card (no code-level approve exists).

Single canonical import surface: every project API (the override tier) is
importable from ``peaksMCP.overrides``.  The implementation modules below
(``peaksMCP.plotting.*``, ``peaksMCP.pxt_utils.*``,
``peaksMCP.workflows.*``) stay importable for Python compatibility, but
discovery projects their index entries onto ``peaksMCP.overrides`` and never
shows the implementation module to the model.
"""

from __future__ import annotations

from peaksMCP.plotting.layout import plot_batch
from peaksMCP.plotting.validation import plot_validation_pair
from peaksMCP.pxt_utils.metadata import (
    classify_data_format,
    is_gold_format,
    read_meta,
    theta_offset_deg,
)
from peaksMCP.workflows.publication import validate_arpes_metadata
from peaksMCP.workflows.slice_view import show_mapping_slice

from .batch_preprocess import (
    BatchPreprocessItem,
    BatchProcessingReport,
    preprocess_batch,
)
from .calibration import GoldCalibration, fit_gold_reference
from .conversion import convert_experiment
from .inspection import ExperimentSummary, ScanKind, ScanSummary, inspect_experiment
from .load import LoadedScans, ScanEntry, load_data
from .models import Report, report_dict, report_summary
from .preprocess import ProcessingReport, ProcessingResult, preprocess_cut, preprocess_mapping
from .save import save_result

__all__ = [
    # loading and persistence facades
    "load_data",
    "LoadedScans",
    "ScanEntry",
    "save_result",
    # task-level facades
    "convert_experiment",
    "inspect_experiment",
    "fit_gold_reference",
    "preprocess_cut",
    "preprocess_mapping",
    "preprocess_batch",
    # facade result types
    "GoldCalibration",
    "ExperimentSummary",
    "ScanSummary",
    "ScanKind",
    "ProcessingResult",
    "ProcessingReport",
    "BatchPreprocessItem",
    "BatchProcessingReport",
    # conversion and metadata (translation/legacy writers are NOT model
    # verbs: use convert_experiment / save_result; the raw converter and
    # datasheet translator stay importable for legacy/in-process use only)
    "read_meta",
    "classify_data_format",
    "is_gold_format",
    "theta_offset_deg",
    "validate_arpes_metadata",
    # plotting and workflows
    "plot_batch",
    "plot_validation_pair",
    "show_mapping_slice",
    # report plumbing
    "Report",
    "report_dict",
    "report_summary",
]
