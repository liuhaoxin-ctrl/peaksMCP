"""Curated adapter surface on top of peaks.

The model verb surface holds ONLY compatibility adapters - things that add a
real boundary peaks cannot cross itself (data access, format conversion,
inspection, uniform persistence, deterministic plotting conventions).  It
does NOT pre-compose analysis workflows: fitting, coordinate correction,
k-space conversion and batch preprocessing are native peaks steps the model
obtains through search/get and composes in the notebook, then persists
through the ``save_with_consent`` MCP tool (staged, human-approved).

Adapters:

- data access: ``load_data`` (file / experiment index: per-file identity,
  representation and sizes, read from headers only) and
  ``inspect_experiment`` (scans index -> structured experiment summary -
  the single classification owner for gold/cut/mapping kinds, decision
  lists and shape conflicts);
- conversion: ``convert_experiment`` (raw PXT -> NetCDF adapter; pure
  computation, outputs are published only after the consent card);
- persistence: the ``save_with_consent`` MCP tool (staged bytes +
  approval-gated gateway; ``SaveReceipt`` is the result type);
- plotting conventions: ``plot_batch`` / ``plot_validation_pair`` /
  ``show_mapping_slice``.

Low-level metadata helpers (``read_meta`` / ``classify_data_format`` /
``is_gold_format`` / ``theta_offset_deg``) live in ``pxt_utils.metadata`` as
private implementation detail and are NOT re-exported here.  Earlier task
facades (fit_gold_reference, preprocess_cut, preprocess_mapping,
preprocess_batch) were demoted out of the model surface: they fixed a
workflow that the model should compose from native peaks APIs; their code
remains importable in the internal modules for in-process/tests only.
"""

from __future__ import annotations

from peaksMCP.plotting.layout import plot_batch
from peaksMCP.plotting.validation import plot_validation_pair
from peaksMCP.workflows.publication import validate_arpes_metadata
from peaksMCP.workflows.slice_view import show_mapping_slice

from .conversion import convert_experiment
from .inspection import ExperimentSummary, ScanKind, ScanSummary, inspect_experiment
from .load import LoadedScans, ScanEntry, load_data
from .save import SaveReceipt

__all__ = [
    # loading adapters
    "load_data",
    "LoadedScans",
    "ScanEntry",
    # conversion + experiment adapters
    "convert_experiment",
    "inspect_experiment",
    "ExperimentSummary",
    "ScanSummary",
    "ScanKind",
    # save receipt type (the verb is the MCP tool save_with_consent)
    "SaveReceipt",
    # publication metadata validation (model-visible helper)
    "validate_arpes_metadata",
    # plotting conventions
    "plot_batch",
    "plot_validation_pair",
    "show_mapping_slice",
]
