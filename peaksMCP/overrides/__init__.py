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

Surface invariants (enforced by tests):

- ``MODEL_CALLABLE_EXPORTS`` must equal the manifest ``apis`` keys exactly
  (every model-callable export belongs to the manifest, and nothing else
  does); ``CONTRACT_TYPES`` are the non-callable record/result types and are
  explicitly not verbs.

Low-level metadata helpers (``read_meta`` / ``classify_data_format`` /
``is_gold_format`` / ``theta_offset_deg``) live in ``pxt_utils.metadata`` as
private implementation detail and are NOT re-exported here.  Earlier task
facades (fit_gold_reference, preprocess_cut, preprocess_mapping,
preprocess_batch) were demoted out of the model surface: they fixed a
workflow that the model should compose from native peaks APIs; their code
remains importable in the internal modules for in-process/tests only.
"""

from __future__ import annotations

from peaksMCP.plotting.layout import plot_batch  # noqa: F401 (public surface)
from peaksMCP.plotting.validation import plot_validation_pair  # noqa: F401
from peaksMCP.workflows.slice_view import show_mapping_slice  # noqa: F401

from .conversion import convert_experiment  # noqa: F401
from .inspection import (  # noqa: F401
    ExperimentSummary,
    ScanKind,
    ScanSummary,
    inspect_experiment,
)
from .load import LoadedScans, ScanEntry, load_data  # noqa: F401
from .save import SaveReceipt  # noqa: F401

#: The six public adapters registered in ``config/override_manifest.yaml``.
#: Invariant: ``peaksMCP.overrides.MODEL_CALLABLE_EXPORTS == manifest keys``.
MODEL_CALLABLE_EXPORTS = frozenset(
    {
        "load_data",
        "convert_experiment",
        "inspect_experiment",
        "plot_batch",
        "plot_validation_pair",
        "show_mapping_slice",
    }
)

#: Non-callable contract/record types re-exported for typing convenience.
#: They are deliberately NOT model verbs and never appear in the manifest.
CONTRACT_TYPES = frozenset(
    {
        "LoadedScans",
        "ScanEntry",
        "ExperimentSummary",
        "ScanSummary",
        "ScanKind",
        "SaveReceipt",
    }
)

__all__ = sorted(MODEL_CALLABLE_EXPORTS | CONTRACT_TYPES)
