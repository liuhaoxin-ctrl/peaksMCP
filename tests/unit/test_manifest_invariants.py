"""Manifest <-> Python surface invariants (bidirectional).

Locks the Phase-3.5 rule that ``config/override_manifest.yaml`` is the ONLY
owner of the model-callable surface:

- every manifest row declares the FULL contract (export/exposure/summary/
  inputs/returns/preconditions/side_effects/errors/example) - schema-enforced,
  re-checked here against the shipped file;
- every manifest export must exist, be callable and have a resolvable
  signature (runtime verification by importing the export);
- the callables re-exported by ``peaksMCP.overrides`` must equal the manifest
  keys exactly - no extra model-callable export, no missing one; contract
  types (LoadedScans/ScanEntry/ExperimentSummary/ScanSummary/ScanKind/
  SaveReceipt) are explicitly non-callable and are not verbs.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import yaml


def _manifest_document() -> dict:
    path = Path(__file__).parents[2] / "peaksMCP" / "config" / "override_manifest.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_manifest_rows_declare_the_full_contract():
    document = _manifest_document()
    assert document["version"] == 4
    required = {
        "export", "exposure", "summary", "inputs", "returns",
        "preconditions", "side_effects", "errors", "example",
    }
    for name, entry in document["apis"].items():
        missing = {key for key in required if not str(entry.get(key) or "").strip()}
        assert not missing, f"manifest row {name!r} misses: {sorted(missing)}"


def test_every_manifest_export_is_importable_callable_and_signed():
    from peaksMCP.config.schema import validate_override_manifest

    assert validate_override_manifest(_manifest_document()) == []
    for name, entry in _manifest_document()["apis"].items():
        export = entry["export"]
        module_name, _, attr = export.rpartition(".")
        module = importlib.import_module(module_name)
        obj = getattr(module, attr)
        assert callable(obj), f"{export} is not callable"
        inspect.signature(obj)  # resolvable signature
        assert export.endswith(f".{name}")


def test_overrides_callables_equal_manifest_keys_exactly():
    import peaksMCP.overrides as overrides

    manifest_names = set(_manifest_document()["apis"])
    assert overrides.MODEL_CALLABLE_EXPORTS == manifest_names
    assert overrides.CONTRACT_TYPES == {
        "LoadedScans", "ScanEntry", "ExperimentSummary",
        "ScanSummary", "ScanKind", "SaveReceipt",
    }
    exported = set(overrides.__all__)
    assert exported == overrides.MODEL_CALLABLE_EXPORTS | overrides.CONTRACT_TYPES
    # Non-class callables in the public surface are exactly the manifest
    # verbs (contract types are classes and are explicitly not verbs).
    callable_exports = {
        name
        for name in exported
        if callable(getattr(overrides, name))
        and not isinstance(getattr(overrides, name), type)
    }
    assert callable_exports == manifest_names
    assert all(
        isinstance(getattr(overrides, name), type) for name in overrides.CONTRACT_TYPES
    )
    # Types are never verbs (and vice versa).
    assert not (overrides.CONTRACT_TYPES & overrides.MODEL_CALLABLE_EXPORTS)


def test_legacy_or_internal_names_never_leak_into_the_public_surface():
    import peaksMCP.overrides as overrides

    exported = set(overrides.__all__)
    assert "save_result" not in exported
    assert "read_meta" not in exported
    assert "validate_arpes_metadata" not in exported
    assert "Report" not in exported and "report_dict" not in exported
    assert "preprocess_cut" not in exported and "fit_gold_reference" not in exported
