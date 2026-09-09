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


def test_manifest_examples_only_use_real_parameter_names():
    """Contract 与实现一致性：example 里的关键字参数必须是运行时签名的参数名。

    Example 是 agent 的用法模板——如果它用了签名里不存在的参数名（例如旧的
    force= / cpu_limit_percent=），agent 照着写就会失败。这一条把这种漂移挡在
    manifest 层（可解析的 example 才校验；无法解析的记入 skipped 说明）。
    """
    import ast

    document = _manifest_document()
    skipped: list[str] = []
    for name, entry in document["apis"].items():
        export = entry["export"]
        module_name, _, attr = export.rpartition(".")
        module = importlib.import_module(module_name)
        params = set(inspect.signature(getattr(module, attr)).parameters)
        example = entry.get("example") or ""
        try:
            tree = ast.parse(example)
        except SyntaxError:
            skipped.append(name)
            continue
        keywords = {
            node2.arg
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            for node2 in node.keywords
            if node2.arg is not None
        }
        unknown = sorted(keywords - params)
        assert not unknown, (
            f"manifest {name} example 用了签名里不存在的参数：{unknown}；"
            f"签名参数：{sorted(params)}"
        )
    assert skipped == [], f"example 无法解析的 row 应显式处理：{skipped}"
