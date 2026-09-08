"""Strict schema tests for the curated catalogs (native_catalog + override_manifest).

A damaged default configuration must fail loudly (ValueError) instead of
silently degrading to an empty catalog; the explicit-path reader stays
lenient for legacy shapes during the compatibility window.
"""

from __future__ import annotations

import pytest
import yaml

from peaksMCP.config.schema import (
    load_yaml_unique,
    validate_native_catalog,
    validate_override_manifest,
)


def _native(**overrides):
    doc = {"version": 1, "apis": {"k_convert": {"aliases": ["k space", "动量转换"]}}}
    doc.update(overrides)
    return doc


def _override(**overrides):
    doc = {
        "version": 3,
        "apis": {
            "load_data": {
                "module": "peaksMCP.overrides.load",
                "project": True,
                "aliases": ["load data", "加载"],
            }
        },
        "project": [
            {
                "name": "load_data",
                "export": "peaksMCP.overrides.load_data",
                "exposure": "facade",
                "category": "ingestion",
                "kind": "loader",
                "stability": "new",
            }
        ],
    }
    doc.update(overrides)
    return doc


def test_yaml_unique_loader_rejects_duplicate_keys():
    with pytest.raises(yaml.YAMLError, match="duplicate mapping key 'module'"):
        load_yaml_unique("load_data:\n  module: a\n  module: b\n")


def test_native_catalog_rejects_unknown_fields_and_bad_versions():
    errors = validate_native_catalog(_native(apis={"x": {"project": True}}))
    assert any("unknown key" in error for error in errors)
    assert any("version must be 1" in error for error in validate_native_catalog(_native(version=2)))
    assert any("non-empty mapping" in error for error in validate_native_catalog(_native(apis={})))
    errors = validate_native_catalog(_native(apis={"x": {"aliases": ["dup", "DUP"]}}))
    assert any("duplicate alias" in error for error in errors)


def test_override_manifest_rejects_unknown_fields_and_bad_enums():
    errors = validate_override_manifest(_override(apis={"x": {"unknown_thing": 1}}))
    assert any("unknown key" in error for error in errors)
    doc = _override()
    doc["apis"]["load_data"]["exposure"] = "public"
    errors = validate_override_manifest(doc)
    assert any("invalid exposure 'public'" in error for error in errors)
    doc = _override()
    doc["apis"]["load_data"]["category"] = "quantum"
    errors = validate_override_manifest(doc)
    assert any("invalid category 'quantum'" in error for error in errors)


def test_override_manifest_project_seeds_are_cross_checked():
    # A seed that references a ghost apis entry fails.
    doc = _override(project=[{"name": "ghost_api", "export": "peaksMCP.overrides.ghost_api"}])
    errors = validate_override_manifest(doc)
    assert any("does not exist: 'ghost_api'" in error for error in errors)
    # An export that does not end with the entry name fails.
    doc = _override(project=[{"name": "load_data", "export": "peaksMCP.overrides.save_result"}])
    errors = validate_override_manifest(doc)
    assert any("export must end with the entry name" in error for error in errors)
    # An export outside peaksMCP.overrides fails.
    doc = _override(project=[{"name": "load_data", "export": "peaksMCP.plotting.plot_batch"}])
    errors = validate_override_manifest(doc)
    assert any("peaksMCP.overrides.<name>" in error for error in errors)


def test_override_manifest_project_entry_requires_module_or_export():
    doc = _override()
    del doc["apis"]["load_data"]["module"]
    errors = validate_override_manifest(doc)
    assert any("needs module or export+implementation" in error for error in errors)


def test_real_catalogs_pass_strict_validation():
    """The shipped catalogs must validate (the discovery loader now refuses
    to build an index from a damaged default configuration)."""
    import pathlib

    from peaksMCP.config.schema import validate_documents

    config_dir = pathlib.Path("peaksMCP/config")
    native = load_yaml_unique((config_dir / "native_catalog.yaml").read_text(encoding="utf-8"))
    overrides = load_yaml_unique((config_dir / "override_manifest.yaml").read_text(encoding="utf-8"))
    assert validate_documents(native, overrides) == []


def test_default_loader_fails_loudly_on_damaged_config(monkeypatch, tmp_path):
    """A duplicate key in the default catalogs must raise, never return {}."""
    import peaksMCP.discovery.index as index_module

    broken = tmp_path / "native_catalog.yaml"
    broken.write_text("version: 1\napis:\n  k_convert:\n    aliases: [x]\n    aliases: [y]\n", encoding="utf-8")
    (tmp_path / "override_manifest.yaml").write_text(
        "version: 3\napis:\n  load_data:\n    module: m\n    project: true\n", encoding="utf-8"
    )
    monkeypatch.setattr(index_module, "_CONFIG_DIR", tmp_path)
    with pytest.raises(ValueError, match="invalid curated YAML"):
        index_module.load_overrides()


def test_default_loader_falls_back_to_legacy_single_file(monkeypatch, tmp_path):
    """Pre-split checkouts (only config/manifest.yaml) keep working."""
    import peaksMCP.discovery.index as index_module

    (tmp_path / "manifest.yaml").write_text(
        "version: 3\napis:\n  k_convert:\n    aliases: [k space]\n", encoding="utf-8"
    )
    monkeypatch.setattr(index_module, "_CONFIG_DIR", tmp_path)
    document = index_module.load_overrides()
    assert "k_convert" in document["apis"]
    # Both new catalogs absent AND no legacy file -> empty (same as before).
    (tmp_path / "manifest.yaml").unlink()
    assert index_module.load_overrides() == {}
