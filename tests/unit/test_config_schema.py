"""Strict schema tests for the single canonical API catalog."""

from __future__ import annotations

import pytest
import yaml

from peaksMCP.config.schema import load_yaml_unique, validate_api_catalog


def _facade_document() -> dict:
    return {
        "version": 1,
        "apis": {
            "module:peaksMCP.facades:future": {
                "kind": "facade",
                "exposure": "core",
                "export": "peaksMCP.facades.future",
                "summary": "Future facade.",
                "aliases": ["future workflow"],
                "inputs": [{"name": "source", "type": "str", "required": True}],
                "returns": "object",
                "preconditions": "source exists",
                "side_effects": "none",
                "errors": "invalid source",
                "example": "future(source)",
            }
        },
    }


def test_yaml_unique_loader_rejects_duplicate_keys():
    with pytest.raises(yaml.YAMLError, match="duplicate mapping key 'kind'"):
        load_yaml_unique("api:\n  kind: native\n  kind: facade\n")


def test_catalog_rejects_unknown_fields_versions_and_duplicate_aliases():
    doc = {
        "version": 2,
        "apis": {
            "dataarray:peaks.example:run": {
                "kind": "native",
                "exposure": "core",
                "unknown": True,
                "aliases": ["dup", "DUP"],
            }
        },
    }
    errors = validate_api_catalog(doc)
    assert any("version must be 1" in error for error in errors)
    assert any("unknown key" in error for error in errors)
    assert any("duplicate alias" in error for error in errors)


def test_facade_contract_inputs_are_structured():
    assert validate_api_catalog(_facade_document()) == []
    doc = _facade_document()
    next(iter(doc["apis"].values()))["inputs"] = "source: path"
    assert any(
        "inputs must be a non-empty list" in error for error in validate_api_catalog(doc)
    )


def test_real_catalog_passes_strict_validation():
    from pathlib import Path

    document = load_yaml_unique(
        Path("peaksMCP/config/api_catalog.yaml").read_text(encoding="utf-8")
    )
    assert validate_api_catalog(document) == []


def test_default_loader_fails_loudly_on_damaged_catalog(monkeypatch, tmp_path):
    import peaksMCP.discovery.index as index_module

    (tmp_path / "api_catalog.yaml").write_text(
        "version: 1\napis:\n  dataarray:peaks.example:run:\n"
        "    kind: native\n    kind: facade\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(index_module, "_CONFIG_DIR", tmp_path)
    with pytest.raises(ValueError, match="invalid curated YAML"):
        index_module.load_catalog()
