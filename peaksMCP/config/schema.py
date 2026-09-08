"""Strict schema validation for the curated presentation catalogs.

Two documents are curated by hand and consumed by discovery:

- ``native_catalog.yaml`` (version 1) — upstream peaks presentation only:
  every entry is fixed to the native tier and may carry just ``aliases``
  and ``docstring_note``.
- ``override_manifest.yaml`` (version 3) — the project black-box exposure
  record: ``apis`` entries may use the discovery keys (``module``,
  ``project``, ``aliases``, ``docstring_note``) plus the v3 contract fields;
  the ``project`` block holds strict facade contracts.

Validation failures are loud by design: a damaged default configuration must
fail the index build instead of silently degrading to an empty catalog.
"""

from __future__ import annotations

from typing import Any

import yaml

#: v1 native-catalog entry keys (retrieval/display only).
_NATIVE_ENTRY_KEYS = frozenset({"aliases", "docstring_note"})

#: v3 override-manifest apis-entry keys: discovery keys (today's wiring) plus
#: the strict contract vocabulary that facades migrate onto.
_OVERRIDE_ENTRY_KEYS = frozenset(
    {
        "module",
        "project",
        "aliases",
        "docstring_note",
        "export",
        "implementation",
        "exposure",
        "category",
        "kind",
        "summary",
        "inputs",
        "returns",
        "preconditions",
        "side_effects",
        "errors",
        "example",
        "stability",
        "legacy_ids",
        "shadows_native",
    }
)

#: v3 strict facade-contract keys (the ``project`` block).
_PROJECT_CONTRACT_KEYS = frozenset(
    {
        "name",
        "export",
        "exposure",
        "category",
        "kind",
        "summary",
        "inputs",
        "returns",
        "preconditions",
        "side_effects",
        "errors",
        "example",
        "stability",
    }
)

_EXPOSURES = frozenset({"facade", "advanced", "internal"})
_CATEGORIES = frozenset(
    {
        "ingestion",
        "metadata",
        "calibration",
        "preprocessing",
        "visualization",
        "persistence",
        "batch",
    }
)
_KINDS = frozenset(
    {
        "loader",
        "converter",
        "inspector",
        "classifier",
        "calibrator",
        "transformer",
        "visualizer",
        "serializer",
        "batch",
    }
)
_STABILITIES = frozenset({"new", "stable", "deprecated"})


def _unique_key_loader() -> type[yaml.SafeLoader]:
    """SafeLoader subclass that refuses duplicate mapping keys."""

    class _UniqueKeyLoader(yaml.SafeLoader):  # type: ignore[misc]
        pass

    def _construct_mapping(loader: yaml.Loader, node: yaml.MappingNode, deep: bool = False) -> dict:  # type: ignore[no-untyped-def]
        mapping: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in mapping:
                raise yaml.YAMLError(f"duplicate mapping key {key!r} at line {key_node.start_mark.line + 1}")
            mapping[key] = loader.construct_object(value_node, deep=deep)
        return mapping

    _UniqueKeyLoader.add_constructor(  # type: ignore[attr-defined]
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
    )
    return _UniqueKeyLoader


_UNIQUE_LOADER = _unique_key_loader()


def load_yaml_unique(text: str) -> dict[str, Any]:
    """Parse YAML, refusing duplicate mapping keys (yaml.safe_load silently
    keeps the last value, which turns a typo'd duplicate into a silent edit)."""
    return yaml.load(text, Loader=_UNIQUE_LOADER) or {}


def _check_aliases(entry: dict[str, Any], where: str, errors: list[str]) -> None:
    aliases = entry.get("aliases")
    if aliases is None:
        return
    if not isinstance(aliases, list) or not aliases:
        errors.append(f"{where}: aliases must be a non-empty list of strings")
        return
    seen: set[str] = set()
    for value in aliases:
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{where}: aliases must be non-empty strings")
            continue
        key = value.strip().lower()
        if key in seen:
            errors.append(f"{where}: duplicate alias {value!r}")
        seen.add(key)


def validate_native_catalog(document: dict[str, Any]) -> list[str]:
    """Validate a v1 native catalog; returns a list of human-readable errors."""
    errors: list[str] = []
    if document.get("version") != 1:
        errors.append("native_catalog: version must be 1")
    apis = document.get("apis")
    if not isinstance(apis, dict) or not apis:
        errors.append("native_catalog: apis must be a non-empty mapping")
        return errors
    for name, entry in apis.items():
        if not isinstance(entry, dict):
            errors.append(f"native_catalog: {name} must be a mapping")
            continue
        unknown = set(entry) - _NATIVE_ENTRY_KEYS
        if unknown:
            errors.append(f"native_catalog: {name} has unknown key(s) {sorted(unknown)}")
        _check_aliases(entry, f"native_catalog: {name}", errors)
    return errors


def validate_override_manifest(document: dict[str, Any]) -> list[str]:
    """Validate a v3 override manifest; returns a list of human-readable errors."""
    errors: list[str] = []
    if document.get("version") != 3:
        errors.append("override_manifest: version must be 3")
    apis = document.get("apis")
    if not isinstance(apis, dict) or not apis:
        errors.append("override_manifest: apis must be a non-empty mapping")
        return errors
    for name, entry in apis.items():
        if not isinstance(entry, dict):
            errors.append(f"override_manifest: {name} must be a mapping")
            continue
        unknown = set(entry) - _OVERRIDE_ENTRY_KEYS
        if unknown:
            errors.append(f"override_manifest: {name} has unknown key(s) {sorted(unknown)}")
        if entry.get("project") and not (
            entry.get("module") or (entry.get("export") and entry.get("implementation"))
        ):
            errors.append(f"override_manifest: project entry {name} needs module or export+implementation")
        _check_aliases(entry, f"override_manifest: {name}", errors)
        for field, allowed, label in (
            ("exposure", _EXPOSURES, "exposure"),
            ("category", _CATEGORIES, "category"),
            ("kind", _KINDS, "kind"),
            ("stability", _STABILITIES, "stability"),
        ):
            value = entry.get(field)
            if value is not None and value not in allowed:
                errors.append(
                    f"override_manifest: {name} has invalid {label} {value!r} "
                    f"(allowed: {sorted(allowed)})"
                )
    # The project block: strict facade contracts that must reference apis
    # entries and use only contract fields.
    project = document.get("project")
    if project is not None:
        if not isinstance(project, list):
            errors.append("override_manifest: project must be a list")
        else:
            for index, contract in enumerate(project):
                where = f"override_manifest: project[{index}]"
                if not isinstance(contract, dict):
                    errors.append(f"{where} must be a mapping")
                    continue
                unknown = set(contract) - _PROJECT_CONTRACT_KEYS
                if unknown:
                    errors.append(f"{where} has unknown key(s) {sorted(unknown)}")
                name = contract.get("name")
                if not isinstance(name, str) or name not in apis:
                    errors.append(f"{where} references an apis entry that does not exist: {name!r}")
                export = contract.get("export")
                if not isinstance(export, str) or not export.startswith("peaksMCP.overrides."):
                    errors.append(f"{where} export must be peaksMCP.overrides.<name>: {export!r}")
                if name and export and export.rsplit(".", 1)[-1] != name:
                    errors.append(f"{where} export must end with the entry name: {export!r}")
                for field, allowed, label in (
                    ("exposure", _EXPOSURES, "exposure"),
                    ("category", _CATEGORIES, "category"),
                    ("kind", _KINDS, "kind"),
                    ("stability", _STABILITIES, "stability"),
                ):
                    value = contract.get(field)
                    if value is not None and value not in allowed:
                        errors.append(
                            f"{where} has invalid {label} {value!r} (allowed: {sorted(allowed)})"
                        )
    return errors


def validate_documents(native: dict[str, Any], overrides: dict[str, Any]) -> list[str]:
    """Validate both curated catalogs together (order matters for messages)."""
    return [
        *validate_native_catalog(native),
        *validate_override_manifest(overrides),
    ]
