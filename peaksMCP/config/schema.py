"""Strict schema validation for the curated presentation catalogs.

Two documents are curated by hand and consumed by discovery:

- ``native_catalog.yaml`` (version 1) — upstream peaks presentation only:
  every entry is fixed to the native tier and may carry just ``aliases``
  and ``docstring_note``.
- ``override_manifest.yaml`` (version 5, breaking) — the single manifest of
  public project APIs.  One row per adapter: ``export`` (the only identity,
  ``peaksMCP.overrides.<name>``) plus the full structured contract
  (``summary``/``inputs``/``returns``/``preconditions``/``side_effects``/
  ``errors``/``example``) and search aliases.  No module hints, docstring
  notes or project seeds — signatures are verified at runtime by importing
  the declared export.

Validation failures are loud by design: a damaged default configuration must
fail the index build instead of silently degrading to an empty catalog.
"""

from __future__ import annotations

from typing import Any

import yaml

#: v1 native-catalog entry keys (retrieval/display only).
_NATIVE_ENTRY_KEYS = frozenset({"aliases", "docstring_note"})

#: v5 project-manifest entry keys: the full structured contract per public
#: adapter.  No module/docstring_note/project seeds: export is the only
#: identity, signatures are verified at runtime by importing export.
_OVERRIDE_ENTRY_KEYS = frozenset(
    {
        "export",
        "exposure",
        "aliases",
        "summary",
        "inputs",
        "returns",
        "preconditions",
        "side_effects",
        "errors",
        "example",
    }
)

#: Every public adapter must declare the FULL contract - a row that only
#: carries export/exposure/summary would leave the model without the
#: usage/limit/side-effect documentation the black-box surface promises.
_OVERRIDE_REQUIRED_KEYS = frozenset(
    {
        "export",
        "exposure",
        "summary",
        "inputs",
        "returns",
        "preconditions",
        "side_effects",
        "errors",
        "example",
    }
)

_EXPOSURES = frozenset({"facade", "advanced", "internal"})

#: v5: one declared parameter of an adapter (``inputs`` entry).
_INPUT_ENTRY_KEYS = frozenset({"name", "type", "required", "default", "note"})


def _check_inputs(entry: dict[str, Any], where: str, errors: list[str]) -> None:
    """Validate the structured ``inputs`` list of one manifest v5 row.

    Free text let the contract name parameters that do not exist (it declared
    ``scans:`` for ``inspect_experiment`` whose first parameter is
    ``experiment``, and ``items:`` for ``plot_batch(data, ...)``).  Structured
    entries are name-checked against the real signature at runtime
    (``discovery.signatures``) and shape-checked here, so a typo fails loudly.
    """
    inputs = entry.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        errors.append(f"{where} inputs must be a non-empty list of parameter entries")
        return
    required_count = 0
    for position, item in enumerate(inputs):
        if not isinstance(item, dict):
            errors.append(f"{where} inputs[{position}] must be a mapping")
            continue
        unknown = set(item) - _INPUT_ENTRY_KEYS
        if unknown:
            errors.append(f"{where} inputs[{position}] has unknown key(s) {sorted(unknown)}")
        name = item.get("name")
        if not isinstance(name, str) or not name.isidentifier():
            errors.append(f"{where} inputs[{position}] needs an identifier name, got {name!r}")
        type_name = item.get("type")
        if not isinstance(type_name, str) or not type_name.strip():
            errors.append(f"{where} inputs[{position}] needs a non-empty type declaration")
        if "required" in item and not isinstance(item["required"], bool):
            errors.append(f"{where} inputs[{position}] required must be a boolean")
        if item.get("required") is True:
            required_count += 1
    if not required_count:
        errors.append(f"{where} inputs must mark at least one parameter required")


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
    """Validate a v5 override manifest; returns a list of human-readable errors."""
    errors: list[str] = []
    if document.get("version") != 5:
        errors.append("override_manifest: version must be 5")
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
        for required in sorted(_OVERRIDE_REQUIRED_KEYS):
            value = entry.get(required)
            if value is None or (isinstance(value, str) and not value.strip()):
                errors.append(f"override_manifest: {name} must declare {required!r}")
        _check_aliases(entry, f"override_manifest: {name}", errors)
        _check_inputs(entry, f"override_manifest: {name}", errors)
        export = entry.get("export")
        if export is not None and (
            not isinstance(export, str)
            or not export.startswith("peaksMCP.overrides.")
            or export.rsplit(".", 1)[-1] != name
        ):
            errors.append(
                f"override_manifest: {name} export must be peaksMCP.overrides.<name>: {export!r}"
            )
        exposure = entry.get("exposure")
        if exposure is not None and exposure not in _EXPOSURES:
            errors.append(
                f"override_manifest: {name} has invalid exposure {exposure!r} "
                f"(allowed: {sorted(_EXPOSURES)})"
            )
    return errors


def validate_documents(native: dict[str, Any], overrides: dict[str, Any]) -> list[str]:
    """Validate both curated catalogs together (order matters for messages)."""
    return [
        *validate_native_catalog(native),
        *validate_override_manifest(overrides),
    ]
