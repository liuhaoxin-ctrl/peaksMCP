"""Build and search a live index of the installed Peaks API.

The index combines runtime descriptor inspection with a static AST scan. It is built once per
kernel so it always reflects the installed Peaks version without requiring a generated catalog.
"""

from __future__ import annotations

import ast
import hashlib
import os
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_SCOPES = {"all", "dataarray", "dataset", "datatree", "top_level", "module"}
_SKIP_DIRS = {"__pycache__", "GUI"}
_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_]*|[\u4e00-\u9fff]+")

#: peaksMCP source files whose edits must invalidate the in-kernel API index.
_INDEX_PACKAGE = Path(__file__).resolve().parent.parent
_ADAPTER_SOURCES = (
    "discovery/index.py",
    "discovery/signatures.py",
    "config/native_catalog.yaml",
    "config/override_manifest.yaml",
    "config/metadata.py",
    "config/metadata_baseline.yaml",
    "server/jupyter_peaks/core/tools.py",
)
#: Curated presentation documents: upstream aliases/notes (native tier) and
#: the project black-box manifest.  ``load_overrides()`` merges their ``apis``
#: blocks; the retired single-file config/manifest.yaml is read only as a
#: compatibility fallback when both new catalogs are absent.
_NATIVE_CATALOG = "config/native_catalog.yaml"
_OVERRIDE_MANIFEST = "config/override_manifest.yaml"
_LEGACY_MANIFEST = "config/manifest.yaml"

#: peaks modules whose entries must never be surfaced by search/get.  The
#: hvplot-based ``iplot`` accessor is intentionally hidden: "interactive"
#: intent resolves to the native Qt viewer ``disp`` instead.  The PXT reader
#: module is hidden too: loading is a single curated verb (``load_data`` in
#: peaksMCP.overrides), so the raw ``load_pxt`` implementation stays internal.
_HIDDEN_MODULES = frozenset(
    {"peaks.core.GUI.iplot.hvplot", "peaksMCP.pxt_utils.loader"}
)

#: Presentation tiers inside one index: ``override`` marks the peaksMCP
#: project-added APIs (black-box tier, preferred by search); everything else
#: is the native ``peaks`` tier.
TIER_OVERRIDE = "override"
TIER_NATIVE = "native"

#: Canonical module for every project (override-tier) API.  Search/get expose
#: project functions ONLY under ``module:peaksMCP.overrides:<name>``; the
#: implementation module (where the function actually lives, e.g.
#: ``peaksMCP.plotting.layout``) is projection detail and is never shown.
#: The original implementation id is kept in each entry's ``legacy_ids`` so
#: :meth:`ApiIndex.get` can still resolve pre-canonical ids.
CANONICAL_MODULE = "peaksMCP.overrides"

#: Searched-namespace label for the fallback stage: no override name/alias hit
#: exactly, so the query ranked against the full index (override candidates
#: plus native).  Reported as ``searched_namespace="mixed"`` — it is no longer
#: mislabelled as "native" (that word now means the entry tier only).
TIER_MIXED = "mixed"

#: Minimum relevance that qualifies a stage-1 override hit in the two-tier
#: search (exact alias tier, 900, and above — i.e. the query equals an
#: override's canonical name or one of its aliases).  Weaker partial matches
#: (name/alias substrings, 650-800) intentionally fall through to the native
#: search so short generic names like ``plot`` are not hijacked by
#: ``plot_batch``.
OVERRIDE_MIN_SCORE = 900

#: Full-tree fingerprint checks are expensive (an os.walk over every Peaks +
#: peaksMCP source file).  ``ApiIndex.is_stale()`` runs on every search and
#: write, so the fingerprint result is cached for this window: source edits are
#: picked up within a few seconds without paying the walk cost per call.
STALE_REFRESH_INTERVAL_S = 5.0


def _tokens(text: str) -> set[str]:
    """Return normalized English and CJK search tokens.

    CJK runs are expanded with bigrams so natural-language phrases can overlap
    aliases/summaries even when the exact word order differs (e.g. ``扣除背景``
    and the alias ``背景扣除`` share ``扣除``/``背景``).
    """
    words = _TOKEN_RE.findall((text or "").lower())
    tokens = set(words)
    for word in words:
        if not word:
            continue
        if word[0].isascii():
            tokens.update(part for part in word.split("_") if part)
        elif len(word) >= 2:
            tokens.update(word[index : index + 2] for index in range(len(word) - 1))
    return tokens


def _stat_signature(path: str) -> tuple[int, int] | None:
    """Return ``(mtime_ns, size)`` for a source file, or None when missing."""
    try:
        stat = os.stat(path)
        return (stat.st_mtime_ns, stat.st_size)
    except OSError:
        return None


def source_signature(pkg_dir: str | os.PathLike[str] | None = None) -> tuple[Any, ...]:
    """Cheap mtime/size signature of the installed Peaks package + adapter sources.

    Used to detect that the index source changed after the in-kernel index was
    built (requiring a kernel restart), without hashing every file's bytes.
    """
    import peaks

    package_dir = os.fspath(pkg_dir) if pkg_dir is not None else os.path.dirname(peaks.__file__)
    signature: list[Any] = [getattr(peaks, "__version__", "?")]
    for root, dirs, files in os.walk(package_dir):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS)
        for filename in sorted(files):
            if not filename.endswith(".py"):
                continue
            path = os.path.join(root, filename)
            signature.append((path, _stat_signature(path)))
    # peaksMCP's own analysis API is indexed too, so its source must be part of
    # the fingerprint (an index built before a plotting/workflow change would
    # otherwise never be flagged stale).
    import peaksMCP

    for root, dirs, files in os.walk(os.path.dirname(peaksMCP.__file__)):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS)
        for filename in sorted(files):
            if not filename.endswith(".py"):
                continue
            path = os.path.join(root, filename)
            signature.append((path, _stat_signature(path)))
    for relative in _ADAPTER_SOURCES:
        path = str(_INDEX_PACKAGE / relative)
        signature.append((path, _stat_signature(path)))
    return tuple(signature)


def source_fingerprint(pkg_dir: str | os.PathLike[str] | None = None) -> str:
    """Deterministic hash of the current Peaks + adapter source tree."""
    return hashlib.sha256(repr(source_signature(pkg_dir)).encode()).hexdigest()


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    try:
        rendered = ast.unparse(node.args)
    except Exception:
        rendered = ""
    return f"{node.name}({rendered})"


def _canonical(scope: str, module: str, name: str) -> str:
    return f"{scope}:{module}:{name}"


def _entry(scope: str, module: str, name: str, **extra: Any) -> dict[str, Any]:
    return {
        "id": _canonical(scope, module, name),
        "scope": scope,
        "module": module,
        "name": name,
        "summary": "",
        "docstring": "",
        **extra,
    }


def _scan_accessor_class(
    entries: list[dict[str, Any]],
    accessor_cls: type,
    module: str,
    scope: str,
    accessor_name: str,
) -> None:
    """Index the public methods/properties of a Peaks custom accessor class.

    ``da.metadata`` is an ``_CachedAccessor`` whose class (``Metadata``) carries
    methods such as ``set_EF_correction`` / ``get_EF_correction`` that the main
    xarray-descriptor scan never sees.  Each becomes an entry with scope
    ``<accessor_name>`` (e.g. ``metadata:peaks.core.metadata.metadata_methods:set_EF_correction``)
    so ``peaks_search_api`` / ``peaks_get_api`` can resolve them.
    """
    for member in sorted(dir(accessor_cls)):
        if member.startswith("_"):
            continue
        obj = getattr(accessor_cls, member, None)
        if not callable(obj) and not isinstance(obj, property):
            continue
        if getattr(obj, "__module__", module) not in (None, module, accessor_cls.__module__):
            # Inherited from elsewhere (e.g. object utilities): skip.
            if not getattr(obj, "__module__", "").startswith("peaks."):
                continue
        doc = (getattr(obj, "__doc__", "") or "").strip()
        entries.append(
            _entry(
                accessor_name,
                accessor_cls.__module__,
                member,
                kind="property" if isinstance(obj, property) else "method",
                func_name=member,
                accessor_class=accessor_cls.__name__,
                summary=doc.splitlines()[0][:240] if doc else "",
                docstring=doc,
            )
        )


def scan_runtime() -> list[dict[str, Any]]:
    """Inspect Peaks-owned xarray descriptors and top-level exports.

    Returns
    -------
    list of dict
        Structured API entries for the running Peaks installation.
    """
    import peaks
    import xarray as xr

    entries: list[dict[str, Any]] = []
    for class_name, scope in (
        ("DataArray", "dataarray"),
        ("Dataset", "dataset"),
        ("DataTree", "datatree"),
    ):
        cls = getattr(xr, class_name)
        for name, descriptor in sorted(cls.__dict__.items()):
            if name.startswith("_"):
                continue
            dtype = type(descriptor).__name__
            module = getattr(descriptor, "module_name", "") or getattr(
                descriptor, "__module__", ""
            )
            accessor = getattr(descriptor, "_accessor", None)
            if dtype == "_CachedAccessor" and isinstance(accessor, type):
                # _CachedAccessor's own __module__ is xarray.core.accessor;
                # the real owning module lives on the accessor class (e.g.
                # peaks.core.metadata.metadata_methods for da.metadata).  Override
                # unconditionally when the accessor class is peaks-owned.
                accessor_module = getattr(accessor, "__module__", "")
                if accessor_module.startswith("peaks."):
                    module = accessor_module
                elif not module:
                    module = accessor_module
            is_peaks = module.startswith("peaks.")
            is_lazy = dtype == "LazyAccessorDescriptor"
            is_cached = dtype == "_CachedAccessor" and (
                is_peaks or isinstance(accessor, str)
            )
            if not (is_peaks or is_lazy or is_cached):
                continue
            func_name = getattr(descriptor, "func_name", None) or name
            doc = getattr(descriptor, "__doc__", "") or ""
            if dtype == "_CachedAccessor" and isinstance(accessor, type):
                # xarray's _CachedAccessor.__doc__ is the descriptor class's own
                # docstring (e.g. da.iplot -> "Custom property-like object ...");
                # the real one lives on the wrapped accessor's __call__ (e.g.
                # da.iplot -> HVPlotAccessor.__call__).  Use that instead so the
                # interactive widget APIs are discoverable by intent.
                call_doc = getattr(accessor.__call__, "__doc__", "")
                doc = call_doc or (getattr(accessor, "__doc__", "") or "")
            doc = doc.strip()
            entries.append(
                _entry(
                    scope,
                    module,
                    name,
                    kind="accessor" if is_cached else "method",
                    func_name=func_name,
                    accessor_type=dtype,
                    summary=doc.splitlines()[0][:240] if doc else "",
                    docstring=doc,
                )
            )
            # Peaks custom accessor classes (e.g. ``da.metadata``) expose their
            # own methods (``set_EF_correction``, ``get_EF_correction``, ...) that
            # are otherwise invisible to the index.  Recursively scan those too,
            # so ``peaks_search_api``/``peaks_get_api`` can find them.
            if is_cached and isinstance(accessor, type) and accessor_module.startswith("peaks."):
                _scan_accessor_class(entries, accessor, accessor_module, scope, name)

    for name in sorted(dir(peaks)):
        if name.startswith("_"):
            continue
        obj = getattr(peaks, name)
        module = getattr(obj, "__module__", "") or "peaks"
        if not module.startswith("peaks"):
            module = "peaks"
        doc = (getattr(obj, "__doc__", "") or "").strip()
        entries.append(
            _entry(
                "top_level",
                module,
                name,
                kind="callable" if callable(obj) else "symbol",
                func_name=name,
                summary=doc.splitlines()[0][:240] if doc else "",
                docstring=doc,
            )
        )
    return entries


def scan_modules(
    package_dir: str | os.PathLike[str],
    package_name: str = "peaks",
    include_prefixes: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Statically scan public module functions without importing every module.

    Parameters
    ----------
    package_dir : path-like
        Root directory of the package to scan (e.g. the installed ``peaks`` or
        the ``peaksMCP`` package).
    package_name : str, default "peaks"
        Import name used to build canonical module ids (e.g. ``peaksMCP``).
    include_prefixes : tuple of str, optional
        Only keep entries whose module starts with one of these prefixes
        (e.g. ``("peaksMCP.plotting",)``); ``None`` keeps everything.

    Returns
    -------
    list of dict
        Public module-level functions with source signatures and summaries.
    """
    package_dir = os.fspath(package_dir)
    entries: list[dict[str, Any]] = []
    for root, dirs, files in os.walk(package_dir):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS)
        for filename in sorted(files):
            if not filename.endswith(".py"):
                continue
            path = os.path.join(root, filename)
            relative = os.path.relpath(root, package_dir)
            module = package_name if relative == "." else f"{package_name}." + relative.replace(os.sep, ".")
            module += "." + filename[:-3]
            if module.startswith("peaks.SARPES") or "._lazy_import" in module:
                continue
            if include_prefixes is not None and not module.startswith(include_prefixes):
                continue
            try:
                source = Path(path).read_text(encoding="utf-8", errors="replace")
                tree = ast.parse(source, filename=path)
            except (OSError, SyntaxError):
                continue
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
                    doc = ast.get_docstring(node, clean=True) or ""
                    entries.append(
                        _entry(
                            "module",
                            module,
                            node.name,
                            kind="function",
                            func_name=node.name,
                            summary=doc.splitlines()[0][:240] if doc else "",
                            docstring=doc.strip(),
                            signature=_signature(node),
                            source_path=path,
                        )
                    )
    return entries


def _read_config_document(path: Path) -> dict[str, Any]:
    """Parse one curated YAML document; an unreadable file yields {}."""
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}


def _load_document_checked(path: Path) -> dict[str, Any]:
    """Parse one curated document strictly for the default configuration.

    Duplicate mapping keys, malformed YAML or missing files raise instead of
    silently degrading to an empty catalog: discovery must never build an
    index from a damaged curated config.
    """
    from peaksMCP.config.schema import load_yaml_unique

    try:
        return load_yaml_unique(path.read_text(encoding="utf-8"))
    except OSError:
        raise
    except yaml.YAMLError as exc:
        raise ValueError(f"{path.name}: invalid curated YAML: {exc}") from exc


#: Directory holding the curated catalogs (patched in tests).
_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def load_overrides(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Load the curated API presentation documents.

    Parameters
    ----------
    path : str or os.PathLike, optional
        Explicit override file to read (any legacy shape, including the
        retired single-file ``config/manifest.yaml``, parsed leniently).
        When omitted, the merged view of ``config/native_catalog.yaml``
        (upstream, native tier) and ``config/override_manifest.yaml``
        (project APIs) is returned after strict schema validation; the
        legacy single file is read as a compatibility fallback only when
        both new catalogs are absent.

    Returns
    -------
    dict
        Merged document with ``version``, ``apis`` (native entries merged
        with project entries) and the override manifest's ``project`` seeds.

    Raises
    ------
    ValueError
        When the default configuration is damaged (duplicate keys, malformed
        YAML or schema violations) — loud failure beats an empty catalog.
    """
    if path is not None:
        return _read_config_document(Path(path))
    try:
        native = _load_document_checked(_CONFIG_DIR / "native_catalog.yaml")
        overrides = _load_document_checked(_CONFIG_DIR / "override_manifest.yaml")
    except OSError:
        # Compatibility: pre-split checkouts ship only the single file.
        legacy = _read_config_document(_CONFIG_DIR / "manifest.yaml")
        if legacy:
            return legacy
        return {}
    from peaksMCP.config.schema import validate_documents

    errors = validate_documents(native, overrides)
    if errors:
        raise ValueError("curated config invalid:\n- " + "\n- ".join(errors))
    merged: dict[str, Any] = {"version": overrides.get("version", 3)}
    merged_apis: dict[str, Any] = {}
    for document in (native, overrides):
        merged_apis.update(document.get("apis") or {})
    merged["apis"] = merged_apis
    if overrides.get("project"):
        merged["project"] = overrides["project"]
    return merged


def load_api_overrides(path: str | os.PathLike[str] | None = None) -> dict[str, dict[str, Any]]:
    """Return the per-API presentation entries, keyed by API name.

    Each entry may carry ``aliases`` (extra search terms), ``docstring_note``
    (prepended to the live docstring), ``module`` and ``project``.

    Parameters
    ----------
    path : str or os.PathLike, optional
        Explicit override file to read; defaults to the merged
        native/override catalogs (see :func:`load_overrides`).

    Returns
    -------
    dict of dict
        Mapping of API name to its curated configuration.

    Examples
    --------
    >>> "k_convert" in load_api_overrides()
    True
    """
    entries = load_overrides(path).get("apis") or {}
    return {str(name): dict(config or {}) for name, config in entries.items()}


def load_project_added(path: str | os.PathLike[str] | None = None) -> set[str]:
    """Return the ``module:name`` ids this project adds to the API.

    Derived from the ``project: true`` flag in ``config/override_manifest.yaml``,
    so the audited exposure record cannot drift from the aliases and notes that
    sit next to it. :func:`build_index` marks every matching entry with
    ``project_added=True`` so callers can tell project code from upstream.

    Parameters
    ----------
    path : str or os.PathLike, optional
        Explicit override file to read; defaults to the merged
        native/override catalogs (see :func:`load_overrides`).

    Returns
    -------
    set of str
        ``module:name`` identifiers such as
        ``peaksMCP.plotting.layout:plot_batch``. Entries missing a ``module``
        are ignored.

    Examples
    --------
    >>> "peaksMCP.plotting.layout:plot_batch" in load_project_added()
    True
    """
    return {
        f"{config['module']}:{name}"
        for name, config in load_api_overrides(path).items()
        if config.get("project") and config.get("module")
    }


def _merge_duplicates(entries: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for item in entries:
        canonical = str(item["id"])
        existing = merged.get(canonical)
        if existing is None:
            merged[canonical] = dict(item)
        else:
            for key, value in item.items():
                if value and not existing.get(key):
                    existing[key] = value
    return sorted(merged.values(), key=lambda item: str(item["id"]))


def build_index() -> ApiIndex:
    """Build the complete live API index for the installed Peaks package.

    Returns
    -------
    ApiIndex
        In-memory entries, Peaks version and deterministic fingerprint.

    Examples
    --------
    >>> index = build_index()
    >>> bool(index.entries)
    True
    """
    import peaks

    package_dir = os.path.dirname(peaks.__file__)
    entries = _merge_duplicates([*scan_runtime(), *scan_modules(package_dir)])
    # The agent should also discover peaksMCP's own analysis API (facades /
    # plotting / workflows / conversion) without relying on the skill file:
    # scan the core data layer of this package into the same index.
    import peaksMCP

    peaksmcp_dir = os.path.dirname(peaksMCP.__file__)
    entries = _merge_duplicates(
        [
            *entries,
            *scan_modules(
                peaksmcp_dir,
                package_name="peaksMCP",
                include_prefixes=(
                    "peaksMCP.overrides",
                    "peaksMCP.plotting",
                    "peaksMCP.workflows",
                    "peaksMCP.pxt_utils",
                    "peaksMCP.batch",
                ),
            ),
        ]
    )
    api_overrides = load_api_overrides()
    project_added = load_project_added()
    for item in entries:
        names = {item["name"], item["id"]}
        item_aliases: list[str] = []
        note: str | None = None
        for key, config in api_overrides.items():
            if key in names or key.lower() == str(item["name"]).lower():
                item_aliases.extend(str(value) for value in config.get("aliases") or [])
                if config.get("docstring_note"):
                    note = str(config["docstring_note"])
                if config.get("exposure"):
                    item["exposure"] = str(config["exposure"])
        if note is not None:
            item["docstring_note"] = note
        item["aliases"] = sorted(set([*item.get("aliases", []), *item_aliases]))
        if f"{item.get('module')}:{item.get('name')}" in project_added:
            item["project_added"] = True
        # Override tier = this project's black-box APIs; everything else native.
        item["tier"] = TIER_OVERRIDE if item.get("project_added") else TIER_NATIVE
    # Single-canonical projection: every project entry is exposed ONLY as
    # ``module:peaksMCP.overrides:<name>``.  The implementation-module id is
    # preserved in ``legacy_ids`` so ApiIndex.get still resolves
    # pre-canonical ids (and python imports of the implementation modules stay
    # valid, they are just projection detail now).
    for item in entries:
        if not item.get("project_added"):
            continue
        original_id = str(item["id"])
        item["id"] = f"module:{CANONICAL_MODULE}:{item['name']}"
        item["module"] = CANONICAL_MODULE
        legacy_ids = list(item.get("legacy_ids") or [])
        if original_id not in legacy_ids:
            legacy_ids.append(original_id)
        item["legacy_ids"] = legacy_ids
    fingerprint = source_fingerprint()
    entries = [item for item in entries if item.get("module") not in _HIDDEN_MODULES]
    return ApiIndex(entries=entries, peaks_version=getattr(peaks, "__version__", "?"), fingerprint=fingerprint)


def _rank_entries(
    entries: list[dict[str, Any]],
    query: str,
    scope: str,
    tier: str = "all",
    *,
    include_advanced: bool = False,
) -> list[tuple[int, str, dict[str, Any]]]:
    """Score entries for one query with deterministic lexical ranking.

    Scores mirror the search contract: exact name 1000, exact alias 900,
    name-prefix 800, name-substring 700, alias-substring 650, then the
    token-overlap fallback. ``tier`` restricts the candidate set to ``all`` /
    :data:`TIER_OVERRIDE` / :data:`TIER_NATIVE`.

    Exposure gating: ``advanced`` entries (the curated low-level layer) are
    excluded unless the query hits them exactly (score >= 900) or
    ``include_advanced`` is set — generic/fuzzy queries must never surface
    them by default.
    """
    qtokens = _tokens(query)
    scored: list[tuple[int, str, dict[str, Any]]] = []
    for item in entries:
        if scope != "all" and item["scope"] != scope:
            continue
        if tier == TIER_OVERRIDE and not item.get("project_added"):
            continue
        if tier == TIER_NATIVE and item.get("project_added"):
            continue
        name = str(item.get("name", "")).lower()
        module = str(item.get("module", "")).lower()
        summary = str(item.get("summary", "")).lower()
        docstring = str(item.get("docstring", "")).lower()
        aliases = [str(alias).lower() for alias in item.get("aliases", [])]
        if name == query:
            score = 1000
        elif query in aliases:
            score = 900
        elif name.startswith(query):
            score = 800
        elif query in name:
            score = 700
        elif any(query in alias or alias in query for alias in aliases):
            score = 650
        else:
            if item.get("kind") == "symbol":
                # Non-callable re-exported modules (peaks.xr, peaks.netcdf, ...)
                # are informational only; they must not hijack task queries via
                # token overlap. They stay reachable through exact/prefix names.
                score = 0
            else:
                name_key = str(item.get("name", "")).lower()
                name_tokens = _tokens(name_key)
                if name_key in qtokens:
                    # The API name is itself a token of the phrase query: a strong
                    # signal that outranks any incidental partial token overlap.
                    score = 500 + 40 * len(qtokens & name_tokens)
                else:
                    name_overlap = len(qtokens & name_tokens)
                    alias_overlap = len(qtokens & set().union(*(_tokens(a) for a in aliases))) if aliases else 0
                    summary_overlap = len(qtokens & _tokens(summary))
                    docstring_overlap = len(qtokens & _tokens(docstring))
                    module_overlap = len(qtokens & _tokens(module))
                    score = name_overlap * 100 + alias_overlap * 80 + summary_overlap * 30 + docstring_overlap * 20 + module_overlap * 15
        if score:
            if (
                item.get("exposure") == "advanced"
                and not include_advanced
                and score < 900
            ):
                continue
            scored.append((score, str(item["id"]), item))
    scored.sort(key=lambda row: (-row[0], row[1]))
    return scored


def _trim_rows(
    rows: list[tuple[int, str, dict[str, Any]]], limit: int
) -> list[dict[str, Any]]:
    """Deduplicate ranked rows by (module, name) and cap to ``limit``."""
    seen: set[tuple[str, str]] = set()
    output: list[dict[str, Any]] = []
    for _, _, item in rows:
        key = (str(item.get("module", "")), str(item.get("name", "")))
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
        if len(output) == limit:
            break
    return output


def _filter_entries(
    entries: list[dict[str, Any]],
    scope: str,
    tier: str,
    *,
    include_advanced: bool = False,
) -> list[dict[str, Any]]:
    """Scope/tier filter used by the empty-query listing path.

    Advanced (curated low-level) entries are excluded from listings unless
    ``include_advanced`` is set.
    """

    def keep(entry: dict[str, Any]) -> bool:
        if scope != "all" and entry["scope"] != scope:
            return False
        if not include_advanced and entry.get("exposure") == "advanced":
            return False
        if tier == TIER_OVERRIDE:
            return bool(entry.get("project_added"))
        if tier == TIER_NATIVE:
            return not entry.get("project_added")
        return True

    return [entry for entry in entries if keep(entry)]


def _compact_entry(item: dict[str, Any], score: int | None = None) -> dict[str, Any]:
    """One search result row: canonical id, name, tier and a one-line summary.

    Deliberately NOT the full index entry: no docstring body, aliases, legacy
    ids, signature or source paths.  Model-facing search output stays small
    and black-box; the detail belongs to peaks_get_api (canonical id only).
    """
    summary = str(item.get("summary") or "")
    if not summary:
        summary = (str(item.get("docstring") or "").splitlines() or [""])[0]
    return {
        "id": str(item.get("id") or ""),
        "name": str(item.get("name") or ""),
        "module": item.get("module"),
        "scope": item.get("scope"),
        "tier": item.get("tier")
        or (TIER_OVERRIDE if item.get("project_added") else TIER_NATIVE),
        "exposure": item.get("exposure"),
        "summary": summary[:160],
        "score": score,
    }


def _compact_rows(
    rows: list[tuple[int, str, dict[str, Any]]], limit: int
) -> list[dict[str, Any]]:
    """Deduplicate ranked rows by (module, name), keep score, cap to limit."""
    seen: set[tuple[str, str]] = set()
    output: list[dict[str, Any]] = []
    for score, _, item in rows:
        key = (str(item.get("module", "")), str(item.get("name", "")))
        if key in seen:
            continue
        seen.add(key)
        output.append(_compact_entry(item, score))
        if len(output) == limit:
            break
    return output


def search_index(
    entries: list[dict[str, Any]],
    query: str,
    scope: str = "all",
    limit: int = 5,
    tier: str = "all",
    *,
    include_advanced: bool = False,
) -> list[dict[str, Any]]:
    """Rank API entries using deterministic field-aware lexical matching.

    Parameters
    ----------
    entries : list of dict
        Dynamic API records to rank.
    query : str
        API name, module fragment, alias or natural-language phrase.
    scope : str, default "all"
        Optional DataArray, Dataset, DataTree or top-level scope filter.
    limit : int, default 5
        Maximum number of unique results, clamped to 1 through 20.
    tier : str, default "all"
        Optional ``"all"`` / ``"override"`` / ``"native"`` candidate filter.
        The model-facing two-stage behaviour lives in
        :func:`search_index_tiered`.

    Returns
    -------
    list of dict
        Best matching API records in descending relevance order.

    Examples
    --------
    >>> search_index([{"id": "top:peaks:plot", "name": "plot", "scope": "top", "module": "peaks"}], "plot")[0]["name"]
    'plot'
    """
    if scope not in _SCOPES:
        raise ValueError(f"invalid scope {scope!r}; expected one of {sorted(_SCOPES)}")
    query = query.strip().lower()
    limit = max(1, min(int(limit), 20))
    if not query:
        return [
            _compact_entry(item)
            for item in _filter_entries(entries, scope, tier, include_advanced=include_advanced)[:limit]
        ]
    return _compact_rows(
        _rank_entries(entries, query, scope, tier, include_advanced=include_advanced),
        limit,
    )


def search_index_tiered(
    entries: list[dict[str, Any]],
    query: str,
    scope: str = "all",
    limit: int = 5,
    *,
    include_advanced: bool = False,
) -> tuple[str, list[dict[str, Any]]]:
    """Two-stage override-first search used by the MCP search tool.

    Stage 1 ranks only the override tier. If its best hit reaches
    :data:`OVERRIDE_MIN_SCORE` (the query equals an override's canonical name
    or one of its aliases), the override matches are returned alone under
    searched-namespace ``override``. Otherwise the search falls back to the
    full index under ``mixed`` — override candidates plus native peaks APIs
    are ranked together (this is a mixed namespace, not a native-only list).
    An empty query lists the whole index under ``all``.

    Returns
    -------
    tuple of (str, list of dict)
        Searched-namespace label (``override`` / ``mixed`` / ``all``) followed
        by the best matching records.
    """
    if scope not in _SCOPES:
        raise ValueError(f"invalid scope {scope!r}; expected one of {sorted(_SCOPES)}")
    query = query.strip().lower()
    limit = max(1, min(int(limit), 20))
    if not query:
        return "all", [
            _compact_entry(item)
            for item in _filter_entries(entries, scope, "all", include_advanced=include_advanced)[:limit]
        ]
    override_rows = _rank_entries(
        entries, query, scope, TIER_OVERRIDE, include_advanced=include_advanced
    )
    if override_rows and override_rows[0][0] >= OVERRIDE_MIN_SCORE:
        return TIER_OVERRIDE, _compact_rows(override_rows, limit)
    return TIER_MIXED, _compact_rows(
        _rank_entries(entries, query, scope, "all", include_advanced=include_advanced),
        limit,
    )


@dataclass(slots=True)
class ApiIndex:
    """In-memory API index owned by one Jupyter kernel."""

    entries: list[dict[str, Any]]
    peaks_version: str
    fingerprint: str
    #: TTL cache for the expensive full-tree fingerprint check (is_stale).
    _stale_checked_at: float = field(default=0.0, init=False, repr=False)
    _stale_result: bool = field(default=False, init=False, repr=False)

    def search(
        self,
        query: str,
        scope: str = "all",
        limit: int = 5,
        *,
        include_advanced: bool = False,
    ) -> list[dict[str, Any]]:
        """Two-tier override-first search; returns compact entries only.

        Override (peaksMCP project) APIs win whenever the query exactly matches
        one of their names or aliases; otherwise the full index is searched.
        Advanced entries surface only on exact hits or when
        ``include_advanced=True``.  See :meth:`search_tiered` for the label.
        """
        return search_index_tiered(
            self.entries, query, scope, limit, include_advanced=include_advanced
        )[1]

    def search_tiered(
        self,
        query: str,
        scope: str = "all",
        limit: int = 5,
        *,
        include_advanced: bool = False,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Two-stage override-first search with the searched-namespace label.

        Returns a ``(searched_namespace, matches)`` pair where the label is
        ``"override"`` (query hit an override name/alias exactly), ``"mixed"``
        (fell back to the full index) or ``"all"`` (empty query).
        """
        return search_index_tiered(
            self.entries, query, scope, limit, include_advanced=include_advanced
        )

    def get(self, canonical_id: str) -> dict[str, Any] | None:
        """Return one canonical entry by its EXACT canonical id.

        Strict division of labour with :meth:`search`: search returns compact
        rows (canonical id, name, tier, one-line summary); get accepts only
        such a canonical id (e.g. ``module:peaksMCP.overrides:show_mapping_slice``)
        and returns the internal entry for detail lookup.  Bare names, search
        aliases and legacy implementation ids are refused — a caller that has
        only a name or alias must search first.
        """
        wanted = canonical_id.strip()
        for entry in self.entries:
            if entry["id"] == wanted:
                return entry
        return None

    def is_stale(self) -> bool:
        """Return True when Peaks or adapter source changed after this index was built.

        The full-tree fingerprint walk is cached for ``STALE_REFRESH_INTERVAL_S``
        seconds so per-call staleness checks stay cheap (searches and writes call
        this on every request).  Source edits are reflected within the window;
        the index itself is hot-rebuilt by :func:`ensure_fresh_index` once a
        stale fingerprint is observed.
        """
        now = time.monotonic()
        if now - self._stale_checked_at >= STALE_REFRESH_INTERVAL_S:
            self._stale_checked_at = now
            self._stale_result = source_fingerprint() != self.fingerprint
        return self._stale_result
