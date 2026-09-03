"""Build and search a live index of the installed Peaks API.

The index combines runtime descriptor inspection with a static AST scan. It is built once per
kernel so it always reflects the installed Peaks version without requiring a generated catalog.
"""

from __future__ import annotations

import ast
import hashlib
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
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
    "discovery/api_overrides.yaml",
    "config/metadata.py",
    "config/metadata_baseline.yaml",
    "server/jupyter_peaks/core/tools.py",
)


class IndexStaleError(RuntimeError):
    """Raised when the cached API index no longer matches the source tree.

    The index is intentionally rebuilt only by a kernel restart (never hot-
    rebuilt), so tools fail fast and tell the model to restart instead of
    serving a stale surface mixed with fresh source docs.
    """


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


def _summary(node: ast.AST) -> str:
    doc = ast.get_docstring(node, clean=True) or ""
    return doc.splitlines()[0][:240] if doc else ""


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
        doc = getattr(obj, "__doc__", "") or ""
        entries.append(
            _entry(
                accessor_name,
                accessor_cls.__module__,
                member,
                kind="property" if isinstance(obj, property) else "method",
                func_name=member,
                accessor_class=accessor_cls.__name__,
                summary=doc.strip().splitlines()[0][:240] if doc.strip() else "",
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
            entries.append(
                _entry(
                    scope,
                    module,
                    name,
                    kind="accessor" if is_cached else "method",
                    func_name=func_name,
                    accessor_type=dtype,
                    summary=doc.strip().splitlines()[0][:240] if doc.strip() else "",
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
        doc = getattr(obj, "__doc__", "") or ""
        entries.append(
            _entry(
                "top_level",
                module,
                name,
                kind="callable" if callable(obj) else "symbol",
                func_name=name,
                summary=doc.strip().splitlines()[0][:240] if doc.strip() else "",
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
                    entries.append(
                        _entry(
                            "module",
                            module,
                            node.name,
                            kind="function",
                            func_name=node.name,
                            summary=_summary(node),
                            signature=_signature(node),
                            source_path=path,
                        )
                    )
    return entries


def load_overrides(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Load curated aliases and metadata overrides."""
    target = Path(path) if path else Path(__file__).with_name("api_overrides.yaml")
    try:
        return yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {"aliases": {}, "overrides": {}}


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
    # The agent should also discover peaksMCP's own analysis API (plotting /
    # workflows / conversion) without relying on the skill file: scan the core
    # data layer of this package into the same index.
    import peaksMCP

    peaksmcp_dir = os.path.dirname(peaksMCP.__file__)
    entries = _merge_duplicates(
        [
            *entries,
            *scan_modules(
                peaksmcp_dir,
                package_name="peaksMCP",
                include_prefixes=(
                    "peaksMCP.plotting",
                    "peaksMCP.workflows",
                    "peaksMCP.pxt_utils",
                    "peaksMCP.batch",
                ),
            ),
        ]
    )
    overrides = load_overrides()
    aliases = overrides.get("aliases") or {}
    per_api = overrides.get("overrides") or {}
    for item in entries:
        names = {item["name"], item["id"]}
        item_aliases: list[str] = []
        for key, values in aliases.items():
            if key in names or key.lower() == str(item["name"]).lower():
                item_aliases.extend(str(value) for value in values)
        override = per_api.get(item["id"], per_api.get(item["name"], {})) or {}
        item.update(override)
        item["aliases"] = sorted(set([*item.get("aliases", []), *item_aliases]))
    fingerprint = source_fingerprint()
    return ApiIndex(entries=entries, peaks_version=getattr(peaks, "__version__", "?"), fingerprint=fingerprint)


def search_index(
    entries: list[dict[str, Any]], query: str, scope: str = "all", limit: int = 5
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
        return [entry for entry in entries if scope == "all" or entry["scope"] == scope][:limit]
    qtokens = _tokens(query)
    scored: list[tuple[int, str, dict[str, Any]]] = []
    for item in entries:
        if scope != "all" and item["scope"] != scope:
            continue
        name = str(item.get("name", "")).lower()
        module = str(item.get("module", "")).lower()
        summary = str(item.get("summary", "")).lower()
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
                    module_overlap = len(qtokens & _tokens(module))
                    score = name_overlap * 100 + alias_overlap * 80 + summary_overlap * 30 + module_overlap * 15
        if score:
            scored.append((score, str(item["id"]), item))
    scored.sort(key=lambda row: (-row[0], row[1]))
    seen: set[tuple[str, str]] = set()
    output: list[dict[str, Any]] = []
    for _, _, item in scored:
        key = (str(item.get("module", "")), str(item.get("name", "")))
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
        if len(output) == limit:
            break
    return output


@dataclass(slots=True)
class ApiIndex:
    """In-memory API index owned by one Jupyter kernel."""

    entries: list[dict[str, Any]]
    peaks_version: str
    fingerprint: str

    def search(self, query: str, scope: str = "all", limit: int = 5) -> list[dict[str, Any]]:
        """Search the index and return compact entries."""
        return search_index(self.entries, query, scope, limit)

    def get(self, canonical_id: str) -> dict[str, Any] | None:
        """Return one canonical entry.

        Accepts the full canonical ID returned by :meth:`search` (e.g.
        ``module:peaksMCP.workflows.cut_preprocessing:process_cut``), a bare
        API name (``process_cut``), or one of its search aliases
        (``preprocess_cut``) — aliases resolve to the canonical entry so a
        typo'd ``peaks_get_api`` still returns the real API instead of an
        unknown-ID error.
        """
        wanted = canonical_id.strip()
        for entry in self.entries:
            if entry["id"] == wanted or entry["name"] == wanted:
                return entry
        lowered = wanted.lower()
        if lowered:
            for entry in self.entries:
                if lowered in {str(alias).lower() for alias in entry.get("aliases", [])}:
                    return entry
        return None

    def is_stale(self) -> bool:
        """Return True when Peaks or adapter source changed after this index was built.

        The index is never hot-rebuilt; a stale index must be rebuilt by a kernel
        restart (``peaksMCP restart kernel``).
        """
        return source_fingerprint() != self.fingerprint
