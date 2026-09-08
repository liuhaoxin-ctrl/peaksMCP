"""Read the curated presentation layer for MCP tools and runtime prompts."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


@lru_cache(maxsize=1)
def _document() -> dict[str, Any]:
    return yaml.safe_load(Path(__file__).with_name("metadata_baseline.yaml").read_text(encoding="utf-8")) or {}


@lru_cache(maxsize=1)
def _prompts_document() -> dict[str, Any]:
    return yaml.safe_load(Path(__file__).with_name("prompts.yaml").read_text(encoding="utf-8")) or {}


def prompts() -> dict[str, Any]:
    """Return the curated runtime prompt text (the L3 layer).

    Model/user-facing copy that used to be hardcoded in Python source
    (``tools.py`` interactive/guidance text, ``notebook_unsafe.py`` hard-block
    replies and the code/ipython scanner issue descriptions) is stored in
    ``prompts.yaml`` so wording can be reviewed and edited without touching
    code. Python only formats the ``{placeholders}``.

    Returns
    -------
    dict
        Nested prompt groups: ``interactive_omitted_note``,
        ``list_resources_guidance``, ``notebook_unsafe``, ``scanner``,
        ``ipython``.

    Examples
    --------
    >>> "scanner" in prompts()
    True
    """
    return _prompts_document().get("prompts") or {}


def tool_metadata(name: str) -> dict[str, str]:
    """Return presentation metadata for one MCP tool.

    Parameters
    ----------
    name : str
        Stable MCP tool name.

    Returns
    -------
    dict of str
        Curated ``title`` and ``description`` values.

    Examples
    --------
    >>> tool_metadata("peaks_search_api")["title"]
    'Search Peaks API'
    """
    item = (_document().get("tools") or {}).get(name, {})
    return {"title": str(item.get("title") or name), "description": str(item.get("description") or "")}


def list_resources() -> list[str]:
    """Return the canonical plotting-format resource ids.

    Examples
    --------
    >>> "dispersion_grid" in list_resources()
    True
    """
    return sorted((_document().get("resources") or {}).keys())


def resource_metadata(resource_id: str) -> dict[str, Any]:
    """Return one canonical plotting-format template.

    Parameters
    ----------
    resource_id : str
        Resource id from :func:`list_resources`.

    Returns
    -------
    dict
        ``title``, ``when_to_use``, ``figure`` styling contract and the
        ``template`` code snippet to run verbatim.

    Raises
    ------
    KeyError
        If the resource id is unknown.
    """
    item = (_document().get("resources") or {}).get(resource_id)
    if item is None:
        raise KeyError(f"unknown resource {resource_id!r}; known: {list_resources()}")
    return item
