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
        ``notebook_unsafe``, ``scanner``, ``ipython``.

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
    >>> tool_metadata("search")["title"]
    'Search Peaks API'
    """
    item = (_document().get("tools") or {}).get(name, {})
    return {"title": str(item.get("title") or name), "description": str(item.get("description") or "")}


def tool_names() -> frozenset[str]:
    """Return the canonical set of MCP tool names.

    The ``tools:`` block in ``metadata_baseline.yaml`` is the single source of
    truth for which tools exist and how they are presented.  Registration
    (``core/tools.py``) and the STDIO-proxy inventory guard both derive their
    tool-name set from here so a renamed/added/removed tool only needs one edit.

    Returns
    -------
    frozenset of str
        Stable MCP tool names declared in the baseline.

    Examples
    --------
    >>> "search" in tool_names()
    True
    """
    return frozenset((_document().get("tools") or {}).keys())
