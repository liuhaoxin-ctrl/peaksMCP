"""Read the curated presentation layer for MCP tools."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


@lru_cache(maxsize=1)
def _document() -> dict[str, Any]:
    return yaml.safe_load(Path(__file__).with_name("metadata_baseline.yaml").read_text(encoding="utf-8")) or {}


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
