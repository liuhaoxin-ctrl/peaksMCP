"""Persistence support retained behind the ``save_with_consent`` MCP tool.

Scientific analysis is exposed directly by ``peaks`` through the canonical
API catalog.  This package deliberately exports no model-callable facade.
Legacy implementation modules remain internal during the migration so the
save gateway can evolve independently.
"""

from __future__ import annotations

from .save import SaveReceipt  # noqa: F401

__all__ = ["SaveReceipt"]
