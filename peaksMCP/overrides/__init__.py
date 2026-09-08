"""Curated project surface (facades) on top of peaks.

The model composes these functions instead of re-implementing analysis; native
peaks functions stay reachable through search/get.  Every facade:

- returns a small JSON-safe dict or a DataArray/Dataset (never prints noise);
- renders at most one canonical summary line (Show convention);
- never writes to disk by itself — persistence goes through ``save_result``
  with an explicit preview -> approve flow (Save convention).
"""

from __future__ import annotations

from .load import load_data
from .models import Report, report_dict, report_summary
from .save import save_result

__all__ = ["load_data", "save_result", "Report", "report_dict", "report_summary"]
