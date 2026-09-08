"""Notebook backends that do not depend on the MCP protocol."""

from .base import SharedState, ensure_fresh_index
from .notebook import NotebookBackend
from .notebook_unsafe import UnsafeNotebookBackend

__all__ = [
    "NotebookBackend",
    "SharedState",
    "UnsafeNotebookBackend",
    "ensure_fresh_index",
]

