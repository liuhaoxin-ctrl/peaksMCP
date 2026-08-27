"""Register the deliberately small peaksMCP tool surface."""

from __future__ import annotations

import base64
from typing import Any

from fastmcp import FastMCP
from mcp.types import ImageContent, TextContent

from peaksMCP.config import tool_metadata
from peaksMCP.discovery.index import IndexStaleError, build_index
from peaksMCP.discovery.signatures import describe_api

from ..backend import NotebookBackend, SharedState, UnsafeNotebookBackend


def _require_index(state: SharedState):
    """Return the live index, raising a structured stale error when a restart is needed."""
    if state.api_index is None:
        state.api_index = build_index()
    if state.api_index.is_stale():
        raise IndexStaleError(
            "INDEX_STALE_RESTART_REQUIRED: the installed Peaks or peaksMCP adapter "
            "source changed after the API index was built. Restart the kernel to "
            "rebuild the index (peaksMCP restart kernel) before searching again."
        )
    return state.api_index


def _register(mcp: FastMCP, name: str, function: Any) -> None:
    metadata = tool_metadata(name)
    mcp.tool(name=name, title=metadata["title"], description=metadata["description"])(function)


def _output_content(notebook: NotebookBackend) -> list[TextContent | ImageContent]:
    """Convert Jupyter MIME bundles to native MCP text and image content."""
    blocks: list[TextContent | ImageContent] = []
    for output in notebook.active_cell_output().get("outputs", []):
        data = output.get("data", {}) if isinstance(output, dict) else {}
        text = output.get("text") if isinstance(output, dict) else None
        if text:
            blocks.append(TextContent(type="text", text="".join(text) if isinstance(text, list) else str(text)))
        for mime in ("image/png", "image/jpeg", "image/svg+xml"):
            payload = data.get(mime)
            if not payload:
                continue
            payload = "".join(payload) if isinstance(payload, list) else str(payload)
            if mime == "image/svg+xml":
                payload = base64.b64encode(payload.encode("utf-8")).decode("ascii")
            blocks.append(ImageContent(type="image", mimeType=mime, data=payload))
        plain = data.get("text/plain")
        if plain and not any(isinstance(block, ImageContent) for block in blocks[-3:]):
            blocks.append(TextContent(type="text", text="".join(plain) if isinstance(plain, list) else str(plain)))
    return blocks or [TextContent(type="text", text="No active-cell output.")]


def register_safe_tools(mcp: FastMCP, state: SharedState, notebook: NotebookBackend) -> None:
    """Register the twelve read-only and guidance tools.

    Parameters
    ----------
    mcp : fastmcp.FastMCP
        In-kernel MCP server receiving the registrations.
    state : SharedState
        Kernel state and lazily built Peaks API index.
    notebook : NotebookBackend
        Read-only notebook implementation.

    Examples
    --------
    >>> register_safe_tools(mcp, state, notebook)
    """
    def peaks_search_api(query: str, scope: str = "all", limit: int = 5) -> dict[str, Any]:
        index = _require_index(state)
        matches = index.search(query, scope, limit)
        return {"query": query, "scope": scope, "count": len(matches), "peaks_version": index.peaks_version, "fingerprint": index.fingerprint, "matches": matches}

    def peaks_get_api(canonical_id: str) -> dict[str, Any]:
        index = _require_index(state)
        entry = index.get(canonical_id)
        if entry is None:
            raise KeyError(f"unknown canonical API ID: {canonical_id}")
        return describe_api(entry)

    def askuserquestion(prompt: str, hint: str | None = None, options: list[str] | None = None) -> dict[str, Any]:
        return {"status": "needs_input", "prompt": prompt, "hint": hint, "options": options or []}

    functions = {
        "peaks_search_api": peaks_search_api,
        "peaks_get_api": peaks_get_api,
        "askuserquestion": askuserquestion,
        "notebook_list_variables": notebook.list_variables,
        "notebook_read_variable": notebook.read_variable,
        "notebook_read_active_cell": notebook.active_cell,
        "notebook_read_active_cell_output": lambda: _output_content(notebook),
        "notebook_read_content": notebook.notebook_content,
        "notebook_move_cursor": notebook.move_cursor,
        "notebook_server_status": notebook.server_status,
        "notebook_kernel_status": notebook.kernel_status,
        "notebook_wait_for_kernel": notebook.wait_for_kernel,
    }
    for name, function in functions.items():
        _register(mcp, name, function)


def register_unsafe_tools(mcp: FastMCP, notebook: UnsafeNotebookBackend) -> None:
    """Register the five consent-gated mutation tools.

    Parameters
    ----------
    mcp : fastmcp.FastMCP
        In-kernel MCP server receiving the registrations.
    notebook : UnsafeNotebookBackend
        Consent, scanning and audit protected notebook implementation.

    Examples
    --------
    >>> register_unsafe_tools(mcp, unsafe_notebook)
    """
    functions = {
        "notebook_execute_code": notebook.execute_code,
        "notebook_execute_active_cell": notebook.execute_active_cell,
        "notebook_add_cell": notebook.add_cell,
        "notebook_delete_cell": notebook.delete_cell,
        "notebook_apply_patch": notebook.apply_patch,
    }
    for name, function in functions.items():
        _register(mcp, name, function)
