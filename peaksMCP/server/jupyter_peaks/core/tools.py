"""Register the deliberately small peaksMCP tool surface."""

from __future__ import annotations

import base64
import json
import re
from functools import wraps
from typing import Any

from fastmcp import FastMCP
from mcp.types import ImageContent, TextContent

from peaksMCP.config import tool_metadata
from peaksMCP.discovery.signatures import describe_api

from ..backend import NotebookBackend, SharedState, UnsafeNotebookBackend, ensure_fresh_index
from ..security import AuditLogger

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_OMITTED_IMAGE_MIME = "application/vnd.peaksmcp.image-omitted+json"
_MAX_IMAGE_BYTES = 8 * 1024 * 1024
_MAX_RESPONSE_IMAGE_BYTES = 16 * 1024 * 1024


def _clean_output_text(value: Any) -> str:
    """Join Jupyter text fragments and remove terminal colour escapes."""
    text = "".join(value) if isinstance(value, list) else str(value)
    return _ANSI_ESCAPE.sub("", text)


def _base64_decoded_size(payload: str) -> int:
    """Return decoded size without allocating the decoded image."""
    compact = "".join(payload.split())
    padding = 2 if compact.endswith("==") else 1 if compact.endswith("=") else 0
    return max(0, len(compact) * 3 // 4 - padding)


def _image_omitted_content(
    mime: str,
    size: int,
    *,
    reason: str,
) -> TextContent:
    """Describe an intentionally omitted oversized inline image."""
    return TextContent(
        type="text",
        text=json.dumps(
            {
                "output_type": "image_omitted",
                "mime_type": mime,
                "decoded_bytes": size,
                "per_image_limit_bytes": _MAX_IMAGE_BYTES,
                "response_limit_bytes": _MAX_RESPONSE_IMAGE_BYTES,
                "reason": reason,
            },
            ensure_ascii=False,
        ),
    )


def _require_index(state: SharedState):
    """Return the live index, hot-rebuilding it in the kernel when stale."""
    return ensure_fresh_index(state)


def _summarize_arguments(args: tuple, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Compact, JSON-safe argument summary for the audit trail."""
    summary: dict[str, Any] = {}
    for index, value in enumerate(args):
        text = str(value)
        summary[str(index)] = text[:200]
    for key, value in kwargs.items():
        text = str(value)
        summary[str(key)] = text[:200]
    return summary


def _register(mcp: FastMCP, name: str, function: Any, audit: AuditLogger) -> None:
    metadata = tool_metadata(name)

    @wraps(function)
    def audited(*args, **kwargs):
        # Every tool call is audited (called/ok/error), not only authorisation
        # decisions — the audit log is the full tool-call trail.
        audit.write(name, "called", {"args": _summarize_arguments(args, kwargs)})
        try:
            result = function(*args, **kwargs)
        except Exception as exc:
            audit.write(
                name, "error",
                {"error_type": type(exc).__name__, "error": str(exc)[:500]},
            )
            raise
        audit.write(name, "ok", {})
        return result

    mcp.tool(name=name, title=metadata["title"], description=metadata["description"])(audited)


def _output_content(notebook: NotebookBackend) -> list[TextContent | ImageContent]:
    """Convert Jupyter MIME bundles to native MCP text and image content."""
    blocks: list[TextContent | ImageContent] = []
    included_image_bytes = 0
    for output in notebook.active_cell_output().get("outputs", []):
        if not isinstance(output, dict):
            continue
        if output.get("output_type") == "error":
            traceback_lines = output.get("traceback") or []
            if not isinstance(traceback_lines, list):
                traceback_lines = [traceback_lines]
            error = {
                "output_type": "error",
                "ename": _clean_output_text(output.get("ename", "Error")),
                "evalue": _clean_output_text(output.get("evalue", "")),
                "traceback": [_clean_output_text(line) for line in traceback_lines],
            }
            blocks.append(
                TextContent(
                    type="text",
                    text=json.dumps(error, ensure_ascii=False),
                )
            )
            continue
        data = output.get("data", {})
        if not isinstance(data, dict):
            data = {}
        omitted_by_frontend = data.get(_OMITTED_IMAGE_MIME, [])
        if isinstance(omitted_by_frontend, list):
            for item in omitted_by_frontend:
                if isinstance(item, dict):
                    try:
                        omitted_size = max(0, int(item.get("decoded_bytes", 0)))
                    except (TypeError, ValueError):
                        omitted_size = 0
                    blocks.append(
                        _image_omitted_content(
                            str(item.get("mime_type", "image/unknown")),
                            omitted_size,
                            reason=str(item.get("reason", "frontend_limit")),
                        )
                    )
        text = output.get("text")
        if text:
            blocks.append(TextContent(type="text", text="".join(text) if isinstance(text, list) else str(text)))
        output_has_image = False
        for mime in ("image/png", "image/jpeg", "image/svg+xml"):
            payload = data.get(mime)
            if not payload:
                continue
            payload = "".join(payload) if isinstance(payload, list) else str(payload)
            if mime == "image/svg+xml":
                raw = payload.encode("utf-8")
                image_bytes = len(raw)
                encoded_payload = base64.b64encode(raw).decode("ascii")
            else:
                image_bytes = _base64_decoded_size(payload)
                encoded_payload = payload
            if image_bytes > _MAX_IMAGE_BYTES:
                blocks.append(
                    _image_omitted_content(mime, image_bytes, reason="per_image_limit")
                )
                continue
            if included_image_bytes + image_bytes > _MAX_RESPONSE_IMAGE_BYTES:
                blocks.append(
                    _image_omitted_content(mime, image_bytes, reason="response_limit")
                )
                continue
            blocks.append(
                ImageContent(type="image", mimeType=mime, data=encoded_payload)
            )
            included_image_bytes += image_bytes
            output_has_image = True
        plain = data.get("text/plain")
        if plain and not output_has_image:
            blocks.append(TextContent(type="text", text="".join(plain) if isinstance(plain, list) else str(plain)))
    return blocks or [TextContent(type="text", text="No active-cell output.")]


def register_safe_tools(mcp: FastMCP, state: SharedState, notebook: NotebookBackend, audit: AuditLogger) -> None:
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

    def mcp_list_resources() -> dict[str, Any]:
        """Discover the canonical publication plotting formats.

        Returns every resource (uri, when-to-use, example, figure contract and the
        full ``template`` inline) plus guidance.  Templates are embedded directly
        because some clients (Claude Desktop) reject custom-scheme resource URIs
        like ``peaksmcp://plot/<id>``, so the model can run the template from the
        tool output without a client-side resource fetch.  Call this FIRST
        whenever a figure is needed — ``notebook_write_with_api_check`` requires
        it to have been read before any plotting code is executed.
        """
        from peaksMCP.config.metadata import list_resources, resource_metadata

        state.read_plot_resources = True

        resources = [
            {
                "uri": f"peaksmcp://plot/{resource_id}",
                "name": resource_id,
                "use_when": str(meta.get("when_to_use") or ""),
                "example": str(meta.get("example") or ""),
                "figure": str(meta.get("figure") or {}),
                "template": str(meta.get("template") or ""),
            }
            for resource_id in list_resources()
            for meta in [resource_metadata(resource_id)]
        ]
        return {
            "total_resources": len(resources),
            "resources": resources,
            "guidance": {
                "resources_vs_tools": {
                    "resources": (
                        "Read-only reference data, templates and documentation — each "
                        "resource below includes its template inline; run it verbatim."
                    ),
                    "tools": (
                        "Active operations: peaks_search_api / peaks_get_api to look up "
                        "APIs, notebook_write_with_api_check to write and run analysis cells."
                    ),
                },
                "when_to_use_resources": [
                    "Before drawing any figure, pick a plotting format here and run its inline template verbatim.",
                    "For a publication-style figure, run the selected template verbatim (adjust variable names only).",
                    "Comparing many cuts -> dispersion_grid; one cut -> dispersion_single; EF slice -> fermi_surface.",
                ],
                "first_use": "Call mcp_list_resources() BEFORE plotting to select the canonical format.",
            },
        }

    functions = {
        "peaks_search_api": peaks_search_api,
        "peaks_get_api": peaks_get_api,
        "askuserquestion": askuserquestion,
        "mcp_list_resources": mcp_list_resources,
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
        _register(mcp, name, function, audit)


def register_unsafe_tools(mcp: FastMCP, notebook: UnsafeNotebookBackend, audit: AuditLogger) -> None:
    """Register the two mutation tools.

    The notebook is a STRICTLY APPEND-ONLY log: ``notebook_write_with_api_check``
    and ``notebook_add_cell`` always add a new cell at the END and never edit,
    delete or reorder an existing cell.  This preserves the agent's full work
    history top-to-bottom.  Code execution is scanned and audit-logged, and
    follows the active mode's consent policy. Explicit-consent findings are
    confirmed in every mode.

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
        # Model-generated code is written through the API-checked entry point.
        "notebook_write_with_api_check": notebook.write_with_api_check,
        "notebook_add_cell": notebook.add_cell,
    }
    for name, function in functions.items():
        _register(mcp, name, function, audit)
