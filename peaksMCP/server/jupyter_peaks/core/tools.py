"""Register the deliberately small peaksMCP tool surface."""

from __future__ import annotations

import json
import re
from functools import wraps
from typing import Any

from fastmcp import FastMCP
from mcp.types import TextContent

from peaksMCP.config import tool_metadata
from peaksMCP.discovery.signatures import describe_api

from ..backend import NotebookBackend, SharedState, UnsafeNotebookBackend, ensure_fresh_index
from ..security import AuditLogger

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_OMITTED_IMAGE_MIME = "application/vnd.peaksmcp.image-omitted+json"
#: MIME types that only a live browser frontend can render (ipywidgets,
#: HoloViews/hvplot/Plotly/Bokeh).  The model cannot see interactive widgets
#: through the text channel, so they are reported as a text marker instead.
_INTERACTIVE_MIMES = (
    "application/vnd.jupyter.widget-view+json",
    "application/vnd.holoviews_load.v0+json",
    "application/vnd.plotly.v1+json",
    "application/vnd.bokehjs_exec.v0+json",
)
_IMAGE_MIMES = ("image/png", "image/jpeg", "image/svg+xml")


def _clean_output_text(value: Any) -> str:
    """Join Jupyter text fragments and remove terminal colour escapes."""
    text = "".join(value) if isinstance(value, list) else str(value)
    return _ANSI_ESCAPE.sub("", text)


def _markdown_to_text(payload: Any) -> str:
    """Extract readable text from a Jupyter ``text/markdown`` payload.

    Colored analysis boxes (peaks' ``analysis_warning``) are rendered as
    markdown/HTML and carry no ``text/plain``, so without this the model reads
    nothing from them.  Strip tags and unescape entities to plain text.
    """
    import html

    text = "".join(payload) if isinstance(payload, list) else str(payload)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _require_index(state: SharedState):
    """Return the live index, hot-rebuilding it in the kernel when stale."""
    return ensure_fresh_index(state)


def _record_verified_api(state: SharedState, entry: dict[str, Any]) -> None:
    """Remember that a canonical API was fetched via peaks_get_api this session.

    Unlocking covers the canonical name and every search alias, so a later
    cell that writes the alias (e.g. ``preprocess_cut`` for ``process_cut``)
    is not blocked as an unverifiable name.
    """
    names = {str(entry.get("name") or "")}
    names.update(str(alias) for alias in entry.get("aliases", []) if alias)
    names.discard("")
    state.verified_peaks_names.update(names)
    for name in names:
        state.unknown_api_attempts.pop(name, None)


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


def _interactive_omitted_content(mime: str) -> TextContent:
    """Report an interactive widget/panel that only the notebook can render.

    The model cannot receive the live widget through the MCP text channel, so
    the tool returns a short marker instead of an empty or misleading output —
    the same pattern used for oversized images.
    """
    return TextContent(
        type="text",
        text=json.dumps(
            {
                "output_type": "interactive_omitted",
                "mime_type": mime,
                "note": (
                    "Interactive panel/widget rendered in the notebook for the "
                    "user; it cannot be embedded here. Consider it done and "
                    "describe the figure to the user."
                ),
            },
            ensure_ascii=False,
        ),
    )


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


def _output_content(notebook: NotebookBackend) -> list[TextContent]:
    """Convert Jupyter cell outputs to the text the model may read.

    Rules (keep the model's view simple):
    - image pixels are never sent; each output that rendered one counts toward
      a single closing "figure rendered in the notebook" line;
    - ``text/markdown`` boxes (peaks' colored analysis boxes) are converted to
      plain text so the model can read the numbers inside them;
    - interactive widgets stay as short markers (frontend-only);
    - errors and stream/text output pass through unchanged.
    """
    blocks: list[TextContent] = []
    rendered_images = 0
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
        markdown = data.get("text/markdown")
        if markdown:
            readable = _markdown_to_text(markdown)
            if readable:
                blocks.append(TextContent(type="text", text=readable))
        interactive_mime = next(
            (mime for mime in _INTERACTIVE_MIMES if data.get(mime)), None
        )
        if interactive_mime is not None:
            blocks.append(_interactive_omitted_content(interactive_mime))
        text = output.get("text")
        if text:
            blocks.append(
                TextContent(
                    type="text",
                    text="".join(text) if isinstance(text, list) else str(text),
                )
            )
        has_image = any(data.get(mime) for mime in _IMAGE_MIMES) or bool(
            data.get(_OMITTED_IMAGE_MIME)
        )
        if has_image:
            rendered_images += 1
        plain = data.get("text/plain")
        if plain and not has_image and interactive_mime is None:
            blocks.append(
                TextContent(
                    type="text",
                    text="".join(plain) if isinstance(plain, list) else str(plain),
                )
            )
    if rendered_images:
        blocks.append(
            TextContent(
                type="text",
                text=(
                    f"Inline figure rendered in the notebook for the user "
                    f"({rendered_images} image(s)); image pixels are not sent "
                    f"to the model. A bare '<Figure ...>' repr instead of this "
                    f"line means the figure was NOT displayed."
                ),
            )
        )
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
        _record_verified_api(state, entry)
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
