"""Register the deliberately small peaksMCP tool surface."""

from __future__ import annotations

import json
import re
import time
from functools import wraps
from typing import Any

from fastmcp import FastMCP
from mcp.types import TextContent

from peaksMCP.config import prompts, tool_metadata
from peaksMCP.discovery.signatures import describe_api

from ..backend import NotebookBackend, SharedState, UnsafeNotebookBackend, ensure_fresh_index
from ..security import AuditLogger

#: Curated model/user-facing runtime text (config/prompts.yaml), read once at
#: import so per-call lookups stay cheap and wording lives outside Python.
_PROMPTS = prompts()
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


def _search_match_mode(query: Any, searched: str, matches: list[dict[str, Any]]) -> str:
    """Classify how one search resolved: exact_name / exact_alias / fuzzy / list.

    ``searched == "override"`` means the query hit an override name or alias
    exactly; name equality decides which.  Any other non-empty query resolved
    through the mixed full-index fallback (fuzzy).  An empty query is a list.
    """
    query_text = query.strip().lower() if isinstance(query, str) else ""
    if not query_text:
        return "list"
    primary = matches[0]["name"] if matches else ""
    if searched == "override":
        return "exact_name" if primary.lower() == query_text else "exact_alias"
    return "fuzzy"


def _clean_output_text(value: Any) -> str:
    """Join Jupyter text fragments and remove terminal colour escapes."""
    text = "".join(value) if isinstance(value, list) else str(value)
    return _ANSI_ESCAPE.sub("", text)


def _normalize_outputs(outputs: list[dict[str, Any]]) -> list[TextContent]:
    """Convert Jupyter cell outputs to the text the model may read.

    Output is normalised so the model is not flooded with review noise:
    - errors are returned (the model must see why execution failed);
    - each rendered figure counts toward a single closing "figure rendered in
      the notebook" line; image pixels and interactive widgets stay as short
      markers (frontend-only);
    - plain text, ``text/plain`` reprs and markdown boxes are **not** echoed to
      the model — they are the analysis result the user reads in the notebook,
      not something to be re-stated inline.  Returning nothing for a text-only
      cell is the intended behaviour, not a missing output.
    """
    blocks: list[TextContent] = []
    rendered_images = 0
    has_error = False
    saw_output = False
    for output in outputs:
        if not isinstance(output, dict):
            continue
        saw_output = True
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
            has_error = True
            continue
        data = output.get("data", {})
        if not isinstance(data, dict):
            data = {}
        has_image = any(data.get(mime) for mime in _IMAGE_MIMES) or bool(
            data.get(_OMITTED_IMAGE_MIME)
        )
        interactive_mime = next(
            (mime for mime in _INTERACTIVE_MIMES if data.get(mime)), None
        )
        # Plain text / text/plain / markdown boxes are intentionally NOT echoed.
        if interactive_mime is not None:
            blocks.append(_interactive_omitted_content(interactive_mime))
            rendered_images += 1
        if has_image:
            rendered_images += 1
    # An error always wins: the model needs to see it regardless of figures.
    if has_error:
        return blocks
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
        return blocks
    if not saw_output:
        return [TextContent(type="text", text="No active-cell output.")]
    # Output was text-only: suppressed by design (the user reads it in the notebook).
    return []


def _text_blocks(outputs: list[dict[str, Any]]) -> list[str]:
    """JSON-safe rendering of the normalised output blocks (plain text strings)."""
    return [block.text for block in _normalize_outputs(outputs)]


def _settle_executed_outputs(
    state: SharedState, result: dict[str, Any]
) -> list[dict[str, Any]]:
    """Return the executed cell's outputs after the frontend push settles.

    ``execute_code`` returns its snapshot at shell-reply time; a trailing
    inline image can land in the frontend output model a moment later and be
    pushed back through the Comm.  Wait a short bounded window for the cache
    copy of the executed cell to stabilise, then return the last copy.
    Falls back to the response snapshot when no push ever arrives.
    """
    snapshot = result.get("outputs")
    if not isinstance(snapshot, list):
        return []
    cell_id = result.get("id")
    if not (
        isinstance(cell_id, str)
        and cell_id
        and state.bridge
        and state.bridge.connected
    ):
        return snapshot
    latest = snapshot
    cache_seen = False
    changed_at = time.monotonic()
    deadline = changed_at + 2.0
    while time.monotonic() < deadline:
        cached = state.cell_outputs.get(cell_id)
        if cached is None:
            if cache_seen or time.monotonic() - changed_at >= 0.6:
                # No push pipeline for this cell: trust the response snapshot.
                return latest
        elif cached == latest:
            if not cache_seen:
                # Push already matches the response: nothing more to wait for.
                return latest
            if time.monotonic() - changed_at >= 0.2:
                return latest
        else:
            cache_seen = True
            latest = cached
            changed_at = time.monotonic()
        time.sleep(0.05)
    return latest


def _require_index(state: SharedState):
    """Return the live index, hot-rebuilding it in the kernel when stale."""
    return ensure_fresh_index(state)


def _record_verified_api(state: SharedState, entry: dict[str, Any]) -> None:
    """Remember that a canonical API was fetched via peaks_get_api this session.

    Only the canonical executable name is unlocked.  Natural-language aliases
    never unlock Python symbols: they are search vocabulary, not callable
    identifiers, so a later cell that writes an alias (e.g. ``mapping slice``)
    stays blocked until the model fetches and uses the real function name.
    """
    name = str(entry.get("name") or "")
    if not name:
        return
    state.verified_peaks_names.add(name)
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
                "note": _PROMPTS["interactive_omitted_note"],
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


def _read_active_cell_normalized(notebook: NotebookBackend) -> dict[str, Any]:
    """Return the current frontend cell with a normalised ``output`` list.

    The raw ``outputs`` are dropped: the model only receives the same
    normalised view the write tool returns (errors / figure markers), so a
    user-run cell is readable without flooding the conversation with its
    original text.
    """
    cell = notebook.active_cell()
    if isinstance(cell, dict) and isinstance(cell.get("outputs"), list):
        cell = {**cell, "output": _text_blocks(cell.pop("outputs"))}
    return cell


def register_safe_tools(mcp: FastMCP, state: SharedState, notebook: NotebookBackend, audit: AuditLogger) -> None:
    """Register the six read-only and guidance tools.

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
        searched, matches = index.search_tiered(query, scope, limit)
        match_mode = _search_match_mode(query, searched, matches)
        return {
            "query": query,
            "scope": scope,
            # Backwards-compatible: searched_tier keeps the same label values.
            "searched_tier": searched,
            # Canonical names for the two search dimensions:
            # searched_namespace is override / mixed / all; match_mode is
            # exact_name / exact_alias / fuzzy / list.
            "searched_namespace": searched,
            "match_mode": match_mode,
            "count": len(matches),
            "peaks_version": index.peaks_version,
            "fingerprint": index.fingerprint,
            "matches": matches,
        }

    def peaks_get_api(canonical_id: str) -> dict[str, Any]:
        index = _require_index(state)
        entry = index.get(canonical_id)
        if entry is None:
            raise KeyError(f"unknown canonical API ID: {canonical_id}")
        _record_verified_api(state, entry)
        return describe_api(entry)

    def askuserquestion(prompt: str, hint: str | None = None, options: list[str] | None = None) -> dict[str, Any]:
        return {"status": "needs_input", "prompt": prompt, "hint": hint, "options": options or []}

    def read_active_cell() -> dict[str, Any]:
        return _read_active_cell_normalized(notebook)

    functions = {
        "peaks_search_api": peaks_search_api,
        "peaks_get_api": peaks_get_api,
        "askuserquestion": askuserquestion,
        "notebook_list_variables": notebook.list_variables,
        "notebook_read_variable": notebook.read_variable,
        "notebook_read_active_cell": read_active_cell,
        "notebook_server_status": notebook.server_status,
    }
    for name, function in functions.items():
        _register(mcp, name, function, audit)


def register_unsafe_tools(mcp: FastMCP, notebook: UnsafeNotebookBackend, audit: AuditLogger) -> None:
    """Register the two mutation tools.

    The notebook is a STRICTLY APPEND-ONLY log: ``notebook_write_with_api_check``
    and ``notebook_add_cell`` always add a new cell at the END and never edit,
    delete or reorder an existing cell.  This preserves the agent's full work
    history top-to-bottom.  Code execution is scanned and audit-logged;
    write-to-disk intents (SAVE001/SAVE002, network egress) always require
    explicit user approval in the notebook, while plain execution is governed
    by the ``require_consent`` master switch (profile ``mcp.require_consent``).

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
    def notebook_write_with_api_check(code: str, timeout: float = 120.0) -> dict[str, Any]:
        """Run the API-checked write and return the normalised output summary.

        Raw Jupyter outputs are never echoed to the model: the response replaces
        them with an ``output`` list containing only errors, the "Inline figure
        rendered ..." line and interactive markers; a text-only cell returns
        ``[]`` by design (its content is displayed in the notebook for the
        user).  Cell identity and execution flags stay on the response so the
        agent knows what was appended and whether it ran.
        """
        result = notebook.write_with_api_check(code=code, timeout=timeout)
        if isinstance(result, dict) and isinstance(result.get("outputs"), list):
            cell = {
                key: result[key]
                for key in (
                    "id", "index", "cell_type", "source",
                    "execution_success", "saved", "save_error",
                )
                if key in result
            }
            return {
                **cell,
                "output": _text_blocks(_settle_executed_outputs(notebook.state, result)),
                "api_check": result.get("api_check"),
            }
        return result

    functions = {
        # Model-generated code is written through the API-checked entry point.
        "notebook_write_with_api_check": notebook_write_with_api_check,
        "notebook_add_cell": notebook.add_cell,
    }
    for name, function in functions.items():
        _register(mcp, name, function, audit)
