"""Register the deliberately small peaksMCP tool surface."""

from __future__ import annotations

import json
import re
import uuid
from functools import wraps
from typing import Any

from fastmcp import FastMCP
from mcp.types import TextContent

from peaksMCP.config import prompts, tool_metadata
from peaksMCP.discovery.signatures import describe_api

from ..backend import NotebookBackend, SharedState, UnsafeNotebookBackend, ensure_fresh_index
from ..security import AuditLogger
from ..security.audit import operation_context as _operation_context

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
#: MIME families that are part of the notebook text archive and are silently
#: not echoed to the model (reprs, markdown, html); every OTHER payload MIME
#: is normalized to one generic "omitted" marker per distinct type.
_TEXT_ARCHIVE_MIMES = frozenset(
    {"text/plain", "text/markdown", "text/html", "text/latex"}
)
_MAX_OMITTED_MIMES = 4


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

    Output is normalised so the model is not flooded with review noise, while
    the notebook stays the shared context for the user AND the agent:
    - errors are returned (the model must see why execution failed);
    - each rendered figure counts toward a single closing "figure rendered in
      the notebook" line; image pixels and interactive widgets stay as short
      markers (frontend-only);
    - stdout text is echoed ONLY as a short summary: at most three non-empty
      lines of at most 200 chars each, appearing before the first figure.
      That is exactly the shape of the facades' one-line summaries (and of an
      agent's own brief prints), so the agent knows what it did, what was
      produced and where the variables are. Longer text, ``text/plain`` reprs
      and markdown boxes stay in the notebook for the user and are not
      re-stated to the model.
    """
    blocks: list[TextContent] = []
    rendered_images = 0
    has_error = False
    saw_output = False
    first_media_order: int | None = None
    text_events: list[tuple[int, str]] = []
    omitted_mimes: list[str] = []
    for order, output in enumerate(outputs):
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
        if output.get("output_type") == "stream" and output.get("name") == "stdout":
            raw = output.get("text", "")
            text = _clean_output_text(raw)
            for line in text.splitlines():
                if line.strip():
                    text_events.append((order, line.rstrip()))
            continue
        data = output.get("data", {})
        if not isinstance(data, dict):
            data = {}
        data_mimes = {str(mime) for mime in data if data.get(mime)}
        has_image = bool(data_mimes & set(_IMAGE_MIMES)) or bool(
            data.get(_OMITTED_IMAGE_MIME)
        )
        interactive_mime = next(
            (mime for mime in _INTERACTIVE_MIMES if data.get(mime)), None
        )
        # text/plain reprs and markdown boxes are intentionally NOT echoed.
        if interactive_mime is not None:
            if first_media_order is None:
                first_media_order = order
            blocks.append(_interactive_omitted_content(interactive_mime))
            rendered_images += 1
        if has_image:
            if first_media_order is None:
                first_media_order = order
            rendered_images += 1
        # Generic MIME normalization: any payload MIME outside the text
        # archive, the image family and the interactive markers collapses to
        # one bounded "omitted" marker per distinct MIME type - the notebook
        # keeps the real payload, the model gets a uniform note.
        for mime in sorted(
            data_mimes
            - _TEXT_ARCHIVE_MIMES
            - set(_INTERACTIVE_MIMES)
            - set(_IMAGE_MIMES)
            - {_OMITTED_IMAGE_MIME}
        ):
            if mime not in omitted_mimes and len(omitted_mimes) < _MAX_OMITTED_MIMES:
                omitted_mimes.append(mime)
    summary = _stdout_summary(text_events, first_media_order)
    # An error always wins: the model needs to see it regardless of figures.
    if has_error:
        return blocks
    if summary is not None:
        blocks.append(TextContent(type="text", text=summary))
    # Generic MIME normalization: any payload MIME outside the text archive,
    # the image family and the interactive markers collapses to one bounded
    # "omitted" marker per distinct MIME type - the notebook keeps the real
    # payload, the model gets a uniform note.
    for mime in sorted(omitted_mimes):
        blocks.append(
            TextContent(
                type="text",
                text=json.dumps(
                    {
                        "output_type": "omitted_mime",
                        "mime_type": mime,
                        "note": _PROMPTS.get("mime_omitted_note")
                        or "Additional output of this MIME type is rendered in the notebook only.",
                    },
                    ensure_ascii=False,
                ),
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
        return blocks
    if not saw_output:
        return [TextContent(type="text", text="No active-cell output.")]
    if summary is not None or omitted_mimes:
        return blocks
    # Text-only output that is long or figure-interleaved: it stays in the
    # notebook (shared context) but is not re-stated to the model.
    return []


#: Max lines / chars of stdout text echoed to the model as a cell summary.
_SUMMARY_MAX_LINES = 3
_SUMMARY_MAX_LINE_LENGTH = 200


def _stdout_summary(
    text_events: list[tuple[int, str]],
    first_media_order: int | None,
) -> str | None:
    """Return a short stdout summary for the model, or None.

    Only text printed BEFORE the first figure counts: figures are the
    deliverable of plotting cells and trailing prints are usually noise.
    At most :data:`_SUMMARY_MAX_LINES` non-empty lines of at most
    :data:`_SUMMARY_MAX_LINE_LENGTH` chars are echoed; anything longer stays
    in the notebook as the archive for the user.
    """
    if not text_events:
        return None
    before_figure = [
        line
        for order, line in text_events
        if first_media_order is None or order < first_media_order
    ]
    if not before_figure:
        return None
    if len(before_figure) > _SUMMARY_MAX_LINES:
        return None
    if any(len(line) > _SUMMARY_MAX_LINE_LENGTH for line in before_figure):
        return None
    return "\n".join(before_figure)


def _text_blocks(outputs: list[dict[str, Any]]) -> list[str]:
    """JSON-safe rendering of the normalised output blocks (plain text strings)."""
    return [block.text for block in _normalize_outputs(outputs)]


def _stdout_metrics(outputs: list[dict[str, Any]]) -> tuple[int, str]:
    """stdout 行数与前 80 字符（给 agent 一个"有长输出被归档"的显式信号）。

    只统计 stream.stdout；正文不回传，避免把 notebook 的归档内容塞回对话。
    """
    chunks: list[str] = []
    for output in outputs if isinstance(outputs, list) else []:
        if not isinstance(output, dict) or output.get("output_type") != "stream":
            continue
        if output.get("name") != "stdout":
            continue
        text = output.get("text") or []
        if isinstance(text, str):
            chunks.append(text)
        elif isinstance(text, list):
            chunks.extend(str(part) for part in text)
    joined = "".join(chunks)
    lines = sum(1 for line in joined.splitlines() if line.strip())
    return lines, joined[:80]


def _require_index(state: SharedState):
    """Return the live index, hot-rebuilding it in the kernel when stale."""
    return ensure_fresh_index(state)


def _record_verified_api(state: SharedState, entry: dict[str, Any]) -> None:
    """Record a canonical API proof after a successful get.

    The ledger is keyed by the CANONICAL ID (with its scope/module snapshot),
    never by a bare name: run_cell unlocks an exact-name call only when an
    id exists whose name AND scope match the call site, so same-name APIs in
    different modules/scopes cannot be confused, and natural-language aliases
    never unlock Python symbols (they are search vocabulary, not callable
    identifiers).
    """
    canonical_id = str(entry.get("id") or "")
    name = str(entry.get("name") or "")
    if not canonical_id or not name:
        return
    state.verified_apis[canonical_id] = {
        "id": canonical_id,
        "name": name,
        "scope": entry.get("scope"),
        "module": entry.get("module"),
        "tier": entry.get("tier"),
        "exposure": entry.get("exposure"),
    }
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
        # Every tool call is audited with ONE operation_id spanning the whole
        # flow (call -> proof -> scanner -> cell -> consent ticket -> publish),
        # so log lines for one request can be correlated.  The final line
        # carries the SEMANTIC outcome (executed / blocked / denied / saved /
        # failed / ok), not a blanket "ok".
        operation_id = uuid.uuid4().hex
        audit.write(name, "called", {"operation_id": operation_id, "args": _summarize_arguments(args, kwargs)})
        token = _operation_context.set(operation_id)
        try:
            result = function(*args, **kwargs)
        except Exception as exc:
            audit.write(
                name, "error",
                {
                    "operation_id": operation_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:500],
                },
            )
            raise
        finally:
            _operation_context.reset(token)
        outcome, details = _semantic_outcome(result)
        audit.write(name, outcome, {"operation_id": operation_id, **details})
        return result

    mcp.tool(name=name, title=metadata["title"], description=metadata["description"])(audited)


def _semantic_outcome(result: Any) -> tuple[str, dict[str, Any]]:
    """Map a tool result to the semantic audit outcome + correlation fields.

    Outcomes: ``executed`` (a cell ran), ``saved`` / ``denied`` / ``failed`` /
    ``blocked`` (persistence or refusal), ``ok`` otherwise.  Correlation
    fields: executed cell id, canonical api ids verified by the api check,
    and the save ticket id/hash when a persistence verb produced one.
    """
    if not isinstance(result, dict):
        return "ok", {}
    details: dict[str, Any] = {}
    if result.get("execution_timed_out"):
        details["note"] = str(result.get("note") or "kernel reply timed out")
        return "failed", details
    cell_id = result.get("id")
    if isinstance(cell_id, str) and cell_id:
        details["cell_id"] = cell_id
    api_check = result.get("api_check")
    if isinstance(api_check, dict):
        verified = api_check.get("verified_peaks_apis") or []
        ids: list[str] = []
        for item in verified:
            for match in item.get("matches") or []:
                if isinstance(match, str) and match not in ids:
                    ids.append(match)
        if ids:
            details["api_ids"] = ids
    ticket_id = result.get("ticket_id")
    if isinstance(ticket_id, str) and ticket_id:
        details["ticket_id"] = ticket_id
    sha256 = result.get("sha256")
    if isinstance(sha256, str) and sha256:
        details["sha256"] = sha256[:16]
    status = result.get("status")
    if result.get("blocked"):
        return "blocked", details
    if status in {"saved", "denied", "failed", "blocked"}:
        return str(status), details
    if result.get("execution_success") is not None or "id" in result:
        return "executed", details
    return "ok", details


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
    """Register the read-only/guidance tools (search/get/inspect_notebook).

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
    def search(
        query: str,
        scope: str = "all",
        limit: int = 5,
        include_advanced: bool = False,
    ) -> dict[str, Any]:
        index = _require_index(state)
        searched, matches = index.search_tiered(
            query, scope, limit, include_advanced=include_advanced
        )
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
            "include_advanced": include_advanced,
            "count": len(matches),
            "peaks_version": index.peaks_version,
            "fingerprint": index.fingerprint,
            "matches": matches,
        }

    def get(canonical_id: str) -> dict[str, Any]:
        index = _require_index(state)
        entry = index.get(canonical_id)
        if entry is None:
            raise KeyError(
                f"unknown canonical API ID: {canonical_id}. get accepts ONLY "
                "the canonical id returned by search "
                "(module:peaksMCP.overrides:<name> or a peaks module:name) - "
                "run search first, then get the id from its results."
            )
        _record_verified_api(state, entry)
        detail = describe_api(entry)
        if detail.get("project_added") and not detail.get("signature_resolved"):
            raise RuntimeError(
                f"manifest export {detail.get('export')!r} is not importable/resolvable "
                "- the override manifest and its implementation have drifted; fix the "
                "manifest row or the adapter before use."
            )
        if detail.get("project_added") and detail.get("contract_input_issues"):
            raise RuntimeError(
                "manifest inputs do not match the adapter signature: "
                + "; ".join(detail["contract_input_issues"])
                + " - fix the manifest row before use."
            )
        return detail

    def inspect_notebook(
        target: str = "variables",
        variable_name: str | None = None,
        detail: str = "summary",
        limit: int = 10,
        cell: str | int | None = None,
        offset: int = 0,
        with_text_outputs: bool = False,
    ) -> dict[str, Any]:
        """Inspect the live notebook through the generic object-summary protocol.

        ``target`` discriminates the request: ``variables`` (namespace rows),
        ``variable`` (one named variable, requires ``variable_name``),
        ``active_cell`` / ``cells`` / ``cell`` (cell identity/source; ``cells``
        pages tail history via ``limit``/``offset``, ``cell`` takes an id or
        index).  ``detail`` selects ``summary`` (one bounded line/item) or
        ``preview`` (structural detail).  ``with_text_outputs=True`` adds each
        cell's bounded TEXT outputs (streams + text/plain only, ~8KB/cell) so
        the model can re-read what a cell printed - the notebook is the shared
        context center.  Image payloads are NEVER returned by any target (they
        travel once, settled inside the run reply).
        """
        return notebook.inspect(
            target,
            variable_name=variable_name,
            detail=detail,
            limit=limit,
            cell=cell,
            offset=offset,
            with_text_outputs=with_text_outputs,
        )

    functions = {
        "search": search,
        "get": get,
        "inspect_notebook": inspect_notebook,
    }
    for name, function in functions.items():
        _register(mcp, name, function, audit)


def register_unsafe_tools(mcp: FastMCP, notebook: UnsafeNotebookBackend, audit: AuditLogger) -> None:
    """Register the two mutation tools (run_cell / save_with_consent).

    The notebook is a STRICTLY APPEND-ONLY log: ``run_cell`` always appends a
    new cell at the END and never edits, deletes or reorders an existing
    cell, preserving the agent's full work history top-to-bottom.  Code
    execution is scanned and audit-logged; file-write intents
    (SAVE001/SAVE002/FILE002) are hard-blocked (run_cell is never a
    persistence path) while network egress (NET001) always requires
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
    def run_cell(
        code: str,
        timeout: float = 120.0,
        api_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run code in one new notebook cell (API-checked) and return the
        normalised output summary.

        ``api_ids`` optionally declares the canonical Peaks API ids this cell
        relies on; every declared id must already be proven by a successful
        ``get`` this session (unproven ids are refused).  Raw Jupyter outputs
        are never echoed to the model: the response replaces them with an
        ``output`` list containing only errors, the "Inline figure rendered
        ..." line and interactive markers, while ``stdout_lines`` /
        ``stdout_head`` report the cell's own print output (a text-only cell
        therefore returns ``[]`` plus the stdout signal).  Cell identity and
        execution flags stay on the response, and ``kernel_state`` /
        ``kernel_busy_s`` say whether the kernel is still executing - a timeout
        never interrupts it, so inspect before retrying (or use
        ``inspect_notebook(target="kernel")``).
        """
        result = notebook.write_with_api_check(code=code, timeout=timeout, api_ids=api_ids)
        if isinstance(result, dict) and isinstance(result.get("outputs"), list):
            cell = {
                key: result[key]
                for key in (
                    "id", "index", "cell_type", "source",
                    "execution_success", "saved", "save_error",
                )
                if key in result
            }
            stdout_lines, stdout_head = _stdout_metrics(result.get("outputs") or [])
            return {
                **cell,
                # The response carries the frontend's ONE settled output
                # snapshot (quiet window after kernel idle, 2s cap); nothing
                # is polled or cached server-side any more.
                "output": _text_blocks(result.get("outputs") or []),
                # 显式信号：这一格 stdout 的行数与开头（正文不回传，只有被归档
                # 的长文本需要时用 inspect_notebook with_text_outputs 回读）。
                "stdout_lines": stdout_lines,
                "stdout_head": stdout_head,
                "kernel_state": result.get("kernel_state"),
                "kernel_busy_s": result.get("kernel_busy_s"),
                "api_check": result.get("api_check"),
            }
        return result

    def save_with_consent(
        variable_name: str,
        path: str,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """Persist one notebook variable through staged, human-approved saving.

        The result type is ``SaveReceipt`` (operation / status saved|denied|
        blocked|failed / path / kind / size_bytes / sha256 / dims /
        dtype / units / overwrite / ticket_id).  One variable, one file, per
        call - there is no batch save.  Nothing is written unless the user
        approves the save card in the notebook; overwrite=True only permits
        replacing an existing target AFTER that approval.
        """
        return notebook.save_with_consent(
            variable_name=variable_name,
            path=path,
            overwrite=overwrite,
        )

    functions = {
        # The single agent execution entry point (never a persistence path).
        "run_cell": run_cell,
        # The single persistence verb (staged bytes + approval card).
        "save_with_consent": save_with_consent,
    }
    for name, function in functions.items():
        _register(mcp, name, function, audit)
