from __future__ import annotations

import asyncio

from peaksMCP.config.metadata import tool_names
from peaksMCP.server.jupyter_peaks.backend import SharedState
from peaksMCP.server.jupyter_peaks.mcp_server import JupyterPeaksMCPServer


class FakeIPython:
    user_ns = {}


def names(server):
    return asyncio.run(server.tool_names())


def test_tool_surface_matches_metadata_baseline():
    """The exposed tool set is exactly the ``tools:`` block of metadata_baseline.yaml.

    Registration and the STDIO-proxy inventory guard both derive their tool-name
    set from that single source, so this is the only place the literal list lives.
    """
    server = JupyterPeaksMCPServer(SharedState(FakeIPython()))
    assert names(server) == sorted(tool_names())


def test_five_tool_surface_is_exact():
    """The model surface is EXACTLY search/get/inspect_notebook/run_cell/
    save_with_consent - positive list plus negative assertions for every
    retired tool name."""
    server = JupyterPeaksMCPServer(SharedState(FakeIPython()))
    exposed = set(names(server))
    assert exposed == {"search", "get", "inspect_notebook", "run_cell", "save_with_consent"}
    for retired in (
        # guidance / read / mutation tools folded into the five.
        "askuserquestion", "notebook_list_variables", "notebook_read_variable",
        "notebook_read_active_cell", "notebook_server_status", "notebook_add_cell",
        # pre-shrink names.
        "peaks_search_api", "peaks_get_api", "notebook_write_with_api_check",
        # even older entry points and mutation tools.
        "notebook_execute_code", "notebook_execute_with_api_check",
        "notebook_execute_active_cell", "notebook_delete_cell", "notebook_apply_patch",
    ):
        assert retired not in exposed, retired


def test_tool_metadata_is_nonempty():
    server = JupyterPeaksMCPServer(SharedState(FakeIPython()))
    tools = asyncio.run(server.mcp.list_tools())
    assert all(tool.title and tool.description for tool in tools)


def test_normalize_outputs_suppresses_plain_text_even_after_an_image():
    from peaksMCP.server.jupyter_peaks.core.tools import _normalize_outputs

    outputs = [
        {"data": {"image/png": "aW1hZ2U=", "text/plain": "<Figure>"}},
        {"data": {"text/plain": "later text"}},
    ]
    blocks = _normalize_outputs(outputs)
    # One closing line reports the rendered figure; the duplicate "<Figure>"
    # repr is suppressed; plain text after the figure is NOT echoed.
    text = "\n".join(getattr(block, "text", "") for block in blocks)
    assert "Inline figure rendered" in text and "(1 image(s))" in text
    assert "<Figure>" not in text
    assert "later text" not in text


def test_normalize_outputs_reports_interactive_widget_as_text():
    import json

    from mcp.types import TextContent

    from peaksMCP.server.jupyter_peaks.core.tools import _normalize_outputs

    outputs = [
        {
            "data": {
                "application/vnd.jupyter.widget-view+json": {
                    "model_id": "abc",
                    "version_major": 2,
                },
                "text/plain": "HoloViews Layout",
            }
        }
    ]
    blocks = _normalize_outputs(outputs)
    markers = [
        block.text
        for block in blocks
        if isinstance(block, TextContent) and "interactive_omitted" in block.text
    ]
    assert len(markers) == 1
    payload = json.loads(markers[0])
    assert payload["output_type"] == "interactive_omitted"
    assert "model_id" not in payload  # no widget state leaks back
    assert not any(
        isinstance(block, TextContent) and "HoloViews Layout" in block.text
        for block in blocks
    )


def test_normalize_outputs_generic_mime_marker_is_bounded_and_uniform():
    """Any payload MIME outside the text archive / images / widgets collapses
    to one 'omitted_mime' marker per distinct type (bounded), never raw data."""
    import json

    from mcp.types import TextContent

    from peaksMCP.server.jupyter_peaks.core.tools import _normalize_outputs

    outputs = [
        {"data": {"application/pdf": "JVBERi0x", "text/plain": "<Figure>"}},
        {"data": {"application/json": {"a": 1}, "text/plain": "json result"}},
        {"data": {"application/pdf": "second"}},  # duplicate MIME -> still 1 marker
    ]
    blocks = _normalize_outputs(outputs)
    markers = [
        json.loads(block.text)
        for block in blocks
        if isinstance(block, TextContent) and "omitted_mime" in block.text
    ]
    assert [marker["mime_type"] for marker in markers] == ["application/json", "application/pdf"]
    for marker in markers:
        assert marker["output_type"] == "omitted_mime"
        assert marker["note"] and "JVBERi0x" not in marker["note"]
    # The figure-less text reprs stay suppressed; nothing else is echoed.
    text = "\n".join(getattr(block, "text", "") for block in blocks)
    assert "<Figure>" not in text and "json result" not in text


def test_normalize_outputs_preserves_structured_cell_errors():
    import json

    from mcp.types import TextContent

    from peaksMCP.server.jupyter_peaks.core.tools import _normalize_outputs

    outputs = [
        {
            "output_type": "error",
            "ename": "ValueError",
            "evalue": "bad calibration",
            "traceback": [
                "\x1b[31mTraceback (most recent call last):\x1b[0m",
                "ValueError: bad calibration",
            ],
        }
    ]
    blocks = _normalize_outputs(outputs)
    assert len(blocks) == 1
    assert isinstance(blocks[0], TextContent)
    error = json.loads(blocks[0].text)
    assert error["output_type"] == "error"
    assert error["ename"] == "ValueError"
    assert error["evalue"] == "bad calibration"
    assert error["traceback"][0] == "Traceback (most recent call last):"


def test_normalize_outputs_reports_inline_images_without_payloads():
    from mcp.types import TextContent

    from peaksMCP.server.jupyter_peaks.core.tools import _normalize_outputs

    outputs = [
        {
            "output_type": "display_data",
            "data": {"image/png": "QUJDREVGRw=="},
        }
    ]
    blocks = _normalize_outputs(outputs)
    marker = next(block for block in blocks if isinstance(block, TextContent))
    assert marker.text.startswith("Inline figure rendered")
    assert "pixels are not sent to the model" in marker.text


def test_normalize_outputs_suppresses_markdown_boxes_without_figure_or_error():
    """peaks' colored analysis boxes are text/markdown only (no figure, no error);
    under output normalisation they are NOT echoed — the user reads them in the
    notebook, so they must not be re-stated inline as model review noise."""
    from peaksMCP.server.jupyter_peaks.core.tools import _normalize_outputs

    outputs = [
        {
            "output_type": "display_data",
            "data": {
                "text/markdown": (
                    '<div class="alert alert-block alert-success">'
                    "<b>Au fitting results: </b> Resolution (1st fit) "
                    "9.08 meV, accuracy_by_2nd_fitting 9.08 meV</div>"
                )
            },
        }
    ]
    blocks = _normalize_outputs(outputs)
    readable = "\n".join(getattr(block, "text", "") for block in blocks)
    assert "Au fitting results:" not in readable
    assert "Resolution (1st fit) 9.08 meV" not in readable
    assert "<div>" not in readable and "<b>" not in readable


def test_normalize_outputs_counts_frontend_omitted_image_as_rendered():
    """A frontend-omitted (oversized) image still counts as a rendered figure —
    one closing line, no per-item byte/limit bookkeeping."""
    from peaksMCP.server.jupyter_peaks.core.tools import _normalize_outputs

    outputs = [
        {
            "output_type": "display_data",
            "data": {
                "application/vnd.peaksmcp.image-omitted+json": [
                    {
                        "mime_type": "image/svg+xml",
                        "decoded_bytes": 9000000,
                        "reason": "per_image_limit",
                    }
                ]
            },
        }
    ]
    block_text = "\n".join(
        getattr(block, "text", "") for block in _normalize_outputs(outputs)
    )
    assert "Inline figure rendered" in block_text and "(1 image(s))" in block_text


def test_server_status_exposes_index_stale():
    """server_status must expose ``index_stale`` so the model can self-diagnose
    the INDEX_STALE_RESTART_REQUIRED condition instead of only search/get."""
    from unittest.mock import MagicMock

    from peaksMCP.server.jupyter_peaks.backend.notebook import NotebookBackend

    state = SharedState(FakeIPython())
    notebook = NotebookBackend(state)
    status = notebook.server_status()
    assert "index_stale" in status
    # No index built yet -> not stale.
    assert status["index_stale"] is False
    # An index whose fingerprint no longer matches is reported stale.
    stale_index = MagicMock()
    stale_index.entries = []
    stale_index.is_stale.return_value = True
    state.api_index = stale_index
    assert notebook.server_status()["index_stale"] is True
    # A fresh index reports not stale.
    fresh_index = MagicMock()
    fresh_index.entries = [object()]
    fresh_index.is_stale.return_value = False
    state.api_index = fresh_index
    status = notebook.server_status()
    assert status["index_stale"] is False
    assert status["api_count"] == 1


def test_normalize_outputs_echoes_short_stdout_summary():
    """Short stdout before any figure is the agent's first-hand knowledge of
    what the cell did (e.g. a facade summary) and IS echoed."""
    from peaksMCP.server.jupyter_peaks.core.tools import _normalize_outputs

    outputs = [
        {"output_type": "stream", "name": "stdout",
         "text": "load_data: BP_0020.nc (NetCDF) dims [eV=168, theta_par=902]\n"},
    ]
    blocks = _normalize_outputs(outputs)
    text = "\n".join(getattr(b, "text", "") for b in blocks)
    assert "load_data: BP_0020.nc" in text


def test_normalize_outputs_suppresses_long_stdout():
    """Long listings (e.g. per-file archive rows) stay in the notebook for the
    user and are NOT re-stated to the model."""
    from peaksMCP.server.jupyter_peaks.core.tools import _normalize_outputs

    lines = "\n".join(f"  BP_{i:04d}   netcdf (eV x168, theta_par x902)" for i in range(8))
    outputs = [{"output_type": "stream", "name": "stdout", "text": "header\n" + lines + "\n"}]
    blocks = _normalize_outputs(outputs)
    assert blocks == []


def test_normalize_outputs_echoes_summary_before_figure_only():
    """A facade summary printed before the figure is echoed; text printed
    after the figure is not (it is archive for the user)."""
    from peaksMCP.server.jupyter_peaks.core.tools import _normalize_outputs

    outputs = [
        {"output_type": "stream", "name": "stdout",
         "text": "fit_gold_reference: c0=2.659 eV, rendered\n"},
        {"data": {"image/png": "aW1hZ2U="}},
        {"output_type": "stream", "name": "stdout", "text": "trailing debug\n"},
    ]
    text = "\n".join(getattr(b, "text", "") for b in _normalize_outputs(outputs))
    assert "fit_gold_reference:" in text
    assert "trailing debug" not in text
    assert "Inline figure rendered" in text


def test_normalize_outputs_oversized_summary_line_is_suppressed():
    from peaksMCP.server.jupyter_peaks.core.tools import _normalize_outputs

    long = "x" * 250
    outputs = [{"output_type": "stream", "name": "stdout", "text": long + "\n"}]
    assert _normalize_outputs(outputs) == []


def test_audit_keeps_the_executed_cell_code_in_full():
    """The audit trail is the reviewable record of what ran: a 200-character cap
    hid every call past it (a cell that imported first and called load_data()
    later looked like it never used the curated verb)."""
    from peaksMCP.server.jupyter_peaks.core.tools import _summarize_arguments

    long_code = "import os\n" + "x = 1\n" * 200 + "scans = load_data('data_netcdf')\n"
    summary = _summarize_arguments((), {"code": long_code, "timeout": 120.0})
    assert summary["code"] == long_code          # full text, not a summary
    assert len(summary["code"]) > 200
    # other arguments stay bounded summaries
    assert summary["timeout"] == "120.0"
