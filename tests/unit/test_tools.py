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


def test_removed_tools_stay_removed():
    """Regression guards for tool-surface reductions already shipped."""
    server = JupyterPeaksMCPServer(SharedState(FakeIPython()))
    exposed = set(names(server))
    assert {"notebook_write_with_api_check", "notebook_add_cell"} <= exposed
    # The old model-generated code entry points are no longer exposed.
    assert "notebook_execute_code" not in exposed
    assert "notebook_execute_with_api_check" not in exposed
    assert "notebook_execute_active_cell" not in exposed
    # delete_cell / apply_patch were removed: the notebook is a strictly
    # append-only log and patching an existing cell would overwrite it.
    assert "notebook_delete_cell" not in exposed
    assert "notebook_apply_patch" not in exposed


def test_tool_metadata_is_nonempty():
    server = JupyterPeaksMCPServer(SharedState(FakeIPython()))
    tools = asyncio.run(server.mcp.list_tools())
    assert all(tool.title and tool.description for tool in tools)


def test_output_content_suppresses_plain_text_even_after_an_image():
    from peaksMCP.server.jupyter_peaks.core.tools import _output_content

    class Notebook:
        def active_cell_output(self):
            return {
                "outputs": [
                    {"data": {"image/png": "aW1hZ2U=", "text/plain": "<Figure>"}},
                    {"data": {"text/plain": "later text"}},
                ]
            }

    blocks = _output_content(Notebook())
    # One closing line reports the rendered figure; the duplicate "<Figure>"
    # repr is suppressed; plain text after the figure is NOT echoed.
    text = "\n".join(getattr(block, "text", "") for block in blocks)
    assert "Inline figure rendered" in text and "(1 image(s))" in text
    assert "<Figure>" not in text
    assert "later text" not in text


def test_output_content_reports_interactive_widget_as_text():
    import json

    from mcp.types import TextContent

    from peaksMCP.server.jupyter_peaks.core.tools import _output_content

    class Notebook:
        def active_cell_output(self):
            return {
                "outputs": [
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
            }

    blocks = _output_content(Notebook())
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


def test_output_content_preserves_structured_cell_errors():
    import json

    from mcp.types import TextContent

    from peaksMCP.server.jupyter_peaks.core.tools import _output_content

    class Notebook:
        def active_cell_output(self):
            return {
                "outputs": [
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
            }

    blocks = _output_content(Notebook())
    assert len(blocks) == 1
    assert isinstance(blocks[0], TextContent)
    error = json.loads(blocks[0].text)
    assert error["output_type"] == "error"
    assert error["ename"] == "ValueError"
    assert error["evalue"] == "bad calibration"
    assert error["traceback"][0] == "Traceback (most recent call last):"


def test_output_content_reports_inline_images_without_payloads():
    from mcp.types import TextContent

    import peaksMCP.server.jupyter_peaks.core.tools as tools

    class Notebook:
        def active_cell_output(self):
            return {
                "outputs": [
                    {
                        "output_type": "display_data",
                        "data": {"image/png": "QUJDREVGRw=="},
                    }
                ]
            }

    blocks = tools._output_content(Notebook())
    marker = next(block for block in blocks if isinstance(block, TextContent))
    assert marker.text.startswith("Inline figure rendered")
    assert "pixels are not sent to the model" in marker.text


def test_output_content_suppresses_markdown_boxes_without_figure_or_error():
    """peaks' colored analysis boxes are text/markdown only (no figure, no error);
    under output normalisation they are NOT echoed — the user reads them in the
    notebook, so they must not be re-stated inline as model review noise."""
    from peaksMCP.server.jupyter_peaks.core.tools import _output_content

    class Notebook:
        def active_cell_output(self):
            return {
                "outputs": [
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
            }

    blocks = _output_content(Notebook())
    readable = "\n".join(getattr(block, "text", "") for block in blocks)
    assert "Au fitting results:" not in readable
    assert "Resolution (1st fit) 9.08 meV" not in readable
    assert "<div>" not in readable and "<b>" not in readable


def test_output_content_counts_frontend_omitted_image_as_rendered():
    """A frontend-omitted (oversized) image still counts as a rendered figure —
    one closing line, no per-item byte/limit bookkeeping."""
    from peaksMCP.server.jupyter_peaks.core.tools import _output_content

    class Notebook:
        def active_cell_output(self):
            return {
                "outputs": [
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
            }

    block_text = "\n".join(
        getattr(block, "text", "") for block in _output_content(Notebook())
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
