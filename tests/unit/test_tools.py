from __future__ import annotations

import asyncio
from unittest.mock import Mock

import pytest

from peaksMCP.config.metadata import tool_metadata, tool_names
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


def test_tool_metadata_blocks_round_one_discovery_and_notebook_noise_patterns():
    search = tool_metadata("search")["description"].lower()
    get = tool_metadata("get")["description"].lower()
    inspect = tool_metadata("inspect_notebook")["description"].lower()
    run_cell = tool_metadata("run_cell")["description"].lower()

    assert "preprocess all 2d data" in search
    for broad_query in ("single-letter", "wildcard", "broad catalog enumeration"):
        assert broad_query in search
    assert "session_proof_ledger" in search and "next_action" in search
    assert "already-proven" in search
    assert "never through mcpscript" in search
    assert "current live kernel" in search and "fresh kernel" in search
    assert "extension/server restart" in search
    assert "only api ids you will call" in get
    assert "speculative helpers" in get and "same canonical id again" in get
    assert "proof_status" in get and "session_proof_ledger" in get
    assert "sole next_action" in get
    assert "do not immediately re-read a successful run_cell" in inspect
    assert "runtime status" in inspect and 'target="kernel"' in inspect
    assert "does not browse files" in inspect
    assert '"notebook" is an exact alias for "cells"' in inspect
    assert "repository" in run_cell and "datasheet" in run_cell
    assert "at most three non-empty stdout lines" in run_cell
    assert "each at most 200 characters" in run_cell
    assert "code=..., never source=..." in run_cell
    assert 'run_cell(code=text, cell_type="markdown")' in run_cell
    assert "never pass code_language or language" in run_cell
    assert "coordinate `.values`" in run_cell
    assert "one compact grid" in run_cell
    assert "processed_stems" in run_cell


def test_model_api_entry_adds_get_parameter_name_without_removing_legacy_id():
    from peaksMCP.server.jupyter_peaks.core.tools import _model_api_entry

    row = _model_api_entry({"id": "top_level:peaks:load_experiment", "name": "load_experiment"})

    assert row == {
        "id": "top_level:peaks:load_experiment",
        "canonical_id": "top_level:peaks:load_experiment",
        "name": "load_experiment",
    }


def test_registered_search_and_get_expose_canonical_id_at_model_boundary(
    monkeypatch, tmp_path
):
    """The registered tools, not only the helper, must use get's parameter name."""
    from peaksMCP.server.jupyter_peaks.core.tools import register_safe_tools
    from peaksMCP.server.jupyter_peaks.security import AuditLogger

    canonical_id = "top_level:peaks.core.fileIO.experiment:load_experiment"
    entry = {
        "id": canonical_id,
        "name": "load_experiment",
        "module": "peaks.core.fileIO.experiment",
        "scope": "top_level",
        "tier": "native",
        "exposure": "core",
        "score": 1000,
    }
    index = Mock(peaks_version="test", fingerprint="fingerprint")
    index.is_stale.return_value = False
    index.search_tiered.return_value = ("catalog", [entry])
    index.get.return_value = entry
    state = SharedState(FakeIPython(), api_index=index)

    registered = {}
    fake_mcp = Mock()

    def tool(**metadata):
        def decorate(function):
            registered[metadata["name"]] = function
            return function

        return decorate

    fake_mcp.tool.side_effect = tool
    monkeypatch.setattr(
        "peaksMCP.server.jupyter_peaks.core.tools.describe_api",
        lambda row: {**row, "signature": "load_experiment(source)"},
    )
    register_safe_tools(
        fake_mcp,
        state,
        Mock(),
        AuditLogger(tmp_path / "audit.jsonl"),
    )

    search_result = registered["search"](query="load experiment")
    get_result = registered["get"](canonical_id=canonical_id)
    proven_search = registered["search"](query="load experiment")
    duplicate_get = registered["get"](canonical_id=canonical_id)

    assert search_result["matches"][0]["canonical_id"] == canonical_id
    assert search_result["matches"][0]["proof_status"] == "needs_get"
    assert search_result["session_proof_ledger"] == []
    assert search_result["next_action"]["eligible_canonical_ids"] == [canonical_id]
    assert search_result["matches"][0]["id"] == canonical_id
    assert get_result["canonical_id"] == canonical_id
    assert get_result["id"] == canonical_id
    assert get_result["proof_status"] == "newly_proven"
    assert get_result["session_proof_ledger"] == [
        {
            "canonical_id": canonical_id,
            "name": "load_experiment",
            "scope": "top_level",
        }
    ]
    assert get_result["next_action"]["action"] == "use_api_without_get_again"
    assert proven_search["matches"][0]["proof_status"] == "already_proven"
    assert proven_search["next_action"]["action"] == "use_proven_match_without_get"
    assert duplicate_get["proof_status"] == "already_proven"
    # The proof ledger remains internal and therefore keeps the discovery id.
    assert state.verified_apis[canonical_id]["id"] == canonical_id


@pytest.mark.parametrize(
    ("description", "error_match"),
    [
        pytest.param(RuntimeError("describe exploded"), "describe exploded", id="describe"),
        pytest.param(
            {
                "project_added": True,
                "signature_resolved": False,
                "export": "peaksMCP.bad.missing",
            },
            "not importable/resolvable",
            id="signature-validation",
        ),
        pytest.param(
            {
                "project_added": True,
                "signature_resolved": True,
                "contract_input_issues": ["declared input 'typo' is invalid"],
            },
            "inputs do not match",
            id="contract-validation",
        ),
    ],
)
def test_get_failure_never_records_a_proof(
    monkeypatch, tmp_path, description, error_match
):
    from peaksMCP.server.jupyter_peaks.core.tools import register_safe_tools
    from peaksMCP.server.jupyter_peaks.security import AuditLogger

    canonical_id = "top_level:peaks.example:example"
    entry = {
        "id": canonical_id,
        "name": "example",
        "module": "peaks.example",
        "scope": "top_level",
        "tier": "native",
        "exposure": "core",
        "score": 1000,
    }
    index = Mock(peaks_version="test", fingerprint="fingerprint")
    index.is_stale.return_value = False
    index.get.return_value = entry
    index.search_tiered.return_value = ("catalog", [entry])
    state = SharedState(FakeIPython(), api_index=index)
    registered = {}
    fake_mcp = Mock()

    def tool(**metadata):
        def decorate(function):
            registered[metadata["name"]] = function
            return function

        return decorate

    def describe(_entry):
        if isinstance(description, Exception):
            raise description
        return {**entry, **description}

    fake_mcp.tool.side_effect = tool
    monkeypatch.setattr(
        "peaksMCP.server.jupyter_peaks.core.tools.describe_api", describe
    )
    register_safe_tools(
        fake_mcp,
        state,
        Mock(),
        AuditLogger(tmp_path / "audit.jsonl"),
    )

    with pytest.raises(RuntimeError, match=error_match):
        registered["get"](canonical_id=canonical_id)

    assert state.verified_apis == {}
    result = registered["search"](query="example")
    assert result["session_proof_ledger"] == []
    assert result["matches"][0]["proof_status"] == "needs_get"


def test_exact_search_next_action_excludes_prior_proofs_and_lower_fuzzy_hits(
    monkeypatch, tmp_path
):
    """The latest search response must prevent the r002 duplicate-get pattern."""
    from peaksMCP.server.jupyter_peaks.core.tools import register_safe_tools
    from peaksMCP.server.jupyter_peaks.security import AuditLogger

    k_convert_id = "dataarray:peaks.core.process.k_conversion:k_convert"
    assign_id = "metadata:peaks.core.metadata.metadata_methods:assign_normal_emission"
    lower_fuzzy_id = "metadata:peaks.core.metadata.metadata_methods:set_normal_emission"
    matches = [
        {
            "id": assign_id,
            "name": "assign_normal_emission",
            "module": "peaks.core.metadata.metadata_methods",
            "scope": "metadata",
            "tier": "native",
            "exposure": "core",
            "score": 1000,
        },
        {
            "id": lower_fuzzy_id,
            "name": "set_normal_emission",
            "module": "peaks.core.metadata.metadata_methods",
            "scope": "metadata",
            "tier": "native",
            "exposure": "core",
            "score": 450,
        },
    ]
    index = Mock(peaks_version="test", fingerprint="fingerprint")
    index.is_stale.return_value = False
    index.search_tiered.return_value = ("catalog", matches)
    state = SharedState(FakeIPython(), api_index=index)
    state.verified_apis[k_convert_id] = {
        "id": k_convert_id,
        "name": "k_convert",
        "module": "peaks.core.process.k_conversion",
        "scope": "dataarray",
    }
    registered = {}
    fake_mcp = Mock()

    def tool(**metadata):
        def decorate(function):
            registered[metadata["name"]] = function
            return function

        return decorate

    fake_mcp.tool.side_effect = tool
    register_safe_tools(
        fake_mcp,
        state,
        Mock(),
        AuditLogger(tmp_path / "audit.jsonl"),
    )

    result = registered["search"](query="assign_normal_emission")

    assert result["match_mode"] == "exact_name"
    assert result["session_proof_ledger"] == [
        {
            "canonical_id": k_convert_id,
            "name": "k_convert",
            "scope": "dataarray",
        }
    ]
    assert [row["proof_status"] for row in result["matches"]] == [
        "needs_get",
        "needs_get",
    ]
    assert result["next_action"] == {
        "action": "get_only_selected_unproven",
        "eligible_canonical_ids": [assign_id],
        "instruction": (
            "Choose only APIs you will call, then get each eligible id once. "
            "Never get an id listed in session_proof_ledger."
        ),
    }


def test_invalid_catalog_enumeration_requires_a_descriptive_query(monkeypatch, tmp_path):
    from peaksMCP.server.jupyter_peaks.core.tools import register_safe_tools
    from peaksMCP.server.jupyter_peaks.security import AuditLogger

    entry = {
        "id": "top_level:peaks.example:example",
        "name": "example",
        "module": "peaks.example",
        "scope": "top_level",
        "tier": "native",
        "exposure": "core",
        "score": 300,
    }
    index = Mock(peaks_version="test", fingerprint="fingerprint")
    index.is_stale.return_value = False
    index.search_tiered.return_value = ("all", [entry])
    state = SharedState(FakeIPython(), api_index=index)
    registered = {}
    fake_mcp = Mock()

    def tool(**metadata):
        def decorate(function):
            registered[metadata["name"]] = function
            return function

        return decorate

    fake_mcp.tool.side_effect = tool
    register_safe_tools(fake_mcp, state, Mock(), AuditLogger(tmp_path / "audit.jsonl"))

    for query in ("", "*", "a", "list all"):
        result = registered["search"](query=query)
        assert result["matches"], query
        assert result["next_action"]["action"] == "refine_search"
        assert "eligible_canonical_ids" not in result["next_action"]


def test_fuzzy_search_only_offers_the_highest_ranked_get_target(monkeypatch, tmp_path):
    from peaksMCP.server.jupyter_peaks.core.tools import register_safe_tools
    from peaksMCP.server.jupyter_peaks.security import AuditLogger

    entries = [
        {
            "id": f"top_level:peaks.example:{name}",
            "name": name,
            "module": "peaks.example",
            "scope": "top_level",
            "tier": "native",
            "exposure": "core",
            "score": score,
        }
        for name, score in (("load_experiment", 700), ("bin_data", 500), ("extract_cut", 400))
    ]
    index = Mock(peaks_version="test", fingerprint="fingerprint")
    index.is_stale.return_value = False
    index.search_tiered.return_value = ("catalog", entries)
    state = SharedState(FakeIPython(), api_index=index)
    registered = {}
    fake_mcp = Mock()

    def tool(**metadata):
        def decorate(function):
            registered[metadata["name"]] = function
            return function

        return decorate

    fake_mcp.tool.side_effect = tool
    register_safe_tools(fake_mcp, state, Mock(), AuditLogger(tmp_path / "audit.jsonl"))

    result = registered["search"](query="preprocess all 2D data")

    assert result["match_mode"] == "fuzzy"
    assert result["next_action"]["eligible_canonical_ids"] == [entries[0]["id"]]


def test_live_tool_schemas_use_exact_model_facing_parameter_names():
    server = JupyterPeaksMCPServer(SharedState(FakeIPython()))
    schemas = {
        tool.name: tool.parameters for tool in asyncio.run(server.mcp.list_tools())
    }

    assert set(schemas["search"]["properties"]) == {
        "query", "scope", "limit", "include_advanced"
    }
    assert set(schemas["get"]["properties"]) == {"canonical_id"}
    assert set(schemas["run_cell"]["properties"]) == {
        "code", "timeout", "api_ids", "cell_type"
    }
    assert set(schemas["save_with_consent"]["properties"]) == {
        "variable_name", "path", "overwrite"
    }
    assert all(schema.get("additionalProperties") is False for schema in schemas.values())


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
                    "<b>Au fitting results: </b> Resolution (1st fit): FWHM "
                    "9.08 meV (instrument);   Resolution (2nd fit): FWHM "
                    "11.20 meV (effective temperature 22.0 K)</div>"
                )
            },
        }
    ]
    blocks = _normalize_outputs(outputs)
    readable = "\n".join(getattr(block, "text", "") for block in blocks)
    assert "Au fitting results:" not in readable
    assert "Resolution (1st fit): FWHM 9.08 meV" not in readable
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
