"""Single-source API catalog and model-visibility contracts."""

from __future__ import annotations

from unittest.mock import Mock

from peaksMCP.config.schema import validate_api_catalog


def _catalog(**overrides):
    document = {
        "version": 1,
        "apis": {
            "dataarray:peaks.core.process.k_conversion:k_convert": {
                "kind": "native",
                "exposure": "core",
                "aliases": ["momentum conversion", "k space"],
            }
        },
    }
    document.update(overrides)
    return document


def test_api_catalog_schema_supports_zero_facades_and_strict_enums():
    assert validate_api_catalog(_catalog()) == []

    bad_kind = _catalog()
    next(iter(bad_kind["apis"].values()))["kind"] = "override"
    assert any("invalid kind" in issue for issue in validate_api_catalog(bad_kind))

    bad_exposure = _catalog()
    next(iter(bad_exposure["apis"].values()))["exposure"] = "internal"
    assert any("invalid exposure" in issue for issue in validate_api_catalog(bad_exposure))


def test_native_catalog_rows_use_canonical_ids_and_facades_require_contracts():
    short_name = _catalog(apis={"k_convert": {"kind": "native", "exposure": "core"}})
    assert any("canonical id" in issue for issue in validate_api_catalog(short_name))

    facade = _catalog(
        apis={
            "module:peaksMCP.facades:future": {
                "kind": "facade",
                "exposure": "core",
            }
        }
    )
    issues = validate_api_catalog(facade)
    assert any("must declare 'export'" in issue for issue in issues)
    assert any("must declare 'inputs'" in issue for issue in issues)


def test_catalog_filters_dynamic_scan_and_never_exposes_hidden(monkeypatch):
    import peaksMCP.discovery.index as index_module

    scanned = [
        {
            "id": "dataarray:peaks.core.process.k_conversion:k_convert",
            "scope": "dataarray",
            "module": "peaks.core.process.k_conversion",
            "name": "k_convert",
            "kind": "method",
            "summary": "Convert to momentum coordinates.",
            "docstring": "Convert to momentum coordinates.",
        },
        {
            "id": "module:peaks.secret:leaked_helper",
            "scope": "module",
            "module": "peaks.secret",
            "name": "leaked_helper",
            "kind": "function",
            "summary": "Must not leak.",
            "docstring": "Must not leak.",
        },
        {
            "id": "module:peaks.secret:hidden_helper",
            "scope": "module",
            "module": "peaks.secret",
            "name": "hidden_helper",
            "kind": "function",
            "summary": "Explicitly hidden.",
            "docstring": "Explicitly hidden.",
        },
    ]
    catalog = _catalog(
        apis={
            "dataarray:peaks.core.process.k_conversion:k_convert": {
                "kind": "native",
                "exposure": "core",
            },
            "module:peaks.secret:hidden_helper": {
                "kind": "native",
                "exposure": "hidden",
            },
        }
    )
    monkeypatch.setattr(index_module, "scan_runtime", lambda: scanned)
    monkeypatch.setattr(index_module, "scan_modules", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(index_module, "load_catalog", lambda *_args, **_kwargs: catalog)
    monkeypatch.setattr(index_module, "source_fingerprint", lambda *_args: "fingerprint")

    index = index_module.build_index()
    assert [entry["id"] for entry in index.entries] == [
        "dataarray:peaks.core.process.k_conversion:k_convert"
    ]
    assert index.get("module:peaks.secret:leaked_helper") is None
    assert index.get("module:peaks.secret:hidden_helper") is None


def test_advanced_requires_exact_query_or_explicit_opt_in():
    from peaksMCP.discovery.index import search_index

    entry = {
        "id": "module:peaks.tools:advanced_helper",
        "scope": "module",
        "module": "peaks.tools",
        "name": "advanced_helper",
        "kind": "native",
        "summary": "low level calibration helper",
        "docstring": "",
        "aliases": ["calibration primitive"],
        "exposure": "advanced",
        "tier": "native",
    }
    assert search_index([entry], "low level calibration") == []
    assert search_index([entry], "advanced_helper")[0]["name"] == "advanced_helper"
    assert search_index([entry], "calibration primitive")[0]["name"] == "advanced_helper"
    assert search_index([entry], "low level calibration", include_advanced=True)[0]["name"] == "advanced_helper"

    hidden = {**entry, "id": "module:peaks.tools:hidden_helper", "name": "hidden_helper", "exposure": "hidden"}
    assert search_index([hidden], "hidden_helper", include_advanced=True) == []


def test_natural_2d_preprocessing_discovers_the_native_workflow_entrypoint():
    """One user-intent query must lead to the complete native cut recipe."""
    from peaksMCP.discovery.index import build_index
    from peaksMCP.discovery.signatures import describe_api

    index = build_index()
    canonical_id = "top_level:peaks.core.fileIO.experiment:load_experiment"
    matches = index.search("preprocess all 2D data", limit=5)

    assert matches[0]["id"] == canonical_id
    assert matches[0]["tier"] == "native"

    entry = index.get(canonical_id)
    assert entry is not None
    recipe = describe_api(entry)["docstring"]
    for required in (
        "fit_gold",
        "record.theta_offset_deg",
        "assign_normal_emission(theta_par=record.theta_offset_deg)",
        "k_convert(EF_correction=gold_fit, quiet=True)",
        "stem-keyed",
        "compact",
    ):
        assert required in recipe
    assert "from peaks import load_experiment" in recipe
    assert "do not search or get pxt2nc" in recipe.lower()
    assert ".sel(deflector_perp=0.0)" in recipe
    assert "all processed stems" in recipe
    assert "gold=SELECTED_STEM (index SELECTED_INDEX)" in recipe
    assert '"gold scan fitted once" alone is incomplete' in recipe
    assert 'print("processed_stems=" + ",".join(sorted(result_dict)))' in recipe
    assert "gold_index = exp.gold[0]" in recipe
    assert "gold = exp[gold_index]" in recipe
    assert "gold_record = records_by_index[gold_index]" in recipe
    assert "gold_record.stem" in recipe
    assert "for cut_index in exp.cuts" in recipe
    assert "record = records_by_index[cut_index]" in recipe
    assert "direct search/get calls rather than mcpscript" in recipe.lower()
    assert "before any analysis cell" in recipe.lower()
    assert "fit_gold, assign_normal_emission, and k_convert" in recipe
    assert "not already proven" in recipe.lower()
    assert "never use a blocked run_cell as proof" in recipe.lower()
    assert "copy the complete returned `processed_stems=...` token verbatim" in recipe.lower()
    assert "never infer consecutive stems from the count" in recipe
    assert "at most 2000 characters" in recipe
    assert "do not repeat runtime/tool/api guidance" in recipe.lower()
    assert "do not hand-type a representative stem" in recipe.lower()
    assert "do not inspect exp after loading" in recipe.lower()
    assert "raw.metadata.theta_offset_deg" in recipe
    assert "does not expose" in recipe.lower()
    assert "theta_by_stem[rep_stem]" in recipe
    assert "do not reread datasheet or persist processed data or figures" in recipe.lower()


def test_experiment_contract_names_conflict_fields_and_processing_target_scope():
    """The entrypoint must make conflicts actionable without widening the batch."""
    from peaksMCP.discovery.index import load_api_catalog

    canonical_id = "top_level:peaks.core.fileIO.experiment:load_experiment"
    note = load_api_catalog()[canonical_id]["docstring_note"]

    assert len(note) < 3400
    assert "ExperimentConflict exposes only index/data_format/dims/issue" in note
    assert "records_by_index = {record.index: record for record in exp.records}" in note
    assert "process exactly exp.cuts" in note.lower()
    assert "never gold, mappings, or unsupported records" in note
    assert ".sel(deflector_perp=0.0)" in note
    assert "do not search for an additional Peaks extraction API" in note


def test_native_usage_notes_prevent_round_two_call_failures():
    from peaksMCP.discovery.index import build_index
    from peaksMCP.discovery.signatures import describe_api

    index = build_index()
    by_name = {entry["name"]: entry for entry in index.entries}

    gold_doc = describe_api(by_name["fit_gold"])["docstring"]
    k_doc = describe_api(by_name["k_convert"])["docstring"]
    grid_doc = describe_api(by_name["plot_grid"])["docstring"]
    assert "gold.fit_gold(show=False, quiet=True)" in gold_doc
    assert "never print either complete mapping" in gold_doc
    assert "start_eV/stop_eV/lower_points/upper_points" in gold_doc
    assert "outlier_fraction/uniform" in gold_doc
    assert 'gold_fit.attrs["figure"]' in gold_doc
    assert "Do not call display" in gold_doc
    assert "renders it once" in gold_doc
    assert "quiet=True" in k_doc
    assert "progress widget" in k_doc
    assert "never k_par" in k_doc
    assert by_name["plot_grid"]["exposure"] == "advanced"
    assert "bound DataTree method" in grid_doc
    assert "stem-keyed result dictionary is not a DataTree" in grid_doc
    assert "do not import or call a bare/module-level plot_grid" in grid_doc


def test_run_cell_requires_declared_ids_for_every_direct_peaks_call(monkeypatch, tmp_path):
    from peaksMCP.discovery.index import ApiIndex
    from peaksMCP.server.jupyter_peaks.backend import SharedState, UnsafeNotebookBackend
    from peaksMCP.server.jupyter_peaks.security import AuditLogger, ConsentManager

    canonical_id = "dataarray:peaks.core.process.k_conversion:k_convert"
    entry = {
        "id": canonical_id,
        "scope": "dataarray",
        "module": "peaks.core.process.k_conversion",
        "name": "k_convert",
        "kind": "native",
        "summary": "",
        "docstring": "",
        "aliases": [],
        "exposure": "core",
        "tier": "native",
    }
    state = SharedState(Mock(user_ns={}))
    state.require_consent = False
    state.api_index = ApiIndex([entry], "test", "test")
    state.bridge = Mock()
    state.bridge.request.return_value = {"ok": True}
    state.verified_apis[canonical_id] = dict(entry)
    backend = UnsafeNotebookBackend(
        state, ConsentManager(), AuditLogger(tmp_path / "audit.jsonl")
    )
    monkeypatch.setattr(
        "peaksMCP.server.jupyter_peaks.backend.notebook_unsafe.ensure_fresh_index",
        lambda _state: state.api_index,
    )

    missing = backend.write_with_api_check("da.k_convert()", timeout=5)
    assert missing["blocked"] is True
    assert missing["missing_api_ids"] == [canonical_id]
    state.bridge.request.assert_not_called()

    executed = backend.write_with_api_check(
        "da.k_convert()", timeout=5, api_ids=[canonical_id]
    )
    assert executed["ok"] is True

    generic = backend.write_with_api_check("print(len([1, 2]))", timeout=5)
    assert generic["ok"] is True


def test_run_cell_rejects_partially_declared_multi_api_cell(monkeypatch, tmp_path):
    from peaksMCP.discovery.index import ApiIndex
    from peaksMCP.server.jupyter_peaks.backend import SharedState, UnsafeNotebookBackend
    from peaksMCP.server.jupyter_peaks.security import AuditLogger, ConsentManager

    ids = {
        "fit_gold": "dataarray:peaks.core.fitting.fit:fit_gold",
        "k_convert": "dataarray:peaks.core.process.k_conversion:k_convert",
    }
    entries = [
        {
            "id": canonical_id,
            "scope": "dataarray",
            "module": canonical_id.split(":", 2)[1],
            "name": name,
            "kind": "method",
            "catalog_kind": "native",
            "summary": "",
            "docstring": "",
            "aliases": [],
            "exposure": "core",
            "tier": "native",
        }
        for name, canonical_id in ids.items()
    ]
    state = SharedState(Mock(user_ns={}))
    state.require_consent = False
    state.api_index = ApiIndex(entries, "test", "test")
    state.bridge = Mock()
    state.verified_apis.update({entry["id"]: dict(entry) for entry in entries})
    backend = UnsafeNotebookBackend(
        state, ConsentManager(), AuditLogger(tmp_path / "audit.jsonl")
    )
    monkeypatch.setattr(
        "peaksMCP.server.jupyter_peaks.backend.notebook_unsafe.ensure_fresh_index",
        lambda _state: state.api_index,
    )

    result = backend.write_with_api_check(
        "fit = gold.fit_gold(plot=False)\nout = cut.k_convert(EF_correction=fit)",
        timeout=5,
        api_ids=[ids["fit_gold"]],
    )
    assert result["blocked"] is True
    assert result["missing_api_ids"] == [ids["k_convert"]]
    state.bridge.request.assert_not_called()
