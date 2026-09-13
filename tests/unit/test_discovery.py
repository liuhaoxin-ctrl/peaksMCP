from __future__ import annotations

from peaksMCP.discovery.index import build_index, load_api_catalog
from peaksMCP.discovery.signatures import describe_api


def test_live_index_matches_visible_catalog_and_has_stable_unique_ids():
    first = build_index()
    second = build_index()
    ids = [item["id"] for item in first.entries]
    declared = {
        canonical_id
        for canonical_id, config in load_api_catalog().items()
        if config["exposure"] != "hidden"
    }
    assert set(ids) == declared
    assert len(ids) == len(set(ids))
    assert ids == [item["id"] for item in second.entries]
    assert first.fingerprint == second.fingerprint
    assert all(not item.get("project_added") for item in first.entries)


def test_every_public_name_is_searchable_in_top_three():
    index = build_index()
    for entry in index.entries:
        names = [match["name"] for match in index.search(entry["name"], limit=3)]
        assert entry["name"] in names, entry["id"]


def test_every_curated_alias_ranks_its_canonical_api_in_top_three():
    index = build_index()
    cases = [
        (query, canonical_id)
        for canonical_id, config in load_api_catalog().items()
        if config["exposure"] != "hidden"
        for query in (config.get("aliases") or [])
    ]
    assert len(cases) >= 60
    for query, expected_id in cases:
        ids = [match["id"] for match in index.search(query, limit=3)]
        assert expected_id in ids, (query, expected_id, ids)


def test_get_api_detail_is_a_clean_whitelist():
    index = build_index()
    entry = index.get("dataarray:peaks.core.process.k_conversion:k_convert")
    assert entry is not None
    detail = describe_api(entry)
    assert "k_convert(" in detail["signature"]
    assert detail["docstring"]
    for hidden in ("source_path", "aliases", "legacy_ids", "docstring_note", "func_name"):
        assert hidden not in detail, hidden


def test_experiment_entry_points_use_the_peaks_top_level_contract():
    index = build_index()
    for name in ("pxt2nc", "load_experiment"):
        canonical_id = f"top_level:peaks.core.fileIO.experiment:{name}"
        entry = index.get(canonical_id)
        assert entry is not None
        detail = describe_api(entry)
        assert detail["scope"] == "top_level"
        assert detail["module"] == "peaks.core.fileIO.experiment"
        assert detail["signature"].startswith(f"{name}(")


def test_get_is_canonical_only_and_search_is_compact():
    index = build_index()
    canonical_id = "dataarray:peaks.core.process.k_conversion:k_convert"
    entry = index.get(canonical_id)
    assert entry is not None
    assert index.get("k_convert") is None
    rows = index.search("momentum conversion", limit=3)
    assert rows and rows[0]["id"] == canonical_id
    assert rows[0]["tier"] == "native"
    assert rows[0]["score"] == 900
    for hidden in ("docstring", "aliases", "source_path", "signature", "legacy_ids"):
        assert hidden not in rows[0], hidden


def test_uncurated_and_legacy_facade_apis_are_not_model_visible():
    index = build_index()
    names = {entry["name"] for entry in index.entries}
    for retired in (
        "load_data",
        "convert_experiment",
        "inspect_experiment",
        "plot_batch",
        "plot_validation_pair",
        "show_mapping_slice",
        "iplot",
    ):
        assert retired not in names
    assert index.search("savefig", limit=10) == []


def test_bound_drops_receiver_by_name_not_scope():
    from peaksMCP.discovery.signatures import _bound

    assert (
        _bound("set_EF_correction(self, EF_correction)", "set_EF_correction", "metadata")
        == "set_EF_correction(EF_correction)"
    )
    assert _bound("k_convert(da, eV=None)", "k_convert", "dataarray") == "k_convert(eV=None)"


def test_index_fingerprint_detects_source_changes(monkeypatch):
    import peaksMCP.discovery.index as index_module

    index = build_index()
    assert index.is_stale() is False
    monkeypatch.setattr(index_module, "source_fingerprint", lambda *a, **k: "changed")
    monkeypatch.setattr(index_module, "STALE_REFRESH_INTERVAL_S", 0.0)
    assert index.is_stale() is True


def test_search_namespace_is_single_catalog():
    index = build_index()
    namespace, rows = index.search_tiered("gold reference fit", limit=3)
    assert namespace == "catalog"
    assert rows[0]["name"] == "fit_gold"
    assert index.search_tiered("", limit=3)[0] == "all"


def test_ef_correction_handoff_notes_match_real_signatures():
    index = build_index()
    signatures = {
        name: describe_api(next(item for item in index.entries if item["name"] == name))["signature"]
        for name in ("k_convert", "fit_gold")
    }
    assert "EF_correction=" in signatures["k_convert"], signatures["k_convert"]
    assert "EF_correction_type" in signatures["fit_gold"], signatures["fit_gold"]
