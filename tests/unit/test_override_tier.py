from __future__ import annotations

from peaksMCP.discovery.index import (
    CANONICAL_MODULE,
    TIER_MIXED,
    TIER_NATIVE,
    TIER_OVERRIDE,
    build_index,
    load_project_added,
)
from peaksMCP.discovery.signatures import describe_api


def test_every_entry_is_tagged_with_a_tier():
    index = build_index()
    # The tagged set must mirror the curated declaration exactly.  Compare
    # against the declaration source (manifest) instead of a hard-coded count,
    # so adding/removing a facade only touches the manifest, never this test.
    declared = load_project_added()
    override = [item for item in index.entries if item.get("project_added")]
    # Project entries live under ONE canonical module (module:peaksMCP.overrides:<name>).
    assert {item["module"] for item in override} == {CANONICAL_MODULE}
    assert len(override) == len(declared)
    for item in index.entries:
        expected = TIER_OVERRIDE if item.get("project_added") else TIER_NATIVE
        assert item["tier"] == expected, item["id"]
    assert all(item["tier"] == TIER_OVERRIDE for item in override)


def _names(matches: list[dict]) -> list[str]:
    return [m["name"] for m in matches]


def test_override_alias_query_wins_stage_one_and_returns_only_overrides():
    index = build_index()
    tier, matches = index.search_tiered("batch grid", limit=5)
    assert tier == TIER_OVERRIDE
    assert "plot_batch" in _names(matches)
    assert matches and all(m["tier"] == TIER_OVERRIDE for m in matches)


def test_override_exact_name_wins_stage_one():
    index = build_index()
    tier, matches = index.search_tiered("read_meta", limit=3)
    assert tier == TIER_OVERRIDE
    assert matches[0]["name"] == "read_meta"


def test_native_query_falls_back_to_mixed_namespace():
    index = build_index()
    # A native peaks alias must not be shadowed by any override.
    searched, matches = index.search_tiered("energy distribution curve", limit=5)
    assert searched == TIER_MIXED
    assert "EDC" in _names(matches)
    # A native exact name likewise falls through (no override equals it).
    searched, matches = index.search_tiered("k_convert", limit=3)
    assert searched == TIER_MIXED
    assert matches[0]["name"] == "k_convert"


def test_weak_partial_override_match_does_not_hijack_short_names():
    """'plot' must not be routed to plot_batch — only exact name/alias wins."""
    index = build_index()
    searched, matches = index.search_tiered("plot", limit=5)
    assert searched == TIER_MIXED
    # The fallback is the normal full-index ranking, not an override-only list.
    assert not (matches and all(m["tier"] == TIER_OVERRIDE for m in matches))


def test_empty_query_lists_under_all_tier():
    index = build_index()
    tier, matches = index.search_tiered("", limit=3)
    assert tier == "all"
    assert len(matches) == 3


def test_override_apis_are_black_box_without_source_path():
    index = build_index()
    override = next(item for item in index.entries if item["name"] == "plot_batch")
    detail = describe_api(override)
    assert "source_path" not in detail, "override APIs must not leak source paths"
    assert detail["tier"] == TIER_OVERRIDE
    assert detail["module"].startswith("peaksMCP.")
    assert detail["docstring"]

    # Native peaks APIs keep their source-backed documentation.
    native = next(item for item in index.entries if item["name"] == "k_convert")
    native_detail = describe_api(native)
    assert native_detail["source_path"].endswith(".py")


def test_search_match_mode_classification():
    """peaks_search_api's match_mode must reflect how the query resolved."""
    from peaksMCP.server.jupyter_peaks.core.tools import _search_match_mode

    override_hit = [{"name": "load_data"}]
    mixed_hit = [{"name": "k_convert"}]
    assert _search_match_mode("load_data", "override", override_hit) == "exact_name"
    assert _search_match_mode("加载数据", "override", override_hit) == "exact_alias"
    assert _search_match_mode("load the dataset", "mixed", mixed_hit) == "fuzzy"
    assert _search_match_mode("", "all", []) == "list"
    assert _search_match_mode("anything", "mixed", []) == "fuzzy"


def test_advanced_apis_are_hidden_until_exact_or_opt_in():
    """exposure=advanced: hidden from generic/fuzzy search and listings;
    reachable by exact name and via include_advanced=True."""
    index = build_index()
    advanced_names = {
        item["name"] for item in index.entries if item.get("exposure") == "advanced"
    }
    assert "convert_pxt" in advanced_names and "read_meta" in advanced_names
    assert advanced_names <= {
        item["name"] for item in index.entries if item.get("project_added")
    }
    # Generic fuzzy query must not surface an advanced API by default.
    for query in ("batch conversion helper", "translate a datasheet file"):
        names = _names(index.search(query, limit=10))
        assert not any(name in advanced_names for name in names), (query, names)
    # ... but include_advanced=True lets them participate.
    names = _names(index.search("translate datasheet", limit=10, include_advanced=True))
    assert "translate_datasheet" in names
    # Exact-name queries are always allowed (score 1000).
    tier, matches = index.search_tiered("read_meta", limit=3)
    assert tier == TIER_OVERRIDE and matches[0]["name"] == "read_meta"
    # Exact aliases resolve too (score 900).
    tier, matches = index.search_tiered("convert pxt", limit=3)
    assert tier == TIER_OVERRIDE and matches[0]["name"] == "convert_pxt"
    # Broad user-intent aliases reach the facade, not the advanced twin.
    names = _names(index.search("batch convert a folder", limit=10))
    assert "convert_path" not in names
    names = _names(index.search("batch convert a folder", limit=10, include_advanced=True))
    assert "convert_path" in names


def test_facade_apis_stay_fully_searchable():
    index = build_index()
    facade_names = {
        item["name"] for item in index.entries if item.get("exposure") == "facade"
    }
    assert {"load_data", "preprocess_cut", "plot_batch"} <= facade_names
    # A generic intent query still lands on the facade (not the advanced twin).
    names = _names(index.search("convert a file to netcdf", limit=5))
    assert "convert_experiment" in names
    assert "convert_pxt" not in names
