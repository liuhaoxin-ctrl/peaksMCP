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
    from peaksMCP.discovery.index import load_api_overrides

    declared = load_project_added()
    override = [item for item in index.entries if item.get("project_added")]
    # Project entries live under ONE canonical module (module:peaksMCP.overrides:<name>);
    # internal (write-verb) entries are not model-facing.
    assert {item["module"] for item in override} == {CANONICAL_MODULE}
    internal = {
        name
        for name, config in load_api_overrides().items()
        if config.get("project") and config.get("exposure") == "internal"
    }
    assert len(override) == len(declared) - len(internal)
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
    tier, matches = index.search_tiered("inspect_experiment", limit=3)
    assert tier == TIER_OVERRIDE
    assert matches[0]["name"] == "inspect_experiment"


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
    """Project APIs are described from their manifest contract, never source.

    The export is imported at detail time and the signature runtime-verified;
    the docstring-free whitelist carries the structured contract instead.
    """
    index = build_index()
    override = next(item for item in index.entries if item["name"] == "plot_batch")
    detail = describe_api(override)
    assert "source_path" not in detail, "override APIs must not leak source paths"
    assert "docstring" not in detail
    assert detail["tier"] == TIER_OVERRIDE
    assert detail["module"] == CANONICAL_MODULE
    assert detail["export"] == f"{CANONICAL_MODULE}.plot_batch"
    assert detail["signature_resolved"] is True, "manifest export must import"
    assert "plot_batch(" in detail["signature"]
    assert detail["contract"]["summary"]
    assert detail["contract"]["inputs"]
    assert detail["contract"]["returns"]

    # Native peaks APIs get the same clean whitelist: no source path either.
    native = next(item for item in index.entries if item["name"] == "k_convert")
    native_detail = describe_api(native)
    assert "source_path" not in native_detail
    assert native_detail["signature"]
    assert native_detail["docstring"]


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
    reachable by exact name/alias and via include_advanced=True.

    The v4 manifest curates exactly six public facades today, so the gate is
    exercised against synthetic advanced rows (the ranker must keep enforcing
    it the day a low-level row is curated again).
    """
    from peaksMCP.discovery.index import load_api_overrides, search_index_tiered

    index = build_index()
    # Today's manifest declares no advanced/internal project rows.
    assert {
        name
        for name, config in load_api_overrides().items()
        if config.get("export") and config.get("exposure") != "facade"
    } == set()
    assert all(
        item.get("exposure") == "facade"
        for item in index.entries
        if item.get("project_added")
    )

    advanced = {
        "id": "module:peaksMCP.overrides:advanced_helper",
        "scope": "module",
        "module": CANONICAL_MODULE,
        "name": "advanced_helper",
        "kind": "function",
        "func_name": "advanced_helper",
        "summary": "low-level conversion helper",
        "docstring": "low-level conversion helper",
        "aliases": ["classify format"],
        "exposure": "advanced",
        "tier": TIER_OVERRIDE,
        "project_added": True,
    }
    entries = [*index.entries, advanced]
    # Generic fuzzy query must not surface the advanced API by default.
    for query in ("batch conversion helper", "translate a datasheet file"):
        _, matches = search_index_tiered(entries, query, limit=10)
        names = [m["name"] for m in matches]
        assert "advanced_helper" not in names, (query, names)
    # ... but include_advanced=True lets it participate.
    from peaksMCP.discovery.index import search_index

    names = [m["name"] for m in search_index(entries, "low-level helper", limit=10, include_advanced=True)]
    assert "advanced_helper" in names
    # Exact-name queries are always allowed (score 1000).
    tier, matches = search_index_tiered(entries, "advanced_helper", limit=3)
    assert tier == TIER_OVERRIDE and matches[0]["name"] == "advanced_helper"
    # Exact aliases resolve too (score 900).
    tier, matches = search_index_tiered(entries, "classify format", limit=3)
    assert tier == TIER_OVERRIDE and matches[0]["name"] == "advanced_helper"



def test_facade_apis_stay_fully_searchable():
    index = build_index()
    facade_names = {
        item["name"] for item in index.entries if item.get("exposure") == "facade"
    }
    assert {"load_data", "convert_experiment", "inspect_experiment", "plot_batch"} <= facade_names
    # Task facades were demoted to internal: absent from the model surface.
    assert "preprocess_cut" not in facade_names
    assert "fit_gold_reference" not in facade_names
    # A generic intent query still lands on the facade (not the advanced twin).
    names = _names(index.search("convert a file to netcdf", limit=5))
    assert "convert_experiment" in names
    assert "convert_pxt" not in names
