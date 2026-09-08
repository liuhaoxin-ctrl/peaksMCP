from __future__ import annotations

from peaksMCP.discovery.index import (
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
    assert {f"{item['module']}:{item['name']}" for item in override} == declared
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


def test_native_query_falls_back_to_native_tier():
    index = build_index()
    # A native peaks alias must not be shadowed by any override.
    tier, matches = index.search_tiered("energy distribution curve", limit=5)
    assert tier == TIER_NATIVE
    assert "EDC" in _names(matches)
    # A native exact name likewise falls through (no override equals it).
    tier, matches = index.search_tiered("k_convert", limit=3)
    assert tier == TIER_NATIVE
    assert matches[0]["name"] == "k_convert"


def test_weak_partial_override_match_does_not_hijack_short_names():
    """'plot' must not be routed to plot_batch — only exact name/alias wins."""
    index = build_index()
    tier, matches = index.search_tiered("plot", limit=5)
    assert tier == TIER_NATIVE
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
