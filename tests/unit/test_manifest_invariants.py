"""Catalog-to-Python surface invariants."""

from __future__ import annotations

from peaksMCP.discovery.index import load_api_catalog


def test_current_catalog_has_no_facade_rows():
    assert not {
        canonical_id
        for canonical_id, row in load_api_catalog().items()
        if row["kind"] == "facade"
    }


def test_overrides_exports_only_the_save_receipt_contract():
    import peaksMCP.overrides as overrides

    assert set(overrides.__all__) == {"SaveReceipt"}
    for retired in (
        "load_data",
        "convert_experiment",
        "inspect_experiment",
        "plot_batch",
        "plot_validation_pair",
        "show_mapping_slice",
    ):
        assert not hasattr(overrides, retired)


def test_all_catalog_keys_are_exact_discovery_ids():
    from peaksMCP.discovery.index import build_index

    index = build_index()
    visible = {
        canonical_id
        for canonical_id, row in load_api_catalog().items()
        if row["exposure"] != "hidden"
    }
    assert {entry["id"] for entry in index.entries} == visible
