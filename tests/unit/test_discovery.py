from __future__ import annotations

from pathlib import Path

import yaml

from peaksMCP.discovery.index import build_index
from peaksMCP.discovery.signatures import describe_api


def test_live_index_has_stable_unique_ids():
    first = build_index()
    second = build_index()
    ids = [item["id"] for item in first.entries]
    assert len(ids) >= 100
    assert len(ids) == len(set(ids))
    assert ids == [item["id"] for item in second.entries]
    assert first.fingerprint == second.fingerprint


def test_every_public_name_is_searchable_in_top_three():
    index = build_index()
    for entry in index.entries:
        names = [match["name"] for match in index.search(entry["name"], limit=3)]
        assert entry["name"] in names, entry["id"]


def test_at_least_sixty_natural_language_aliases_rank_top_three():
    index = build_index()
    document = yaml.safe_load(Path(__file__).parents[2].joinpath("peaksMCP/discovery/api_overrides.yaml").read_text())
    cases = [(query, name) for name, queries in document["aliases"].items() for query in queries]
    assert len(cases) >= 60
    reciprocal_ranks = []
    for query, expected in cases[:60]:
        names = [match["name"] for match in index.search(query, limit=3)]
        assert expected in names, (query, expected, names)
        reciprocal_ranks.append(1 / (names.index(expected) + 1))
    assert sum(reciprocal_ranks) / len(reciprocal_ranks) >= 0.9


def test_get_api_returns_source_signature_and_docstring():
    index = build_index()
    entry = next(item for item in index.entries if item["name"] == "k_convert")
    detail = describe_api(entry)
    assert "k_convert(" in detail["signature"]
    assert detail["source_path"].endswith(".py")
    assert detail["docstring"]


def test_interactive_widget_apis_are_discoverable_by_intent():
    index = build_index()
    # The hvplot-based `iplot` accessor is intentionally hidden from search/get
    # entirely: "interactive ..." intent resolves to the native Qt viewer disp.
    assert all(item.get("name") != "iplot" for item in index.entries)
    assert not any(
        "peaks.core.GUI.iplot.hvplot" in str(item.get("module", ""))
        for item in index.entries
    )
    for query in ("interactive", "interactive panel", "interactive viewer", "widget"):
        names = [match["name"] for match in index.search(query, limit=5)]
        assert "disp" in names, (query, names)
    for query in ("iplot", "hvplot"):
        names = [match["name"] for match in index.search(query, limit=5)]
        assert "iplot" not in names, (query, names)


def test_get_resolves_canonical_id_name_and_alias():
    """peaks_get_api must accept the full canonical ID, the bare API name and
    any search alias (e.g. mapping slice) — all resolve to the same entry."""
    index = build_index()
    entry = next(
        item
        for item in index.entries
        if item["id"] == "module:peaksMCP.workflows.slice_view:show_mapping_slice"
    )
    assert entry["name"] == "show_mapping_slice"
    assert "mapping slice" in entry.get("aliases", [])
    assert index.get(entry["id"]) is not None
    resolved_name = index.get("show_mapping_slice")
    assert resolved_name is not None and resolved_name["id"] == entry["id"]
    resolved_alias = index.get("mapping slice")
    assert resolved_alias is not None and resolved_alias["id"] == entry["id"]


def test_bound_drops_receiver_by_name_not_scope():
    from peaksMCP.discovery.signatures import _bound

    # Accessor-class methods: a leading self is dropped in any accessor scope.
    assert (
        _bound("set_EF_correction(self, EF_correction)", "set_EF_correction", "metadata")
        == "set_EF_correction(EF_correction)"
    )
    assert (
        _bound("linear(self, independent_var=None)", "linear", "quick_fit")
        == "linear(independent_var=None)"
    )
    # xarray accessor: the receiver is dropped for accessor scopes.
    assert _bound("k_convert(da, eV=None)", "k_convert", "dataarray") == "k_convert(eV=None)"
    # module-level functions keep a real data argument.
    assert (
        _bound("plot_batch(data, ncols=3)", "plot_batch", "module")
        == "plot_batch(data, ncols=3)"
    )


def test_index_fingerprint_detects_source_changes(monkeypatch):
    import peaksMCP.discovery.index as index_module

    index = build_index()
    assert index.is_stale() is False
    # A source change must flip the fingerprint without touching the entries.
    monkeypatch.setattr(index_module, "source_fingerprint", lambda *a, **k: "changed-after-build")
    assert index_module.source_fingerprint() == "changed-after-build"
    assert index.is_stale() is True


class _StubNotebook:
    def list_variables(self) -> list:
        return []

    def read_variable(self, name: str) -> dict:
        return {}

    def active_cell(self) -> dict:
        return {}

    def active_cell_output(self) -> dict:
        return {"outputs": []}

    def notebook_content(self) -> dict:
        return {}

    def move_cursor(self, where: str) -> dict:
        return {}

    def server_status(self) -> dict:
        return {}

    def kernel_status(self) -> dict:
        return {}

    def wait_for_kernel(self) -> dict:
        return {}


def test_stale_index_is_hot_rebuilt_by_search_and_get(monkeypatch, tmp_path):
    from peaksMCP.server.jupyter_peaks.backend.base import ExecutionMode, SharedState
    from peaksMCP.server.jupyter_peaks.core.tools import register_safe_tools

    class FakeMCP:
        def __init__(self) -> None:
            self.registered: dict[str, object] = {}

        def tool(self, name: str, **kwargs):
            def wrap(function):
                self.registered[name] = function
                return function
            return wrap

    state = SharedState(ipython=object())
    state.mode = ExecutionMode.SAFE
    state.api_index = build_index()
    state.api_index.fingerprint = "changed-after-build"  # simulate stale
    mcp = FakeMCP()
    from peaksMCP.server.jupyter_peaks.security import AuditLogger
    register_safe_tools(mcp, state, _StubNotebook(), AuditLogger(tmp_path / "test-audit.log"))  # type: ignore[arg-type]
    search = mcp.registered["peaks_search_api"]
    get = mcp.registered["peaks_get_api"]
    # A stale index is hot-rebuilt in-kernel instead of erroring: search/get
    # succeed and the index is fresh again.
    result = search("k_convert")
    assert result["count"] >= 1
    detail = get("dataarray:peaks.core.process.k_conversion:k_convert")
    assert "k_convert" in detail["signature"]
    assert state.api_index.is_stale() is False
    # A fresh index is not stale and search works end to end.
    state.api_index = build_index()
    assert search("k_convert")["count"] >= 1
