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


def test_stale_error_is_raised_by_search_and_get(monkeypatch):
    from peaksMCP.discovery.index import IndexStaleError
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
    register_safe_tools(mcp, state, _StubNotebook(), AuditLogger("/tmp/test-audit.log"))  # type: ignore[arg-type]
    search = mcp.registered["peaks_search_api"]
    get = mcp.registered["peaks_get_api"]
    for call in (
        lambda: search("k_convert"),
        lambda: get("dataarray:peaks.core.process.k_conversion:k_convert"),
    ):
        try:
            call()
        except IndexStaleError as exc:
            assert "INDEX_STALE_RESTART_REQUIRED" in str(exc)
        else:
            raise AssertionError("expected IndexStaleError")
    # A fresh index is not stale and search works end to end.
    state.api_index = build_index()
    assert search("k_convert")["count"] >= 1
