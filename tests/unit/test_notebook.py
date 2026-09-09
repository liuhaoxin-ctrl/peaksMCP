from __future__ import annotations

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from peaksMCP.server.jupyter_peaks.backend import NotebookBackend, SharedState
from peaksMCP.server.jupyter_peaks.backend.notebook import summarize_xarray


class FakeIPython:
    def __init__(self, namespace):
        self.user_ns = namespace


def test_xarray_summary_includes_structure_units_and_lazy_state():
    value = xr.DataArray(
        da.ones((4, 5), chunks=(2, 5)), dims=("eV", "theta_par"),
        coords={"eV": xr.DataArray(np.arange(4), dims="eV", attrs={"units": "eV"})},
        attrs={"units": "counts", "sample": "BP"}, name="scan",
    )
    summary = summarize_xarray(value)
    assert summary["dims"] == ["eV", "theta_par"]
    assert summary["sizes"] == {"eV": 4, "theta_par": 5}
    assert summary["coords"]["eV"]["units"] == "eV"
    assert summary["lazy"] is True
    assert value.data.__dask_graph__() is not None


def test_dataset_and_datatree_summary():
    dataset = xr.Dataset({"a": ("x", [1, 2]), "b": ("x", [3, 4])})
    assert set(summarize_xarray(dataset)["variables"]) == {"a", "b"}
    assert summarize_xarray(dataset)["lazy"] is False
    tree = xr.DataTree.from_dict({"/scan": xr.Dataset({"a": ("x", [1])})})
    assert "/scan" in summarize_xarray(tree)["groups"]


def test_summary_lazy_check_never_materializes_backing_data(monkeypatch):
    """The lazy probe inspects storage types only; reading .data would force a
    full backend read (get_duck_array) of disk arrays just to answer it."""
    def raise_if_read(_self):
        raise AssertionError("summarize_xarray must not touch .data/.values")

    monkeypatch.setattr(xr.Variable, "data", property(raise_if_read))

    eager_ds = xr.Dataset({"a": ("x", [1, 2, 3])})
    eager_da = xr.DataArray([1, 2], dims="eV")
    assert summarize_xarray(eager_ds)["lazy"] is False
    assert summarize_xarray(eager_da)["lazy"] is False

    # A non-dask lazy backend (storage not in memory) is flagged lazy without
    # any data access — previously this path read value.data to decide.
    monkeypatch.setattr(xr.Variable, "_in_memory", property(lambda _self: False))
    assert summarize_xarray(eager_ds)["lazy"] is True
    assert summarize_xarray(eager_da)["lazy"] is True


def test_xarray_summary_reports_real_peaks_apis_without_invoking_them():
    import peaks  # noqa: F401  # installs Peaks descriptors on xarray

    summary = summarize_xarray(xr.DataArray([1], dims="eV"))
    assert {"metadata", "fit_gold", "k_convert"} <= set(summary["peaks_apis"])


def test_namespace_listing_and_missing_variable():
    backend = NotebookBackend(SharedState(FakeIPython({"scan": xr.DataArray([1]), "_private": 2})))
    assert [item["name"] for item in backend.list_variables()["variables"]] == ["scan"]
    with pytest.raises(KeyError):
        backend.read_variable("missing")


def test_read_active_cell_keeps_cursor_metadata_output_free():
    """active_cell returns the frontend snapshot verbatim for the read tool to
    normalise; the kernel-side cursor state never stores raw outputs
    (executed-cell outputs travel once, settled inside the execute response)."""
    class Bridge:
        connected = True

        def request(self, operation, timeout=5):
            assert operation == "read_active_cell"
            return {
                "id": "cell-7",
                "source": "data.plot()",
                "outputs": [{"output_type": "stream", "text": "fresh"}],
            }

    state = SharedState(FakeIPython({}))
    state.bridge = Bridge()
    backend = NotebookBackend(state)

    cell = backend.active_cell()
    assert cell["id"] == "cell-7"
    assert cell["outputs"] == [{"output_type": "stream", "text": "fresh"}]
    # Cursor metadata is stored without outputs; the duplicate output cache is gone.
    assert state.active_cell == {"id": "cell-7", "source": "data.plot()"}
    assert not hasattr(state, "active_cell_output")
    # Fallback path when the bridge is absent returns whatever was last stored.
    state.bridge = None
    assert backend.active_cell()["id"] == "cell-7"


def test_loaded_scans_surfaces_through_the_generic_index_protocol():
    """A LoadedScans index in the notebook must be visible and readable
    through the generic object-summary protocol (no special case, no
    classification): representation counts, conversion state and provenance.
    The notebook is the shared context and the agent needs programmatic
    situation awareness (is the required data already loaded?)."""
    from peaksMCP.overrides import LoadedScans, ScanEntry

    exp = LoadedScans(
        [
            ScanEntry(stem="BP_0020", path="/d/BP_0020.nc", representation="netcdf",
                      experiment_index=20, sizes={"eV": 168, "theta_par": 902}),
            ScanEntry(stem="BP_0015", path="/d/BP_0015.pxt", representation="raw_pxt",
                      experiment_index=15, sizes={"eV": 168, "theta_par": 902}),
        ],
        source="BP260623/data",
        metadata_source="datasheet",
        metadata_path="/d/datasheet.csv",
    )
    backend = NotebookBackend(SharedState(FakeIPython({"exp": exp, "scan": xr.DataArray([1])})))

    listed = backend.list_variables()["variables"]
    row = next(item for item in listed if item["name"] == "exp")
    assert row["type"].endswith("LoadedScans")
    # Structural only: representations, not classification.
    assert row["indexed"] == {"n": 2, "representations": {"raw_pxt": 1, "netcdf": 1},
                              "needs_conversion": 1}

    detail = backend.read_variable("exp")
    assert detail["name"] == "exp"
    assert detail["n_files"] == 2
    assert detail["representations"] == {"raw_pxt": 1, "netcdf": 1}
    assert detail["needs_conversion"] == ["BP_0015"]
    assert detail["metadata_source"] == "datasheet"
    assert detail["summary"].startswith("load_data:")
    # No classification leaks into the notebook surface.
    assert "gold" not in detail and "cuts" not in detail


def test_index_protocol_works_for_any_object_without_imports():
    """The protocol is duck-typed: any entries/stems object summarizes, and
    ordinary objects still fall back to a bounded repr."""
    class Entry:
        def __init__(self, stem, representation):
            self.stem = stem
            self.representation = representation

    class MyIndex:
        entries = [Entry("a.pxt", "raw_pxt")]
        source = "mine"
        metadata_source = "none"
        stems = ["a.pxt"]

        def summary_line(self):
            return "my summary line"

        @property
        def needs_conversion(self):
            return ["a.pxt"]

    backend = NotebookBackend(SharedState(FakeIPython({"mine": MyIndex(), "plain": object()})))
    listed = backend.list_variables()["variables"]
    row = next(item for item in listed if item["name"] == "mine")
    assert row["indexed"]["representations"] == {"raw_pxt": 1}
    detail = backend.read_variable("plain")
    assert "repr" in detail and len(detail["repr"]) <= 4000


def test_inspect_notebook_targets_and_details_are_bounded():
    from peaksMCP.overrides import LoadedScans, ScanEntry

    exp = LoadedScans(
        [ScanEntry(stem="BP_0015", path="/d/BP_0015.nc", representation="netcdf",
                   experiment_index=15, sizes={"eV": 168, "theta_par": 902})],
        source="data_netcdf",
    )
    state = SharedState(FakeIPython({"exp": exp, "scan": xr.DataArray([1], dims="eV")}))
    backend = NotebookBackend(state)

    rows = backend.inspect("variables", detail="summary", limit=1)
    assert rows["target"] == "variables" and rows["count"] == 2 and rows["truncated"] is True
    assert len(rows["variables"]) == 1
    summary_row = rows["variables"][0]
    assert summary_row["name"] == "exp"
    assert "summary" in summary_row and "dims" not in summary_row

    preview = backend.inspect("variables", detail="preview")
    preview_rows = {item["name"]: item for item in preview["variables"]}
    assert preview_rows["exp"]["structural"]["representations"]["netcdf"] == 1
    assert preview_rows["scan"]["structural"]["dims"] == ["eV"]

    variable = backend.inspect("variable", variable_name="exp", detail="summary")
    assert variable["summary"].startswith("LoadedScans n=1")
    variable_preview = backend.inspect("variable", variable_name="scan", detail="preview")
    assert variable_preview["dims"] == ["eV"]

    cell = backend.inspect("active_cell", detail="summary")
    assert cell["target"] == "active_cell"
    assert cell["n_outputs"] is None

    with pytest.raises(ValueError, match="unknown target"):
        backend.inspect("variablesx")
    with pytest.raises(ValueError, match="requires variable_name"):
        backend.inspect("variable")
    with pytest.raises(KeyError):
        backend.inspect("variable", variable_name="missing")
    # limit is clamped to 1..50 (0 clamps up to 1).
    assert len(backend.inspect("variables", detail="summary", limit=0)["variables"]) == 1


def test_inspect_active_cell_never_returns_raw_outputs():
    """Executed outputs travel once (settled inside the run_cell reply);
    inspect_notebook(active_cell, preview) must never reopen that channel -
    no raw outputs, no image payloads, just identity/source metadata."""
    class Bridge:
        connected = True

        def request(self, operation, timeout=5):
            assert operation == "read_active_cell"
            return {
                "id": "cell-7",
                "index": 3,
                "cell_type": "code",
                "execution_count": 4,
                "source": "data.plot()",
                "outputs": [
                    {"output_type": "stream", "text": "secret"},
                    {"output_type": "display_data",
                     "data": {"image/png": "QUJDREVGRw=="}},
                ],
            }

    state = SharedState(FakeIPython({}))
    state.bridge = Bridge()
    backend = NotebookBackend(state)

    summary = backend.inspect("active_cell", detail="summary")
    assert summary["outputs_omitted"] is True
    assert summary["n_outputs"] == 2
    assert "outputs" not in summary and "secret" not in str(summary)

    preview = backend.inspect("active_cell", detail="preview")
    assert preview["id"] == "cell-7" and preview["source"].startswith("data.plot()")
    assert "outputs" not in preview
    assert "secret" not in str(preview) and "QUJDREVGRw" not in str(preview)


def test_inspect_notebook_cell_history_targets_are_bounded_and_output_free():
    """cells/cell targets read bounded history through the frontend document
    model: identity + source + execution metadata only, never raw outputs;
    pagination via limit/offset; missing cells raise KeyError."""
    class Bridge:
        connected = True

        def request(self, operation, payload=None, timeout=5):
            if operation == "read_cells":
                limit, offset = payload["limit"], payload["offset"]
                all_cells = [
                    {"id": f"c{i}", "index": i, "cell_type": "code",
                     "execution_count": i, "source": f"source {i}"}
                    for i in range(5)
                ]
                end = len(all_cells) - offset
                start = max(0, end - limit)
                return {"cells": all_cells[start:end], "truncated": start > 0}
            if operation == "read_cell":
                wanted = payload["cell"]
                for row in [
                    {"id": f"c{i}", "index": i, "cell_type": "code",
                     "execution_count": i,
                     "source": "data.plot()", "outputs": ["secret-output"]}
                    for i in range(5)
                ]:
                    if wanted == row["id"] or wanted == row["index"]:
                        return {"cell": row}
                return {"cell": None}
            raise AssertionError(f"unexpected op {operation}")

    state = SharedState(FakeIPython({}))
    state.bridge = Bridge()
    backend = NotebookBackend(state)

    rows = backend.inspect("cells", detail="summary", limit=2, offset=1)
    assert rows["count"] == 2 and rows["truncated"] is True
    assert [cell["index"] for cell in rows["cells"]] == [2, 3]
    assert rows["cells"][0]["source_preview"].startswith("source 2")
    assert "outputs" not in rows["cells"][0]

    preview = backend.inspect("cells", detail="preview", limit=50)
    assert preview["cells"][0]["source"] == "source 0"
    assert "outputs" not in preview["cells"][0]

    one = backend.inspect("cell", cell="c3", detail="preview")
    assert one["index"] == 3 and one["source"].startswith("data.plot()")
    assert "outputs" not in one and "secret-output" not in str(one)

    summary = backend.inspect("cell", cell=1, detail="summary")
    assert summary["source_preview"] and "outputs" not in summary

    with pytest.raises(KeyError, match="does not exist"):
        backend.inspect("cell", cell="missing")
    with pytest.raises(ValueError, match="requires cell"):
        backend.inspect("cell")
    with pytest.raises(ValueError, match="unknown target"):
        backend.inspect("nonsense")
