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


def test_wait_for_idle_timeout_and_success():
    state = SharedState(FakeIPython({}))
    backend = NotebookBackend(state)
    assert backend.wait_for_kernel(0.1)["ready"]
    state.kernel_state = "busy"
    assert not backend.wait_for_kernel(0.05, 0.01)["ready"]


def test_read_active_cell_refreshes_identity_scoped_output_cache():
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

    assert backend.active_cell()["id"] == "cell-7"
    assert backend.active_cell_output() == {
        "cell_id": "cell-7",
        "outputs": [{"output_type": "stream", "text": "fresh"}],
    }
