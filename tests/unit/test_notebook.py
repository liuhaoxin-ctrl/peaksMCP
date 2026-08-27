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
    tree = xr.DataTree.from_dict({"/scan": xr.Dataset({"a": ("x", [1])})})
    assert "/scan" in summarize_xarray(tree)["groups"]


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

