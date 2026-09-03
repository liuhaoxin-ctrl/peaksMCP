from __future__ import annotations

import pytest


# --------------------------------------------------------------------------- #
# CommBridge (active_cell_bridge.py)                                          #
# --------------------------------------------------------------------------- #
def test_comm_bridge_request_reply_and_state_update():
    import threading
    import time

    from peaksMCP.server.jupyter_peaks.active_cell_bridge import CommBridge
    from peaksMCP.server.jupyter_peaks.backend import SharedState


    class FakeIPython:
        user_ns = {}

    state = SharedState(FakeIPython())
    bridge = CommBridge(state)

    class FakeComm:
        def __init__(self):
            self.sent = []
            self._closed = False

        def send(self, payload):
            self.sent.append(payload)
            # Deliver through the registered callback on a separate thread.
            if payload.get("type") == "request":
                reply = {"request_id": payload["request_id"], "ok": True, "result": {"value": 42}}
                thread = threading.Thread(target=lambda: self.message_callback({"content": {"data": reply}}), daemon=True)
                thread.start()

        def on_msg(self, callback):
            self.message_callback = callback

        def on_close(self, _cb):
            pass

    comm = FakeComm()
    bridge._on_open(comm, {"content": {"data": {}}})
    bridge.last_seen = time.time()

    result = bridge.request("read_active_cell", {}, timeout=5)
    assert result.get("value") == 42
    assert comm.sent[-1]["type"] == "request"

    # A pushed active-cell message updates the state cache.
    comm.message_callback({"content": {"data": {"type": "active_cell", "cell": {"id": "c1", "index": 1}, "outputs": [{"output_type": "display_data"}]}}})
    assert state.active_cell == {"id": "c1", "index": 1}
    assert len(state.active_cell_output) == 1


def test_comm_bridge_fails_closed_when_disconnected():
    from peaksMCP.server.jupyter_peaks.active_cell_bridge import CommBridge
    from peaksMCP.server.jupyter_peaks.backend import SharedState

    state = SharedState(type("_IP", (), {"user_ns": {}})())
    bridge = CommBridge(state)
    with pytest.raises(RuntimeError, match="not connected"):
        bridge.request("read_active_cell")


# --------------------------------------------------------------------------- #
# jupyter_mcp_extension.py: autostart honouring                                #
# --------------------------------------------------------------------------- #
def test_load_extension_autostart_env_controls_mcp_start(monkeypatch):
    import os

    import peaksMCP.server.jupyter_peaks.jupyter_mcp_extension as ext

    class FakeIPython:
        def register_magics(self, _magics):
            pass

    calls: list[str] = []
    real_get = os.environ.get

    def fake_start(ipython):
        calls.append("start")

    monkeypatch.setattr(ext, "_start", fake_start)
    monkeypatch.setattr(
        ext.os.environ, "get",
        lambda key, default=None: "false" if key == "PEAKSMCP_AUTOSTART" else real_get(key, default),
    )

    ext.load_ipython_extension(FakeIPython())
    assert calls == []  # autostart=false -> magics only, no MCP start

    monkeypatch.setattr(
        ext.os.environ, "get",
        lambda key, default=None: real_get(key, default),
    )
    ext.load_ipython_extension(FakeIPython())
    assert calls == ["start"]


# --------------------------------------------------------------------------- #
# workflows (publication.py)                                                  #
# --------------------------------------------------------------------------- #
def test_validate_arpes_metadata_units_required():
    import xarray as xr

    from peaksMCP.workflows.publication import validate_arpes_metadata

    good = xr.DataArray(
        [[1.0, 2.0], [3.0, 4.0]],
        dims=("eV", "theta_par"),
    ).assign_coords(
        eV=("eV", [0, 1], {"units": "eV"}),
        theta_par=("theta_par", [-1, 1], {"units": "deg"}),
    ).assign_attrs(units="counts")
    assert validate_arpes_metadata(good) == []

    bad = xr.DataArray([[1.0, 2.0], [3.0, 4.0]], dims=("eV", "theta_par"))
    issues = validate_arpes_metadata(bad)
    assert any("no units" in issue for issue in issues)


def test_publication_grid_rejects_unvalidated_data():
    import xarray as xr

    from peaksMCP.workflows.publication import publication_grid

    bad = xr.DataArray([[1.0, 2.0], [3.0, 4.0]], dims=("eV", "theta_par"))
    with pytest.raises(ValueError, match="validation failed"):
        publication_grid([bad], titles=["bad"])


def test_save_processed_roundtrips_metadata_attrs(tmp_path):
    from pathlib import Path

    import numpy as np
    import peaks
    import xarray as xr

    from peaksMCP.workflows import save_processed

    da = xr.DataArray(
        np.random.rand(10, 10),
        dims=("eV", "kx"),
        coords={
            "eV": xr.DataArray(np.linspace(-1, 0, 10), dims="eV", attrs={"units": "eV"}),
            "kx": xr.DataArray(np.linspace(-0.3, 0.3, 10), dims="kx", attrs={"units": "1 / angstrom"}),
        },
        attrs={
            "units": "counts",
            "experiment_metadata_json": {"a": 1},
            "_EF_correction": {"c0": 2.6597},
            "calibration": {"ef_correction": {"c0": 2.6597}},
            "history": "AnalysisHistoryRecordCollection(entries=[...])",
        },
    )

    path = save_processed(da, str(tmp_path / "out"))
    assert path.endswith(".nc") and Path(path).is_file()
    # Raw to_netcdf would reject the dict attrs; save_processed must round-trip.
    back = peaks.load(path)
    assert back.attrs["_EF_correction"] == {"c0": 2.6597}
    import pint

    assert isinstance(back.coords["kx"].attrs["units"], pint.Unit)
