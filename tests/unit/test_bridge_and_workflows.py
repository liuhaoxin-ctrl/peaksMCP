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
            # Reply synchronously so the request event resolves.
            if payload.get("type") == "request":
                reply = {"request_id": payload["request_id"], "ok": True, "result": {"value": 42}}
                thread = threading.Thread(target=lambda: bridge._on_message({"content": {"data": reply}}), daemon=True)
                thread.start()

        def on_msg(self, _cb):
            pass

        def on_close(self, _cb):
            pass

    comm = FakeComm()
    bridge._on_open(comm, {"content": {"data": {}}})
    bridge.last_seen = time.time()

    result = bridge.request("read_active_cell", {}, timeout=5)
    assert result.get("value") == 42
    assert comm.sent[-1]["type"] == "request"

    # A pushed active-cell message updates the state cache.
    bridge._on_message({"content": {"data": {"type": "active_cell", "cell": {"id": "c1", "index": 1}, "outputs": [{"output_type": "display_data"}]}}})
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
