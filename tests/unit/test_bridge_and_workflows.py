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

    # A pushed active-cell message updates the cursor state (no raw outputs
    # stored on it) and caches outputs per cell id for the settle buffer.
    comm.message_callback({"content": {"data": {"type": "active_cell", "cell": {"id": "c1", "index": 1}, "outputs": [{"output_type": "display_data"}]}}})
    assert state.active_cell == {"id": "c1", "index": 1}
    assert len(state.cell_outputs["c1"]) == 1


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


def test_load_extension_enables_matplotlib_inline_before_autostart(monkeypatch):
    """Plotting must render as inline png, never surface bare '<Figure>' reprs:
    the extension turns matplotlib inline on during load, before autostart."""
    import os

    import peaksMCP.server.jupyter_peaks.jupyter_mcp_extension as ext

    class FakeIPython:
        def register_magics(self, _magics):
            pass

        def run_line_magic(self, name, line):
            recorded.append((name, line))

    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(ext, "_start", lambda _ip: None)
    monkeypatch.setattr(
        ext.os.environ, "get",
        lambda key, default=None: "false" if key == "PEAKSMCP_AUTOSTART" else os.environ.get(key, default),
    )
    ext.load_ipython_extension(FakeIPython())
    assert ("matplotlib", "inline") in recorded


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


def test_read_meta_classifies_records_and_dimensionality(tmp_path):
    import json

    import numpy as np
    import xarray as xr

    from peaksMCP.workflows import read_meta

    meta = {
        "notes": ["Cut theta_offset=1.5"],
        "records": {
            "5": {"experiment": {"data_format": "sweep", "energy_start_eV": 2.2, "energy_stop_eV": 2.7},
                  "photon": {"polarisation": "P"}, "theta_offset_deg": 1.5, "is_gold_reference": False},
            "20": {"experiment": {"data_format": "Au sweep", "energy_start_eV": 2.2, "energy_stop_eV": 2.7},
                   "photon": {"polarisation": "P"}, "theta_offset_deg": 1.5, "is_gold_reference": True},
            "7": {"experiment": {"data_format": "mapping", "energy_start_eV": 2.2, "energy_stop_eV": 5.0},
                  "photon": {"polarisation": "S"}, "theta_offset_deg": 1.5, "is_gold_reference": False},
            "26": {"experiment": {"data_format": "sweep", "energy_start_eV": 2.2, "energy_stop_eV": 2.7},
                   "photon": {"polarisation": "S"}, "theta_offset_deg": 1.5, "is_gold_reference": False},
        },
    }
    path = tmp_path / "experiment_metadata.json"
    path.write_text(json.dumps(meta), encoding="utf-8")

    scans = {
        5: xr.DataArray(np.zeros((10, 10)), dims=("eV", "theta_par")),
        26: xr.DataArray(np.zeros((10, 61, 10)), dims=("eV", "deflector_perp", "theta_par")),
    }
    summary = read_meta(str(path), data=scans)

    assert summary["sweeps"] == [5, 26]
    assert summary["gold"] == [20]
    assert summary["mappings"] == [7]
    assert summary["energy_windows_eV"] == [(2.2, 2.7), (2.2, 5.0)]
    assert summary["notes"] == ["Cut theta_offset=1.5"]
    rec26 = next(r for r in summary["records"] if r["index"] == 26)
    assert rec26["ndim"] == 3 and rec26["dims"] == ["eV", "deflector_perp", "theta_par"]
    rec20 = next(r for r in summary["records"] if r["index"] == 20)
    assert rec20["is_gold"] is True

    # Without data, no dimensionality field but classification still works.
    bare = read_meta(meta)
    assert bare["sweeps"] == [5, 26]
    assert all("ndim" not in r for r in bare["records"])
