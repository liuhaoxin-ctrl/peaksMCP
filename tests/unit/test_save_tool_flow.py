"""Save-flow ordering contract for the save_with_consent backend.

Locks the Phase-3.5 fix: intent record cell FIRST, then staging/consent,
then the outcome record cell - and internal record cells must NOT pass
through the model mutation consent gate (no second confirmation when
require_consent is on).
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from peaksMCP.overrides import save as save_module
from peaksMCP.server.jupyter_peaks.backend import SharedState
from peaksMCP.server.jupyter_peaks.backend.notebook_unsafe import UnsafeNotebookBackend
from peaksMCP.server.jupyter_peaks.security import AuditLogger, ConsentManager


class _FakeIPython:
    def __init__(self, namespace):
        self.user_ns = namespace


class _FakeBridge:
    """Records every operation in order."""

    connected = True

    def __init__(self):
        self.events: list[str] = []

    def request(self, operation, payload=None, timeout=30.0):
        if operation == "add_cell":
            self.events.append(f"cell:{payload.get('cell_type')}")
            return {"id": "c", "cell_type": payload.get("cell_type")}
        raise AssertionError(f"unexpected operation {operation!r}")


def _array() -> xr.DataArray:
    return xr.DataArray(
        np.arange(6, dtype=float).reshape(2, 3),
        dims=("eV", "kx"),
        coords={"eV": np.linspace(-1, 0, 2), "kx": np.linspace(-0.3, 0.3, 3)},
        attrs={"units": "counts"},
    )


def _backend(tmp_path, namespace, bridge):
    state = SharedState(_FakeIPython(namespace))
    state.bridge = bridge
    return UnsafeNotebookBackend(
        state, ConsentManager(bridge), AuditLogger(tmp_path / "audit.log")
    )


def _approval_with_log(bridge, approved: bool):
    """Install an approval channel on the active gateway that logs its call."""
    def channel(payload):
        bridge.events.append("consent")
        return approved

    save_module._set_approval_channel(channel)
    return channel


def test_save_flow_order_intent_then_consent_then_outcome(tmp_path):
    """Order: precheck -> intent record cell -> (staging) consent -> publish ->
    outcome record cell.  The record cells never pass the model consent gate:
    exactly ONE consent event (the save card) sits between the two cells."""
    bridge = _FakeBridge()
    _approval_with_log(bridge, True)
    backend = _backend(tmp_path, {"scan": _array()}, bridge)
    target = tmp_path / "scan.nc"

    receipt = backend.save_with_consent("scan", str(target))

    assert receipt["status"] == "saved"
    assert target.exists()
    cells = [event for event in bridge.events if event.startswith("cell:")]
    assert cells == ["cell:markdown", "cell:markdown"]  # intent + outcome only
    consents = [event for event in bridge.events if event == "consent"]
    assert len(consents) == 1  # exactly ONE consent: the save card
    assert bridge.events.index("cell:markdown") < bridge.events.index("consent")
    assert bridge.events.index("consent") < len(bridge.events) - 1 - bridge.events[::-1].index("cell:markdown")


def test_save_denied_records_denied_outcome(tmp_path):
    bridge = _FakeBridge()
    _approval_with_log(bridge, False)
    backend = _backend(tmp_path, {"scan": _array()}, bridge)
    target = tmp_path / "scan.nc"

    receipt = backend.save_with_consent("scan", str(target))

    assert receipt["status"] == "denied"
    assert not target.exists()
    assert len([e for e in bridge.events if e == "consent"]) == 1
    assert bridge.events.count("cell:markdown") == 2


def test_save_blocked_before_staging_records_blocked_only(tmp_path):
    """An existing target without overwrite: no staging, no consent card; the
    record cell reports the block and the receipt carries the preview info."""
    bridge = _FakeBridge()
    _approval_with_log(bridge, True)
    backend = _backend(tmp_path, {"scan": _array()}, bridge)
    target = tmp_path / "scan.nc"
    target.write_bytes(b"existing")

    receipt = backend.save_with_consent("scan", str(target))

    assert receipt["status"] == "blocked"
    assert "exists" in (receipt["note"] or "")
    assert receipt["kind"] == "netcdf" and receipt["dims"] == {"eV": 2, "kx": 3}
    assert target.read_bytes() == b"existing"
    assert bridge.events == ["cell:markdown"]  # one record cell, no consent


def test_save_without_channel_fails_closed_after_intent_cell(tmp_path):
    """No approval channel: intent cell first, then blocked (nothing staged,
    no consent attempted, no outcome bytes)."""
    bridge = _FakeBridge()
    save_module._set_approval_channel(None)
    backend = _backend(tmp_path, {"scan": _array()}, bridge)
    target = tmp_path / "scan.nc"

    receipt = backend.save_with_consent("scan", str(target))

    assert receipt["status"] == "blocked"
    assert receipt["note"] and "no approval channel" in receipt["note"]
    assert not target.exists()
    assert len([e for e in bridge.events if e.startswith("cell:")]) == 2
    assert "consent" not in bridge.events
    assert save_module._active_gateway()._staged == {}


def test_save_unknown_variable_and_unsupported_target_raise_before_cells(tmp_path):
    bridge = _FakeBridge()
    backend = _backend(tmp_path, {"scan": _array()}, bridge)

    with pytest.raises(KeyError, match="does not exist"):
        backend.save_with_consent("ghost", str(tmp_path / "x.nc"))
    # xarray requires .nc: refused before any cell/staging.
    with pytest.raises(ValueError, match="cannot serialise"):
        backend.save_with_consent("scan", str(tmp_path / "x.txt"))
    assert bridge.events == []


def test_figure_serialisation_via_gateway(tmp_path):
    """save_with_consent supports matplotlib Figures (Agg render to bytes):
    a figure variable can be persisted to an image target like any result."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from peaksMCP.overrides.save import _save_result

    figure = plt.figure(figsize=(2, 1))
    figure.add_subplot(111).plot([0, 1], [0, 1])
    save_module._set_approval_channel(lambda payload: True)
    target = tmp_path / "fig.png"
    receipt = _save_result(figure, str(target))
    plt.close("all")

    assert receipt.status == "saved"
    assert receipt.kind == "figure"
    assert target.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert receipt.structure["n_axes"] == 1


def test_run_cell_timeout_returns_structured_note(tmp_path):
    """run_cell 超时 ≠ 停止：返回结构化提示而不是让 agent 以为执行结束。"""
    class TimeoutBridge:
        connected = True

        def request(self, operation, payload=None, timeout=30.0):
            raise TimeoutError("no kernel reply")

    from peaksMCP.server.jupyter_peaks.backend import UnsafeNotebookBackend
    from peaksMCP.server.jupyter_peaks.security import AuditLogger, ConsentManager

    state = SharedState(_FakeIPython({}))
    state.bridge = TimeoutBridge()
    backend = UnsafeNotebookBackend(
        state, ConsentManager(TimeoutBridge()), AuditLogger(tmp_path / "a.log")
    )
    result = backend.execute_code("import time; time.sleep(999)", timeout=0.05)
    assert result["execution_timed_out"] is True
    assert "TIMEOUT IS NOT STOP" in result["note"]


def test_save_with_unreachable_approver_is_blocked_not_denied(tmp_path):
    """The channel exists but the frontend is gone: nobody could approve, so the
    receipt must say blocked - ``denied`` means a human actively refused, which
    would make the agent believe it was rejected on purpose."""
    bridge = _FakeBridge()
    backend = _backend(tmp_path, {"scan": _array()}, bridge)
    target = tmp_path / "out.nc"

    def unreachable(payload):
        bridge.events.append("consent-unreachable")
        return None

    save_module._set_approval_channel(unreachable)
    receipt = backend.save_with_consent("scan", str(target))

    assert receipt["status"] == "blocked", receipt
    assert "reachable" in receipt["note"]
    assert not target.exists()


def test_consent_request_reports_nobody_reachable_without_a_frontend():
    """``ConsentManager.request`` returns None (not False) when no approver can
    be reached, so callers can tell "no channel" from "human said no"."""
    from peaksMCP.server.jupyter_peaks.security import ConsentManager

    assert ConsentManager().request("save_ticket", {}) is None

    class _Disconnected:
        connected = False

    assert ConsentManager(_Disconnected()).request("save_ticket", {}) is None
    assert ConsentManager(callback=lambda *_: False).request("save_ticket", {}) is False
    assert ConsentManager(callback=lambda *_: None).request("save_ticket", {}) is None
