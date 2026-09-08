"""Connection-identity regressions without a real Jupyter kernel or browser."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from peaksMCP.server.jupyter_peaks.active_cell_bridge import CommBridge
from peaksMCP.server.jupyter_peaks.backend import SharedState


class _Comm:
    def __init__(self, on_send=None):
        self.sent = []
        self._closed = False
        self.on_send = on_send

    def on_msg(self, callback):
        self._message_callback = callback

    def on_close(self, callback):
        self._close_callback = callback

    def send(self, data):
        self.sent.append(data)
        if self.on_send is not None and data.get("type") == "request":
            self.on_send(data)

    def emit(self, data):
        self._message_callback({"content": {"data": data}})

    def close(self):
        self._closed = True
        self._close_callback({})


def _open(bridge, comm, cell="current"):
    bridge._on_open(comm, {"content": {"data": {
        "type": "notebook_state", "cell": {"id": cell},
        "outputs": [{"output_type": "stream", "text": cell}],
    }}})


@pytest.fixture
def bridge():
    return CommBridge(SharedState(SimpleNamespace(user_ns={})))


@pytest.mark.parametrize("notification", ["frontend_closing", "transport_close"])
def test_old_close_does_not_disconnect_new_comm(bridge, notification):
    old, current = _Comm(), _Comm()
    _open(bridge, old, "old")
    _open(bridge, current, "new")
    last_seen = bridge.last_seen
    if notification == "frontend_closing":
        old.emit({"type": "frontend_closing"})
    else:
        old.close()
    assert bridge.comm is current
    assert bridge.connected
    assert bridge.last_seen == last_seen
    assert bridge.state.active_cell == {"id": "new"}


@pytest.mark.parametrize("message_type", ["active_cell", "notebook_state", "heartbeat"])
def test_stale_messages_cannot_update_state_or_heartbeat(bridge, message_type):
    old, current = _Comm(), _Comm()
    _open(bridge, old, "old")
    _open(bridge, current, "new")
    bridge.last_seen = 0
    old.emit({"type": message_type, "cell": {"id": "stale"}, "outputs": []})
    assert bridge.state.active_cell == {"id": "new"}
    assert bridge.state.active_cell_output[0]["text"] == "new"
    assert bridge.last_seen == 0
    assert not bridge.connected
    current.emit({"type": "heartbeat"})
    assert bridge.connected


def test_delayed_execution_output_does_not_replace_the_active_cell(bridge):
    comm = _Comm()
    _open(bridge, comm, "active")
    comm.emit({
        "type": "active_cell",
        "cell": {"id": "active", "source": "current = 1"},
        "outputs": [{"output_type": "stream", "text": "current"}],
    })

    comm.emit({
        "type": "cell_output",
        "cell_id": "executed",
        "cell": {"id": "executed", "source": "data.plot()"},
        "outputs": [{"output_type": "display_data", "data": {"image/png": "AA=="}}],
    })

    assert bridge.state.active_cell["id"] == "active"
    assert bridge.state.active_cell_output[0]["text"] == "current"
    assert bridge.state.cell_outputs["executed"][0]["output_type"] == "display_data"


def test_stale_reply_cannot_complete_new_request(bridge):
    old = _Comm()
    _open(bridge, old)

    def respond(request):
        old.emit({"request_id": request["request_id"], "ok": True, "result": {"value": "stale"}})
        old.emit({"type": "frontend_closing"})
        old.close()
        assert not bridge._pending[request["request_id"]][0].is_set()
        current.emit({"request_id": request["request_id"], "ok": True, "result": {"value": "current"}})

    current = _Comm(respond)
    _open(bridge, current)
    assert bridge.request("read_active_cell", timeout=0.1) == {"value": "current"}
    assert bridge.connected
    assert bridge._pending == {}


def test_replacement_fails_old_request_and_new_connection_recovers(bridge):
    current = _Comm()

    def replace(request):
        _open(bridge, current)
        old.emit({"request_id": request["request_id"], "ok": True, "result": {"stale": True}})

    old = _Comm(replace)
    _open(bridge, old)
    with pytest.raises(RuntimeError, match="connection was replaced"):
        bridge.request("read_active_cell", timeout=0.1)
    assert bridge.comm is current
    assert bridge._pending == {}
    current.on_send = lambda request: current.emit({
        "request_id": request["request_id"], "ok": True, "result": {"recovered": True},
    })
    assert bridge.request("read_active_cell", timeout=0.1) == {"recovered": True}


@pytest.mark.parametrize("notification", ["frontend_closing", "transport_close"])
def test_current_close_fails_pending_request_immediately(bridge, notification):
    def close(_request):
        if notification == "frontend_closing":
            comm.emit({"type": "frontend_closing"})
        else:
            comm.close()

    comm = _Comm(close)
    _open(bridge, comm)
    with pytest.raises(RuntimeError, match="connection was closed"):
        bridge.request("read_active_cell", timeout=0.1)
    assert bridge.comm is None
    assert bridge.last_seen is None
    assert bridge._pending == {}
    assert not bridge.connected


@pytest.mark.parametrize("change", ["replacement", "frontend_closing", "transport_close"])
def test_connection_change_wakes_a_waiting_caller(bridge, change):
    sent = threading.Event()
    comm = _Comm(lambda _request: sent.set())
    _open(bridge, comm)
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(bridge.request, "read_active_cell", timeout=3)
        assert sent.wait(1)
        if change == "replacement":
            _open(bridge, _Comm())
        elif change == "frontend_closing":
            comm.emit({"type": "frontend_closing"})
        else:
            comm.close()
        with pytest.raises(RuntimeError, match="operation outcome is unknown"):
            result.result(timeout=1)
    assert bridge._pending == {}


def test_duplicate_reply_and_close_do_not_overwrite_completed_result(bridge):
    def respond(request):
        comm.emit({"request_id": request["request_id"], "ok": True, "result": {"value": 42}})
        comm.emit({"request_id": request["request_id"], "ok": False, "error": "late reply"})
        comm.close()

    comm = _Comm(respond)
    _open(bridge, comm)
    assert bridge.request("read_active_cell", timeout=0.1) == {"value": 42}
    assert bridge._pending == {}


def test_timeout_and_send_failure_clean_up_pending_requests(bridge):
    comm = _Comm()
    _open(bridge, comm)
    with pytest.raises(TimeoutError):
        bridge.request("read_active_cell", timeout=0.001)
    assert bridge._pending == {}
    comm.emit({"request_id": comm.sent[-1]["request_id"], "ok": True, "result": {}})
    assert bridge._pending == {}

    def fail(_request):
        raise OSError("send failed")

    comm.on_send = fail
    with pytest.raises(OSError, match="send failed"):
        bridge.request("read_active_cell", timeout=0.1)
    assert bridge._pending == {}


def test_concurrent_requests_receive_their_own_out_of_order_replies(bridge):
    ready = threading.Event()
    requests = []

    def record(request):
        requests.append(request)
        if len(requests) == 2:
            ready.set()

    comm = _Comm(record)
    _open(bridge, comm)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(bridge.request, "first", timeout=3)
        second = pool.submit(bridge.request, "second", timeout=3)
        assert ready.wait(2)
        for request in reversed(requests):
            comm.emit({
                "request_id": request["request_id"], "ok": True,
                "result": {"operation": request["operation"]},
            })
        assert first.result(timeout=1) == {"operation": "first"}
        assert second.result(timeout=1) == {"operation": "second"}
    assert bridge._pending == {}
