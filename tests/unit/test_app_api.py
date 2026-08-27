from __future__ import annotations

from collections import deque

from starlette.testclient import TestClient

from peaksMCP.app.api import create_app


class _FakeJupyter:
    def poll(self):  # process alive -> None
        return None


class _FakeProfile:
    class _MCP:
        host = "127.0.0.1"
        port = 8123

    class _Jupyter:
        kernel_name = "peaksmcp"

    class _Dashboard:
        host = "127.0.0.1"
        port = 8765

    mcp = _MCP()
    jupyter = _Jupyter()
    dashboard = _Dashboard()
    name = "default"


class _FakeSupervisor:
    def __init__(self) -> None:
        self.jupyter = _FakeJupyter()
        self.jupyter_url = "http://127.0.0.1:8888"
        self.dashboard_url = "http://127.0.0.1:8765"
        self.kernel_id = "k1"
        self.notebook_path = "peaksMCP-runtime.ipynb"
        self.token = "test-token"
        self.dashboard_token = "test-dashboard-token"
        self.profile = _FakeProfile()
        self.logs: deque = deque(
            [{"sequence": 1, "timestamp": 1, "component": "jupyter", "message": "booted"}],
            maxlen=2,
        )
        self.restart_calls: list[tuple[float, bool]] = []

    def status(self) -> dict:
        return {
            "status": "RUNNING", "pid": 42, "profile": "default", "kernel_id": "k1",
            "session_id": "s1", "notebook_path": "peaksMCP-runtime.ipynb",
            "jupyter_url": self.jupyter_url, "dashboard_url": self.dashboard_url,
            "notebook_url": f"{self.jupyter_url}/lab/tree/peaksMCP-runtime.ipynb",
            "mcp_url": "http://127.0.0.1:8123/mcp",
        }

    def restart_mcp(self, timeout: float = 45) -> dict:
        return {"ready": True}

    def restart_kernel(self, timeout: float = 90, require_comm: bool = False) -> dict:
        self.restart_calls.append((timeout, require_comm))
        return {"ready": True, "kernel_id": self.kernel_id}

    def start_mcp(self, timeout: float = 45) -> dict:
        return {"ready": True}

    def extension_status(self, timeout: float = 3) -> dict:
        return {"loaded": True, "detail": "IPython extension loaded"}


async def _online_mcp(*_a, **_k):
    return {"ok": True, "tool_count": 12, "status": {"comm_connected": True}}


async def _offline_mcp(*_a, **_k):
    return {"ok": False, "error": "MCP endpoint unreachable"}


async def _idle_kernel(_supervisor) -> str:
    return "idle"


class _FakeClient:
    def __init__(self, *args, **kwargs) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args) -> None:
        pass

    async def call_tool(self, name: str, arguments: dict):
        return type("Result", (), {"data": {"ok": True, "tool": name, "arguments": arguments}})()


def _authenticated_client(supervisor: _FakeSupervisor | None = None) -> tuple[TestClient, _FakeSupervisor]:
    active = supervisor or _FakeSupervisor()
    client = TestClient(create_app(active))
    assert client.get("/").status_code == 401
    assert client.get(f"/?token={active.dashboard_token}").status_code == 200
    return client, active


def test_dashboard_assets_and_status_online(monkeypatch):
    monkeypatch.setattr("peaksMCP.app.api._mcp_probe", _online_mcp)
    monkeypatch.setattr("peaksMCP.app.api._jupyter_kernel_state", _idle_kernel)
    client, _supervisor = _authenticated_client()
    assert client.get("/").status_code == 200
    assert "MCP Inspector" in client.get("/").text
    assert client.get("/assets/app.js").status_code == 200
    status = client.get("/api/status").json()
    assert status["supervisor_running"] is True
    assert status["aggregate"] == "ready"
    assert status["components"]["jupyter"]["state"] == "ready"
    assert status["components"]["kernel"]["state"] == "ready"
    assert status["components"]["mcp"]["state"] == "ready"
    assert status["components"]["comm"]["state"] == "ready"
    assert status["notebook_open_url"] == "/open-notebook"
    assert "token" not in status
    opened = client.get("/open-notebook", follow_redirects=False)
    assert opened.status_code == 303
    assert opened.headers["location"].endswith("token=test-token")
    assert "default" in client.get("/api/profiles").json()["profiles"]
    assert client.get("/api/logs").json()["logs"][0]["component"] == "jupyter"


def test_status_reports_mcp_offline(monkeypatch):
    monkeypatch.setattr("peaksMCP.app.api._mcp_probe", _offline_mcp)
    monkeypatch.setattr("peaksMCP.app.api._jupyter_kernel_state", _idle_kernel)
    client, _supervisor = _authenticated_client()
    status = client.get("/api/status").json()
    assert status["components"]["mcp"]["state"] == "error"
    assert status["aggregate"] == "error"
    assert status["components"]["extension"]["state"] == "ready"


def test_start_mcp_and_restart_delegate(monkeypatch):
    monkeypatch.setattr("peaksMCP.app.api._mcp_probe", _online_mcp)
    client, supervisor = _authenticated_client()
    assert client.post("/api/start-mcp").json()["ready"] is True
    assert client.post("/api/restart/mcp").json()["ready"] is True
    assert client.post("/api/restart/kernel").json()["ready"] is True
    assert client.post("/api/restart/all").json()["ready"] is True
    assert supervisor.restart_calls[-1] == (90, True)
    assert client.post("/api/restart/unknown").status_code == 400


def test_inspector_whitelist_and_call(monkeypatch):
    monkeypatch.setattr("peaksMCP.app.api._mcp_probe", _online_mcp)
    monkeypatch.setattr("peaksMCP.app.api.Client", lambda *a, **k: _FakeClient())
    client, _supervisor = _authenticated_client()
    blocked = client.post("/api/mcp/tool", json={"name": "notebook_execute_code", "arguments": {}})
    assert blocked.status_code == 403
    allowed = client.post("/api/mcp/tool", json={"name": "peaks_search_api", "arguments": {"query": "norm"}})
    assert allowed.status_code == 200
    assert allowed.json()["result"]["ok"] is True
    assert allowed.json()["result"]["tool"] == "peaks_search_api"


def test_api_rejects_missing_token_and_cross_origin_control():
    supervisor = _FakeSupervisor()
    anonymous = TestClient(create_app(supervisor))
    assert anonymous.get("/api/status").status_code == 401

    client, _active = _authenticated_client(supervisor)
    response = client.post(
        "/api/restart/mcp",
        headers={"Origin": "https://attacker.example"},
    )
    assert response.status_code == 403


def test_log_websocket_uses_sequence_after_deque_rotation():
    client, supervisor = _authenticated_client()
    with client.websocket_connect("/ws/logs") as socket:
        assert socket.receive_json()["logs"][0]["sequence"] == 1
        supervisor.logs.append(
            {"sequence": 2, "timestamp": 2, "component": "mcp", "message": "second"}
        )
        assert socket.receive_json()["logs"][-1]["sequence"] == 2
        supervisor.logs.append(
            {"sequence": 3, "timestamp": 3, "component": "mcp", "message": "third"}
        )
        assert socket.receive_json()["logs"][-1]["sequence"] == 3
        supervisor.logs.append(
            {"sequence": 4, "timestamp": 4, "component": "mcp", "message": "fourth"}
        )
        assert socket.receive_json()["logs"][-1]["sequence"] == 4
