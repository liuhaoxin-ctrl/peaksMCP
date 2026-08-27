from __future__ import annotations

from starlette.testclient import TestClient

from peaksMCP.app.api import create_app
from peaksMCP.app.profiles import load_profile


class FakeProcess:
    def poll(self):
        return None


class FakeSupervisor:
    def __init__(self):
        self.profile = load_profile()
        self.jupyter = FakeProcess()
        self.kernel_id = "missing-test-kernel"
        self.logs = []
        self.token = "test-token"
        self.jupyter_url = "http://127.0.0.1:1"

    def _headers(self):
        return {"Authorization": "token test-token"}

    def status(self):
        return {"status": "RUNNING", "pid": 1, "profile": "default", "notebook_url": "http://127.0.0.1:1/lab/tree/test.ipynb", "mcp_url": "http://127.0.0.1:2/mcp"}


def test_dashboard_assets_profiles_and_status(monkeypatch):
    async def fake_ping(*_args, **_kwargs):
        return {"ok": True, "tool_count": 12, "status": {"comm_connected": True}}

    monkeypatch.setattr("peaksMCP.app.api.check_http_mcp_server", fake_ping)
    client = TestClient(create_app(FakeSupervisor()))
    assert client.get("/").status_code == 200
    assert "MCP Inspector" in client.get("/").text
    assert client.get("/assets/app.js").status_code == 200
    status = client.get("/api/status").json()
    assert status["components"]["mcp"]["state"] == "ready"
    assert status["components"]["comm"]["state"] == "ready"
    assert status["notebook_open_url"].endswith("token=test-token")
    assert "default" in client.get("/api/profiles").json()["profiles"]
