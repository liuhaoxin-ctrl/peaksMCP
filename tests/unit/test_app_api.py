from __future__ import annotations

import asyncio

import pytest
from starlette.testclient import TestClient

from peaksMCP.app.api import create_app


def test_frontend_save_must_be_confirmed():
    from peaksMCP.app.api import _flush_frontend_save

    class Supervisor:
        def execute_kernel(self, code, timeout=10):
            assert "r.get('saved') is True" in code
            return {"status": "error"}

    import pytest

    with pytest.raises(RuntimeError, match="Notebook save failed"):
        _flush_frontend_save(Supervisor())


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
    def __init__(self, jupyter_state: str = "running", jupyter_alive: bool = True) -> None:
        self.jupyter = _FakeJupyter() if jupyter_alive else None
        self.jupyter_state = jupyter_state
        self.jupyter_url = "http://127.0.0.1:8888"
        self.dashboard_url = "http://127.0.0.1:8765"
        self.kernel_id = "k1"
        self.notebook_path = "peaksMCP-runtime.ipynb"
        self.token = "test-token"
        self.dashboard_token = "test-dashboard-token"
        self.profile = _FakeProfile()
        self.restart_calls: list[tuple[float, bool]] = []

    def status(self) -> dict:
        return {
            "status": "RUNNING", "pid": 42, "profile": "default", "kernel_id": "k1",
            "session_id": "s1", "notebook_path": "peaksMCP-runtime.ipynb",
            "jupyter_url": self.jupyter_url, "dashboard_url": self.dashboard_url,
            "notebook_url": f"{self.jupyter_url}/lab/tree/peaksMCP-runtime.ipynb",
            "mcp_url": "http://127.0.0.1:8123/mcp",
            "jupyter_state": self.jupyter_state,
        }

    def restart_mcp(self, timeout: float = 45) -> dict:
        return {"ready": True}

    def restart_kernel(self, timeout: float = 90, require_comm: bool = False) -> dict:
        self.restart_calls.append((timeout, require_comm))
        return {"ready": True, "kernel_id": self.kernel_id}

    def start_mcp(self, timeout: float = 45) -> dict:
        return {"ready": True}

    def export_variable(self, name: str, value: object) -> None:
        self.exported = {name: value}

    def extension_status(self, timeout: float = 3) -> dict:
        return {"loaded": True, "detail": "IPython extension loaded"}

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"token {self.token}"}


def test_snapshot_uses_async_contents_api_and_unique_non_overwriting_names(monkeypatch):
    from peaksMCP.app.api import _create_notebook_snapshot

    class Response:
        def __init__(self, status_code: int, payload: dict | None = None) -> None:
            self.status_code = status_code
            self._payload = payload or {}

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

        def json(self) -> dict:
            return self._payload

    class ContentsAPI:
        def __init__(self) -> None:
            self.created: set[str] = set()
            self.put_payloads: list[dict] = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args) -> None:
            return None

        async def get(self, url: str, **_kwargs):
            if url.endswith("peaksMCP-runtime.ipynb"):
                return Response(
                    200,
                    {
                        "name": "peaksMCP-runtime.ipynb",
                        "path": "peaksMCP-runtime.ipynb",
                        "type": "notebook",
                        "format": "json",
                        "content": {"cells": []},
                    },
                )
            return Response(200 if url in self.created else 404)

        async def put(self, url: str, *, json: dict, **_kwargs):
            assert url not in self.created
            self.created.add(url)
            self.put_payloads.append(json)
            return Response(201)

    api = ContentsAPI()
    monkeypatch.setattr("peaksMCP.app.api._flush_frontend_save", lambda _s: None)
    monkeypatch.setattr("peaksMCP.app.api.httpx.AsyncClient", lambda **_k: api)
    supervisor = _FakeSupervisor()

    first = asyncio.run(_create_notebook_snapshot(supervisor))
    second = asyncio.run(_create_notebook_snapshot(supervisor))

    assert first != second
    assert first.startswith("peaksMCP-snapshot-") and first.endswith(".ipynb")
    assert len(api.created) == 2
    assert api.put_payloads == [
        {"type": "notebook", "format": "json", "content": {"cells": []}},
        {"type": "notebook", "format": "json", "content": {"cells": []}},
    ]


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
    assert "default-src 'self'" in client.get("/").headers["content-security-policy"]
    # Content assertions are intentionally structural: the operator-console
    # copy is evolving (dashboard redesign) and must not pin the test to it.
    assert '<script src="/assets/app.js"></script>' in client.get("/").text
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


def test_status_reports_mcp_offline(monkeypatch):
    monkeypatch.setattr("peaksMCP.app.api._mcp_probe", _offline_mcp)
    monkeypatch.setattr("peaksMCP.app.api._jupyter_kernel_state", _idle_kernel)
    client, _supervisor = _authenticated_client()
    status = client.get("/api/status").json()
    assert status["components"]["mcp"]["state"] == "error"
    assert status["aggregate"] == "error"
    assert status["components"]["extension"]["state"] == "ready"


@pytest.mark.parametrize(
    "jupyter_state,jupyter_alive,expected_ui,expected_top,open_url",
    [
        ("starting", True, "starting", "STARTING", False),
        ("starting", False, "starting", "STARTING", False),
        ("stopped", False, "stopped", "STOPPED", False),
        ("stopped", True, "stopped", "STOPPED", False),
        # A process that died after being declared running is an error, never
        # a silent "ready".
        ("running", False, "error", "STOPPED", False),
    ],
)
def test_status_jupyter_readiness_from_declared_state(
    monkeypatch, jupyter_state, jupyter_alive, expected_ui, expected_top, open_url
):
    """Readiness reflects the declared group state, not just process existence:
    "starting" is a real state (never an error) and the managed-notebook open
    URL is only exposed once Jupyter is truly ready."""
    monkeypatch.setattr("peaksMCP.app.api._mcp_probe", _online_mcp)
    monkeypatch.setattr("peaksMCP.app.api._jupyter_kernel_state", _idle_kernel)
    supervisor = _FakeSupervisor(jupyter_state=jupyter_state, jupyter_alive=jupyter_alive)
    client, _supervisor = _authenticated_client(supervisor)
    status = client.get("/api/status").json()
    assert status["components"]["jupyter"]["state"] == expected_ui
    assert status["status"] == expected_top
    assert ("notebook_open_url" in status) is open_url
    if expected_ui == "ready":
        assert status["aggregate"] == "ready"
    elif expected_ui == "error":
        assert status["aggregate"] == "error"
    else:
        assert status["aggregate"] == "degraded"


def test_start_mcp_and_restart_delegate(monkeypatch):
    monkeypatch.setattr("peaksMCP.app.api._mcp_probe", _online_mcp)
    client, supervisor = _authenticated_client()
    assert client.post("/api/start-mcp").json()["ready"] is True
    assert client.post("/api/restart/mcp").json()["ready"] is True
    assert client.post("/api/restart/kernel").json()["ready"] is True
    assert client.post("/api/restart/all").json()["ready"] is True
    assert supervisor.restart_calls[-1] == (90, True)
    assert client.post("/api/restart/unknown").status_code == 400


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


# =====================
# Conversion
# =====================


# --------------------------------------------------------------------------- #
# Recent-activity audit chains (Phase 6c)                                     #
# --------------------------------------------------------------------------- #

def test_recent_audit_chains_groups_by_operation_id(tmp_path):
    from peaksMCP.app.api import recent_audit_chains

    audit = tmp_path / "tool_audit.log"
    audit.write_text(
        "\n".join([
            '{"timestamp": "2026-01-01T00:00:01", "tool": "run_cell", "outcome": "called",'
            ' "details": {"operation_id": "op-1", "args": {"code": "scans = load_data(..."}}}',
            '{"timestamp": "2026-01-01T00:00:02", "tool": "run_cell", "outcome": "executed",'
            ' "details": {"operation_id": "op-1", "cell_id": "c3"}}',
            '{"timestamp": "2026-01-01T00:00:05", "tool": "save_with_consent", "outcome": "called",'
            ' "details": {"operation_id": "op-2", "args": {"path": "/out/BP_0005_processed.nc"}}}',
            '{"timestamp": "2026-01-01T00:00:06", "tool": "save_with_consent", "outcome": "saved",'
            ' "details": {"operation_id": "op-2", "ticket_id": "t-abc", "sha256": "deadbeef1234"}}',
            '{"timestamp": "2026-01-01T00:00:03", "tool": "run_cell", "outcome": "executed",'
            ' "details": {"cell_id": "old"}}',
            "not-json-line",
        ]) + "\n",
        encoding="utf-8",
    )
    payload = recent_audit_chains(audit, max_chains=10)
    by_id = {chain["operation_id"]: chain for chain in payload["chains"]}
    # 按 last_at 倒序：op-2 (06) 在前，op-1 (02) 在后。
    assert [chain["operation_id"] for chain in payload["chains"]] == ["op-2", "op-1"]
    chain = by_id["op-2"]
    assert chain["ticket_id"] == "t-abc" and chain["sha256"] == "deadbeef1234"
    assert chain["target_path"] == "/out/BP_0005_processed.nc"
    assert chain["tools"] == ["save_with_consent"] and chain["outcomes"] == ["called", "saved"]
    chain1 = by_id["op-1"]
    assert chain1["cell_ids"] == ["c3"]
    assert any(event.get("code_head", "").startswith("scans = load_data") for event in chain1["events"])
    # 无 operation_id 的旧事件进 unattached；坏行被跳过。
    assert len(payload["unattached"]) == 1
    assert payload["unattached"][0]["tool"] == "run_cell"


def test_activity_endpoint_requires_auth_and_returns_chains(tmp_path, monkeypatch):
    from peaksMCP.app.api import create_app

    monkeypatch.setenv("PEAKSMCP_HOME", str(tmp_path))
    audit = tmp_path / "audit" / "tool_audit.log"
    audit.parent.mkdir()
    audit.write_text(
        '{"timestamp": "2026-01-01T00:00:01", "tool": "run_cell", "outcome": "called",'
        ' "details": {"operation_id": "op-1", "args": {"code": "x=1"}}}\n',
        encoding="utf-8",
    )
    supervisor = _FakeSupervisor()
    app = create_app(supervisor)
    with TestClient(app) as client:
        assert client.get("/api/activity/recent").status_code == 401
        response = client.get(
            "/api/activity/recent",
            headers={"Authorization": f"Bearer {supervisor.dashboard_token}"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["chains"] and body["chains"][0]["operation_id"] == "op-1"
        assert body["source"].endswith("tool_audit.log")
