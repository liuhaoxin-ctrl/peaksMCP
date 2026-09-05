from __future__ import annotations

import asyncio

from starlette.testclient import TestClient

from peaksMCP.app.api import create_app


def test_load_bridge_code_is_valid_multiline_python():
    """The kernel bridge code generated for load-into-notebook must compile as
    exec (a single-line ``def _do(): try:`` chain is invalid Python and used to
    be silently swallowed)."""
    from peaksMCP.app.api import load_into_notebook

    captured: dict[str, list[str]] = {"codes": []}

    class FakeSupervisor:
        def execute_kernel(self, code, timeout=10):
            captured["codes"].append(code)
            return {"status": "ok"}

    result = load_into_notebook(FakeSupervisor(), "/data/BP_0001.nc")
    assert result is True
    bridge_code = captured["codes"][0]
    compile(bridge_code, "<load-into-notebook>", "exec")  # must be valid Python
    assert "def run():" in bridge_code
    assert "from peaks import load" in bridge_code
    assert "data = load(" in bridge_code
    assert "execution_success" in bridge_code
    assert 'result.get("saved") is not True' in bridge_code
    assert "'data' not in" not in "\n".join(captured["codes"])
    for code in captured["codes"]:
        compile(code, "<load-control>", "exec")


def test_load_embeds_experiment_metadata_when_present(tmp_path):
    from peaksMCP.app.api import load_into_notebook

    nc = tmp_path / "BP_0001.nc"
    nc.write_text("dummy")
    (tmp_path / "experiment_metadata.json").write_text("{}", encoding="utf-8")

    captured: dict[str, list[str]] = {"codes": []}

    class FakeSupervisor:
        def execute_kernel(self, code, timeout=10):
            captured["codes"].append(code)
            return {"status": "ok"}

    assert load_into_notebook(FakeSupervisor(), [str(nc)], timeout=2) is True
    bridge_code = captured["codes"][0]
    compile(bridge_code, "<load-into-notebook>", "exec")
    assert "import json" in bridge_code
    assert "metadata = json.load(open(" in bridge_code
    assert str(nc.parent / "experiment_metadata.json") in bridge_code


def test_load_skips_metadata_when_absent(tmp_path):
    from peaksMCP.app.api import load_into_notebook

    nc = tmp_path / "BP_0001.nc"
    nc.write_text("dummy")

    captured: dict[str, list[str]] = {"codes": []}

    class FakeSupervisor:
        def execute_kernel(self, code, timeout=10):
            captured["codes"].append(code)
            return {"status": "ok"}

    assert load_into_notebook(FakeSupervisor(), [str(nc)], timeout=2) is True
    bridge_code = captured["codes"][0]
    assert "metadata = json.load" not in bridge_code


def test_auto_load_skips_without_output():
    from peaksMCP.app.api import load_into_notebook

    class _Never:
        def execute_kernel(self, *_a, **_k):
            raise AssertionError("execute_kernel must not be called without an output")

    assert load_into_notebook(_Never(), None) is False


def test_load_notebook_accepts_paths_list(monkeypatch):
    calls: list[object] = []

    def fake_load(_supervisor, nc_path, **_kwargs):
        calls.append(nc_path)
        # A single visible cell loads all paths together (all-or-nothing):
        # one failing file fails the whole load.
        paths = nc_path if isinstance(nc_path, list) else [nc_path]
        return not any("bad" in str(p) for p in paths)

    monkeypatch.setattr("peaksMCP.app.api.load_into_notebook", fake_load)
    client, _supervisor = _authenticated_client()
    resp = client.post(
        "/api/notebook/load",
        json={"paths": ["/data/a.nc", "/data/bad.nc"]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["paths"] == ["/data/a.nc", "/data/bad.nc"]
    # All paths share the single-cell outcome.
    assert body["results"] == {"/data/a.nc": False, "/data/bad.nc": False}
    assert body["loaded"] is False
    assert calls == [["/data/a.nc", "/data/bad.nc"]]


def test_load_notebook_single_path_legacy(monkeypatch):
    monkeypatch.setattr(
        "peaksMCP.app.api.load_into_notebook", lambda _s, nc_path, **_k: True
    )
    client, _supervisor = _authenticated_client()
    resp = client.post("/api/notebook/load", json={"path": "/data/a.nc"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["loaded"] is True
    assert body["results"] == {"/data/a.nc": True}


def test_load_notebook_requires_path(monkeypatch):
    client, _supervisor = _authenticated_client()
    assert client.post("/api/notebook/load", json={}).status_code == 400
    assert client.post("/api/notebook/load", json={"paths": []}).status_code == 400
    assert client.post("/api/notebook/load", json={"paths": "/data/a.nc"}).status_code == 400
    assert client.post("/api/notebook/load", json={"paths": [1]}).status_code == 400


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
    def __init__(self) -> None:
        self.jupyter = _FakeJupyter()
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
    # Execution tools are NOT in the inspector whitelist: rejected.
    blocked = client.post(
        "/api/mcp/tool",
        json={"name": "notebook_write_with_api_check", "arguments": {"code": "1+1"}},
    )
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


# =====================
# Conversion
# =====================
