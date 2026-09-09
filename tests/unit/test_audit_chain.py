"""Audit chain: one operation_id spans wrapper + internal audit events."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_audit_write_injects_active_operation_id(tmp_path):
    from peaksMCP.server.jupyter_peaks.security import AuditLogger
    from peaksMCP.server.jupyter_peaks.security.audit import operation_context

    path = tmp_path / "audit.log"
    logger = AuditLogger(path)
    token = operation_context.set("op-chain-1")
    try:
        logger.write("scanner", "blocked", {"reason": "x"})
        logger.write("consent", "approved", {"ticket_id": "t1"})
    finally:
        operation_context.reset(token)
    logger.write("outside", "ok", {})
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    inside = [e for e in events if e["details"].get("operation_id") == "op-chain-1"]
    assert len(inside) == 2
    outside = [e for e in events if e["tool"] == "outside"][0]
    assert "operation_id" not in outside["details"]


def test_run_cell_chain_events_share_one_operation_id(tmp_path):
    """一次被 run_cell 硬阻止的写盘尝试：called→blocked(内部)→error 同 op id。"""
    import asyncio

    from fastmcp import Client
    from fastmcp.exceptions import ToolError

    from peaksMCP.server.jupyter_peaks.backend import SharedState
    from peaksMCP.server.jupyter_peaks.mcp_server import JupyterPeaksMCPServer

    class FakeIPython:
        user_ns = {}

    class Bridge:
        connected = True

        def request(self, operation, payload=None, timeout=30.0):
            raise AssertionError(f"不应执行：{operation}")

    state = SharedState(FakeIPython())
    state.bridge = Bridge()
    server = JupyterPeaksMCPServer(state)

    async def _call():
        async with Client(server.mcp) as client:
            with pytest.raises(ToolError):
                await client.call_tool(
                    "run_cell", {"code": "import matplotlib.pyplot as plt\nplt.savefig('x.png')"}
                )

    asyncio.run(_call())

    import os

    home = os.environ["PEAKSMCP_HOME"]
    candidate = Path(home) / "audit" / "tool_audit.log"  # conftest 隔离根下的默认路径
    events = []
    for line in candidate.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("tool") == "run_cell":
            events.append(event)
    outcomes = {event["outcome"] for event in events}
    op_ids = {str((event.get("details") or {}).get("operation_id")) for event in events}
    assert {"called", "blocked", "error"} <= outcomes, events
    assert len(op_ids) == 1 and "" not in op_ids, op_ids
