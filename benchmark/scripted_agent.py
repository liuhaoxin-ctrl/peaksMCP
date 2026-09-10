#!/usr/bin/env python3
"""Deterministic generic agent for campaign acceptance runs.

Drives the five model-facing MCP tools over the trial's managed kernel and
performs the golden cut-preprocessing chain - no LLM involved, so the campaign
runner, the approval harness, the audit trail and the grader can be exercised
end to end in CI-adjacent environments.  Every scientific decision (which scan
is gold, which is the cut, which angular offset) is derived from
``inspect_experiment`` inside the notebook, exactly as an agent must.

Usage (the campaign runner fills the placeholders)::

    python benchmark/scripted_agent.py --trial {trial} --output {output}
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

FIT_GOLD = "dataarray:peaks.core.fitting.fit:fit_gold"
K_CONVERT = "dataarray:peaks.core.process.k_conversion:k_convert"
SET_EF = "metadata:peaks.core.metadata.metadata_methods:set_EF_correction"
FACADES = (
    "module:peaksMCP.overrides:load_data",
    "module:peaksMCP.overrides:inspect_experiment",
)


class Client:
    """One persistent MCP session (the API-proof ledger lives in the session)."""

    def __init__(self, url: str) -> None:
        import threading

        self.url = url
        self.loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._stack = None
        self._client = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if not self._ready.wait(90):
            raise RuntimeError("MCP session did not open")

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._open())
        self.loop.run_forever()

    async def _open(self) -> None:
        import contextlib

        from fastmcp import Client as FastMCPClient

        self._stack = contextlib.AsyncExitStack()
        self._client = await self._stack.enter_async_context(
            FastMCPClient(self.url, timeout=900)
        )
        self._ready.set()

    def call(self, name: str, arguments: dict) -> dict:
        future = asyncio.run_coroutine_threadsafe(
            self._client.call_tool(name, arguments), self.loop
        )
        result = future.result(timeout=600)
        data = getattr(result, "data", None)
        if isinstance(data, dict):
            return data
        return {"content": [str(item) for item in getattr(result, "content", [])]}


def _endpoint() -> str:
    """Resolve the managed kernel's MCP endpoint from the runfile."""
    home = Path(os.environ.get("PEAKSMCP_HOME", Path.home() / ".peaksMCP"))
    state = json.loads((home / "run.json").read_text(encoding="utf-8"))
    return f"http://127.0.0.1:{state['mcp_port']}/mcp"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--input", default="")
    parser.add_argument("--endpoint", default="")
    args = parser.parse_args()

    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dir = args.input or str(Path(args.trial) / "workspace" / "input")
    client = Client(args.endpoint or _endpoint())

    def cell(label: str, code: str, api_ids=None, timeout: float = 300.0) -> dict:
        result = client.call(
            "run_cell", {"code": code, "timeout": timeout, "api_ids": api_ids or []}
        )
        if result.get("blocked") or result.get("execution_success") is not True:
            raise SystemExit(f"{label} failed: {json.dumps(result, ensure_ascii=False)[:600]}")
        print(f"[scripted-agent] {label}: ok", flush=True)
        return result

    # 1. prove the contract ids this run relies on, then compose the chain
    for canonical_id in (FIT_GOLD, K_CONVERT, SET_EF, *FACADES):
        client.call("get", {"canonical_id": canonical_id})

    cell(
        "classify",
        "from peaksMCP.overrides import load_data, inspect_experiment\n"
        f"scans = load_data({input_dir!r})\n"
        "summary = inspect_experiment(scans)\n"
        "assert summary.gold, 'no gold reference classified'\n"
        "present = {int(s[-4:]) for s in scans.stems}\n"
        "gold_index = next(i for i in summary.gold if i in present)\n"
        "gold_stem = f'BP_{gold_index:04d}'\n"
        "cut_stems = [f'BP_{i:04d}' for i in summary.cuts if i in present]\n"
        "assert cut_stems, 'no cut in this index'\n"
        "cut_index = int(cut_stems[0][-4:])\n"
        "theta_offset = next(r.theta_offset_deg for r in summary.records if r.index == cut_index)\n"
        "assert theta_offset, 'angular offset missing from the metadata'",
    )
    cell(
        "inventory",
        "import json\n"
        "print(json.dumps({'gold': gold_stem, 'cuts': cut_stems, 'theta': theta_offset}))",
    )
    # The printed inventory is ARCHIVED, not echoed: run_cell's stdout_head is a
    # deliberately short "there was long output" signal (first 80 chars).  Read
    # the cell back through the designed path - inspect_notebook with
    # with_text_outputs - exactly as the model has to.
    archive = client.call(
        "inspect_notebook",
        {"target": "active_cell", "detail": "preview", "with_text_outputs": True},
    )
    text = ""
    if isinstance(archive, dict):
        text = str(archive.get("text_outputs") or "")
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise SystemExit(
            "inventory not recoverable from the notebook archive: "
            + json.dumps(archive, ensure_ascii=False)[:600]
        )
    inventory = json.loads(text[start : end + 1])
    cut_stems = list(inventory["cuts"])
    print(f"[scripted-agent] processing {len(cut_stems)} cut(s): {cut_stems[:4]}...", flush=True)

    # One gold fit, reused for every cut (the contract the grader checks).
    cell(
        "fit gold once",
        "gold = scans[gold_stem]\n"
        "fit = gold.fit_gold(plot=False, show=False)\n"
        "ef = dict(fit.attrs['EF_correction'])\n"
        "assert 'c0' in ef, ef",
        api_ids=[FIT_GOLD],
    )

    for cut_stem in cut_stems:
        cell(
            f"level+zero {cut_stem}",
            f"cut = scans[{cut_stem!r}]\n"
            # A record declared sweep may carry the scanned deflector axis
            # (L112: (eV, theta_par, deflector_perp)).  The human product is the
            # CENTRE PLANE along it; integrating would mix different k_y
            # regions.  Attributes must survive the selection.
            "extra = [d for d in cut.dims if d not in ('eV', 'theta_par')]\n"
            "if extra:\n"
            "    cut = cut.isel({extra[0]: cut.sizes[extra[0]] // 2})\n"
            "cut.metadata.set_EF_correction(ef)\n"
            "shifted = cut.assign_coords(theta_par=cut.theta_par - theta_offset)",
            api_ids=[SET_EF],
        )
        cell(
            f"k-space {cut_stem}",
            "kcut = shifted.k_convert(quiet=True)\n"
            "assert kcut.dims == ('eV', 'kx'), kcut.dims\n"
            "assert float(kcut.eV.min()) <= 0.0 <= float(kcut.eV.max())\n"
            "assert abs(float(kcut.kx.min()) + float(kcut.kx.max())) <= 0.05",
            api_ids=[K_CONVERT],
        )
        # Consent-gated persistence: the benchmark harness answers each card.
        target = output_dir / f"{cut_stem}_processed.nc"
        receipt = client.call(
            "save_with_consent", {"variable_name": "kcut", "path": str(target)}
        )
        if receipt.get("status") != "saved":
            raise SystemExit(f"save {cut_stem} did not complete: {receipt}")
        print(f"[scripted-agent] saved {target.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
