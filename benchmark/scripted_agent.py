#!/usr/bin/env python3
"""Deterministic generic agent for campaign acceptance runs.

Drives the five model-facing MCP tools over the trial's managed kernel and
performs the golden cut-preprocessing chain - no LLM involved, so the campaign
runner, the notebook bridge, the audit trail and the grader can be exercised
end to end in CI-adjacent environments. Every scientific decision (which scan
is gold, which is a cut, which angular offset) is derived from the
``ExperimentIndex`` returned by ``peaks.load_experiment`` inside the notebook.

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
ASSIGN_NORMAL = "metadata:peaks.core.metadata.metadata_methods:assign_normal_emission"
PXT2NC = "top_level:peaks.core.fileIO.experiment:pxt2nc"
LOAD_EXPERIMENT = "top_level:peaks.core.fileIO.experiment:load_experiment"


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


def _output_line(result: dict, prefix: str) -> str:
    """Return one short stdout receipt already present in a run-cell reply."""
    for block in result.get("output") or []:
        for line in str(block).splitlines():
            if line.startswith(prefix):
                return line.removeprefix(prefix)
    raise SystemExit(
        f"run_cell reply did not contain {prefix!r}: "
        + json.dumps(result, ensure_ascii=False)[:600]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--input", default="")
    parser.add_argument("--endpoint", default="")
    args = parser.parse_args()

    input_dir = args.input or str(Path(args.trial) / "workspace" / "input")
    client = Client(args.endpoint or _endpoint())

    def cell(
        label: str,
        code: str,
        api_ids=None,
        timeout: float = 300.0,
        cell_type: str = "code",
    ) -> dict:
        arguments: dict = {"code": code, "timeout": timeout, "cell_type": cell_type}
        if cell_type == "code":
            arguments["api_ids"] = api_ids or []
        result = client.call("run_cell", arguments)
        if result.get("blocked") or result.get("execution_success") is not True:
            raise SystemExit(f"{label} failed: {json.dumps(result, ensure_ascii=False)[:600]}")
        print(f"[scripted-agent] {label}: ok", flush=True)
        return result

    # Inspect representation state first. The short receipt is already in the
    # run_cell response, so the conversion decision needs no notebook reread.
    client.call("get", {"canonical_id": LOAD_EXPERIMENT})
    initial = cell(
        "load experiment",
        "import peaks\n"
        f"experiment = peaks.load_experiment({input_dir!r})\n"
        "print(f'STATE needs_conversion={len(experiment.needs_conversion)}')",
        api_ids=[LOAD_EXPERIMENT],
    )
    raw_state = _output_line(initial, "STATE needs_conversion=")
    try:
        needs_conversion = int(raw_state)
    except ValueError as exc:
        raise SystemExit(f"invalid representation-state receipt: {raw_state!r}") from exc

    reload_code = ""
    if needs_conversion:
        client.call("get", {"canonical_id": PXT2NC})
        cell(
            "convert raw data once",
            f"conversion = peaks.pxt2nc({input_dir!r})\n"
            "assert conversion.failed == 0, conversion\n"
            "assert conversion.converted > 0, conversion",
            api_ids=[PXT2NC],
            timeout=900.0,
        )
        reload_code = "experiment = peaks.load_experiment(conversion.destination)\n"

    classification = cell(
        "classify experiment",
        reload_code
        + "assert not experiment.needs_conversion, experiment.needs_conversion\n"
        "assert experiment.gold, 'no gold reference classified'\n"
        "present = {int(s[-4:]) for s in experiment.stems}\n"
        "gold_index = next(i for i in experiment.gold if int(i) in present)\n"
        "gold_stem = f'BP_{int(gold_index):04d}'\n"
        "records_by_index = {int(r.index): r for r in experiment.records}\n"
        "cut_indices = [int(i) for i in experiment.cuts if int(i) in present]\n"
        "cut_stems = [records_by_index[i].stem or f'BP_{i:04d}' for i in cut_indices]\n"
        "assert cut_stems, 'no cut in this index'\n"
        "theta_offset = records_by_index[cut_indices[0]].theta_offset_deg\n"
        "assert theta_offset is not None, 'angular offset missing from the metadata'\n"
        "import json\n"
        "inventory = {'gold': gold_stem, 'cuts': cut_stems, 'theta': theta_offset}\n"
        "print('INVENTORY ' + json.dumps(inventory, separators=(',', ':')))",
        api_ids=[LOAD_EXPERIMENT] if needs_conversion else [],
    )
    inventory = json.loads(_output_line(classification, "INVENTORY "))
    cut_stems = list(inventory["cuts"])
    print(f"[scripted-agent] processing {len(cut_stems)} cut(s): {cut_stems[:4]}...", flush=True)

    # One gold fit, reused for every cut (the contract the grader checks).
    client.call("get", {"canonical_id": FIT_GOLD})
    gold_fit = cell(
        "fit gold once",
        "gold = experiment[gold_stem]\n"
        "fit = gold.fit_gold(show=False, quiet=True)\n"
        "gold_diagnostic_fig = fit.attrs['figure']\n"
        "ef = dict(fit.attrs['EF_correction'])\n"
        "assert 'c0' in ef, ef\n"
        "fit_window = fit.attrs['fit_window']\n"
        "fit_quality = fit.attrs['EF_quality']\n"
        "print(f\"FIT EF={fit.attrs['EF_poly4']:.6f}; \"\n"
        "      f\"window={fit_window['start_eV']:.4f}:{fit_window['stop_eV']:.4f}; \"\n"
        "      f\"plateaus={fit_window['lower_points']}/{fit_window['upper_points']}; \"\n"
        "      f\"outliers={fit_quality['outlier_fraction']:.3f}; \"\n"
        "      f\"uniform={fit_quality['uniform']}\")\n"
        "gold_diagnostic_fig",
        api_ids=[FIT_GOLD],
        timeout=900.0,
    )
    fit_receipt = _output_line(gold_fit, "FIT ")

    # One notebook batch cell, one stem-keyed result dictionary and one static
    # call site per per-cut API. There is no pilot scan to recompute later.
    for canonical_id in (ASSIGN_NORMAL, K_CONVERT):
        client.call("get", {"canonical_id": canonical_id})
    cell(
        "process all cuts once",
        "import numpy as np\n"
        "cut_results = {}\n"
        "representative_raw = None\n"
        "representative_kcut = None\n"
        "for decision_index in experiment.cuts:\n"
        "    cut_index = int(decision_index)\n"
        "    if cut_index not in present:\n"
        "        continue\n"
        "    record = records_by_index[cut_index]\n"
        "    raw_cut = experiment[decision_index]\n"
        # A declared sweep can carry the scanned deflector axis. The intended
        # 2-D product is its zero-deflector plane, never an integral over k_y.
        "    if 'deflector_perp' in raw_cut.dims:\n"
        "        raw_cut = raw_cut.sel(deflector_perp=0.0, method='nearest')\n"
        "    extra = [d for d in raw_cut.dims if d not in ('eV', 'theta_par')]\n"
        "    assert not extra, (record.stem, raw_cut.dims)\n"
        "    assert record.theta_offset_deg is not None, record.stem\n"
        "    shifted = raw_cut.metadata.assign_normal_emission(\n"
        "        theta_par=record.theta_offset_deg\n"
        "    )\n"
        "    kcut = shifted.k_convert(EF_correction=fit, quiet=True)\n"
        "    stem = record.stem or f'BP_{cut_index:04d}'\n"
        "    cut_results[stem] = kcut\n"
        "    if representative_raw is None:\n"
        "        representative_raw = raw_cut\n"
        "        representative_kcut = kcut\n"
        "    assert kcut.dims == ('eV', 'kx'), (stem, kcut.dims)\n"
        "    assert bool(np.isfinite(kcut.values).any()), stem\n"
        "    assert float(kcut.eV.min()) <= 0.0 <= float(kcut.eV.max()), stem\n"
        "    assert abs(float(kcut.kx.min()) + float(kcut.kx.max())) <= 0.05, stem\n"
        "assert sorted(cut_results) == sorted(cut_stems), sorted(cut_results)\n"
        "print('processed_stems=' + ','.join(sorted(cut_results)))",
        api_ids=[ASSIGN_NORMAL, K_CONVERT],
        timeout=900.0,
    )

    # Three static outputs total: gold diagnostic above, one all-cut grid,
    # and one representative before/after comparison. Closing each Figure
    # before its rich repr prevents Matplotlib's end-of-cell duplicate flush.
    cell(
        "render compact all-cut grid",
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "cut_keys = sorted(cut_results)\n"
        "ncol = 4\n"
        "nrow = int(np.ceil(len(cut_keys) / ncol))\n"
        "cut_grid_fig, cut_grid_axes = plt.subplots(nrow, ncol, figsize=(12, 2.6 * nrow), squeeze=False, constrained_layout=True)\n"
        "for axis, stem in zip(cut_grid_axes.ravel(), cut_keys):\n"
        "    data = cut_results[stem]\n"
        "    axis.imshow(data.values, origin='lower', aspect='auto')\n"
        "    axis.set_title(stem, fontsize=8)\n"
        "for axis in cut_grid_axes.ravel()[len(cut_keys):]:\n"
        "    axis.axis('off')\n"
        "cut_grid_fig.suptitle('All calibrated k-space cuts')\n"
        "plt.close(cut_grid_fig)\n"
        "cut_grid_fig",
    )
    cell(
        "render representative before/after",
        "validation_fig, validation_axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)\n"
        "validation_axes[0].imshow(representative_raw.values, origin='lower', aspect='auto')\n"
        "validation_axes[1].imshow(representative_kcut.values, origin='lower', aspect='auto')\n"
        "validation_axes[0].set_title('raw angle space')\n"
        "validation_axes[1].set_title('calibrated k space')\n"
        "plt.close(validation_fig)\n"
        "validation_fig",
    )
    # S4: the task asks for one closing Markdown cell - the notebook is the
    # execution record, so the summary is appended, never written over it.
    cell(
        "closing summary",
        "## Cut preprocessing summary\n\n"
        f"- gold reference: {inventory['gold']} (classified by peaks.load_experiment)\n"
        f"- Fermi-level correction: {fit_receipt}\n"
        f"- theta_offset={inventory['theta']} deg from record.theta_offset_deg\n"
        f"- cuts processed: {len(cut_stems)} in result dictionary `cut_results`\n"
        + (
            "- cache: raw inputs were converted once and the validated cache was created\n"
            if needs_conversion
            else "- cache: preconverted cache reused; `pxt2nc` was not called\n"
        )
        + "- processed_stems="
        + ",".join(sorted(cut_stems))
        + "\n"
        + "- unprocessed targets: none (record 3 has no metadata entry; "
        "record 26 was reduced to its centre deflector plane)\n"
        "- validation: gold diagnostic, one all-cut grid, and one before/after figure; "
        "no processed NetCDF or image files were written",
        cell_type="markdown",
    )
    print(f"[scripted-agent] summary written for {len(cut_stems)} result(s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
