"""One-command live pi trial: fresh stack -> run pi -> verify figures -> restore.

All operational lessons are encoded here instead of a manual:

- every trial gets a FRESH notebook + FRESH kernel (a warm notebook makes the
  agent skip the workflow and produce no figures),
- the notebook file is created when missing (the host refuses to start
  otherwise) with the required kernelspec (name "peaksmcp"),
- the pre-trial host (profile/root/notebook) is captured before switching and
  restored afterwards -- including failure paths -- via cli._ensure_host
  (no browser popup, same flow as benchmark/run_campaign.py),
- a host that came up broken (STOPPED) is stopped and retried once,
- pi runs non-interactively (--mode json --print) from the repo root so the
  root .mcp.json exposes the peaksMCP MCP server; provider/model/thinking and
  the prompt have the locked defaults below,
- credential-store failures from pi are detected and reported as actionable
  errors instead of silent empty runs,
- figures are verified deterministically (no vision model needed): the trial
  notebook must contain the gold diagnostic plus grid and before/after cells
  that CALL peaks.plot_cut_grid / peaks.plot_before_after, render at the
  locked sizes (13.6x12.0 / 11.0x4.0 in) on the viridis colormap with zero
  text overlaps and zero text clipped outside the canvas,
- the acceptance loop is built in: `run` repeats --repetitions (default 3)
  times and stops at the first failure, so "three consecutive clean runs"
  is the exit criterion.

Usage:
    python tools/trial.py status
    python tools/trial.py run [--repetitions 3] [--name trial1] [--prompt ...]
    python tools/trial.py verify <notebook.ipynb> [--strict]
    python tools/trial.py restore
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
STATE = Path(
    os.environ.get("PEAKSMCP_HOME", Path.home() / ".peaksMCP")
) / "trial_state.json"
DEFAULT_PROMPT = (
    "use tools, help me preprocess all 2d data in "
    "/Users/haoxin/Documents/实验数据/BP260623"
)
DEFAULT_STEMS = [
    f"BP_{i:04d}" for i in (5, 6, 9, 10, 11, 12, 15, 16, 19, 21, 22, 25, 26, 27)
]
LOCKED_METRICS = {
    "grid": {"size_in": (13.6, 12.0), "aspect": 1.133, "png_aspect": 1.143, "cmap": "viridis"},
    "before_after": {"size_in": (11.0, 4.0), "aspect": 2.75, "png_aspect": 2.746, "cmap": "viridis"},
}
SIZE_TOL = 0.1
ASPECT_TOL = 0.03
PNG_ASPECT_TOL = 0.03
CMAP_MATCH_MIN = 0.95

NB_TEMPLATE = {
    "cells": [{"cell_type": "markdown", "metadata": {}, "source": ["# {notebook}\n"]}],
    "metadata": {
        "kernelspec": {
            "display_name": "Python (peaksMCP)",
            "language": "python",
            "name": "peaksmcp",
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


class TrialError(RuntimeError):
    """A trial failure with an actionable, pre-digested message."""


# --------------------------------------------------------------------------
# environment guards
# --------------------------------------------------------------------------


def _ensure_numba_cache_env() -> None:
    """numba cache=True needs a writable cache dir; point it into the workspace
    when the home cache locations are not writable (sandboxed shells)."""
    if os.environ.get("NUMBA_CACHE_DIR"):
        return
    for candidate in (
        Path.home() / ".cache" / "numba",
        Path.home() / "Library" / "Caches" / "numba",
    ):
        if candidate.exists() and os.access(candidate, os.W_OK):
            return
        if not candidate.exists() and os.access(candidate.parent, os.W_OK):
            return
    fallback = REPO_ROOT / ".trial_numba_cache"
    fallback.mkdir(parents=True, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str(fallback)


def _require_pi() -> str:
    pi = shutil.which("pi")
    if not pi:
        raise TrialError("`pi` executable not found on PATH; install it first")
    return pi


# --------------------------------------------------------------------------
# managed stack switching
# --------------------------------------------------------------------------


def _ns(profile: str, root_dir: str, notebook: str, timeout: float) -> argparse.Namespace:
    return argparse.Namespace(
        profile=profile, root_dir=root_dir, notebook=notebook, timeout=timeout
    )


def _read_runfile() -> dict | None:
    from peaksMCP.observability import read_runfile

    return read_runfile()


def _capture() -> dict:
    cur = _read_runfile()
    return {
        "profile": (cur or {}).get("profile") or "default",
        "root_dir": (cur or {}).get("root_dir"),
        "notebook": (cur or {}).get("notebook_path"),
    }


def _save_capture(captured: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(captured, indent=2), encoding="utf-8")


def _load_capture() -> dict:
    if not STATE.exists():
        raise TrialError(
            f"no pre-trial host capture at {STATE}; run `trial.py run` or `fresh` first"
        )
    return json.loads(STATE.read_text(encoding="utf-8"))


def _wait_ready(timeout: float) -> dict:
    state = _read_runfile()
    if not state or state.get("stale"):
        raise TrialError("no live runfile after host start")
    headers = {"Authorization": f"Bearer {state['dashboard_token']}"}
    deadline = time.monotonic() + timeout
    last: dict = {}
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(
                f"{state['dashboard_url']}/api/status", headers=headers, timeout=5
            )
            resp.raise_for_status()
            last = resp.json()
            comps = last.get("components") or {}
            needed = ("jupyter", "kernel", "extension", "mcp")
            if all((comps.get(n) or {}).get("state") == "ready" for n in needed):
                print(
                    "stack ready:", {n: comps[n]["state"] for n in needed},
                    "kernel_id:", last.get("kernel_id"),
                )
                return last
            last_error = last.get("last_error")
            if last.get("jupyter_state") == "stopped" and last_error:
                raise TrialError(f"host came up STOPPED: {last_error}")
        except TrialError:
            raise
        except Exception as exc:  # noqa: BLE001 - host still starting
            last["_err"] = type(exc).__name__
        time.sleep(1.0)
    raise TrialError(f"stack not ready in {timeout}s: {json.dumps(last)[:400]}")


def _ensure_fresh_stack(root_dir: Path, notebook: str, timeout: float) -> None:
    """Capture the pre-trial host, create the fresh notebook if missing, and
    switch the managed host to it. Retries once across a stopped-broken host."""
    from peaksMCP import cli

    if not STATE.exists():
        _save_capture(_capture())
    root_dir.mkdir(parents=True, exist_ok=True)
    nb = root_dir / notebook
    if not nb.exists():
        payload = json.loads(json.dumps(NB_TEMPLATE).replace("{notebook}", notebook))
        nb.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        print("created fresh notebook:", nb)
    cli._ensure_host(_ns("default", str(root_dir), notebook, timeout))
    try:
        _wait_ready(timeout)
    except TrialError as exc:
        # A half-dead host matches its own runfile so _ensure_host will not
        # respawn it: stop the whole tree once and retry on a clean slate.
        print(f"first attempt failed ({exc}); stopping host and retrying once")
        subprocess.run([sys.executable, "-m", "peaksMCP", "stop"], check=False)
        time.sleep(2)
        cli._ensure_host(_ns("default", str(root_dir), notebook, timeout))
        _wait_ready(timeout)


def restore_stack(timeout: float) -> None:
    """Bring the captured pre-trial host back (safe to call repeatedly)."""
    from peaksMCP import cli

    captured = _load_capture()
    if not captured.get("root_dir"):
        raise TrialError("captured host has no root_dir; nothing to restore")
    print("restoring pre-trial host:", captured)
    cli._ensure_host(
        _ns(captured["profile"], captured["root_dir"], captured["notebook"], timeout)
    )
    _wait_ready(timeout)


# --------------------------------------------------------------------------
# pi execution
# --------------------------------------------------------------------------


def run_pi(run_name: str, prompt: str, args: argparse.Namespace, workdir: Path) -> dict:
    pi = _require_pi()
    session_dir = Path(args.sessions_dir).resolve()
    session_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = workdir / f"{run_name}.stdout.jsonl"
    stderr_path = workdir / f"{run_name}.stderr.log"
    command = [
        pi,
        "--mode", "json",
        "--print",
        "--provider", args.provider,
        "--model", args.model,
        "--thinking", args.thinking,
        "--session-dir", str(session_dir),
        "--name", run_name,
        prompt,
    ]
    print("running:", " ".join(command[:8]) + " ...")
    started = time.monotonic()
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open(
        "w", encoding="utf-8"
    ) as err:
        process = subprocess.run(
            command, cwd=str(REPO_ROOT), stdout=out, stderr=err, timeout=args.agent_timeout
        )
    elapsed = time.monotonic() - started
    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
    if "Credential store read failed" in stderr_text:
        raise TrialError(
            "pi could not read its credential store (~/.pi): it needs write access to "
            "~/.pi even to read credentials; run this from a normal user shell"
        )
    errors = _scan_pi_errors(stdout_path)
    return {
        "exit_code": process.returncode,
        "elapsed_seconds": round(elapsed, 1),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "error_results": errors,
    }


def _scan_pi_errors(stdout_path: Path) -> list[str]:
    """Count toolResults whose content mentions error/blocked/failed."""
    hits = []
    for line in stdout_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "message":
            continue
        for part in (event.get("message") or {}).get("content") or []:
            if part.get("type") != "toolResult":
                continue
            content = part.get("content")
            text = content if isinstance(content, str) else json.dumps(content)
            low = text.lower()
            if any(k in low for k in ("error", "blocked", "failed", "rejected")):
                hits.append(text[:160].replace("\n", " "))
    return hits


# --------------------------------------------------------------------------
# deterministic figure verification
# --------------------------------------------------------------------------


def _load_plotting_env(stems: list[str]) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402
    import numpy as np  # noqa: E402
    import peaks  # noqa: E402
    import xarray as xr  # noqa: E402

    rng = np.random.default_rng(0)

    def dummy_cut(kdim: str = "kx") -> xr.DataArray:
        return xr.DataArray(
            rng.random((300, 200)),
            dims=("eV", kdim),
            coords={"eV": np.linspace(-0.5, 0.3, 300), kdim: np.linspace(-1.2, 1.2, 200)},
        )

    def dummy_raw() -> xr.DataArray:
        return xr.DataArray(
            rng.random((300, 200)),
            dims=("eV", "theta_par"),
            coords={
                "eV": np.linspace(-0.5, 0.3, 300),
                "theta_par": np.linspace(-15, 15, 200),
            },
        )

    return {
        "plt": plt,
        "np": np,
        "peaks": peaks,
        "result_dict": {s: dummy_cut() for s in stems},
        "theta_by_stem": {s: 1.5 for s in stems},
        "representative_raw": dummy_raw(),
        "representative_kcut": dummy_cut(),
        "rep_stem": stems[0],
        "rep_index": 12,
        "gold_fit": type("GoldFit", (), {"attrs": {"EF_poly4": 2.6608}})(),
    }


def _render_cell(src: str, kind: str, env: dict) -> dict:
    """Re-render a plotting cell on synthetic data; measure text geometry."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    plt = env["plt"]
    lines = src.rstrip().splitlines()
    while lines and re.fullmatch(r"\s*(plt\.close\(fig\d*\)|[A-Za-z_]\w*)\s*", lines[-1]):
        lines.pop()
    exec("\n".join(lines), env)  # noqa: S102 - synthetic data only
    fig = None
    for value in env.values():
        if isinstance(value, plt.Figure):
            fig = value
            break
    if fig is None:
        raise TrialError(f"no figure object produced by {kind} cell")
    FigureCanvasAgg(fig)  # helpers close their own figure, detaching its canvas
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fbb = fig.bbox
    artists = []
    for ax in fig.axes:
        for t in [*ax.texts, ax.title, ax.xaxis.label, ax.yaxis.label]:
            if t is None or not t.get_text() or not t.get_visible():
                continue
            try:
                bb = t.get_window_extent(renderer=renderer)
            except Exception:  # noqa: BLE001
                continue
            artists.append((t.get_text()[:28], bb))
        for axis_name in ("xaxis", "yaxis"):
            axis = getattr(ax, axis_name)
            lo, hi = axis.get_view_interval()
            for lbl, tick in zip(
                axis.get_ticklabels(), axis.get_majorticklocs(), strict=False
            ):
                if not lbl.get_text() or not lbl.get_visible():
                    continue
                if tick < lo or tick > hi:
                    continue  # out-of-view tick labels are never painted
                try:
                    bb = lbl.get_window_extent(renderer=renderer)
                except Exception:  # noqa: BLE001
                    continue
                artists.append((lbl.get_text()[:28], bb))
    overlaps: list = []
    clipped: list[str] = []
    for i, a in enumerate(artists):
        if not fbb.contains(a[1].x0, a[1].y0) or not fbb.contains(a[1].x1, a[1].y1):
            clipped.append(a[0])
        for j in range(i + 1, len(artists)):
            b = artists[j]
            if a[1].overlaps(b[1]):
                inter = a[1].intersection(a[1], b[1])
                area = inter.width * inter.height if inter is not None else 0
                if area > 0:
                    overlaps.append((a[0], b[0], round(area, 0)))
    w, h = fig.get_size_inches()
    plt.close(fig)
    return {
        "kind": kind,
        "size_in": [round(float(w), 2), round(float(h), 2)],
        "aspect": round(float(w) / float(h), 3),
        "overlaps": overlaps,
        "clipped_text": clipped,
    }


def _png_cmap_match(png_bytes: bytes, cmap_name: str) -> float:
    """Fraction of saturated (data) pixels whose color sits on the colormap LUT.

    White background, black text and gray axes are excluded so the metric
    measures the intensity map itself.
    """
    import io

    import numpy as np
    from PIL import Image

    img = np.asarray(Image.open(io.BytesIO(png_bytes)).convert("RGB"), dtype=np.int16)
    flat = img.reshape(-1, 3)[::11]
    saturated = flat[flat.max(axis=1) - flat.min(axis=1) > 20]
    if len(saturated) == 0:
        return 0.0
    from matplotlib import colormaps

    lut = (np.asarray(colormaps[cmap_name](np.linspace(0, 1, 256)))[:, :3] * 255).astype(
        np.int16
    )
    hit = 0
    for start in range(0, len(saturated), 4000):
        chunk = saturated[start : start + 4000]
        dist = np.abs(chunk[:, None, :] - lut[None, :, :]).sum(axis=2)
        hit += int((dist.min(axis=1) <= 12).sum())
    return hit / len(saturated)


def verify_notebook(notebook_path: Path, first_cell: int, strict: bool = True) -> dict:
    """Run the locked-format checks against a trial notebook. Raises TrialError
    on any violation when strict; otherwise returns the collected report."""
    if not notebook_path.exists():
        raise TrialError(f"notebook not found: {notebook_path}")
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    report = {"figures": [], "problems": [], "cells": len(nb["cells"])}
    used_grid = used_ba = False
    env: dict | None = None
    for i, cell in enumerate(nb["cells"]):
        if i < first_cell:
            continue
        src = "".join(cell.get("source", []))
        used_grid = used_grid or "plot_cut_grid(" in src
        used_ba = used_ba or "plot_before_after(" in src
        if any(o.get("output_type") == "error" for o in cell.get("outputs", [])):
            report["problems"].append(f"cell {i} has an error output")
        for out in cell.get("outputs", []):
            if "image/png" not in out.get("data", {}):
                continue
            png = out["data"]["image/png"]
            if isinstance(png, list):
                png = "".join(png)
            raw = base64.b64decode(png)
            w, h = struct.unpack(">II", raw[16:24])
            entry = {"cell": i, "px": [w, h], "aspect": round(w / h, 3)}
            if "plot_cut_grid(" in src:
                entry["figure"] = "grid"
            elif "plot_before_after(" in src:
                entry["figure"] = "before_after"
            else:
                entry["figure"] = "gold_or_other"
            if entry["figure"] in LOCKED_METRICS:
                entry["cmap_match"] = round(
                    _png_cmap_match(raw, LOCKED_METRICS[entry["figure"]]["cmap"]), 3
                )
                env = env or _load_plotting_env(DEFAULT_STEMS)
                try:
                    entry["render"] = _render_cell(src, entry["figure"], dict(env))
                except Exception as exc:  # noqa: BLE001
                    entry["render"] = {"error": f"{type(exc).__name__}: {exc}"}
            report["figures"].append(entry)
    report["helpers"] = {"plot_cut_grid": used_grid, "plot_before_after": used_ba}

    gold = [f for f in report["figures"] if f["figure"] == "gold_or_other"]
    if len(gold) != 1:
        report["problems"].append(f"expected exactly 1 gold diagnostic figure, got {len(gold)}")
    for name, metrics in LOCKED_METRICS.items():
        figs = [f for f in report["figures"] if f["figure"] == name]
        if len(figs) != 1:
            report["problems"].append(f"expected exactly 1 {name} figure, got {len(figs)}")
            continue
        fig = figs[0]
        if abs(fig["aspect"] - metrics["png_aspect"]) > PNG_ASPECT_TOL:
            report["problems"].append(
                f"{name} PNG aspect {fig['aspect']} outside "
                f"{metrics['png_aspect']} +/- {PNG_ASPECT_TOL}"
            )
        if fig.get("cmap_match", 1.0) < CMAP_MATCH_MIN:
            report["problems"].append(
                f"{name} colors do not match {metrics['cmap']} "
                f"(match={fig.get('cmap_match')})"
            )
        render = fig.get("render") or {}
        if "error" in render:
            report["problems"].append(f"{name} render failed: {render['error']}")
            continue
        size = render.get("size_in") or [0, 0]
        if (
            abs(size[0] - metrics["size_in"][0]) > SIZE_TOL
            or abs(size[1] - metrics["size_in"][1]) > SIZE_TOL
        ):
            report["problems"].append(f"{name} size {size} != {metrics['size_in']} +/- {SIZE_TOL}")
        if render.get("overlaps"):
            report["problems"].append(f"{name} has text overlaps: {render['overlaps']}")
        if render.get("clipped_text"):
            report["problems"].append(f"{name} has clipped text: {render['clipped_text']}")
    if strict:
        if not used_grid:
            report["problems"].append("no cell calls peaks.plot_cut_grid")
        if not used_ba:
            report["problems"].append("no cell calls peaks.plot_before_after")
    return report


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def cmd_status(_args: argparse.Namespace) -> int:
    print(json.dumps(_read_runfile() or {"status": "no runfile"}, indent=2, default=str))
    if STATE.exists():
        print("capture:", STATE.read_text(encoding="utf-8").strip())
    return 0


def cmd_restore(args: argparse.Namespace) -> int:
    restore_stack(args.timeout)
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    _ensure_numba_cache_env()
    report = verify_notebook(Path(args.notebook), args.first_cell, strict=args.strict)
    print(json.dumps(report, indent=2, default=str, ensure_ascii=False))
    if args.strict and report["problems"]:
        print("VERIFY FAIL:", report["problems"])
        return 1
    print("VERIFY OK" if args.strict else "report only")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    _ensure_numba_cache_env()
    if args.repetitions < 1:
        raise TrialError("--repetitions must be >= 1")
    results = []
    base = Path(args.workdir)
    try:
        for rep in range(1, args.repetitions + 1):
            run_name = f"{args.name}-r{rep}"
            workdir = base / args.name / f"r{rep}"
            workdir.mkdir(parents=True, exist_ok=True)
            print(f"\n=== trial {run_name} (repetition {rep}/{args.repetitions}) ===")
            _ensure_fresh_stack(workdir, "plotfmt.ipynb", args.timeout)
            metadata = run_pi(run_name, args.prompt, args, workdir)
            print(
                f"pi exit {metadata['exit_code']} in {metadata['elapsed_seconds']}s; "
                f"error-ish toolResults: {len(metadata['error_results'])}"
            )
            for hit in metadata["error_results"][:3]:
                print("  ERR:", hit)
            if metadata["exit_code"] != 0:
                raise TrialError(f"pi exited {metadata['exit_code']} for {run_name}")
            if metadata["error_results"]:
                raise TrialError(f"{run_name} has {len(metadata['error_results'])} error toolResults")
            report = verify_notebook(workdir / "plotfmt.ipynb", 1, strict=True)
            print(json.dumps(
                {"figures": report["figures"], "helpers": report["helpers"]},
                indent=2, default=str, ensure_ascii=False,
            ))
            if report["problems"]:
                results.append({"run": run_name, "ok": False, "problems": report["problems"]})
                raise TrialError(f"{run_name} verification failed: {report['problems']}")
            results.append({"run": run_name, "ok": True})
            print(f"{run_name}: PASS")
        print("\nall repetitions passed:", json.dumps(results, ensure_ascii=False))
        return 0
    except TrialError as exc:
        print(f"\nTRIAL FAILED: {exc}")
        return 1
    except Exception as exc:  # noqa: BLE001 - surface the real error, never mask it
        print(f"\nTRIAL CRASHED: {type(exc).__name__}: {exc}")
        return 1
    finally:
        if not args.keep_stack:
            try:
                print("\nrestoring pre-trial host...")
                restore_stack(args.timeout)
            except TrialError as exc:
                print(f"restore failed (recover with `tools/trial.py restore`): {exc}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="fresh stack -> pi -> verify -> restore, N times")
    r.add_argument("--name", default="trial")
    r.add_argument("--repetitions", type=int, default=3)
    r.add_argument("--prompt", default=DEFAULT_PROMPT)
    r.add_argument("--provider", default="deepseek")
    r.add_argument("--model", default="deepseek-v4-flash")
    r.add_argument("--thinking", default="low")
    r.add_argument("--workdir", default=str(REPO_ROOT / ".pi_sessions" / "trials"))
    r.add_argument("--sessions-dir", default=str(REPO_ROOT / ".pi_sessions"))
    r.add_argument("--timeout", type=float, default=300.0)
    r.add_argument("--agent-timeout", type=float, default=1800.0)
    r.add_argument("--keep-stack", action="store_true", help="skip the final restore")
    r.set_defaults(func=cmd_run)

    v = sub.add_parser("verify", help="check one notebook against the locked format")
    v.add_argument("notebook")
    v.add_argument("first_cell", type=int, nargs="?", default=1)
    v.add_argument("--strict", action="store_true", default=True)
    v.set_defaults(func=cmd_verify)

    s = sub.add_parser("restore", help="restore the captured pre-trial host")
    s.add_argument("--timeout", type=float, default=300.0)
    s.set_defaults(func=cmd_restore)

    st = sub.add_parser("status", help="show live runfile and restore capture")
    st.set_defaults(func=cmd_status)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except TrialError as exc:
        print(f"error: {exc}")
        sys.exit(1)
