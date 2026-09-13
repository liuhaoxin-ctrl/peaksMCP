#!/usr/bin/env python3
"""Create, execute, grade, aggregate, and compare agent benchmark campaigns."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import random
import shlex
import shutil
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

import httpx

try:
    from benchmark import run_case as rc
except ModuleNotFoundError:  # direct ``python benchmark/run_campaign.py`` invocation
    import run_case as rc  # type: ignore[no-redef]

CAMPAIGN_SCHEMA_VERSION = 1
DEFAULT_CAMPAIGNS = rc.BENCH_DIR / "campaigns"
PI_ALLOWED_TOOLS = ["mcp", "mcpScript"]
LIVE_EVIDENCE_MARKER = "__PEAKSMCP_LIVE_EVIDENCE__="


def now() -> str:
    return datetime.now(UTC).isoformat()


def campaign_manifest_path(campaign_dir: Path) -> Path:
    return campaign_dir / "campaign.json"


def load_campaign(path: str | Path) -> tuple[Path, dict[str, Any]]:
    candidate = Path(path).expanduser().resolve()
    campaign_dir = candidate if candidate.is_dir() else candidate.parent
    manifest_path = candidate if candidate.is_file() else campaign_manifest_path(candidate)
    if not manifest_path.is_file():
        raise SystemExit(f"campaign manifest not found: {manifest_path}")
    return campaign_dir, json.loads(manifest_path.read_text(encoding="utf-8"))


def save_campaign(campaign_dir: Path, payload: dict[str, Any]) -> None:
    rc.atomic_write_json(campaign_manifest_path(campaign_dir), payload)


def capture_managed_host() -> dict[str, Any] | None:
    """Capture the user-visible host workspace before campaign replacement."""
    from peaksMCP.observability import read_runfile

    state = read_runfile()
    if not state or state.get("stale"):
        return None
    return {
        "profile": state.get("profile") or "default",
        "root_dir": state.get("root_dir"),
        "notebook": state.get("notebook_path"),
    }


def restore_managed_host(
    previous: dict[str, Any] | None,
    *,
    timeout: float,
) -> dict[str, Any]:
    """Restore the pre-campaign host, or stop the temporary host if absent."""
    from peaksMCP import cli

    current = cli._runfile(False)
    if previous is None:
        if current and not current.get("stale"):
            cli._terminate_supervisor(int(current["pid"]))
        return {"status": "stopped_temporary_host"}
    restored = cli._ensure_host(
        argparse.Namespace(
            profile=previous["profile"],
            root_dir=previous.get("root_dir"),
            notebook=previous.get("notebook"),
            timeout=timeout,
        )
    )
    return {
        "status": "restored",
        "profile": restored.get("profile"),
        "root_dir": restored.get("root_dir"),
        "notebook": restored.get("notebook_path"),
    }


def _condition_list(raw: str) -> list[str]:
    conditions = [item.strip().lower() for item in raw.split(",") if item.strip()]
    unknown = sorted(set(conditions) - set(rc.CONDITION_PROMPT_FILES))
    if unknown:
        raise SystemExit(f"unknown conditions: {unknown}")
    if len(conditions) != len(set(conditions)):
        raise SystemExit("conditions must not contain duplicates")
    return conditions


def create_campaign(args: argparse.Namespace) -> int:
    campaign_id = args.name or datetime.now().strftime("%Y%m%d-%H%M%S")
    campaign_dir = Path(args.campaigns).expanduser().resolve() / campaign_id
    if campaign_dir.exists() and not args.force:
        raise SystemExit(f"campaign already exists: {campaign_dir}")
    if campaign_dir.exists():
        shutil.rmtree(campaign_dir)
    trials_dir = campaign_dir / "trials"
    trials_dir.mkdir(parents=True)

    conditions = _condition_list(args.conditions)
    rng = random.Random(args.seed)
    trials: list[dict[str, Any]] = []
    for replicate in range(1, args.repetitions + 1):
        order = list(conditions)
        rng.shuffle(order)
        for position, condition in enumerate(order, start=1):
            run_id = f"r{replicate:03d}-{condition}"
            init_args = argparse.Namespace(
                name=run_id,
                case=args.case,
                condition=condition,
                runs=str(trials_dir),
                data=args.data,
                datasheet=args.datasheet,
                reference=args.reference,
                limit=args.limit,
                copy=args.copy,
                force=False,
                campaign=campaign_id,
                replicate=replicate,
                agent_id=args.agent_id,
                provider=args.provider,
                model=args.model,
                thinking=args.thinking,
                approval_mode=args.approval_mode,
                isolation_enforced=False,
                isolation_mechanism="",
                allowed_tools=[],
            )
            trial_dir, _key = rc.initialize_trial(init_args)
            trials.append(
                {
                    "run_id": run_id,
                    "replicate": replicate,
                    "condition": condition,
                    "order": position,
                    "path": str(trial_dir),
                    "status": "initialized",
                }
            )

    campaign = {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "created_at": now(),
        "objective": args.objective,
        "case": args.case,
        "conditions": conditions,
        "repetitions": args.repetitions,
        "seed": args.seed,
        "approval_mode": args.approval_mode,
        "agent": {
            "id": args.agent_id,
            "provider": args.provider,
            "model": args.model,
            "thinking": args.thinking,
        },
        "freeze": {
            "experiment_sha256": rc.sha256_file(rc.EXPERIMENT_FILE),
            "rubric_sha256": rc.sha256_file(rc.RUBRIC_FILE),
            "common_prompt_sha256": rc.sha256_file(rc.COMMON_PROMPT_FILE),
            "condition_prompt_sha256": {
                name: rc.sha256_file(path) for name, path in rc.CONDITION_PROMPT_FILES.items()
            },
        },
        "trials": trials,
    }
    save_campaign(campaign_dir, campaign)
    print(f"Campaign created: {campaign_dir}")
    print(f"Trials: {len(trials)} ({args.repetitions} paired repetitions)")
    return 0


def _run_state_for_notebook(notebook: Path, profile: str, timeout: float) -> dict[str, Any]:
    """Start the trial through the same ``peaksMCP dash`` command a human uses."""
    from peaksMCP.observability import read_runfile

    environment = {**os.environ, "BROWSER": "true"}
    launched = subprocess.run(
        [
            sys.executable,
            "-m",
            "peaksMCP",
            "dash",
            notebook.name,
            "--root-dir",
            str(notebook.parent.resolve()),
            "--profile",
            profile,
            "--timeout",
            str(timeout),
        ],
        cwd=rc.ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=timeout + 15,
    )
    if launched.returncode != 0:
        raise RuntimeError(
            f"peaksMCP dash failed ({launched.returncode}): "
            f"{(launched.stdout + launched.stderr)[-2000:]}"
        )
    state = read_runfile()
    if not state or state.get("stale"):
        raise RuntimeError("peaksMCP dash returned without a live runfile")
    headers = {"Authorization": f"Bearer {state['dashboard_token']}"}
    deadline = time.monotonic() + timeout
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        try:
            response = httpx.get(f"{state['dashboard_url']}/api/status", headers=headers, timeout=5)
            response.raise_for_status()
            last = response.json()
            components = last.get("components") or {}
            required = ("jupyter", "kernel", "extension", "mcp")
            ready = all((components.get(name) or {}).get("state") == "ready" for name in required)
            # Readiness alone is not enough: the trial's fresh-kernel claim needs
            # the live kernel identity, so a stack that never names its kernel
            # must fail here instead of being graded with empty evidence.
            kernel_id = last.get("kernel_id")
            if ready and kernel_id:
                # The host reports notebook_path RELATIVE to its Jupyter root
                # ("work.ipynb"); store the absolute path so grading never has
                # to guess a base directory (resolving it against the benchmark
                # CWD rejected correctly isolated trials).
                host_root = Path(state.get("root_dir") or notebook.parent).expanduser().resolve()
                reported = Path(str(last.get("notebook_path") or notebook.name))
                live = reported if reported.is_absolute() else host_root / reported
                return {
                    **state,
                    "root_dir": str(host_root),
                    "kernel_id": str(kernel_id),
                    "status_payload": last,
                    "live_notebook": str(live.resolve()),
                }
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.5)
    raise RuntimeError(
        "peaksMCP stack did not become ready with an identified kernel "
        f"(kernel_id={last.get('kernel_id')!r}): {last}"
    )


def _dashboard_url(run_state: dict[str, Any]) -> str:
    return f"{run_state['dashboard_url']}/?token={run_state['dashboard_token']}"


def _wait_for_file(path: Path, process: subprocess.Popen[Any], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file():
            return
        if process.poll() is not None:
            raise RuntimeError(f"approval harness exited with code {process.returncode}")
        time.sleep(0.2)
    raise TimeoutError(f"timed out waiting for {path}")


def _wait_for_comm(run_state: dict[str, Any], timeout: float) -> None:
    headers = {"Authorization": f"Bearer {run_state['dashboard_token']}"}
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        payload = httpx.get(
            f"{run_state['dashboard_url']}/api/status", headers=headers, timeout=5
        ).json()
        if ((payload.get("components") or {}).get("comm") or {}).get("state") == "ready":
            return
        time.sleep(0.5)
    raise TimeoutError("JupyterLab Comm bridge did not become ready")


def start_approval_harness(
    trial_dir: Path,
    notebook_url: str,
    *,
    headed: bool,
    timeout: float,
) -> tuple[subprocess.Popen[Any], Path]:
    manifest = rc.load_manifest(trial_dir)
    operator = rc.operator_dir(trial_dir)
    ready_file = operator / "harness-ready.json"
    stop_file = operator / "harness-stop"
    ready_file.unlink(missing_ok=True)
    stop_file.unlink(missing_ok=True)
    command = [
        sys.executable,
        str(rc.BENCH_DIR / "approval_harness.py"),
        "--notebook-url",
        notebook_url,
        "--output-dir",
        manifest["paths"]["output"],
        "--log",
        str(operator / "approvals.jsonl"),
        "--ready-file",
        str(ready_file),
        "--stop-file",
        str(stop_file),
        "--timeout",
        str(int(timeout)),
    ]
    if headed:
        command.append("--headed")
    stdout = (operator / "harness.stdout.log").open("w", encoding="utf-8")
    stderr = (operator / "harness.stderr.log").open("w", encoding="utf-8")
    process = subprocess.Popen(command, cwd=rc.ROOT, stdout=stdout, stderr=stderr)
    stdout.close()
    stderr.close()
    _wait_for_file(ready_file, process, timeout)
    return process, stop_file


def stop_approval_harness(process: subprocess.Popen[Any] | None, stop_file: Path | None) -> None:
    if process is None:
        return
    if stop_file is not None:
        stop_file.touch()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def pi_command(args: argparse.Namespace, trial_dir: Path, prompt: str) -> list[str]:
    manifest = rc.load_manifest(trial_dir)
    session_dir = rc.operator_dir(trial_dir).parent / "agent" / "session"
    session_dir.mkdir(parents=True, exist_ok=True)
    command = [
        args.pi_executable,
        "--no-builtin-tools",
        "--tools",
        ",".join(PI_ALLOWED_TOOLS),
        "--no-context-files",
        "--no-skills",
        "--no-prompt-templates",
        "--no-themes",
        "--session-dir",
        str(session_dir),
        "--name",
        str(manifest.get("run_id") or trial_dir.name),
    ]
    if getattr(args, "runner", "pi") == "pi":
        command[1:1] = ["--mode", "json", "--print"]
    else:
        command.extend(
            [
                "--tui-mode",
                "regular",
                "--extension",
                str(rc.BENCH_DIR / "pi_tui_probe.ts"),
            ]
        )
    for flag, value in (
        ("--provider", args.provider),
        ("--model", args.model),
        ("--thinking", args.thinking),
    ):
        if value:
            command.extend([flag, value])
    command.append(
        prompt
        if getattr(args, "runner", "pi") == "pi"
        else (
            "Before doing any scientific work, call the MCP wrapper target named exactly "
            "peaksMCP_inspect_notebook with target=\"kernel\" exactly once. Do not call "
            "the unprefixed name. Then call the mcp wrapper's server-instructions operation "
            "with instructions=\"peaksMCP\" exactly once. Do not list or search tools or "
            "guess another status-tool name first. Report whether Jupyter, kernel, MCP and "
            "Comm are ready; if they are not, stop and state which Dashboard recovery "
            "control the operator must use."
        )
    )
    return command


def generic_command(args: argparse.Namespace, trial_dir: Path) -> tuple[list[str], str | None, Path]:
    manifest = rc.load_manifest(trial_dir)
    values = {
        "trial": str(trial_dir),
        "workspace": manifest["paths"]["workspace"],
        "input": manifest["paths"]["input"],
        "output": manifest["paths"]["output"],
        "notebook": manifest["paths"]["notebook"],
        "prompt_file": manifest["paths"]["prompt"],
        "session_dir": str(trial_dir / "agent" / "session"),
        "run_id": manifest["run_id"],
        "condition": manifest["condition"],
    }
    command = [part.format(**values) for part in shlex.split(args.command_template)]
    prompt = Path(manifest["paths"]["prompt"]).read_text(encoding="utf-8")
    stdin = None if "{prompt_file}" in args.command_template else prompt
    cwd = Path(args.command_cwd.format(**values)).expanduser().resolve() if args.command_cwd else Path(values["workspace"])
    return command, stdin, cwd


def parse_pi_events(path: Path) -> dict[str, Any]:
    counts: dict[str, int] = {}
    session_id = None
    last_usage: dict[str, Any] | None = None
    final_message: dict[str, Any] | None = None
    malformed = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        event_type = str(event.get("type") or "unknown")
        counts[event_type] = counts.get(event_type, 0) + 1
        if event_type == "session":
            session_id = event.get("id")
        usage = event.get("usage") or (event.get("message") or {}).get("usage")
        if isinstance(usage, dict):
            last_usage = usage
        if event_type == "message_end" and (event.get("message") or {}).get("role") == "assistant":
            final_message = event.get("message")
    return {
        "session_id": session_id,
        "event_counts": counts,
        "last_usage": last_usage,
        "final_message": final_message,
        "malformed_lines": malformed,
    }


def _record_pi_client_tool_evidence(
    trial_dir: Path, metadata: dict[str, Any]
) -> dict[str, Any]:
    """Persist Pi's structured client-side tool failures beside runner data."""
    evidence = rc.collect_pi_client_tool_evidence(trial_dir)
    rc.atomic_write_json(trial_dir / "agent" / "client_tool_evidence.json", evidence)
    metadata["client_tool_evidence"] = evidence
    return metadata


def _submit_tui_prompt(process: Any, prompt: str, *, chunk_size: int = 1024) -> None:
    """Paste a long prompt into Pi without deadlocking the PTY on redraws."""
    process.send("\x1b[200~")
    for start in range(0, len(prompt), chunk_size):
        process.send(prompt[start : start + chunk_size])
    process.send("\x1b[201~")
    process.send("\r")


def _submit_tui_command(process: Any, command: str) -> None:
    """Submit a Pi TUI slash command with the carriage return it recognizes."""
    process.send(command)
    process.send("\r")


def execute_agent(
    args: argparse.Namespace,
    trial_dir: Path,
) -> dict[str, Any]:
    manifest = rc.load_manifest(trial_dir)
    prompt = Path(manifest["paths"]["prompt"]).read_text(encoding="utf-8")
    if args.runner in {"pi", "pi-tui"}:
        command = pi_command(args, trial_dir, prompt)
        stdin_text = None
        cwd = rc.ROOT
    else:
        command, stdin_text, cwd = generic_command(args, trial_dir)

    agent_dir = trial_dir / "agent"
    stdout_path = agent_dir / "stdout.jsonl" if args.runner == "pi" else agent_dir / "stdout.log"
    stderr_path = agent_dir / "stderr.log"
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "BENCH_TRIAL_DIR": str(trial_dir),
            "BENCH_WORKSPACE": manifest["paths"]["workspace"],
            "BENCH_PROMPT_FILE": manifest["paths"]["prompt"],
        }
    )
    started = time.monotonic()
    started_at = now()
    timed_out = False
    if args.runner == "pi-tui":
        import pexpect

        with stdout_path.open("w", encoding="utf-8") as stdout:
            process = pexpect.spawn(
                command[0],
                command[1:],
                cwd=str(cwd),
                env=environment,
                encoding="utf-8",
                timeout=args.agent_timeout,
            )
            process.logfile_read = stdout
            try:
                process.expect_exact("__PEAKSMCP_PI_AGENT_END__")
                send_errors: list[BaseException] = []

                def _send_prompt() -> None:
                    try:
                        _submit_tui_prompt(process, prompt)
                    except BaseException as exc:  # noqa: BLE001 - forwarded below
                        send_errors.append(exc)

                sender = threading.Thread(target=_send_prompt, daemon=True)
                sender.start()
                # Pi redraws while accepting a paste. Reading concurrently is
                # required or both sides can fill the PTY buffers and block.
                process.expect_exact("__PEAKSMCP_PI_AGENT_END__")
                sender.join(timeout=5)
                if sender.is_alive():
                    raise RuntimeError("Pi TUI prompt sender did not finish")
                if send_errors:
                    raise RuntimeError("Pi TUI prompt submission failed") from send_errors[0]
                _submit_tui_command(process, "/quit")
                process.expect(pexpect.EOF, timeout=30)
            except pexpect.TIMEOUT:
                timed_out = True
                process.close(force=True)
            if not process.closed:
                process.close()
            exit_code = process.exitstatus if process.exitstatus is not None else process.signalstatus
        stderr_path.write_text("", encoding="utf-8")
        metadata = {
            "runner": args.runner,
            "command": command[:-1] + ["<prompt>"],
            "cwd": str(cwd),
            "started_at": started_at,
            "finished_at": now(),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "exit_code": exit_code,
            "timed_out": timed_out,
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        }
        _record_pi_client_tool_evidence(trial_dir, metadata)
        rc.atomic_write_json(agent_dir / "runner.json", metadata)
        return metadata

    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            text=True,
        )
        try:
            process.communicate(input=stdin_text, timeout=args.agent_timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
    metadata = {
        "runner": args.runner,
        "command": command[:-1] + ["<prompt>"] if args.runner == "pi" else command,
        "cwd": str(cwd),
        "started_at": started_at,
        "finished_at": now(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "exit_code": process.returncode,
        "timed_out": timed_out,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }
    if args.runner == "pi":
        _record_pi_client_tool_evidence(trial_dir, metadata)
        rc.atomic_write_json(agent_dir / "usage.json", parse_pi_events(stdout_path))
    rc.atomic_write_json(agent_dir / "runner.json", metadata)
    return metadata


def capture_live_kernel_evidence(
    trial_dir: Path,
    run_state: dict[str, Any],
    key: dict[str, Any],
    reference_dir: Path | None,
    *,
    timeout: float = 180.0,
) -> dict[str, Any]:
    """Measure scientific results in the live namespace without a notebook cell.

    This is an evaluator-only channel used after the agent exits. It stores
    bounded numeric metrics, never arrays, and executes with ``store_history``
    disabled so the user's append-only notebook remains the sole work record.
    """
    from jupyter_client import BlockingKernelClient
    from jupyter_client.connect import find_connection_file

    expected = [str(item["stem"]) for item in key["expected_outputs"]]
    home = Path(os.environ.get("PEAKSMCP_HOME", str(Path.home() / ".peaksMCP")))
    runtime_dir = home / "jupyter" / "runtime"
    connection = find_connection_file(
        f"kernel-{run_state['kernel_id']}.json",
        path=[str(runtime_dir)],
    )
    client = BlockingKernelClient(connection_file=connection)
    client.load_connection_file()
    client.start_channels()
    code = f"""
import json as _ev_json
import re as _ev_re
from pathlib import Path as _EvPath
import numpy as _ev_np
import xarray as _ev_xr

_ev_expected = {expected!r}
_ev_reference_dir = {_safe_path(reference_dir)!r}
_ev_seen = set()
_ev_arrays = []
_ev_reports = []
_ev_gold_fits = []

def _ev_walk(value, label, depth=0):
    marker = id(value)
    if marker in _ev_seen or depth > 2:
        return
    _ev_seen.add(marker)
    if isinstance(value, _ev_xr.DataArray):
        _ev_arrays.append((label, value))
        return
    if isinstance(value, _ev_xr.Dataset):
        if isinstance(value.attrs.get('fit_window'), dict):
            _ev_gold_fits.append((label, value))
        return
    if type(value).__name__ == 'ConversionReport':
        _ev_reports.append(value)
        return
    if isinstance(value, dict):
        for key, item in list(value.items())[:200]:
            _ev_walk(item, f'{{label}}[{{key!r}}]', depth + 1)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value[:200]):
            _ev_walk(item, f'{{label}}[{{index}}]', depth + 1)

for _ev_name, _ev_value in list(get_ipython().user_ns.items()):
    if not _ev_name.startswith('_'):
        _ev_walk(_ev_value, _ev_name)

def _ev_stems(label, data):
    text = ' '.join((label, str(data.name or ''), str(data.attrs.get('source_path', '')),
                     str(data.attrs.get('_scan', ''))))
    return set(_ev_re.findall(r'BP_\\d{{4}}', text, flags=_ev_re.IGNORECASE))

def _ev_metrics(stem, data):
    dims = list(data.dims)
    row = {{'variable': stem, 'dims': dims, 'shape': list(data.shape)}}
    ev = data.coords.get('eV')
    row['ef_landmark'] = float(_ev_np.nanmin(_ev_np.abs(ev.values))) if ev is not None and ev.size else None
    kname = next((name for name in ('kx', 'k_par', 'kp', 'kparallel', 'kx_par') if name in data.coords), None)
    kval = data.coords.get(kname) if kname else None
    row['kx_landmark'] = float(_ev_np.nanmin(_ev_np.abs(kval.values))) if kval is not None and kval.size else None
    if not _ev_reference_dir:
        return row
    path = _EvPath(_ev_reference_dir) / f'{{stem}}_processed.nc'
    if not path.is_file():
        row['reference_missing'] = True
        return row
    with _ev_xr.open_dataset(path) as opened:
        ref = opened[list(opened.data_vars)[0]].load()
    row['same_dims'] = set(ref.dims) == set(data.dims)
    if row['same_dims']:
        ref = ref.transpose(*data.dims)
    deltas = []
    for dim in data.dims:
        if dim not in ref.coords or dim not in data.coords:
            continue
        left = _ev_np.asarray(data.coords[dim], dtype=float)
        right = _ev_np.asarray(ref.coords[dim], dtype=float)
        if left.size and right.size:
            deltas.append(max(abs(float(left.min()) - float(right.min())),
                              abs(float(left.max()) - float(right.max())),
                              abs(float(left.mean()) - float(right.mean()))))
    row['coord_delta'] = max(deltas) if deltas else None
    if row['same_dims'] and data.shape != ref.shape:
        ref = ref.interp_like(data, method='linear')
    left = _ev_np.asarray(data.values, dtype=float)
    right = _ev_np.asarray(ref.values, dtype=float)
    if left.shape != right.shape:
        row.update({{'corr': None, 'mask_overlap': 0.0, 'efficiency': None}})
        return row
    ml, mr = _ev_np.isfinite(left), _ev_np.isfinite(right)
    union, both = ml | mr, ml & mr
    row['mask_overlap'] = float(both.sum() / union.sum()) if union.any() else 1.0
    if both.any():
        a, b = left[both], right[both]
        row['corr'] = float(_ev_np.corrcoef(a, b)[0, 1]) if a.size > 1 and a.std() and b.std() else None
        row['efficiency'] = float(_ev_np.mean(_ev_np.abs(a)) / (_ev_np.mean(_ev_np.abs(b)) or 1.0))
    return row

_ev_products = {{}}
_ev_product_priorities = {{}}
_ev_all_kspace_stems = set()
for _ev_label, _ev_data in _ev_arrays:
    if not any(dim in _ev_data.dims for dim in ('kx', 'k_par', 'kp', 'kparallel', 'kx_par')):
        continue
    _ev_label_stems = set(_ev_re.findall(r'BP_\\d{{4}}', _ev_label, flags=_ev_re.IGNORECASE))
    for _ev_stem in _ev_stems(_ev_label, _ev_data):
        canonical = _ev_stem.upper()
        _ev_all_kspace_stems.add(canonical)
        if canonical in _ev_expected:
            # Prefer explicit outputs over aliases/views whose attrs retain a
            # source stem. Plotting variables often hold cropped or transposed
            # views and must not overwrite a stem-keyed result dictionary.
            _ev_priority = 3 if _ev_label.upper() == canonical else (
                2 if canonical in {{stem.upper() for stem in _ev_label_stems}} else 1
            )
            if _ev_priority > _ev_product_priorities.get(canonical, 0):
                _ev_products[canonical] = _ev_metrics(canonical, _ev_data)
                _ev_product_priorities[canonical] = _ev_priority

_ev_report_rows = []
for _ev_report in _ev_reports:
    _ev_report_rows.append({{
        'converted': int(getattr(_ev_report, 'converted', 0)),
        'cached': int(getattr(_ev_report, 'cached', 0)),
        'failed': int(getattr(_ev_report, 'failed', 0)),
    }})
_ev_gold_rows = []
for _ev_label, _ev_fit in _ev_gold_fits:
    _ev_window = _ev_fit.attrs.get('fit_window') or {{}}
    _ev_quality = _ev_fit.attrs.get('EF_quality') or {{}}
    _ev_gold_rows.append({{
        'variable': _ev_label,
        'fit_window': {{
            'start_eV': float(_ev_window.get('start_eV')),
            'center_eV': float(_ev_window.get('center_eV')),
            'stop_eV': float(_ev_window.get('stop_eV')),
            'lower_points': int(_ev_window.get('lower_points', 0)),
            'upper_points': int(_ev_window.get('upper_points', 0)),
            'total_points': int(_ev_window.get('total_points', 0)),
        }},
        'outlier_fraction': float(
            _ev_fit.attrs.get(
                'EF_correction_outlier_fraction',
                _ev_quality.get('outlier_fraction', 1.0),
            )
        ),
        'uniform': bool(_ev_quality.get('uniform', False)),
    }})
def _ev_clean(value):
    if isinstance(value, float) and not _ev_np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {{key: _ev_clean(item) for key, item in value.items()}}
    if isinstance(value, list):
        return [_ev_clean(item) for item in value]
    return value
_ev_payload = {{'status': 'ok', 'processed_stems': sorted(_ev_all_kspace_stems),
               'unexpected_processed_stems': sorted(_ev_all_kspace_stems - set(_ev_expected)),
               'products': _ev_products, 'conversion_reports': _ev_report_rows,
               'gold_fits': _ev_gold_rows}}
print({LIVE_EVIDENCE_MARKER!r} + _ev_json.dumps(_ev_clean(_ev_payload), allow_nan=False))
"""
    stdout: list[str] = []
    try:
        client.wait_for_ready(timeout=timeout)
        message_id = client.execute(code, silent=False, store_history=False)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            message = client.get_iopub_msg(timeout=max(0.1, deadline - time.monotonic()))
            if message.get("parent_header", {}).get("msg_id") != message_id:
                continue
            msg_type = message.get("msg_type")
            content = message.get("content") or {}
            if msg_type == "stream":
                stdout.append(str(content.get("text") or ""))
            elif msg_type == "error":
                raise RuntimeError(str(content.get("evalue") or "live evidence failed"))
            elif msg_type == "status" and content.get("execution_state") == "idle":
                break
        joined = "".join(stdout)
        line = next(
            (item for item in joined.splitlines() if item.startswith(LIVE_EVIDENCE_MARKER)),
            None,
        )
        if line is None:
            raise RuntimeError("live evidence marker missing from kernel output")
        payload = json.loads(line[len(LIVE_EVIDENCE_MARKER):])
    except Exception as exc:  # noqa: BLE001 - evidence failure must be recorded
        payload = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        with contextlib.suppress(Exception):
            client.stop_channels()
    rc.atomic_write_json(rc.evaluator_dir(trial_dir) / "live_evidence.json", payload)
    return payload


def _safe_path(path: Path | None) -> str | None:
    return str(path.resolve()) if path is not None else None


def _mark_trial_started(
    trial_dir: Path,
    *,
    runner: str,
    fresh_kernel: bool,
    kernel_evidence: str,
    generic_isolation: bool,
    isolation_mechanism: str,
    allowed_tools: list[str],
    kernel_id: str | None = None,
    live_notebook: str | None = None,
    kernel_root: str | None = None,
) -> None:
    pi = runner in {"pi", "pi-tui"}
    rc.cmd_start(
        argparse.Namespace(
            run=str(trial_dir),
            fresh_session=True,
            session_evidence=f"new {runner} process with a trial-local session directory",
            fresh_kernel=fresh_kernel,
            kernel_evidence=kernel_evidence,
            kernel_id=kernel_id,
            live_notebook=live_notebook,
            kernel_root=kernel_root,
            isolation_enforced=pi or generic_isolation,
            isolation_mechanism=(
                "pi --no-builtin-tools with only mcp,mcpScript enabled; evaluator outside workspace"
                if pi
                else isolation_mechanism
            ),
            allowed_tools=PI_ALLOWED_TOOLS if pi else allowed_tools,
        )
    )


def run_one_trial(args: argparse.Namespace, trial: dict[str, Any]) -> dict[str, Any]:
    trial_dir = Path(trial["path"])
    manifest = rc.load_manifest(trial_dir)
    case = manifest["case"]
    key = rc.build_answer_key(
        Path(case["data_dir"]), Path(case["datasheet"]), case.get("limit")
    )
    notebook = Path(manifest["paths"]["notebook"])
    harness: subprocess.Popen[Any] | None = None
    stop_file: Path | None = None
    fresh_kernel = False
    kernel_evidence = args.kernel_evidence or ""
    #: Managed-stack state; empty for manual/generic runs so the marker call
    #: below never touches an unbound name.
    run_state: dict[str, Any] = {}

    try:
        if args.manage_stack:
            run_state = _run_state_for_notebook(notebook, args.profile, args.stack_timeout)
            fresh_kernel = True
            kernel_evidence = (
                f"managed peaksMCP host selected unique notebook {notebook}; "
                f"live notebook {run_state.get('live_notebook')}; "
                f"kernel_id={run_state.get('kernel_id')}"
            )
            if args.approval_mode == "harness_allowlist":
                harness, stop_file = start_approval_harness(
                    trial_dir,
                    _dashboard_url(run_state),
                    headed=args.headed,
                    timeout=args.stack_timeout,
                )
                _wait_for_comm(run_state, args.stack_timeout)
        else:
            fresh_kernel = bool(args.fresh_kernel)

        _mark_trial_started(
            trial_dir,
            runner=args.runner,
            fresh_kernel=fresh_kernel,
            kernel_evidence=kernel_evidence,
            kernel_id=run_state.get("kernel_id"),
            live_notebook=run_state.get("live_notebook"),
            kernel_root=run_state.get("root_dir"),
            generic_isolation=args.generic_isolation_enforced,
            isolation_mechanism=args.isolation_mechanism or "",
            allowed_tools=args.allowed_tools or [],
        )
        metadata = execute_agent(args, trial_dir)
        if args.manage_stack:
            reference = manifest.get("case", {}).get("reference_dir")
            capture_live_kernel_evidence(
                trial_dir,
                run_state,
                key,
                Path(reference) if reference else None,
                timeout=min(float(args.stack_timeout), 300.0),
            )
    finally:
        stop_approval_harness(harness, stop_file)

    manifest = rc.load_manifest(trial_dir)
    audit_path = Path(manifest["audit"]["path"])
    manifest["audit"]["end_offset"] = rc.audit_offset(audit_path)
    manifest["audit"]["end_recorded_at"] = now()
    manifest["finished_at"] = now()
    manifest["status"] = "completed"
    manifest["agent"]["runner"] = args.runner
    manifest["agent"]["execution"] = metadata
    rc.save_manifest(trial_dir, manifest)

    rc.cmd_grade(
        argparse.Namespace(
            run=str(trial_dir),
            notebook=None,
            output=None,
            audit=None,
            reference=None,
            since=None,
            json=False,
            quiet=True,
        )
    )
    return json.loads((rc.evaluator_dir(trial_dir) / "result.json").read_text(encoding="utf-8"))


def run_campaign(args: argparse.Namespace) -> int:
    campaign_dir, campaign = load_campaign(args.campaign)
    previous_host = capture_managed_host() if args.manage_stack else None
    for field in ("provider", "model", "thinking"):
        if getattr(args, field) is None:
            setattr(args, field, (campaign.get("agent") or {}).get(field))
    campaign["agent"] = {
        **(campaign.get("agent") or {}),
        "provider": args.provider,
        "model": args.model,
        "thinking": args.thinking,
        "runner": args.runner,
    }
    selected = set(_condition_list(args.conditions)) if args.conditions else None
    trials = sorted(campaign["trials"], key=lambda item: (item["replicate"], item["order"]))

    def stop_after_non_strict(trial: dict[str, Any]) -> int:
        summarize_campaign_dir(campaign_dir, campaign)
        print(
            f"Stopping before the next trial: {trial['run_id']} "
            f"valid={trial.get('valid')!r}, "
            f"strict_success={trial.get('strict_success')!r}",
            flush=True,
        )
        return 1

    try:
        for trial in trials:
            if selected and trial["condition"] not in selected:
                continue
            result_path = rc.evaluator_dir(Path(trial["path"])) / "result.json"
            if result_path.is_file():
                if getattr(args, "stop_on_non_strict", False):
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                    trial["status"] = "graded"
                    trial["strict_success"] = result.get("strict_success")
                    trial["valid"] = (result.get("validity") or {}).get("valid")
                    save_campaign(campaign_dir, campaign)
                    if (
                        trial["valid"] is not True
                        or trial["strict_success"] is not True
                    ):
                        return stop_after_non_strict(trial)
                continue
            print(
                f"Running {trial['run_id']} (replicate {trial['replicate']}, "
                f"condition {trial['condition']})"
            )
            try:
                result = run_one_trial(args, trial)
                trial["status"] = "graded"
                trial["strict_success"] = result["strict_success"]
                trial["valid"] = result["validity"]["valid"]
            except Exception as exc:  # noqa: BLE001
                trial["status"] = "runner_error"
                trial["error"] = f"{type(exc).__name__}: {exc}"
                save_campaign(campaign_dir, campaign)
                if not args.keep_going:
                    raise
            save_campaign(campaign_dir, campaign)
            if getattr(args, "stop_on_non_strict", False) and (
                trial.get("valid") is not True or trial.get("strict_success") is not True
            ):
                return stop_after_non_strict(trial)
        summarize_campaign_dir(campaign_dir, campaign)
    finally:
        if args.manage_stack:
            try:
                campaign["stack_restoration"] = restore_managed_host(
                    previous_host,
                    timeout=args.stack_timeout,
                )
            except KeyboardInterrupt:
                raise
            except BaseException as exc:  # noqa: BLE001
                # Housekeeping: a host that refuses to die (or a helper that
                # signals through SystemExit) is a cleanup warning, not the
                # campaign's outcome - the trials were already graded.
                campaign["stack_restoration"] = {
                    "status": "warning",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                print(
                    f"warning: managed-host teardown failed ({type(exc).__name__}: {exc}); "
                    "trials were graded normally",
                    flush=True,
                )
            save_campaign(campaign_dir, campaign)
    return 0


def grade_campaign(args: argparse.Namespace) -> int:
    campaign_dir, campaign = load_campaign(args.campaign)
    for trial in campaign["trials"]:
        trial_dir = Path(trial["path"])
        rc.cmd_grade(
            argparse.Namespace(
                run=str(trial_dir), notebook=None, output=None, audit=None,
                reference=None, since=None, json=False, quiet=True,
            )
        )
        payload = json.loads(
            (rc.evaluator_dir(trial_dir) / "result.json").read_text(encoding="utf-8")
        )
        trial["status"] = "graded"
        trial["strict_success"] = payload["strict_success"]
        trial["valid"] = payload["validity"]["valid"]
    save_campaign(campaign_dir, campaign)
    summarize_campaign_dir(campaign_dir, campaign)
    return 0


def _mean(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return round(mean(present), 3) if present else None


def summarize_campaign_dir(campaign_dir: Path, campaign: dict[str, Any]) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for trial in campaign["trials"]:
        path = rc.evaluator_dir(Path(trial["path"])) / "result.json"
        if path.is_file():
            results.append(json.loads(path.read_text(encoding="utf-8")))

    conditions: dict[str, Any] = {}
    for condition in campaign["conditions"]:
        selected = [result for result in results if result.get("condition") == condition]
        valid = [result for result in selected if result["validity"]["valid"]]
        elapsed: list[float | None] = []
        for result in selected:
            runner_path = Path(result["run"]) / "agent" / "runner.json"
            runner = json.loads(runner_path.read_text(encoding="utf-8")) if runner_path.is_file() else {}
            elapsed.append(runner.get("elapsed_seconds"))
        dimension_names = sorted(
            {name for result in selected for name in result["score"]["dimensions"]}
        )
        conditions[condition] = {
            "trials": len(selected),
            "valid_trials": len(valid),
            "strict_successes": sum(result["strict_success"] for result in valid),
            "strict_success_rate": (
                round(sum(result["strict_success"] for result in valid) / len(valid), 3)
                if valid
                else None
            ),
            "observed_mean": _mean([result["score"]["observed"] for result in selected]),
            "conservative_mean": _mean(
                [result["score"]["conservative"] for result in selected]
            ),
            "evidence_coverage_mean": _mean(
                [result["score"]["evidence_coverage"] for result in selected]
            ),
            "elapsed_seconds_mean": _mean(elapsed),
            "dimensions": {
                name: {
                    "observed_mean": _mean(
                        [result["score"]["dimensions"][name]["observed"] for result in selected]
                    ),
                    "conservative_mean": _mean(
                        [result["score"]["dimensions"][name]["conservative"] for result in selected]
                    ),
                }
                for name in dimension_names
            },
        }

    failure_frequency: dict[str, dict[str, int]] = {}
    for result in results:
        for check in result["checks"]:
            bucket = failure_frequency.setdefault(check["check"], {"failed": 0, "skipped": 0})
            if check["passed"] is False:
                bucket["failed"] += 1
            elif check["passed"] is None:
                bucket["skipped"] += 1

    paired: list[dict[str, Any]] = []
    by_pair = {(result.get("replicate"), result.get("condition")): result for result in results}
    if {"p1", "p2"} <= set(campaign["conditions"]):
        for replicate in range(1, int(campaign["repetitions"]) + 1):
            p1 = by_pair.get((replicate, "p1"))
            p2 = by_pair.get((replicate, "p2"))
            if not p1 or not p2 or not p1["validity"]["valid"] or not p2["validity"]["valid"]:
                diagnosis = "invalid_or_incomplete_pair"
            elif p1["strict_success"] and p2["strict_success"]:
                diagnosis = "autonomous_path_usable"
            elif not p1["strict_success"] and p2["strict_success"]:
                diagnosis = "discoverability_or_instruction_failure"
            elif not p1["strict_success"] and not p2["strict_success"]:
                diagnosis = "capability_runtime_or_quality_failure"
            else:
                diagnosis = "stochastic_anomaly_repeat_required"
            paired.append({"replicate": replicate, "diagnosis": diagnosis})

    payload = {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "campaign_id": campaign["campaign_id"],
        "generated_at": now(),
        "freeze": campaign["freeze"],
        "agent": campaign.get("agent"),
        "case": campaign["case"],
        "conditions": conditions,
        "paired_diagnosis": paired,
        "failure_frequency": failure_frequency,
        "trial_results": [
            {
                "run_id": result["run_id"],
                "replicate": result.get("replicate"),
                "condition": result.get("condition"),
                "valid": result["validity"]["valid"],
                "strict_success": result["strict_success"],
                "check_map": result.get("check_map") or {
                    row["check"]: row.get("passed") for row in result.get("checks", [])
                },
                "primary_failure": result.get("primary_failure"),
                "result": str(rc.evaluator_dir(Path(result["run"])) / "result.json"),
            }
            for result in results
        ],
    }
    rc.atomic_write_json(campaign_dir / "result.json", payload)

    lines = [
        f"# Campaign `{campaign['campaign_id']}`",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "| Condition | Valid | Strict successes | Strict rate | Conservative | Evidence |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition, data in conditions.items():
        lines.append(
            f"| {condition} | {data['valid_trials']}/{data['trials']} | "
            f"{data['strict_successes']} | {rc._pct(data['strict_success_rate'])} | "
            f"{rc._pct(data['conservative_mean'])} | {rc._pct(data['evidence_coverage_mean'])} |"
        )
    lines += ["", "## Paired diagnosis", ""]
    for pair in paired:
        lines.append(f"- Replicate {pair['replicate']}: `{pair['diagnosis']}`")
    lines += ["", "## Highest-frequency failures", ""]
    ranked = sorted(
        failure_frequency.items(),
        key=lambda item: (item[1]["failed"], item[1]["skipped"]),
        reverse=True,
    )
    for check, counts in ranked[:15]:
        if counts["failed"] or counts["skipped"]:
            lines.append(
                f"- `{check}`: failed {counts['failed']}, skipped {counts['skipped']}"
            )
    (campaign_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Campaign report: {campaign_dir / 'report.md'}")
    return payload


def summarize_campaign(args: argparse.Namespace) -> int:
    campaign_dir, campaign = load_campaign(args.campaign)
    summarize_campaign_dir(campaign_dir, campaign)
    return 0


def compare_campaigns(args: argparse.Namespace) -> int:
    baseline_dir, baseline_manifest = load_campaign(args.baseline)
    candidate_dir, candidate_manifest = load_campaign(args.candidate)
    baseline = summarize_campaign_dir(baseline_dir, baseline_manifest)
    candidate = summarize_campaign_dir(candidate_dir, candidate_manifest)
    experiment = rc.load_yaml(rc.EXPERIMENT_FILE)
    promotion = experiment.get("promotion") or {}
    minimum = int(promotion.get("minimum_valid_trials_per_condition", 3))
    threshold = float(promotion.get("minimum_strict_success_delta", 0.1))
    requested_checks = list(getattr(args, "target_check", None) or [])
    target_checks = requested_checks or list(promotion.get("target_checks") or [])

    p1_before = (baseline["conditions"].get("p1") or {}).get("strict_success_rate")
    p1_after = (candidate["conditions"].get("p1") or {}).get("strict_success_rate")
    delta = None if p1_before is None or p1_after is None else round(p1_after - p1_before, 3)
    enough_valid = all(
        data["valid_trials"] >= minimum for data in candidate["conditions"].values()
    )
    freeze_match = all(
        baseline["freeze"].get(key) == candidate["freeze"].get(key)
        for key in ("experiment_sha256", "rubric_sha256", "common_prompt_sha256", "condition_prompt_sha256")
    )
    agent_match = baseline.get("agent") == candidate.get("agent")
    outcome_before = (baseline["conditions"].get("p1") or {}).get("dimensions", {}).get("Outcome", {}).get("conservative_mean")
    outcome_after = (candidate["conditions"].get("p1") or {}).get("dimensions", {}).get("Outcome", {}).get("conservative_mean")
    safety_before = (baseline["conditions"].get("p1") or {}).get("dimensions", {}).get("Safety", {}).get("conservative_mean")
    safety_after = (candidate["conditions"].get("p1") or {}).get("dimensions", {}).get("Safety", {}).get("conservative_mean")
    no_outcome_regression = None not in (outcome_before, outcome_after) and outcome_after >= outcome_before
    no_safety_regression = None not in (safety_before, safety_after) and safety_after >= safety_before
    elapsed_before = (baseline["conditions"].get("p1") or {}).get("elapsed_seconds_mean")
    elapsed_after = (candidate["conditions"].get("p1") or {}).get("elapsed_seconds_mean")
    efficiency_improved = (
        None not in (elapsed_before, elapsed_after)
        and elapsed_after < elapsed_before
    )
    endpoint_improved = (
        delta is not None
        and (
            delta >= threshold
            or (p1_before == 1.0 and p1_after == 1.0 and efficiency_improved)
        )
    )

    before_trials = {
        (item["replicate"], item["condition"]): item for item in baseline["trial_results"]
    }
    after_trials = {
        (item["replicate"], item["condition"]): item for item in candidate["trial_results"]
    }
    wins = losses = 0
    check_transitions = {
        check: {"fail_to_pass": 0, "pass_to_fail": 0, "unchanged": 0}
        for check in target_checks
    }
    for key in set(before_trials) & set(after_trials):
        before = before_trials[key]
        after = after_trials[key]
        if not before["valid"] or not after["valid"]:
            continue
        if not before["strict_success"] and after["strict_success"]:
            wins += 1
        elif before["strict_success"] and not after["strict_success"]:
            losses += 1
        for check in target_checks:
            old = (before.get("check_map") or {}).get(check)
            new = (after.get("check_map") or {}).get(check)
            if old is not True and new is True:
                check_transitions[check]["fail_to_pass"] += 1
            elif old is True and new is not True:
                check_transitions[check]["pass_to_fail"] += 1
            else:
                check_transitions[check]["unchanged"] += 1

    target_wins = sum(row["fail_to_pass"] for row in check_transitions.values())
    target_losses = sum(row["pass_to_fail"] for row in check_transitions.values())
    target_improved = bool(target_checks) and target_wins > target_losses and target_wins > 0
    candidate_strict = all(
        data.get("strict_success_rate") == 1.0
        for data in candidate["conditions"].values()
    )

    promoted = all(
        (
            enough_valid,
            freeze_match,
            agent_match,
            no_outcome_regression,
            no_safety_regression,
            target_wins >= target_losses if target_checks else wins >= losses,
            target_improved if target_checks else endpoint_improved,
            candidate_strict,
        )
    )
    comparison = {
        "baseline": baseline["campaign_id"],
        "candidate": candidate["campaign_id"],
        "promoted": promoted,
        "p1_strict_success_delta": delta,
        "paired_wins": wins,
        "paired_losses": losses,
        "target_checks": target_checks,
        "target_check_transitions": check_transitions,
        "criteria": {
            "minimum_valid_trials": enough_valid,
            "frozen_design_matches": freeze_match,
            "agent_configuration_matches": agent_match,
            "no_outcome_regression": no_outcome_regression,
            "no_safety_regression": no_safety_regression,
            "paired_wins_not_less_than_losses": wins >= losses,
            "target_wins_not_less_than_losses": target_wins >= target_losses,
            "target_checks_improved": target_improved,
            "candidate_all_valid_trials_strict": candidate_strict,
            "minimum_delta_met": delta is not None and delta >= threshold,
            "perfect_success_with_efficiency_gain": (
                p1_before == 1.0 and p1_after == 1.0 and efficiency_improved
            ),
        },
    }
    output = candidate_dir / f"comparison-vs-{baseline['campaign_id']}.json"
    rc.atomic_write_json(output, comparison)
    print(json.dumps(comparison, indent=2))
    print(f"Comparison saved: {output}")
    return 0 if promoted else 1


def preflight(args: argparse.Namespace) -> int:
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str, *, required: bool = True) -> None:
        checks.append(
            {"check": name, "passed": passed, "required": required, "detail": detail}
        )

    case_file = rc.resolve_case_file(args.case)
    case = rc.load_yaml(case_file)
    resolved_paths: dict[str, Path | None] = {}
    for name in ("data", "datasheet", "reference"):
        path = rc.configured_path(case, name, None, required=name != "reference")
        resolved_paths[name] = path
        add(
            f"case_{name}",
            bool(path and path.exists()),
            str(path),
            required=name != "reference",
        )
    staging = case.get("staging") or {}
    add(
        "input_staging_copy",
        staging.get("strategy", "copy") == "copy",
        f"strategy={staging.get('strategy', 'copy')}",
    )
    source = resolved_paths.get("data")
    include_suffixes = {
        str(value).lower()
        for value in staging.get("include_suffixes")
        or (case.get("parser") or {}).get("input_suffixes")
        or []
    }
    selected = [
        path
        for path in (source.iterdir() if source and source.is_dir() else [])
        if path.is_file()
        and not path.name.startswith(".")
        and (not include_suffixes or path.suffix.lower() in include_suffixes)
        and not any(path.match(pattern) for pattern in staging.get("exclude_globs") or [])
    ]
    input_state = str((case.get("scientific_contract") or {}).get("input_state") or "raw")
    expected_suffix = ".nc" if input_state == "preconverted" else ".pxt"
    add(
        "scientific_input_kind",
        bool(selected) and all(path.suffix.lower() == expected_suffix for path in selected),
        f"state={input_state}; selected {len(selected)} staged files; "
        f"suffixes={sorted(include_suffixes)}",
    )
    leaked_processed = [path.name for path in selected if path.stem.endswith("_processed")]
    add(
        "reference_products_excluded",
        not leaked_processed,
        f"processed products selected for input: {leaked_processed[:5] or 'none'}",
    )
    add("pi_executable", shutil.which(args.pi_executable) is not None, args.pi_executable)
    add("pexpect", importlib.util.find_spec("pexpect") is not None, "Python pexpect package")
    playwright_ready = importlib.util.find_spec("playwright") is not None
    add("playwright_python", playwright_ready, "Python Playwright package")
    add(
        "playwright_cli",
        shutil.which("npx") is not None,
        "npx available for browser diagnostics",
        required=False,
    )
    browser_ready = False
    browser_detail = "not attempted because Python Playwright is unavailable"
    if playwright_ready:
        try:
            from playwright.sync_api import sync_playwright

            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True, channel="chrome")
                browser.close()
            browser_ready = True
            browser_detail = "Chrome launched and closed successfully"
        except Exception as exc:  # noqa: BLE001
            browser_detail = f"{type(exc).__name__}: {exc}"
    add("browser_launch", browser_ready, browser_detail, required=False)
    selftest = subprocess.run(
        [sys.executable, str(rc.BENCH_DIR / "run_case.py"), "selftest"],
        cwd=rc.ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    selftest_lines = [line for line in selftest.stdout.splitlines() if "Self-test passed" in line]
    detail = selftest_lines[-1] if selftest_lines else (selftest.stdout or selftest.stderr)[-300:]
    add("grader_selftest", selftest.returncode == 0, detail)
    if args.require_live_mcp:
        peaks_executable = shutil.which(args.peaks_executable)
        add(
            "peaks_executable",
            peaks_executable is not None,
            (
                f"resolved {args.peaks_executable!r} to {peaks_executable}"
                if peaks_executable
                else f"executable not found: {args.peaks_executable!r}"
            ),
        )
        if peaks_executable is None:
            add(
                "live_mcp",
                False,
                "not attempted because the peaksMCP executable is unavailable",
            )
        else:
            try:
                ping = subprocess.run(
                    [peaks_executable, "mcp-ping"],
                    cwd=rc.ROOT,
                    capture_output=True,
                    text=True,
                    check=False,
                )
            except OSError as exc:
                add("live_mcp", False, f"{type(exc).__name__}: {exc}")
            else:
                add(
                    "live_mcp",
                    ping.returncode == 0,
                    (ping.stdout or ping.stderr)[-500:],
                )
    payload = {
        "passed": all(check["passed"] for check in checks if check["required"]),
        "checks": checks,
    }
    print(json.dumps(payload, indent=2))
    return 0 if payload["passed"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    create = sub.add_parser("create", help="create randomized paired trials")
    create.add_argument("--name")
    create.add_argument("--campaigns", default=str(DEFAULT_CAMPAIGNS))
    create.add_argument("--case", default="bp260623")
    create.add_argument("--conditions", default="p1,p2")
    create.add_argument("--repetitions", type=int, default=3)
    create.add_argument("--seed", type=int, default=20260910)
    create.add_argument("--objective")
    create.add_argument("--data")
    create.add_argument("--datasheet")
    create.add_argument("--reference")
    create.add_argument("--limit", type=int)
    create.add_argument("--copy", action="store_true", help=argparse.SUPPRESS)
    create.add_argument("--force", action="store_true")
    create.add_argument("--approval-mode", choices=("manual_review", "harness_allowlist"), default="harness_allowlist")
    create.add_argument("--agent-id", default="pi")
    create.add_argument("--provider", default="deepseek")
    create.add_argument("--model", default="deepseek-v4-flash")
    create.add_argument("--thinking", default="low")
    create.set_defaults(func=create_campaign)

    run = sub.add_parser("run", help="execute pending trials sequentially")
    run.add_argument("campaign")
    run.add_argument("--conditions")
    run.add_argument("--runner", choices=("pi", "pi-tui", "command"), default="pi-tui")
    run.add_argument("--pi-executable", default="pi")
    run.add_argument("--provider")
    run.add_argument("--model")
    run.add_argument("--thinking")
    run.add_argument("--command-template")
    run.add_argument("--command-cwd")
    run.add_argument("--agent-timeout", type=float, default=3600)
    run.add_argument("--manage-stack", action="store_true")
    run.add_argument("--profile", default="default")
    run.add_argument("--stack-timeout", type=float, default=180)
    run.add_argument("--approval-mode", choices=("manual_review", "harness_allowlist"), default="harness_allowlist")
    run.add_argument("--headed", action="store_true")
    run.add_argument("--fresh-kernel", action="store_true")
    run.add_argument("--kernel-evidence")
    run.add_argument("--generic-isolation-enforced", action="store_true")
    run.add_argument("--isolation-mechanism")
    run.add_argument("--allowed-tools", nargs="*")
    run.add_argument("--keep-going", action="store_true")
    run.add_argument(
        "--stop-on-non-strict",
        action="store_true",
        help="stop before the next trial when a completed trial is invalid or not strict-successful",
    )
    run.set_defaults(func=run_campaign)

    grade = sub.add_parser("grade", help="grade every trial")
    grade.add_argument("campaign")
    grade.set_defaults(func=grade_campaign)

    summarize = sub.add_parser("summarize", help="aggregate existing trial results")
    summarize.add_argument("campaign")
    summarize.set_defaults(func=summarize_campaign)

    compare = sub.add_parser("compare", help="apply the campaign promotion rule")
    compare.add_argument("baseline")
    compare.add_argument("candidate")
    compare.add_argument(
        "--target-check",
        action="append",
        help="rubric check id to compare as paired fail-to-pass evidence (repeatable)",
    )
    compare.set_defaults(func=compare_campaigns)

    flight = sub.add_parser("preflight", help="validate data, grader, and optional live MCP")
    flight.add_argument("--case", default="bp260623")
    flight.add_argument("--pi-executable", default="pi")
    flight.add_argument("--peaks-executable", default="peaksMCP")
    flight.add_argument("--require-live-mcp", action="store_true")
    flight.set_defaults(func=preflight)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run" and args.runner == "command" and not args.command_template:
        parser.error("--command-template is required for --runner command")
    if args.command == "run" and args.manage_stack and args.approval_mode != "harness_allowlist":
        parser.error("--manage-stack requires --approval-mode harness_allowlist; use run_case.py for manual review")
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
