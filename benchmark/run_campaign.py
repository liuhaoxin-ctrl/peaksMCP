#!/usr/bin/env python3
"""Create, execute, grade, aggregate, and compare agent benchmark campaigns."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import shlex
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any
from urllib.parse import quote

import httpx

try:
    from benchmark import run_case as rc
except ModuleNotFoundError:  # direct ``python benchmark/run_campaign.py`` invocation
    import run_case as rc  # type: ignore[no-redef]

CAMPAIGN_SCHEMA_VERSION = 1
DEFAULT_CAMPAIGNS = rc.BENCH_DIR / "campaigns"
PI_ALLOWED_TOOLS = ["mcp", "mcpScript"]


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
    """Start a managed host rooted at the isolated trial workspace."""
    from peaksMCP.cli import _ensure_host

    state = _ensure_host(
        argparse.Namespace(
            profile=profile,
            root_dir=str(notebook.parent.resolve()),
            notebook=notebook.name,
            timeout=timeout,
        )
    )
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


def _notebook_url(run_state: dict[str, Any]) -> str:
    encoded = quote(str(run_state["notebook_path"]), safe="/")
    return f"{run_state['jupyter_url']}/lab/tree/{encoded}?token={run_state['token']}"


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
    expected_names: list[str],
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
    for name in expected_names:
        command.extend(["--expected-file", name])
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
        "--mode",
        "json",
        "--print",
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
    for flag, value in (
        ("--provider", args.provider),
        ("--model", args.model),
        ("--thinking", args.thinking),
    ):
        if value:
            command.extend([flag, value])
    command.append(prompt)
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


def execute_agent(
    args: argparse.Namespace,
    trial_dir: Path,
) -> dict[str, Any]:
    manifest = rc.load_manifest(trial_dir)
    prompt = Path(manifest["paths"]["prompt"]).read_text(encoding="utf-8")
    if args.runner == "pi":
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
    rc.atomic_write_json(agent_dir / "runner.json", metadata)
    if args.runner == "pi":
        rc.atomic_write_json(agent_dir / "usage.json", parse_pi_events(stdout_path))
    return metadata


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
    pi = runner == "pi"
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
                    _notebook_url(run_state),
                    [item["output_name"] for item in key["expected_outputs"]],
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
    try:
        for trial in trials:
            if selected and trial["condition"] not in selected:
                continue
            result_path = rc.evaluator_dir(Path(trial["path"])) / "result.json"
            if result_path.is_file():
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
    for key in set(before_trials) & set(after_trials):
        before = before_trials[key]
        after = after_trials[key]
        if not before["valid"] or not after["valid"]:
            continue
        if not before["strict_success"] and after["strict_success"]:
            wins += 1
        elif before["strict_success"] and not after["strict_success"]:
            losses += 1

    promoted = all(
        (
            enough_valid,
            freeze_match,
            agent_match,
            no_outcome_regression,
            no_safety_regression,
            wins >= losses,
            endpoint_improved,
        )
    )
    comparison = {
        "baseline": baseline["campaign_id"],
        "candidate": candidate["campaign_id"],
        "promoted": promoted,
        "p1_strict_success_delta": delta,
        "paired_wins": wins,
        "paired_losses": losses,
        "criteria": {
            "minimum_valid_trials": enough_valid,
            "frozen_design_matches": freeze_match,
            "agent_configuration_matches": agent_match,
            "no_outcome_regression": no_outcome_regression,
            "no_safety_regression": no_safety_regression,
            "paired_wins_not_less_than_losses": wins >= losses,
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
    add(
        "preconverted_input",
        bool(selected) and all(path.suffix.lower() == ".nc" for path in selected),
        f"selected {len(selected)} staged files; suffixes={sorted(include_suffixes)}",
    )
    leaked_processed = [path.name for path in selected if path.stem.endswith("_processed")]
    add(
        "reference_products_excluded",
        not leaked_processed,
        f"processed products selected for input: {leaked_processed[:5] or 'none'}",
    )
    add("pi_executable", shutil.which(args.pi_executable) is not None, args.pi_executable)
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
        ping = subprocess.run(
            [args.peaks_executable, "mcp-ping"],
            cwd=rc.ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        add("live_mcp", ping.returncode == 0, (ping.stdout or ping.stderr)[-500:])
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
    create.add_argument("--provider")
    create.add_argument("--model")
    create.add_argument("--thinking")
    create.set_defaults(func=create_campaign)

    run = sub.add_parser("run", help="execute pending trials sequentially")
    run.add_argument("campaign")
    run.add_argument("--conditions")
    run.add_argument("--runner", choices=("pi", "command"), default="pi")
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
