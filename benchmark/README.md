# Agent-Agnostic ARPES Preprocessing Benchmark

This benchmark measures whether an AI agent can autonomously discover, execute, verify, and
persist a complete ARPES cut-preprocessing workflow through peaksMCP.

The evaluator never grades agent prose. Evidence comes only from durable artifacts: the notebook,
MCP audit window, saved products, trial manifest, approval log, and intervention log.

## Quick Start

Use the project environment:

```bash
PY=/opt/homebrew/Caskroom/miniforge/base/envs/peaks/bin/python

# 1. Validate the grader and local prerequisites.
$PY benchmark/run_case.py selftest
$PY benchmark/run_campaign.py preflight --case bp260623

# 2. Create three randomized P1/P2 pairs.
$PY benchmark/run_campaign.py create \
  --name baseline-001 \
  --case bp260623 \
  --repetitions 3 \
  --provider openrouter \
  --model <model-id> \
  --thinking high

# 3. Run pi-agent with a fresh managed kernel and deterministic save approvals.
$PY benchmark/run_campaign.py run benchmark/campaigns/baseline-001 \
  --runner pi \
  --provider openrouter \
  --model <model-id> \
  --thinking high \
  --manage-stack \
  --approval-mode harness_allowlist
```

Results are written to `benchmark/campaigns/baseline-001/result.json` and `report.md`.
When `--manage-stack` temporarily replaces an existing peaksMCP host, the runner restores the
previous profile, Jupyter root, and notebook after the campaign, including failure paths.

## Conditions

- **P1 goal-only** gives the scientific goal, deliverables, and operational constraints without
  benchmark-provided tool names or checkpoints. It measures discovery plus execution.
- **P2 tool-aware** adds the semantic five-tool map and explicit execution checkpoints. It controls
  most discoverability and measures execution capability.

Each condition runs in a separate fresh model session, kernel, and notebook. The runner randomizes
condition order within each paired replicate. P2 is never sent as a follow-up to a failed P1.

Interpret paired strict outcomes:

| P1 | P2 | Interpretation |
|---|---|---|
| pass | pass | Autonomous path is usable for this case. Increase case diversity. |
| fail | pass | Discovery, naming, ranking, or server-instruction failure. |
| fail | fail | Workflow, runtime, persistence, review, or result-quality failure. |
| pass | fail | Stochastic anomaly or invalid trial. Repeat before concluding. |

## Trial Layout

```text
trial/
  workspace/                 agent-visible working area
    input/                     staged, preconverted NetCDF input
    output/
    work.ipynb
    prompt.txt               exactly one condition
  evaluator/                 never included in the agent workspace
    manifest.json
    env.json
    answer_key.json          generated only after execution ends
    result.json
    report.md
  agent/
    runner.json
    stdout.jsonl|stdout.log
    stderr.log
    usage.json               pi JSON-event summary when available
    session/
  operator/
    interventions.jsonl
    approvals.jsonl
```

The answer key is independently derived from the datasheet, never from peaksMCP's
`inspect_experiment`. This prevents the system under test from grading itself.

Inputs are copied into each trial rather than symlinked. Case definitions should point to
preconverted NetCDF scans and exclude any `*_processed.nc` reference products. This keeps raw
beamtime data and human references outside the agent's writable workspace and avoids making
conversion approval part of a preprocessing-capability measurement.

## Universal Agent Adapter

Any command-line agent can run the same trial through the generic adapter. The command template
supports these placeholders:

`{trial}`, `{workspace}`, `{input}`, `{output}`, `{notebook}`, `{prompt_file}`, `{session_dir}`,
`{run_id}`, and `{condition}`.

```bash
$PY benchmark/run_campaign.py run benchmark/campaigns/baseline-001 \
  --runner command \
  --command-template 'my-agent --new-session --prompt-file {prompt_file}' \
  --command-cwd '{workspace}' \
  --fresh-kernel \
  --kernel-evidence 'external orchestrator created a new kernel' \
  --generic-isolation-enforced \
  --isolation-mechanism 'container with only the peaksMCP proxy exposed' \
  --allowed-tools mcp
```

If `{prompt_file}` is absent, the runner sends the prompt on stdin. The adapter captures stdout,
stderr, exit status, elapsed time, and timeout state. Conversation text is retained for debugging
but is never grading evidence.

For GUI-only agents, use the manual single-trial flow below and preserve the same artifact layout.

## Manual Single Trial

```bash
$PY benchmark/run_case.py init --name manual-p1 --case bp260623 --condition p1

# Open workspace/work.ipynb in a fresh peaksMCP managed kernel, then immediately before sending
# workspace/prompt.txt to the agent:
$PY benchmark/run_case.py start benchmark/runs/manual-p1 \
  --fresh-session --session-evidence 'new conversation' \
  --fresh-kernel --kernel-evidence 'new managed kernel' \
  --isolation-enforced \
  --isolation-mechanism 'agent filesystem disabled; MCP-only tool allowlist' \
  --allowed-tools mcp

# After the agent stops:
$PY benchmark/run_case.py grade benchmark/runs/manual-p1
```

Record scientific hints, retry instructions, manual code, and manual data edits as JSON objects in
`operator/interventions.jsonl`. A save approval is a policy action and belongs in
`operator/approvals.jsonl`; it is not an assistive intervention.

## Isolation Requirement

Prompt instructions are not isolation. The tested agent must be unable to read evaluator files,
references, benchmark source, prior trials, or unrelated filesystem state.

The pi runner disables all built-in tools and enables only `mcp,mcpScript`:

```text
--no-builtin-tools --tools mcp,mcpScript --no-context-files --no-skills
--no-prompt-templates --no-themes
```

The peaksMCP code scanner is an early rejection layer, not a complete Python sandbox. For a formal
security claim, run both the agent and managed kernel inside an OS-level container or sandbox that
mounts only the trial input and workspace. The benchmark records the declared mechanism and scans
executed code for direct evaluator/reference access markers.

## Approval Harness

`benchmark/approval_harness.py` keeps the trial notebook active, preserving the JupyterLab Comm
bridge. It approves a save only when the card displays exactly one expected filename under the
exact trial output directory. Unexpected targets and every non-save consent dialog are denied.
Every decision is logged.

This mode makes persistence repeatable. It does not count as human scientific review.

## Endpoints

The primary endpoint is `strict_success`. It requires:

- a valid isolated trial with immutable prompt and exact audit byte window;
- all strict rubric checks passing, including no skipped strict checks;
- every expected product, no unexpected or misplaced products, and correct numerical results;
- a validation figure and complete final Markdown summary;
- append-only execution and consent-gated persistence;
- complete evidence, fresh session/kernel, and no assistive intervention.

The weighted score is diagnostic only. Reports expose:

- **observed score**: skipped checks excluded;
- **conservative score**: skipped checks counted as failures;
- **evidence coverage**: fraction of rubric weight that was actually decidable;
- separate Outcome, Safety, Autonomy, Reliability, Evidence, and Diagnostic dimensions.

## Iterative Optimization

Never overwrite a completed trial. Change one subsystem-level factor, create a new campaign, and
compare it against the frozen baseline:

```bash
$PY benchmark/run_campaign.py compare \
  benchmark/campaigns/baseline-001 \
  benchmark/campaigns/candidate-002
```

Promotion requires the same case, prompts, rubric, agent, model, and sampling settings; at least
three valid trials per condition; no Outcome or Safety regression; paired wins not fewer than
losses; and the configured P1 strict-success improvement. If both campaigns have perfect P1 strict
success, lower elapsed time can satisfy the endpoint-improvement clause.

Use failure frequency and each rubric check's `fix` field to select the next change. Do not optimize
the aggregate weighted score directly.

## Generalization

`bp260623` is a development case. It cannot establish universal automation by itself. A release
claim requires at least one independently acquired holdout case with different acquisition data,
scan mix, shapes, and metadata edge cases. Freeze the holdout before evaluation; after observing a
holdout failure, begin a new optimization cycle with a new untouched holdout.

The full research protocol is in `benchmark/PROTOCOL.md`. Machine-readable design settings are in
`benchmark/experiment.yaml`, case definitions are in `benchmark/cases/`, and scoring policy is in
`benchmark/rubric.yaml`.
