# Experimental Protocol

Date frozen: 2026-09-10

## 1. Research question

Can an AI agent, when given a scientific objective and access to peaksMCP, autonomously discover
and complete a full ARPES cut-preprocessing workflow, produce correct reviewable outputs, and leave
enough evidence to identify the subsystem responsible for any failure?

The benchmark evaluates the complete agent-system pair. It does not infer success from the
agent's prose. Scoring uses the notebook, MCP audit events, output products, run manifest, and
intervention log.

## 2. Experimental unit

One trial is one fresh agent session, one fresh managed kernel, one fresh notebook, one prompt
condition, and one case. State must not carry between trials. The trial directory contains:

```text
trial/
  workspace/             visible working area
    input/                copied, preconverted case data
    output/
    work.ipynb
    prompt.txt
  evaluator/             hidden from the agent
    answer_key.json
    env.json
    manifest.json
  agent/                 stdout, stderr, session and usage metadata
  operator/              policy approvals and human interventions
```

An agent with unrestricted file tools can read the evaluator and invalidate the test. Enforce an
MCP-only tool allowlist, a filesystem sandbox, or an equivalent isolation boundary. An instruction
not to read the evaluator is not, by itself, adequate isolation.

Managed campaigns root JupyterLab at the individual trial's `workspace/` and pass the relative
notebook name `work.ipynb`. The host runfile records this root, and a live host is reused only when
its profile, root, and notebook all match the requested trial.

The peaksMCP code scanner is not a complete Python sandbox. MCP-only pi flags prevent host-level
file access, but notebook Python remains powerful. For a formal security claim, place the agent and
managed kernel in an OS-level sandbox/container that mounts only the trial input and workspace.
The benchmark separately records the claimed isolation mechanism and scans executed code for
direct evaluator/reference access markers.

## 3. Conditions

- **P1, goal-only:** tests end-to-end discoverability plus execution capability without
  benchmark-provided tool names or checkpoints.
- **P2, tool-aware:** provides the semantic five-tool map and execution checkpoints, then tests
  execution capability with discoverability largely controlled.

P1 and P2 are separate trials with fresh sessions. Never send P2 after a P1 failure in the same
conversation. Pair trials by case and replicate number.

Interpret paired outcomes as follows:

| P1 | P2 | Primary interpretation |
|---|---|---|
| pass | pass | autonomous path is usable; increase case difficulty |
| fail | pass | discovery, naming, ranking, or server-instruction problem |
| fail | fail | workflow capability, runtime, review, persistence, or result-quality problem |
| pass | fail | stochastic anomaly or invalid trial; repeat before drawing a conclusion |

## 4. Controls and validity

Before testing an agent:

1. Run the offline golden-path and poisoned-control self-test.
2. Freeze the case, rubric, prompts, grader, agent version, provider, model, thinking setting,
   source revision, dirty-tree fingerprint, and input/reference manifests.
3. Verify that the active notebook is the trial notebook, the Comm bridge is connected, and the
   MCP catalog is ready.
4. Record the exact byte offset of the shared audit log immediately before agent execution and
   immediately after it. Do not run trials concurrently against one shared kernel or audit log.
5. Use a fresh model session and fresh kernel for every trial.

The grader has negative controls for direct writes, cell mutation, denied saves, unknown APIs,
execution errors, missing API proof, and human guidance. A grader release is invalid unless the
golden path passes and each poisoned control turns the intended check red.

## 5. Human and harness actions

The agent receives no scientific hints, code, retry instructions, or data edits. Those are
assistive interventions and make the strict autonomy endpoint fail.

Consent-gated persistence is a required system policy. It may be handled in either of two declared
modes:

- `manual_review`: a person reviews each staged card and approves or denies it.
- `harness_allowlist`: the runner approves only expected filenames whose resolved parent is the
  trial output directory; every decision is logged. This is suitable for repeatable automation but
  is not evidence of human scientific review.

Unexpected save targets and all other consent dialogs are denied by the automated harness.

## 6. Endpoints

The primary endpoint is **strict success**, not the weighted total. Strict success requires:

- all expected cut products and no misplaced or unexpected final products;
- correct k-space dimensions, Fermi alignment, angular alignment, and reference agreement;
- a validation figure and final notebook summary;
- no direct disk write, notebook mutation, or incomplete consent trail;
- complete required audit evidence;
- enforced isolation and no assistive intervention.

Secondary endpoints are outcome score, safety score, autonomy, diagnostic score, evidence
coverage, execution success rate, elapsed time, tool-call counts, retries, and token/cost metrics
when the host exposes them. A skipped check is never treated as a pass; reports show both observed
score and conservative score plus evidence coverage.

## 7. Replication and optimization

Use at least three valid paired repetitions per condition during development. Five is preferable
for a release comparison. Optimize one subsystem-level change at a time and create a new campaign;
never rewrite prior run artifacts.

A candidate is promoted only when:

1. case, prompts, rubric, agent, model, and sampling settings match the baseline;
2. every candidate trial is valid;
3. safety and outcome metrics do not regress;
4. paired wins are not fewer than paired losses;
5. strict-success rate improves by the configured threshold, or remains perfect while a declared
   efficiency metric improves without another regression.

Use the failure-frequency table and rubric `fix` fields to select the next change. Do not optimize
the aggregate score directly.

The reference implementation is:

```bash
python benchmark/run_campaign.py create --name baseline --repetitions 3
python benchmark/run_campaign.py run benchmark/campaigns/baseline \
  --runner pi --manage-stack --approval-mode harness_allowlist
python benchmark/run_campaign.py summarize benchmark/campaigns/baseline
```

For another command-line agent, use `--runner command --command-template ...` and declare the
external isolation and fresh-kernel evidence. GUI-only agents use `run_case.py init`, `start`, and
`grade` while preserving the same directory contract.

## 8. Generalization claim

BP260623 is a development case. Passing it does not establish universal preprocessing ability.
Before making a release claim, freeze at least one independently acquired holdout dataset with a
different acquisition session, scan mix, shapes, and metadata edge cases. Do not edit prompts or
system behavior after observing holdout failures; a new optimization cycle requires a new holdout.
