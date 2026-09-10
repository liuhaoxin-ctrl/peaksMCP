# Prompt Contract

The canonical prompts are immutable source files:

- `prompts/common.txt` contains the scientific task and artifact requirements shared by every
  condition.
- `prompts/p1_goal_only.txt` contains the goal-only condition prefix.
- `prompts/p2_tool_aware.txt` contains the tool-aware condition prefix.

`run_case.py init --condition p1|p2` concatenates exactly one condition prefix with the common task,
replaces the trial path placeholders, writes `workspace/prompt.txt`, and freezes its SHA-256 in the
evaluator manifest.

Do not hand-edit a rendered prompt. Do not give one agent both conditions. Do not send P2 as a
follow-up after P1. A prompt hash mismatch invalidates the trial.

P1 measures discovery plus execution. P2 controls most discoverability and measures execution.
Their paired outcome is diagnostic evidence, not an escalation sequence.
