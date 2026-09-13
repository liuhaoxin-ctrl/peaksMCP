from __future__ import annotations

from pathlib import Path

import pytest

from peaksMCP.config import prompts
from peaksMCP.server.jupyter_peaks.security import scan_code


def test_prompts_yaml_exposes_all_runtime_groups():
    """config/prompts.yaml must carry every runtime prompt group the code reads."""
    doc = prompts()
    for group in (
        "server_instructions",
        "interactive_omitted_note",
        "notebook_unsafe",
        "scanner",
        "ipython",
    ):
        assert group in doc, f"missing prompt group {group!r} in config/prompts.yaml"

    unsafe = doc["notebook_unsafe"]
    for key in (
        "index_build_failed",
        "unknown_api_first",
        "unknown_api_retry",
        "api_check_rule",
    ):
        assert unsafe[key], f"missing notebook_unsafe prompt {key!r}"
    assert "cataloged peaks apis" in unsafe["api_check_rule"].lower()


# (snippet, expected rule, prompt key, formatting values) — ties the wording
# the code emits back to the curated YAML template so a rename or a format
# mismatch fails loudly instead of drifting silently.
_SCANNER_CASES = [
    ("import os\nos.system('whoami')", "SYS001", "sys_destructive", {"name": "os.system"}),
    ("import subprocess\nsubprocess.run(['ls'])", "CAP001", "cap_sandbox_via", {"name": "subprocess.run"}),
    ("import os\nos.listdir('.')", "FILE004", "file_read_direct", {"name": "os.listdir"}),
    ("import matplotlib.pyplot as plt\nplt.savefig('o.png')", None, None, {}),
    ("import numpy as np\nx = np.array([1, 2])", None, None, {}),
    ("!ls", "IPY001", "ipy_shell", {}),
]


@pytest.mark.parametrize("code,rule,key,values", _SCANNER_CASES)
def test_scanner_descriptions_render_from_prompts_yaml(code, rule, key, values):
    result = scan_code(code)
    if rule is None:
        assert not result.blocked
        return
    matching = [issue for issue in result.issues if issue.rule_id == rule]
    assert matching, f"{rule} not reported for {code!r}"
    template = prompts()["scanner" if key != "ipy_shell" else "ipython"][key]
    assert matching[0].description == (template.format(**values) if values else template)


def test_savefig_consent_copy_renders_from_prompts_yaml():
    """savefig is now a consent finding (SAVE001 in requires_explicit_consent),
    and its yellow-note copy must render from the curated prompts YAML."""
    result = scan_code("import matplotlib.pyplot as plt\nplt.savefig('o.png')")
    assert not result.blocked
    matching = [issue for issue in result.requires_explicit_consent if issue.rule_id == "SAVE001"]
    assert matching
    template = prompts()["scanner"]["savefig_consent"]
    assert matching[0].description == template


def test_server_instructions_are_delivered_from_prompts_yaml():
    """The always-on FastMCP server instructions must come from the YAML."""
    from peaksMCP.server.jupyter_peaks.mcp_server import _SERVER_INSTRUCTIONS

    template = prompts()["server_instructions"]
    assert _SERVER_INSTRUCTIONS == template
    assert template.startswith("peaksMCP operating contract")
    assert "Use only these five tools: search, get, inspect_notebook, run_cell, and\n  save_with_consent" in template
    assert template.index("Search once with") < template.index("Before submitting any run_cell")
    assert template.index("Start with load_experiment") < template.index("call pxt2nc")
    assert "do not get the same\n  canonical id twice" in template
    assert 'cell_type="markdown"' in template
    assert "A run_cell timeout is not cancellation" in template
    assert "run_cell is not a general persistence path" in template
    assert 'receipt status is "saved"' in template
    assert "gold.fit_gold(show=False, quiet=True)" in template
    assert "Never stringify either mapping" in " ".join(template.split())
    assert "start_eV/stop_eV/lower_points/upper_points" in template
    assert "treat total_points as an energy" in template
    assert 'gold_fit.attrs["figure"]' in template
    assert "k_convert(..., quiet=True)" in template
    assert "plt.close(fig)" in template
    assert "never a defensive getattr" in template


def test_server_instructions_match_exact_live_tool_argument_names():
    template = " ".join(prompts()["server_instructions"].split())

    for call_shape in (
        "search(query=...)",
        "get(canonical_id=...)",
        "inspect_notebook(target=...)",
        'run_cell(code=..., api_ids=[...], cell_type="code")',
        "save_with_consent(variable_name=..., path=...)",
    ):
        assert call_shape in template
    assert "Never substitute id for canonical_id, source for code, or regex for query" in template
    assert 'run_cell(code=TEXT, cell_type="markdown")' in template
    assert "Never pass code_language, language, or any other key" in template


def test_server_instructions_reject_round_one_redundant_exploration():
    template = " ".join(prompts()["server_instructions"].lower().split())

    assert "single-letter" in template
    assert "broad catalog enumeration" in template
    assert "repository files, docs, prompts, or skills" in template
    assert "get only api ids you will call" in template
    assert "before submitting any run_cell" in template
    assert "including calls inside loops or batches" in template
    assert "a search result or an api_ids entry is not proof" in template
    assert "never use a rejected run_cell as api discovery" in template
    assert "at most three non-empty stdout lines" in template
    assert "do not immediately inspect that cell" in template
    assert "capture representative_raw before normal-emission assignment" in template
    assert "representative_kcut from that same single batch loop" in template
    assert "never reload a scan or repeat a scientific call solely for validation" in template
    assert "from peaks import load_experiment" in template
    assert "non-empty needs_conversion" in template
    assert "coordinate `.values`" in template
    assert "every processed stem" in template
    assert "experimentconflict has exactly four public fields" in template
    assert "do not add a separate inventory-only cell" in template
    assert "do not search/get plot_grid" in template
    assert "one value per stem" in template
    assert "gold=selected_stem (index selected_index)" in template
    assert 'saying only "gold scan fitted once" is incomplete' in template
    assert 'print("processed_stems=" + ",".join(sorted(result_dict)))' in template
    assert "copy the complete returned `processed_stems=...` token verbatim" in template
    assert "never infer consecutive stems from the processed count" in template
    assert "never wrap discovery or proof calls in mcpscript" in template
    assert "theta_offset=value deg from record.theta_offset_deg" in template
    assert "naming `record.theta_offset_deg` without its value is incomplete" in template


def test_task_prompts_require_numeric_theta_value_with_exact_record_source():
    root = Path(__file__).resolve().parents[2]
    common = (root / "benchmark/prompts/common.txt").read_text(encoding="utf-8")
    tool_aware = (root / "benchmark/prompts/p2_tool_aware.txt").read_text(encoding="utf-8")

    for raw_prompt in (common, tool_aware):
        prompt = " ".join(raw_prompt.split())
        assert "theta_offset=VALUE deg from record.theta_offset_deg" in prompt
        assert "replacing VALUE with the observed number" in prompt


def test_task_prompts_require_the_selected_gold_identifier():
    root = Path(__file__).resolve().parents[2]
    paths = (
        root / "benchmark/prompts/common.txt",
        root / "benchmark/prompts/p2_tool_aware.txt",
    )

    for path in paths:
        prompt = " ".join(path.read_text(encoding="utf-8").split())
        assert "gold=SELECTED_STEM (index SELECTED_INDEX)" in prompt
        assert '"gold scan fitted once"' in prompt


def test_task_prompts_require_live_processed_stem_receipt_reuse():
    root = Path(__file__).resolve().parents[2]
    paths = (
        root / "benchmark/prompts/common.txt",
        root / "benchmark/prompts/p2_tool_aware.txt",
    )

    for path in paths:
        prompt = " ".join(path.read_text(encoding="utf-8").split())
        assert "exactly one" in prompt
        assert "processed_stems=" in prompt
        assert "sorted" in prompt
        assert "three stdout lines" in prompt
        assert "200 characters" in prompt
        assert "verbatim" in prompt
        assert "consecutive" in prompt


def test_active_prompt_surfaces_do_not_name_retired_tools():
    """Current prompts must not teach names removed from the five-tool surface."""
    root = Path(__file__).resolve().parents[2]
    texts = [prompts()["server_instructions"]]
    for relative in (
        "benchmark/prompts/common.txt",
        "benchmark/prompts/p1_goal_only.txt",
        "benchmark/prompts/p2_tool_aware.txt",
        "benchmark/prompts/u1_natural_2d.txt",
        "claude_plugin/skills/cut-preprocessing/SKILL.md",
    ):
        texts.append((root / relative).read_text(encoding="utf-8"))
    joined = "\n".join(texts)
    for retired in (
        "peaks_search_api",
        "peaks_get_api",
        "mcp_list_resources",
        "notebook_write_with_api_check",
        "notebook_add_cell",
        "fit_gold_reference",
        "preprocess_cut",
        "save_result",
        "publication_grid",
    ):
        assert retired not in joined, retired


def test_interactive_and_block_copy_are_not_empty_strings():
    """The high-traffic L3 strings must stay populated (no accidental blanks)."""
    doc = prompts()
    assert len(doc["interactive_omitted_note"]) > 20
    assert len(doc["scanner"]["savefig_consent"]) > 20
    assert len(doc["notebook_unsafe"]["index_build_failed"]) > 20
