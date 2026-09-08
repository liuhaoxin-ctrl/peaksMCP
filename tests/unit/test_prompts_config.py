from __future__ import annotations

import pytest

from peaksMCP.config import prompts
from peaksMCP.server.jupyter_peaks.security import scan_code


def test_prompts_yaml_exposes_all_runtime_groups():
    """config/prompts.yaml must carry every runtime prompt group the code reads."""
    doc = prompts()
    for group in (
        "server_instructions",
        "interactive_omitted_note",
        "list_resources_guidance",
        "notebook_unsafe",
        "scanner",
        "ipython",
    ):
        assert group in doc, f"missing prompt group {group!r} in config/prompts.yaml"

    unsafe = doc["notebook_unsafe"]
    for key in (
        "index_build_failed",
        "plot_templates_not_read",
        "savefig_forbidden",
        "unknown_api_first",
        "unknown_api_retry",
        "api_check_rule",
    ):
        assert unsafe[key], f"missing notebook_unsafe prompt {key!r}"
    assert "override" in unsafe["api_check_rule"].lower()

    guidance = doc["list_resources_guidance"]
    assert guidance["resources_vs_tools"]["resources"]
    assert guidance["resources_vs_tools"]["tools"]
    assert len(guidance["when_to_use_resources"]) == 3
    assert guidance["first_use"]


# (snippet, expected rule, prompt key, formatting values) — ties the wording
# the code emits back to the curated YAML template so a rename or a format
# mismatch fails loudly instead of drifting silently.
_SCANNER_CASES = [
    ("import os\nos.system('whoami')", "SYS001", "sys_destructive", {"name": "os.system"}),
    ("import subprocess\nsubprocess.run(['ls'])", "CAP001", "cap_sandbox_via", {"name": "subprocess.run"}),
    ("import matplotlib.pyplot as plt\nplt.savefig('o.png')", "SAVE001", "savefig_disabled", {}),
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


def test_server_instructions_are_delivered_from_prompts_yaml():
    """The always-on FastMCP server instructions must come from the YAML."""
    from peaksMCP.server.jupyter_peaks.mcp_server import _SERVER_INSTRUCTIONS

    template = prompts()["server_instructions"]
    assert _SERVER_INSTRUCTIONS == template
    assert template.startswith("Work through peaksMCP tools only.")
    assert "preserve units in every figure" in template


def test_interactive_and_block_copy_are_not_empty_strings():
    """The high-traffic L3 strings must stay populated (no accidental blanks)."""
    doc = prompts()
    assert len(doc["interactive_omitted_note"]) > 20
    assert len(doc["notebook_unsafe"]["savefig_forbidden"]) > 50
    assert len(doc["notebook_unsafe"]["plot_templates_not_read"]) > 100
