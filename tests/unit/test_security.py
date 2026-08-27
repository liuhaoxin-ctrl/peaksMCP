from __future__ import annotations

import json

import pytest

from peaksMCP.server.jupyter_peaks.security import AuditLogger, ConsentManager, scan_code


@pytest.mark.parametrize("code", [
    "!ls", "%%bash\necho bad", "get_ipython().system('ls')", "exec('x=1')", "eval('1+1')",
    "import os as x\nx.system('ls')", "from subprocess import run as r\nr(['ls'])",
    "open('x', 'w')", "import os\nos.environ['A']='B'", "import shutil\nshutil.rmtree('x')",
])
def test_dangerous_patterns_are_blocked(code):
    assert scan_code(code).blocked


@pytest.mark.parametrize("code", [
    "data = data.sel(eV=slice(-1, 0))", "data.plot()", "import numpy as np\nx=np.arange(4)",
    "result = data.S.smooth({'eV': 2})", "print(data.dims)",
])
def test_scientific_code_is_allowed(code):
    assert scan_code(code).is_safe


def test_syntax_errors_are_structured_and_blocked():
    result = scan_code("for")
    assert result.blocked
    assert result.syntax_error["line"] == 1


def test_consent_fails_closed_without_frontend():
    assert ConsentManager().request("delete", {}) is False


def test_audit_is_jsonl_and_private(tmp_path):
    path = tmp_path / "audit.log"
    AuditLogger(path).write("tool", "approved", {"x": 1})
    assert json.loads(path.read_text())["tool"] == "tool"
    assert path.stat().st_mode & 0o777 == 0o600

