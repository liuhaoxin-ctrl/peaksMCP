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
    # Alias bypass of the environment-mutation rule.
    "import os as o\no.environ['A']='B'",
    "import os as o\no.environ.update({'A': 'B'})",
    # environ mutation via call nodes / delete / aug-assign.
    "os.environ.update({'A': 'B'})",
    "os.environ.setdefault('A', 'B')",
    "del os.environ['A']",
    "import os as o\no.environ['A'] += 'B'",
    # open() with the mode passed as keyword.
    "open('x', mode='w')",
    "open('x', mode='a')",
    # pathlib destructive methods, including aliased imports.
    "from pathlib import Path\nPath('x').unlink()",
    "import pathlib\npathlib.Path('x').unlink()",
    "from pathlib import Path as P\nP('x').write_text('evil')",
    "from pathlib import Path\nPath('x').write_bytes(b'e')",
    "from pathlib import Path\nPath('x').rmdir()",
    "from pathlib import Path\nPath('x').rename('y')",
    # indirect fetches of critical callables.
    "getattr(__builtins__, 'exec')('x=1')",
    "globals()['eval']('1+1')",
    "vars()['compile']('x=1', '', 'exec')",
])
def test_bypass_patterns_are_blocked(code):
    """Regression: every previously-reported scanner bypass must now be blocked."""
    assert scan_code(code).blocked


@pytest.mark.parametrize("code", [
    # Path used for read-only traversal stays allowed.
    "from pathlib import Path\np = Path('data')\nfiles = list(p.iterdir())",
    "from pathlib import Path\np = Path('data')\nprint(p.exists())",
    # read-only open stays allowed.
    "open('x', 'r')",
    "open('x', mode='r')",
    # benign attribute chains that share fragments of dangerous names.
    "import os\nprint(os.path.join('a', 'b'))",
    "os.getcwd()",
    "smoothed = data.attrs.get('environ')",
])
def test_read_only_path_and_open_are_allowed(code):
    assert scan_code(code).is_safe


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

