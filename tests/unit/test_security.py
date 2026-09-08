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
    # Capability domains: process/native/deserialization/reflection/star-import
    # are blocked uniformly by module, not by method name.
    "from os import *\nsystem('id')",
    "__builtins__.exec('x=1')",
    "import ctypes\nctypes.CDLL(None).system('id')",
    "globals()['os'].system('id')",
    "import pickle\npickle.loads(x)",
    "import joblib\njoblib.load(path)",
    "import subprocess\nsubprocess.run(['ls'])",
    "import importlib\nimportlib.import_module('os')",
])
def test_capability_domains_block_outside_sandbox(code):
    result = scan_code(code)
    assert result.blocked, result.to_dict()


@pytest.mark.parametrize("code", [
    "os.getcwd()",
    "os.path.join('a', 'b')",
    "import numpy as np\nnp.arange(4)",
    "import xarray as xr\nxr.DataArray([1, 2, 3])",
])
def test_capability_domains_allow_scientific_and_readonly(code):
    result = scan_code(code)
    assert result.is_safe, result.to_dict()
    assert result.requires_explicit_consent == []


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
    # indirect fetch of system / file calls (P0).
    "import os\ngetattr(os, 'system')('id')",
    "import os\nf = os.system\nf('id')",
    "import builtins\ngetattr(builtins, 'open')('/tmp/x', 'w')",
])
def test_bypass_patterns_are_blocked(code):
    """Regression: every previously-reported scanner bypass must now be blocked."""
    assert scan_code(code).blocked


@pytest.mark.parametrize("code, rule", [
    # Container/__dict__ calls and reflection chains (IND002 / REF001).
    ("import os\nos.__dict__['system']('id')", "IND002"),
    ("import builtins\nbuiltins.__dict__['exec']('import os')", "IND002"),
    ("import subprocess\nd = {}\nd['p'] = subprocess.run\nd['p'](['id'])", "IND002"),
    ("import os\nd = os.__dict__\nd['system']('id')", "IND002"),
    ("().__class__.__base__.__subclasses__()", "REF001"),
    ("''.__class__.__mro__[-1].__subclasses__()[0]('/tmp/x', 'w')", "REF001"),
    # Module/class attribute smuggling and dynamic attribute writes (SMUG*).
    ("import os\nos.run_id = os.system\nos.run_id('id')", "SMUG001"),
    ("import os\nclass C:\n    pass\nC.get = os.system\nC.get('id')", "SMUG001"),
    ("import os\nsetattr(os, 'run_id', os.system)", "SMUG002"),
    ("import builtins\ndelattr(builtins, 'open')", "SMUG002"),
    # Dynamic getattr with a computed attribute name (REF003).
    ("import os\ngetattr(os, 'sy' + 'stem')('id')", "REF003"),
    ("import os\nf = getattr(os, 'sy' + 'stem')\nf('id')", "REF003"),
    # Deserialization and raw-descriptor writes (DES001 / FILE003).
    ("import pandas as pd, io\npd.read_pickle(io.BytesIO(b'cos'))", "DES001"),
    ("import os\nfd = os.open('/tmp/x', os.O_WRONLY | os.O_CREAT)\nos.write(fd, b'e')", "FILE003"),
    # Environment mutation through aliases and popitem (ENV001).
    ("import os\nos.environ.popitem()", "ENV001"),
    ("import os\nenv = os.environ\nenv |= {'A': 'B'}", "ENV001"),
])
def test_container_reflection_and_smuggling_bypasses_are_blocked(code, rule):
    """Red-team regression: the 2026 review bypass set must stay hard-blocked."""
    result = scan_code(code)
    assert result.blocked, result.to_dict()
    assert any(issue.rule_id == rule for issue in result.issues), result.to_dict()


@pytest.mark.parametrize("code, rule", [
    ("from pathlib import Path\np = Path('x')\np.write_text('bad')", "SYS001"),
    ("import pathlib as pl\np = pl.Path('x')\np.write_bytes(b'bad')", "SYS001"),
    ("from pathlib import Path as P\np = P('x')\nq = p\nq.unlink()", "SYS001"),
    ("from pathlib import Path\np: Path = Path('x')\np.rename('y')", "SYS001"),
    ("from pathlib import Path\np = q = Path('x')\nq.rmdir()", "SYS001"),
    ("from pathlib import Path\np, n = Path('x'), 1\np.replace('y')", "SYS001"),
    ("from pathlib import Path\np = Path('x') / 'y'\np.write_text('bad')", "SYS001"),
    ("from pathlib import Path\np = Path('x').resolve()\np.write_bytes(b'bad')", "SYS001"),
    ("from pathlib import Path\np = Path.cwd() / 'x'\np.write_text('bad')", "SYS001"),
    ("from pathlib import Path\np = Path('x')\nf = p.unlink\nf()", "SYS001"),
    ("from pathlib import Path\np = Path('x')\nf = getattr(p, 'unlink')\nf()", "SYS001"),
    ("from pathlib import Path\n(p := Path('x')).unlink()", "SYS001"),
    ("from pathlib import Path\np = Path('x')\np.open('w')", "FILE001"),
    ("from pathlib import Path\np = Path('x')\np.open(mode='a')", "FILE001"),
    ("from pathlib import Path\nPath('x').open('r+')", "FILE001"),
    ("import builtins\nbuiltins.open('x', 'w')", "FILE001"),
    ("import builtins as b\nb.open('x', mode='a')", "FILE001"),
    ("from builtins import open as op\nop('x', 'x')", "FILE001"),
    ("import io\nio.open('x', mode='wb')", "FILE001"),
    ("from io import open as op\nf = op\nf('x', 'r+')", "FILE001"),
    ("from os import environ\nenviron['A'] = 'B'", "ENV001"),
    ("from os import environ as env\nenv.update({'A': 'B'})", "ENV001"),
    ("from os import environ as env\ndel env['A']", "ENV001"),
    ("from os import environ as env\nenv['A'] += 'B'", "ENV001"),
    ("import os\nenv = os.environ\nenv.clear()", "ENV001"),
    ("from os import environ\nmutate = environ.update\nmutate({'A': 'B'})", "ENV001"),
    # Later rebinding must not disguise a dangerous call that already occurred.
    ("import os\nf = os.system\nf('id')\nf = print", "SYS001"),
    ("import os as o\no.system('id')\no = None", "SYS001"),
    ("import os\nf = os.system\ndef harmless():\n    f = print\nf('id')", "SYS001"),
    ("import os\nf = os.system\nclass Local:\n    f = print\nf('id')", "SYS001"),
    ("import os\nf = os.system\nf, g = print, f\ng('id')", "SYS001"),
    ("import os\nos = f = os.system\nf('id')", "SYS001"),
    ("import os as o\ndef dangerous(o=o):\n    o.system('id')\ndangerous()", "SYS001"),
    ("import os as o\ndef dangerous(*, o=o):\n    o.system('id')\ndangerous()", "SYS001"),
])
def test_object_and_module_aliases_cannot_bypass_scanner(code, rule):
    result = scan_code(code)
    assert result.blocked
    assert any(issue.rule_id == rule for issue in result.issues), result.to_dict()
    assert result.syntax_error is None


@pytest.mark.parametrize("code", [
    "import os\nf = os.system\nf = print\nf('hello')",
    "import os\nf = os.system\ndef harmless(f):\n    f('hello')",
    "from pathlib import Path\np = Path('x')\nq = p\nq.read_text()",
    "from pathlib import Path\np = Path('x')\np.open()",
    "from pathlib import Path\np = Path('x')\np.open('rb')",
    "import builtins\nbuiltins.open('x', 'r')",
    "from io import open as op\nop('x', mode='rb')",
    "from os import environ as env\nprint(env.get('A'))",
    "from os import environ\nprint(environ['A'])",
    "from os import environ as env\nenv = {}\nenv['A'] = 'B'",
    "from os import environ as env\ndel env",
    "result = data.attrs.get('units')\nselected = data.sel(eV=slice(-1, 0))",
])
def test_readonly_and_local_rebindings_remain_allowed(code):
    result = scan_code(code)
    assert result.is_safe, result.to_dict()
    assert result.requires_explicit_consent == []


@pytest.mark.parametrize("code", [
    "import requests\nsession = requests.Session()\nsession.post('https://example.com', data=data)",
    "from requests import Session as S\ns = S()\nclient = s\nclient.get('https://example.com')",
    "import httpx\nc = httpx.Client()\nc.post('https://example.com', json=data)",
    "from httpx import AsyncClient as C\nc = C()\nc.get('https://example.com')",
    "import requests\nrequests.Session().post('https://example.com', data=data)",
    "import requests\nwith requests.Session() as s:\n    s.post('https://example.com', data=data)",
    "import httpx\nasync def send():\n    async with httpx.AsyncClient() as c:\n        await c.post('https://example.com', json=data)",
])
def test_network_client_aliases_require_explicit_consent(code):
    result = scan_code(code)
    assert result.is_safe
    assert any(issue.rule_id == "NET001" for issue in result.requires_explicit_consent)


@pytest.mark.parametrize("code", [
    "from pathlib import Path\np = Path('x')\np.write_text('bad')",
    "import builtins\nbuiltins.open('x', 'w')",
    "from os import environ\nenviron['A'] = 'B'",
])
def test_alias_bypasses_are_rejected_before_kernel_execution(code):
    """Check the real backend gate without executing any of the unsafe source."""
    from unittest.mock import Mock

    from peaksMCP.server.jupyter_peaks.backend import (
        SharedState,
        UnsafeNotebookBackend,
    )

    state = SharedState(Mock(user_ns={}))
    state.bridge = Mock()
    state.bridge.request.return_value = {"id": "cell-1", "source": code}
    consent, audit = Mock(), Mock()
    notebook = UnsafeNotebookBackend(state, consent, audit)

    with pytest.raises(PermissionError):
        notebook.execute_code(code)

    state.bridge.request.assert_not_called()
    consent.request.assert_not_called()
    audit.write.assert_called_once()
    assert audit.write.call_args.args[1] == "blocked"


@pytest.mark.parametrize("code", [
    # File writes / network egress require explicit consent (never hard-block).
    "import numpy as np\nnp.save('/tmp/x.npy', data)",
    "import pandas as pd\ndf.to_csv('out.csv')",
    "import json\njson.dump(data, f)",
    "import requests\nrequests.post('http://example.com', data=data)",
    "import requests\nrequests.get('http://example.com')",
    "import httpx\nhttpx.post('http://example.com')",
])
def test_file_write_and_network_require_explicit_consent(code):
    result = scan_code(code)
    assert result.is_safe  # never a hard block
    assert any(issue.rule_id in {"SAVE002", "NET001"} for issue in result.requires_explicit_consent)


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
    "import matplotlib.pyplot as plt\nplt.savefig('out.png')",
    "import matplotlib.pyplot as plt\nfig, ax = plt.subplots()\nfig.savefig('out.png')",
    "import matplotlib.pyplot as plt\nplt.figure().savefig('out.pdf')",
])
def test_savefig_requires_explicit_consent(code):
    """savefig is never automatic: it always lands in requires_explicit_consent
    so the user sees the exact cell and approves before a figure is written."""
    result = scan_code(code)
    assert not result.blocked
    assert any(issue.rule_id == "SAVE001" for issue in result.requires_explicit_consent)


@pytest.mark.parametrize("code", [
    # Non-figure uses must not be flagged as savefig consent.
    "print('savefig is a word')",
    "import os\nprint(os.path.exists('out.png'))",
    "import matplotlib.pyplot as plt\nplt.show()",
])
def test_savefig_not_flagged_for_benign_text(code):
    assert scan_code(code).requires_explicit_consent == []


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


def test_notebook_is_append_only_no_delete_or_reorder():
    """The mutation surface is strictly append-only: notebook_delete_cell was
    removed, so an existing cell can never be removed or overwritten by the
    agent (its full work history is preserved top-to-bottom)."""
    from unittest.mock import Mock

    from peaksMCP.server.jupyter_peaks.backend import (
        SharedState,
        UnsafeNotebookBackend,
    )

    state = SharedState(Mock(user_ns={}))
    state.require_consent = False
    state.bridge = Mock()
    state.bridge.request.return_value = {"ok": True}
    notebook = UnsafeNotebookBackend(state, ConsentManager(), AuditLogger("/tmp/t.jsonl"))

    assert not hasattr(notebook, "delete_cell")
    # add_cell is append-only: no position parameter, it always lands at the end.
    import inspect

    assert "position" not in inspect.signature(notebook.add_cell).parameters


def test_execute_code_requires_explicit_consent_for_network_when_consent_enabled():
    """Explicit-consent findings are never bypassed while consent is enabled."""
    from unittest.mock import Mock

    from peaksMCP.server.jupyter_peaks.backend import (
        SharedState,
        UnsafeNotebookBackend,
    )

    state = SharedState(Mock(user_ns={}))
    state.require_consent = True
    state.bridge = Mock()
    consent, audit = Mock(), Mock()
    consent.request.return_value = False
    notebook = UnsafeNotebookBackend(state, consent, audit)

    with pytest.raises(PermissionError):
        notebook.execute_code("import requests\nrequests.post('https://example.com')")

    consent.request.assert_called_once()
    state.bridge.request.assert_not_called()


@pytest.mark.parametrize(
    "code",
    [
        "import os\nf = os.__dict__['system']\nf('echo blocked-by-consent')",
        "import os\nf = getattr(os, 'sys' + 'tem')\nf('echo blocked-by-consent')",
        "from os import system\nlist(map(system, ['echo blocked-by-consent']))",
        "f = __builtins__.__dict__['exec']\nf('x = 1')",
    ],
)
def test_consent_is_hard_boundary_for_unrecognised_python(code):
    """Scanner blind spots must still stop at the user-consent boundary."""
    from unittest.mock import Mock

    from peaksMCP.server.jupyter_peaks.backend import (
        SharedState,
        UnsafeNotebookBackend,
    )

    state = SharedState(Mock(user_ns={}))
    # Consent must be a hard boundary when it is enabled (the profile switch
    # ``mcp.require_consent`` re-enables it; the default is False).
    state.require_consent = True
    state.bridge = Mock()
    state.bridge.request.return_value = {"id": "cell-1", "source": code}
    consent, audit = Mock(), Mock()
    consent.request.return_value = False
    notebook = UnsafeNotebookBackend(state, consent, audit)

    # The code must stop before the kernel: either the scanner now hard-blocks
    # the pattern itself, or the consent boundary rejects it.  Never both a
    # kernel request and a silent pass.
    with pytest.raises(PermissionError):
        notebook.execute_code(code)
    if scan_code(code).blocked:
        consent.request.assert_not_called()
    else:
        consent.request.assert_called_once()
    state.bridge.request.assert_not_called()


def test_audit_is_jsonl_and_private(tmp_path):
    path = tmp_path / "audit.log"
    AuditLogger(path).write("tool", "approved", {"x": 1})
    assert json.loads(path.read_text())["tool"] == "tool"
    assert path.stat().st_mode & 0o777 == 0o600


def test_write_without_exploration_runs_but_invented_apis_still_block(tmp_path):
    """The select-then-run gate was removed: write_with_api_check no longer
    requires prior peaks_search_api/peaks_get_api calls.  The API check itself
    still hard-blocks invented/typo'd Peaks APIs."""
    from unittest.mock import Mock

    from peaksMCP.discovery.index import build_index
    from peaksMCP.server.jupyter_peaks.backend import (
        SharedState,
        UnsafeNotebookBackend,
    )
    from peaksMCP.server.jupyter_peaks.security import AuditLogger, ConsentManager

    state = SharedState(Mock(user_ns={}))
    state.require_consent = False
    state.api_index = build_index()
    state.bridge = Mock()
    state.bridge.request.return_value = {"ok": True}
    nb = UnsafeNotebookBackend(state, ConsentManager(), AuditLogger(tmp_path / "t.jsonl"))

    # No exploration at all: a verified call runs, an invented API blocks.
    assert not nb.write_with_api_check("da.k_convert(quiet=True)", timeout=5).get("blocked")
    blocked = nb.write_with_api_check("da.correct_EF()", timeout=5)
    assert blocked["blocked"] is True
    assert "correct_EF" in str(blocked.get("unknown_refs"))


def test_unknown_api_first_advisory_then_same_name_hard(tmp_path):
    """First unknown occurrence is advisory; repeating the same unproven name
    is a hard refusal until the model proves it with peaks_get_api."""
    from unittest.mock import Mock

    from peaksMCP.discovery.index import build_index
    from peaksMCP.server.jupyter_peaks.backend import (
        SharedState,
        UnsafeNotebookBackend,
    )
    from peaksMCP.server.jupyter_peaks.security import AuditLogger, ConsentManager

    state = SharedState(Mock(user_ns={}))
    state.require_consent = False
    state.api_index = build_index()
    state.bridge = Mock()
    state.bridge.request.return_value = {"ok": True}
    nb = UnsafeNotebookBackend(state, ConsentManager(), AuditLogger(tmp_path / "t.jsonl"))

    code = "correct_EF()"
    first = nb.write_with_api_check(code, timeout=5)
    assert first["blocked"] is True and first.get("requires_search") is True
    assert state.unknown_api_attempts["correct_EF"] == 1
    assert state.bridge.request.call_count == 0

    second = nb.write_with_api_check(code, timeout=5)
    assert second["blocked"] is True
    assert second.get("hard_refusal") is True
    assert state.unknown_api_attempts["correct_EF"] == 2
    assert state.bridge.request.call_count == 0


def test_get_api_proof_unlocks_unknown_name(tmp_path):
    """A successful peaks_get_api for the canonical API unlocks its alias."""
    from unittest.mock import Mock

    from peaksMCP.discovery.index import build_index
    from peaksMCP.server.jupyter_peaks.backend import (
        SharedState,
        UnsafeNotebookBackend,
    )
    from peaksMCP.server.jupyter_peaks.core.tools import _record_verified_api
    from peaksMCP.server.jupyter_peaks.security import AuditLogger, ConsentManager

    state = SharedState(Mock(user_ns={}))
    state.require_consent = False
    state.api_index = build_index()
    state.bridge = Mock()
    state.bridge.request.return_value = {"ok": True}
    nb = UnsafeNotebookBackend(state, ConsentManager(), AuditLogger(tmp_path / "t.jsonl"))

    # A python-callable alias no longer ships, so add one in-test to exercise
    # the unlock path: an unverified name is blocked until peaks_get_api proof.
    entry = next(e for e in state.api_index.entries if e["name"] == "show_mapping_slice")
    aliases = list(entry.get("aliases") or [])
    entry["aliases"] = aliases
    alias_name = "mapping_slice_alias"
    if alias_name not in aliases:
        aliases.append(alias_name)

    first = nb.write_with_api_check(f"{alias_name}(da, dim='eV')", timeout=5)
    assert first["blocked"] is True

    _record_verified_api(state, entry)  # what a successful peaks_get_api records

    again = nb.write_with_api_check(f"{alias_name}(da, dim='eV')", timeout=5)
    assert not again.get("blocked")
    assert again.get("ok") is True
    verified = again.get("api_check", {}).get("verified_peaks_apis", [])
    assert any(item["name"] == alias_name for item in verified)


def test_savefig_requires_user_approval(tmp_path):
    """A savefig cell pauses for explicit user approval (SAVE001 consent); it is
    never executed without it, and executes once the user approves."""
    from unittest.mock import Mock

    from peaksMCP.discovery.index import build_index
    from peaksMCP.server.jupyter_peaks.backend import (
        SharedState,
        UnsafeNotebookBackend,
    )
    from peaksMCP.server.jupyter_peaks.security import AuditLogger, ConsentManager

    code = "import matplotlib.pyplot as plt\nplt.savefig('x.png')"
    state = SharedState(Mock(user_ns={}))
    state.require_consent = False  # the save gate must hold regardless of the switch
    state.api_index = build_index()
    state.bridge = Mock()
    state.bridge.request.return_value = {"ok": True, "outputs": [], "id": "c1"}
    nb = UnsafeNotebookBackend(
        state, ConsentManager(callback=lambda _op, _details: False), AuditLogger(tmp_path / "t.jsonl")
    )

    # Denied: the user rejects the save cell -> it never reaches the kernel.
    with pytest.raises(PermissionError, match="did not approve"):
        nb.write_with_api_check(code, timeout=5)
    state.bridge.request.assert_not_called()

    # Approved: the cell executes and the output is returned for normalisation.
    consent = ConsentManager(callback=lambda _op, _details: True)
    nb = UnsafeNotebookBackend(state, consent, AuditLogger(tmp_path / "t2.jsonl"))
    result = nb.write_with_api_check(code, timeout=5)
    assert not result.get("blocked")
    assert any(op[0][0] == "execute_code" for op in state.bridge.request.call_args_list)



def test_write_with_api_check_classifies_generic_and_verified_calls(tmp_path):
    from unittest.mock import Mock

    from peaksMCP.discovery.index import build_index
    from peaksMCP.server.jupyter_peaks.backend import (
        SharedState,
        UnsafeNotebookBackend,
    )
    from peaksMCP.server.jupyter_peaks.security import AuditLogger, ConsentManager

    state = SharedState(Mock(user_ns={}))
    state.require_consent = False
    state.api_index = build_index()
    state.bridge = Mock()
    state.bridge.request.return_value = {"ok": True}
    nb = UnsafeNotebookBackend(state, ConsentManager(), AuditLogger(tmp_path / "t.jsonl"))

    # Builtins and generic-library calls (including module-member chains like
    # np.linalg.svd / scipy.signal.savgol_filter) never block.
    for code in (
        'print("hi")',
        "len(data.eV)",
        "range(10)",
        "np.linalg.svd(x)",
        "np.fft.fft2(x)",
        "scipy.signal.savgol_filter(x, 5, 2)",
        'xr.concat([a, b], dim="t")',
        "json.dumps(x)",
    ):
        result = nb.write_with_api_check(code, timeout=5)
        assert not result.get("blocked"), (code, result)

    # Peaks APIs are reported as verified regardless of calling convention.
    verified = nb.write_with_api_check("fit_gold(data)", timeout=5)
    assert not verified.get("blocked")
    assert "fit_gold" in [v["name"] for v in verified["api_check"]["verified_peaks_apis"]]
    verified = nb.write_with_api_check("da.k_convert(quiet=True)", timeout=5)
    assert "k_convert" in [v["name"] for v in verified["api_check"]["verified_peaks_apis"]]

    # Invented / typo'd Peaks APIs are hard-blocked after unlocking.
    blocked = nb.write_with_api_check("da.correct_EF()", timeout=5)
    assert blocked["blocked"] is True
    assert "correct_EF" in str(blocked.get("unknown_refs"))


def test_write_with_api_check_receiver_aware_and_scope_aware(monkeypatch, tmp_path):
    import inspect
    from unittest.mock import Mock

    from peaksMCP.discovery.index import build_index
    from peaksMCP.server.jupyter_peaks.backend import (
        SharedState,
        UnsafeNotebookBackend,
    )
    from peaksMCP.server.jupyter_peaks.security import AuditLogger, ConsentManager

    def make_backend(user_ns=None):
        state = SharedState(Mock(user_ns=user_ns or {}))
        state.require_consent = False
        state.api_index = build_index()
        state.bridge = Mock()
        state.bridge.request.return_value = {"ok": True}
        return UnsafeNotebookBackend(state, ConsentManager(), AuditLogger(tmp_path / "t.jsonl"))

    nb = make_backend()

    # 1) strict is removed from the public signature (no bypass path).
    assert "strict" not in inspect.signature(nb.write_with_api_check).parameters

    # 2) import aliases and constructor/variable origins are generic.
    for code in (
        "import numpy as n\nn.linalg.svd(x)",
        "import matplotlib.pyplot as plt\nfig = plt.figure()\nfig.show()",
        "data = load('/x.nc')\ndata.k_convert(quiet=True)",
        "import peaks as pks\ndata = pks.load('/x.nc')\ndata.k_convert(quiet=True)",
    ):
        assert not nb.write_with_api_check(code, timeout=5).get("blocked"), code

    # 4) external-module members never match Peaks APIs by name.
    assert not nb.write_with_api_check("np.linalg.svd(x)", timeout=5).get("blocked")

    # 6) mixed None/str receivers sort by source position without a TypeError.
    assert not nb.write_with_api_check("da[i].mean(); da.k_convert()", timeout=5).get("blocked")

    # 5) fail-closed: invented/typo'd APIs on unprovable receivers are blocked.
    blocked = nb.write_with_api_check("make().correct_EF()", timeout=5)
    assert blocked.get("blocked") and "correct_EF" in str(blocked.get("unknown_refs"))
    blocked = nb.write_with_api_check("da[i].correct_EF()", timeout=5)
    assert blocked.get("blocked") and "correct_EF" in str(blocked.get("unknown_refs"))

    # 3) scope matching: a known DataArray receiver only accepts dataarray-scope
    # Peaks APIs; a top_level API (plot_bz) on it is an unverifiable reference.
    import numpy as np
    import xarray as xr

    nb2 = make_backend({"da": xr.DataArray(np.zeros((4, 4)), dims=("eV", "theta_par"))})
    assert not nb2.write_with_api_check("da.k_convert(quiet=True)", timeout=5).get("blocked")
    blocked = nb2.write_with_api_check("da.plot_bz(...)", timeout=5)
    assert blocked.get("blocked") and "plot_bz" in str(blocked.get("unknown_refs"))
    assert not nb2.write_with_api_check("da[i].mean()", timeout=5).get("blocked")

    # 7) a stale index is hot-rebuilt in-kernel instead of erroring.
    stale = make_backend()
    stale.state.api_index.fingerprint = "changed-after-build"
    result = stale.write_with_api_check("da.k_convert()", timeout=5)
    assert not result.get("blocked")
    assert stale.state.api_index.is_stale() is False
