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
    "open('scan.nc')",
    "open('scan.nc', 'rb')",
    "import builtins\nbuiltins.open('scan.nc', mode='r')",
    "from io import open as op\nop('scan.nc', mode='rb')",
    "from pathlib import Path\np = Path('scan.nc')\nq = p\nq.read_text()",
    "from pathlib import Path\nPath('scan.nc').read_bytes()",
    "from pathlib import Path\nPath('scan.nc').open()",
    "from pathlib import Path\np = Path('data')\nlist(p.iterdir())",
    "from pathlib import Path\nlist(Path('data').glob('*.nc'))",
    "from pathlib import Path\nlist(Path('data').rglob('*.nc'))",
    "import os\nlist(os.walk('data'))",
    "from os import listdir as ls\nls('data')",
    "import os as operating_system\nlist(operating_system.scandir('data'))",
    "import pandas as pd\npd.read_csv('datasheet.csv')",
    "from pandas import read_json as load_json\nload_json('metadata.json')",
    "import xarray as xr\nxr.open_dataset('scan.nc')",
    "from xarray import open_dataarray as oda\noda('scan.nc')",
    "import numpy as np\nnp.load('scan.npy')",
    "import numpy\nnumpy.loadtxt('scan.txt')",
    "from numpy import genfromtxt as load_table\nload_table('scan.csv')",
    "import numpy as np\nnp.fromfile('scan.bin')",
    "import numpy as np\nnp.memmap('scan.bin')",
])
def test_direct_filesystem_reads_and_listings_are_blocked(code):
    result = scan_code(code)

    assert result.blocked, result.to_dict()
    assert any(issue.rule_id == "FILE004" for issue in result.issues), result.to_dict()


@pytest.mark.parametrize("code", [
    "import peaks\nexperiment = peaks.load_experiment('/data')",
    "from peaks import load_experiment as load\nexperiment = load('/data')",
])
def test_peaks_load_experiment_is_not_treated_as_a_direct_file_read(code):
    result = scan_code(code)

    assert result.is_safe, result.to_dict()
    assert not any(issue.rule_id == "FILE004" for issue in result.issues)


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
    "import xarray as xr\nxr.open_dataset('scan.nc')",
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
    "df.to_json('out.json')",
    "df.to_json(path_or_buf='out.json')",
    "df.to_excel('out.xlsx')",
    "df.to_parquet(path='out.parquet')",
    "df.to_hdf('out.h5', key='data')",
    "df.to_feather('out.feather')",
    "df.to_pickle('out.pkl')",
    "dataset.to_netcdf(path='out.nc')",
    "dataset.to_zarr('out.zarr')",
])
def test_dataframe_and_xarray_output_targets_require_explicit_consent(code):
    result = scan_code(code)
    assert result.is_safe
    assert any(
        issue.rule_id == "SAVE002"
        for issue in result.requires_explicit_consent
    ), result.to_dict()


@pytest.mark.parametrize("code", [
    "payload = df.to_json()",
    "payload = df.to_json(path_or_buf=None)",
    "payload = df.to_csv()",
    "payload = df.to_parquet(path=None)",
    "payload = dataset.to_netcdf()",
])
def test_in_memory_serialization_without_output_target_is_allowed(code):
    result = scan_code(code)
    assert result.is_safe, result.to_dict()
    assert result.requires_explicit_consent == []


@pytest.mark.parametrize("code", [
    # Path metadata checks do not expose file contents or directory listings.
    "from pathlib import Path\np = Path('data')\nprint(p.exists())",
    # benign attribute chains that share fragments of dangerous names.
    "import os\nprint(os.path.join('a', 'b'))",
    "os.getcwd()",
    "smoothed = data.attrs.get('environ')",
])
def test_non_reading_path_inspection_is_allowed(code):
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
    """No frontend means nobody can be asked: the answer is ``None`` (no
    approver reachable), never ``False`` (which would claim a human refused).
    Callers still fail closed - a falsy answer refuses the operation."""
    assert ConsentManager().request("delete", {}) is None


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


def test_network_cell_asks_for_consent_with_the_master_switch_off(tmp_path):
    """NET001 always needs approval, and the default master switch must not
    crash that path (``require_consent=False`` used to evaluate ``issue.id`` on
    a dataclass that only has ``rule_id`` -> AttributeError instead of a card)."""
    from unittest.mock import Mock

    from peaksMCP.server.jupyter_peaks.backend import (
        SharedState,
        UnsafeNotebookBackend,
    )

    state = SharedState(Mock(user_ns={}))
    state.require_consent = False
    state.bridge = Mock()
    state.bridge.request.return_value = {"ok": True}
    consent, audit = Mock(), Mock()
    consent.request.return_value = False  # the human declines the card
    notebook = UnsafeNotebookBackend(state, consent, audit)

    with pytest.raises(PermissionError):
        notebook.execute_code("import requests\nrequests.post('https://example.com')")

    # The consent card WAS requested: the guard asked, the human said no.
    consent.request.assert_called_once()
    assert consent.request.call_args.args[0] == "run_cell"


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


def test_write_without_proof_blocks_exact_peaks_calls_until_get(tmp_path):
    """Canonical proof is a HARD gate: an exact-name Peaks call without a
    successful get this session is blocked (advisory).  After the
    canonical id lands in the ledger the call runs; invented/typo'd APIs
    always block."""
    from unittest.mock import Mock

    import numpy as np
    import xarray as xr

    from peaksMCP.discovery.index import build_index
    from peaksMCP.server.jupyter_peaks.backend import (
        SharedState,
        UnsafeNotebookBackend,
    )
    from peaksMCP.server.jupyter_peaks.security import AuditLogger, ConsentManager

    state = SharedState(Mock(user_ns={"da": xr.DataArray(np.zeros((2, 2)), dims=("eV", "theta_par"))}))
    state.require_consent = False
    state.api_index = build_index()
    state.bridge = Mock()
    state.bridge.request.return_value = {"ok": True}
    nb = UnsafeNotebookBackend(state, ConsentManager(), AuditLogger(tmp_path / "t.jsonl"))

    # No exploration at all: an exact Peaks call is blocked until proven.
    first = nb.write_with_api_check("da.k_convert(quiet=True)", timeout=5)
    assert first["blocked"] is True and first.get("requires_search") is True
    assert "k_convert" in str(first.get("unknown_refs"))
    assert state.bridge.request.call_count == 0
    # Invented API stays blocked even after proof of an unrelated name.
    k_convert_id = _prove(state, state.api_index, "k_convert", "dataarray")
    blocked = nb.write_with_api_check("da.correct_EF()", timeout=5)
    assert blocked["blocked"] is True
    assert "correct_EF" in str(blocked.get("unknown_refs"))
    # The proven canonical call now runs.
    second = nb.write_with_api_check(
        "da.k_convert(quiet=True)", timeout=5, api_ids=[k_convert_id]
    )
    assert not second.get("blocked")


def test_cataloged_report_summary_is_generic_after_proven_pxt2nc(tmp_path):
    """The pxt2nc contract's bounded report display must pass the API gate."""
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
    pxt2nc_id = _prove(state, state.api_index, "pxt2nc", "top_level")
    notebook = UnsafeNotebookBackend(
        state, ConsentManager(), AuditLogger(tmp_path / "t.jsonl")
    )

    code = (
        "import peaks as pks\n"
        "first = pks.pxt2nc('/input')\n"
        "print(first.summary_line())"
    )
    result = notebook.write_with_api_check(code, timeout=5, api_ids=[pxt2nc_id])

    assert not result.get("blocked"), result
    assert result["api_check"]["unknown_refs"] == []
    assert "summary_line" in result["api_check"]["generic_refs"]


@pytest.mark.parametrize(
    "method_call",
    [
        "ax.set_aspect('equal')",
        "ax.tick_params(labelsize=8)",
        "fig.supxlabel('Momentum')",
        "fig.supylabel('Intensity')",
    ],
)
def test_matplotlib_formatters_are_generic(tmp_path, method_call):
    from unittest.mock import Mock

    from peaksMCP.discovery.index import build_index
    from peaksMCP.server.jupyter_peaks.backend import SharedState, UnsafeNotebookBackend
    from peaksMCP.server.jupyter_peaks.security import AuditLogger, ConsentManager

    state = SharedState(Mock(user_ns={}))
    state.require_consent = False
    state.api_index = build_index()
    state.bridge = Mock()
    state.bridge.request.return_value = {"ok": True}
    notebook = UnsafeNotebookBackend(
        state, ConsentManager(), AuditLogger(tmp_path / "t.jsonl")
    )

    result = notebook.write_with_api_check(method_call, timeout=5)

    assert not result.get("blocked"), result
    assert method_call.split(".", 1)[1].split("(", 1)[0] in result["api_check"]["generic_refs"]


def test_xarray_null_checks_are_generic(tmp_path):
    from unittest.mock import Mock

    from peaksMCP.discovery.index import build_index
    from peaksMCP.server.jupyter_peaks.backend import SharedState, UnsafeNotebookBackend
    from peaksMCP.server.jupyter_peaks.security import AuditLogger, ConsentManager

    state = SharedState(Mock(user_ns={}))
    state.require_consent = False
    state.api_index = build_index()
    state.bridge = Mock()
    state.bridge.request.return_value = {"ok": True}
    notebook = UnsafeNotebookBackend(
        state, ConsentManager(), AuditLogger(tmp_path / "t.jsonl")
    )

    result = notebook.write_with_api_check(
        "valid = data.notnull()\nmissing = data.isnull()", timeout=5
    )

    assert not result.get("blocked"), result
    assert result["api_check"]["unknown_refs"] == []
    assert {"notnull", "isnull"}.issubset(result["api_check"]["generic_refs"])


def test_unknown_api_first_advisory_then_same_name_hard(tmp_path):
    """First unknown occurrence is advisory; repeating the same unproven name
    is a hard refusal until the model proves it with get."""
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


def _prove(state, index, name, scope=None):
    """Simulate a successful get: record canonical proof ids."""
    from peaksMCP.server.jupyter_peaks.core.tools import _record_verified_api

    found = False
    for entry in index.entries:
        if entry["name"] != name:
            continue
        if scope is not None and entry["scope"] != scope:
            continue
        _record_verified_api(state, entry)
        found = True
        if scope is not None:
            return str(entry["id"])
    if not found:
        raise AssertionError(f"no index entry {name!r} scope={scope!r}")
    return str(next(entry["id"] for entry in index.entries if entry["name"] == name))


def _prove_all(state, index, index_by_name):
    """Record every canonical proof a test relies on (get-first simulation)."""
    for name in index_by_name:
        _prove(state, index, name, index_by_name[name])


def test_get_api_proof_unlocks_only_canonical_name(tmp_path):
    """get proof unlocks the canonical executable name only.

    Natural-language aliases are search vocabulary, not Python identifiers:
    writing an alias as a callable must stay blocked even after the entry was
    fetched, until the model uses the real function name."""
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

    # A python-callable alias does not ship, so add one in-test: the alias must
    # never unlock, while the canonical name must.
    entry = next(e for e in state.api_index.entries if e["name"] == "fit_gold")
    alias_name = "gold_fit_alias"
    entry["aliases"] = list(entry.get("aliases") or []) + [alias_name]

    first = nb.write_with_api_check(f"{alias_name}(gold)", timeout=5)
    assert first["blocked"] is True

    _record_verified_api(state, entry)  # what a successful get records

    # The alias still does not unlock a Python symbol after proof.
    again = nb.write_with_api_check(
        f"{alias_name}(gold)", timeout=5, api_ids=[entry["id"]]
    )
    assert again["blocked"] is True

    # The canonical executable name is unlocked and runs.
    third = nb.write_with_api_check(
        "gold.fit_gold(plot=False)", timeout=5, api_ids=[entry["id"]]
    )
    assert not third.get("blocked")
    verified = third.get("api_check", {}).get("verified_peaks_apis", [])
    assert any(item["name"] == "fit_gold" for item in verified)


def test_savefig_is_hard_blocked_in_run_cell(tmp_path):
    """SAVE001 savefig is hard-blocked: run_cell is NEVER a persistence path
    (save_with_consent is the single persistence verb), so the cell is refused
    before any consent dialog and before any kernel execution - even with
    require_consent=False, and even when the user would have approved."""
    from unittest.mock import Mock

    from peaksMCP.discovery.index import build_index
    from peaksMCP.server.jupyter_peaks.backend import (
        SharedState,
        UnsafeNotebookBackend,
    )
    from peaksMCP.server.jupyter_peaks.security import AuditLogger, ConsentManager

    code = "import matplotlib.pyplot as plt\nplt.savefig('x.png')"
    state = SharedState(Mock(user_ns={}))
    state.require_consent = False  # persistence policy holds regardless
    state.api_index = build_index()
    state.bridge = Mock()
    state.bridge.request.return_value = {"ok": True, "outputs": [], "id": "c1"}
    nb = UnsafeNotebookBackend(
        state, ConsentManager(callback=lambda _op, _details: False), AuditLogger(tmp_path / "t.jsonl")
    )

    with pytest.raises(PermissionError, match="save_with_consent"):
        nb.write_with_api_check(code, timeout=5)
    # Never executed, and no consent dialog was raised for the write either.
    state.bridge.request.assert_not_called()


@pytest.mark.parametrize("code", [
    "df.to_json('out.json')",
    "df.to_excel('out.xlsx')",
    "df.to_parquet('out.parquet')",
    "dataset.to_zarr('out.zarr')",
])
def test_dataframe_and_xarray_writers_are_blocked_before_bridge(code, tmp_path):
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
    state.bridge.request.return_value = {"ok": True, "outputs": [], "id": "c1"}
    notebook = UnsafeNotebookBackend(
        state, ConsentManager(), AuditLogger(tmp_path / "t.jsonl")
    )

    with pytest.raises(PermissionError, match="save_with_consent"):
        notebook.write_with_api_check(code, timeout=5)
    state.bridge.request.assert_not_called()


def test_in_memory_json_serialization_can_reach_bridge(tmp_path):
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
    state.bridge.request.return_value = {"ok": True, "outputs": [], "id": "c1"}
    notebook = UnsafeNotebookBackend(
        state, ConsentManager(), AuditLogger(tmp_path / "t.jsonl")
    )

    result = notebook.write_with_api_check("payload = df.to_json()", timeout=5)
    assert not result.get("blocked"), result
    state.bridge.request.assert_called_once()


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

    # Exact-name Peaks calls need a canonical proof (get) before they run.
    pre = nb.write_with_api_check("da.k_convert(quiet=True)", timeout=5)
    assert pre.get("blocked"), "unproven exact API must not run"
    fit_gold_id = _prove(nb.state, nb.state.api_index, "fit_gold", "dataarray")
    k_convert_id = _prove(nb.state, nb.state.api_index, "k_convert", "dataarray")

    # Proven APIs are reported as verified regardless of calling convention.
    verified = nb.write_with_api_check(
        "fit_gold(data)", timeout=5, api_ids=[fit_gold_id]
    )
    assert not verified.get("blocked")
    assert "fit_gold" in [v["name"] for v in verified["api_check"]["verified_peaks_apis"]]
    verified = nb.write_with_api_check(
        "da.k_convert(quiet=True)", timeout=5, api_ids=[k_convert_id]
    )
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
        load_id = _prove(state, state.api_index, "load_experiment", "top_level")
        k_convert_id = _prove(state, state.api_index, "k_convert", "dataarray")
        backend = UnsafeNotebookBackend(
            state, ConsentManager(), AuditLogger(tmp_path / "t.jsonl")
        )
        return backend, load_id, k_convert_id

    nb, load_id, k_convert_id = make_backend()

    # 1) strict is removed from the public signature (no bypass path).
    assert "strict" not in inspect.signature(nb.write_with_api_check).parameters

    # 2) import aliases and constructor/variable origins are generic.
    for code in (
        "import numpy as n\nn.linalg.svd(x)",
        "import matplotlib.pyplot as plt\nfig = plt.figure()\nfig.show()",
        "data = load_experiment('/x.nc')\ndata.k_convert(quiet=True)",
        "import peaks as pks\ndata = pks.load_experiment('/x.nc')\ndata.k_convert(quiet=True)",
    ):
        ids = [load_id, k_convert_id] if "load_experiment" in code else None
        assert not nb.write_with_api_check(code, timeout=5, api_ids=ids).get("blocked"), code

    # 4) external-module members never match Peaks APIs by name.
    assert not nb.write_with_api_check("np.linalg.svd(x)", timeout=5).get("blocked")

    # 6) mixed None/str receivers sort by source position without a TypeError.
    assert not nb.write_with_api_check(
        "da[i].mean(); da.k_convert()", timeout=5, api_ids=[k_convert_id]
    ).get("blocked")

    # 5) fail-closed: invented/typo'd APIs on unprovable receivers are blocked.
    blocked = nb.write_with_api_check("make().correct_EF()", timeout=5)
    assert blocked.get("blocked") and "correct_EF" in str(blocked.get("unknown_refs"))
    blocked = nb.write_with_api_check("da[i].correct_EF()", timeout=5)
    assert blocked.get("blocked") and "correct_EF" in str(blocked.get("unknown_refs"))

    # 3) scope matching: a known DataArray receiver only accepts dataarray-scope
    # Peaks APIs; a top_level API (plot_bz) on it is an unverifiable reference.
    import numpy as np
    import xarray as xr

    nb2, _, nb2_k_convert_id = make_backend(
        {"da": xr.DataArray(np.zeros((4, 4)), dims=("eV", "theta_par"))}
    )
    assert not nb2.write_with_api_check(
        "da.k_convert(quiet=True)", timeout=5, api_ids=[nb2_k_convert_id]
    ).get("blocked")
    blocked = nb2.write_with_api_check("da.plot_bz(...)", timeout=5)
    assert blocked.get("blocked") and "plot_bz" in str(blocked.get("unknown_refs"))
    assert not nb2.write_with_api_check("da[i].mean()", timeout=5).get("blocked")

    # 7) a stale index is hot-rebuilt in-kernel, but an old proof cannot unlock
    # the refreshed contract until it is proven again.
    stale, _, stale_k_convert_id = make_backend()
    stale.state.api_index.fingerprint = "changed-after-build"
    result = stale.write_with_api_check(
        "da.k_convert()", timeout=5, api_ids=[stale_k_convert_id]
    )
    assert result.get("blocked") is True
    assert result["unproven_api_ids"] == [stale_k_convert_id]
    assert stale.state.verified_apis == {}
    assert stale.state.api_index.is_stale() is False
    refreshed_id = _prove(stale.state, stale.state.api_index, "k_convert", "dataarray")
    assert refreshed_id == stale_k_convert_id
    retry = stale.write_with_api_check(
        "da.k_convert()", timeout=5, api_ids=[refreshed_id]
    )
    assert not retry.get("blocked")


def test_unique_canonical_name_is_provable_on_an_untyped_receiver(tmp_path):
    """A name with exactly one canonical id is reachable from an untyped receiver.

    ``fit_gold`` is indexed once (as a DataArray method) and ``k_convert``
    likewise; neither has a module/top-level twin.  An agent that loaded its
    object through a facade (``gold = scans[stem]``) cannot be told to "get
    the canonical id for this call's scope" - there is only one id and no way
    to infer the call site's scope.  The proof requirement still applies (the
    id must have been fetched this session); what disappears is the demand for
    a twin that does not exist.
    """
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
    backend = UnsafeNotebookBackend(
        state, ConsentManager(), AuditLogger(tmp_path / "t.jsonl")
    )

    # The curated index exposes exactly one canonical id for each name.
    for name in ("fit_gold", "k_convert"):
        assert len([e for e in state.api_index.entries if e["name"] == name]) == 1
        rows = [m for m in state.api_index.search(name, "all", 5) if m.get("name") == name]
        assert len(rows) == 1

    # Unproven: still blocked, and the reply still points at the canonical id.
    blocked = backend.write_with_api_check("gold.fit_gold(plot=False)", timeout=5)
    assert blocked.get("blocked"), blocked
    assert "fit_gold" in str(blocked.get("unknown_refs"))

    # After get, the method form works on a plain variable of unknown type,
    # and so does the bare form of the one-id name.
    fit_gold_id = _prove(state, state.api_index, "fit_gold", "dataarray")
    k_convert_id = _prove(state, state.api_index, "k_convert", "dataarray")
    for code, api_id in (
        ("gold.fit_gold(plot=False)", fit_gold_id),
        ("k_convert(da, quiet=True)", k_convert_id),
        ("shifted.k_convert(quiet=True)", k_convert_id),
    ):
        result = backend.write_with_api_check(code, timeout=5, api_ids=[api_id])
        assert not result.get("blocked"), (code, result)

    # An inconclusive receiver stays fail-closed for names Peaks does not have.
    blocked = backend.write_with_api_check("make().correct_EF()", timeout=5)
    assert blocked.get("blocked"), blocked


def test_project_import_gate_accepts_all_legal_import_forms(tmp_path):
    """Legal peaks import shapes must all pass the API check: plain,
    parenthesised across lines, ``as`` renames, and module-alias calls."""
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
    load_id = _prove(state, state.api_index, "load_experiment", "top_level")
    k_convert_id = _prove(state, state.api_index, "k_convert", "dataarray")

    for code in (
        "from peaks import load_experiment\nload_experiment('data_netcdf')",
        "from peaks import (\n    load_experiment,\n)\nload_experiment('data_netcdf')",
        "from peaks import load_experiment as ld\nld('data_netcdf')",
        "import peaks as pks\npks.load_experiment('data_netcdf')",
        "from peaks import load_experiment\ndata = load_experiment('data_netcdf')\ndata.k_convert(quiet=True)",
    ):
        ids = [load_id, k_convert_id] if "k_convert" in code else [load_id]
        result = nb.write_with_api_check(code, timeout=5, api_ids=ids)
        assert not result.get("blocked"), (code, result)


def test_project_import_gate_rejects_ghost_exports_and_star_imports(tmp_path):
    """Names that are not real peaks/peaksMCP exports are refused before the
    kernel runs, and star imports from peaks/peaksMCP are always refused."""
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

    ghost = nb.write_with_api_check(
        "from peaksMCP.overrides import ghost_api\nghost_api('x')", timeout=5
    )
    assert ghost["blocked"] is True
    assert ghost.get("requires_search") is True
    assert "ghost_api" in ghost["message"]
    state.bridge.request.assert_not_called()

    for code in (
        "from peaksMCP.overrides import *\nload_data('x')",
        "from peaks import *\ndata = load('x.nc')",
    ):
        result = nb.write_with_api_check(code, timeout=5)
        assert result["blocked"] is True
        assert "Star import" in result["message"]


def test_scope_mismatch_hint_distinguishes_proven_name(tmp_path):
    """已 get 同名 API 但 scope 不匹配 → 提示语明确区分，而不是只说"从未验证"。"""
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

    # 只证明 dataarray scope 的 k_convert：未定型调用点（裸调用 / 无法推断
    # 接收者）接受该证明 —— search 只能给出这一条 id，找不到 module twin。
    k_convert_id = _prove(state, state.api_index, "k_convert", "dataarray")
    assert not nb.write_with_api_check(
        "k_convert(da, quiet=True)", timeout=5, api_ids=[k_convert_id]
    ).get("blocked")
    assert not nb.write_with_api_check(
        "shifted.k_convert(quiet=True)", timeout=5, api_ids=[k_convert_id]
    ).get("blocked")

    # 接收者 scope 已知时仍然严格：模块级 drift_correction 用在 DataArray 上。
    import numpy as np
    import xarray as xr

    typed = SharedState(Mock(user_ns={"da": xr.DataArray(np.zeros((4, 4)), dims=("eV", "kx"))}))
    typed.require_consent = False
    typed.api_index = build_index()
    typed.bridge = Mock()
    typed.bridge.request.return_value = {"ok": True}
    nb_typed = UnsafeNotebookBackend(
        typed, ConsentManager(), AuditLogger(tmp_path / "typed.jsonl")
    )
    drift_id = _prove(typed, typed.api_index, "drift_correction", "module")
    result = nb_typed.write_with_api_check(
        "da.drift_correction(...)", timeout=5, api_ids=[drift_id]
    )
    assert result.get("blocked") is True
    assert "不匹配" in result["message"] or "scope" in result["message"]


def test_run_cell_replies_carry_the_kernel_disposition(tmp_path):
    """Every run_cell reply - including refusals - says whether the kernel is
    busy, so a timeout can be triaged instead of guessed at."""
    from unittest.mock import Mock

    from peaksMCP.server.jupyter_peaks.backend import (
        SharedState,
        UnsafeNotebookBackend,
    )

    state = SharedState(Mock(user_ns={}))
    state.require_consent = False
    state.bridge = Mock()
    state.bridge.request.return_value = {"ok": True}
    notebook = UnsafeNotebookBackend(state, ConsentManager(), AuditLogger(str(tmp_path / "a.jsonl")))

    blocked = notebook.write_with_api_check("da.make_up_a_name()", timeout=5)
    assert blocked["blocked"] is True
    assert blocked["kernel_state"] == "idle"

    state.mark_busy()
    busy = notebook.write_with_api_check("da.another_made_up_name()", timeout=5)
    assert busy["kernel_state"] == "busy"
    assert busy["kernel_busy_s"] is not None
