# Test Guide

Use the repository's active Python interpreter for every suite. In the supported environment this is:

```bash
PY=/opt/homebrew/Caskroom/miniforge/base/envs/peaks/bin/python
$PY tools/test.py --list
```

## Test Layers

| Suite | Scope | External requirements |
|---|---|---|
| `quick` | All unit tests and mocked component integration; default development gate | None |
| `unit` | Everything under `tests/unit`, including grader and frontend contracts | Node for the frontend contract |
| `integration` | In-process MCP client/server component boundary | None |
| `realdata` | PXT conversion and Peaks processing; live-kernel checks opt in with `PEAKSMCP_LIVE_KERNEL=1` | Local BP260623 data |
| `benchmark` | Rubric/grader regressions plus golden and poisoned grader self-tests | Local reference data used by the self-test |
| `grader` | Golden and poisoned grader self-tests only | Local reference data used by the self-test |
| `e2e` | Real Dashboard -> JupyterLab -> MCP path in Chrome | Local data, Chrome, free service ports |
| `acceptance` | Managed campaign with the deterministic scripted agent | Local data, Chrome, free service ports |
| `check` | `quality + quick + grader`; required repository gate without duplicate pytest work | Grader prerequisites |
| `all` | Every layer above; explicit only | All requirements above |

Run a layer through the stable entry point:

```bash
$PY tools/test.py quick
$PY tools/test.py benchmark
$PY tools/test.py e2e
```

Plain `pytest` is intentionally equivalent to the fast test selection: it excludes `e2e` and
`slow`. A direct path narrows the collected files but does not bypass those marker exclusions; use
an explicit `-m` expression, or the named `realdata` / `e2e` suite, for an external test.

## Pick Tests By Change

| Changed area | First focused command | Before handoff |
|---|---|---|
| API catalog, search, proof gate | `$PY -m pytest -q tests/unit/test_api_catalog.py tests/unit/test_discovery.py tests/unit/test_manifest_invariants.py` | `quick` |
| `run_cell`, output normalization, security | `$PY -m pytest -q tests/unit/test_notebook.py tests/unit/test_tools.py tests/unit/test_security.py tests/integration/test_mcp_protocol.py` | `quick` |
| Dashboard, profiles, process lifecycle | `$PY -m pytest -q tests/unit/test_app_api.py tests/unit/test_profiles_cli.py tests/unit/test_runtime.py` | `quick` |
| PXT conversion, metadata loading, batch | `$PY -m pytest -q tests/unit/test_pxt.py tests/unit/test_overrides.py tests/unit/test_batch.py` | `realdata` when local data behavior changed |
| JupyterLab Comm or single-cell capture | `$PY tools/test.py frontend` | `e2e` for product-path changes |
| Rubric, oracle, campaign runner | `$PY tools/test.py benchmark` | one fresh campaign when model behavior can change |
| Full Dashboard/Jupyter/MCP behavior | `$PY tools/test.py e2e` | `acceptance` when campaign orchestration changed |

## What Each Layer Proves

- Unit and integration tests prove deterministic code contracts. A skip is not product evidence.
- `benchmark/run_case.py selftest` proves the grader accepts golden evidence and rejects every
  poisoned control. It does not prove an autonomous model succeeds.
- `e2e` proves the human product path works with a deterministic driver. It does not assess model
  planning or API discovery.
- A fresh `benchmark/run_campaign.py` run is the only evidence for autonomous Pi behavior. Never
  regrade or combine old trials after code, prompt, catalog, rubric, or Peaks source changes.

All tests inherit the isolation and native-thread caps in `tests/conftest.py`; do not bypass that
fixture for convenience. Real-data tests copy inputs into temporary directories and must never
write beside raw beamtime data.
