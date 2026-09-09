from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

_ISOLATED_ENV_KEYS = (
    "PEAKSMCP_HOME",
    "JUPYTER_DATA_DIR",
    "JUPYTER_CONFIG_DIR",
    "JUPYTER_RUNTIME_DIR",
    "IPYTHONDIR",
    "MPLCONFIGDIR",
    "NUMBA_CACHE_DIR",
)
_ORIGINAL_ENV = {key: os.environ.get(key) for key in _ISOLATED_ENV_KEYS}
_TEST_ROOT = Path(tempfile.mkdtemp(prefix="peaksmcp-pytest-"))

# Set isolation before test modules are imported. This prevents collection-time
# imports and test-created AuditLogger/KernelSpecManager objects from ever seeing
# the real ~/.peaksMCP or user Jupyter directories.
os.environ["PEAKSMCP_HOME"] = str(_TEST_ROOT / "peaksmcp-home")
os.environ["JUPYTER_DATA_DIR"] = str(_TEST_ROOT / "jupyter-data")
os.environ["JUPYTER_CONFIG_DIR"] = str(_TEST_ROOT / "jupyter-config")
os.environ["JUPYTER_RUNTIME_DIR"] = str(_TEST_ROOT / "jupyter-runtime")
os.environ["IPYTHONDIR"] = str(_TEST_ROOT / "ipython")
os.environ["MPLCONFIGDIR"] = str(_TEST_ROOT / "matplotlib")
os.environ["NUMBA_CACHE_DIR"] = str(_TEST_ROOT / "numba")

os.environ["MPLBACKEND"] = "Agg"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


@pytest.fixture(scope="session", autouse=True)
def isolated_test_environment():
    """Keep all persistent test state in one disposable temporary root."""
    yield _TEST_ROOT
    for key, value in _ORIGINAL_ENV.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    shutil.rmtree(_TEST_ROOT, ignore_errors=True)


@pytest.fixture(autouse=True)
def _no_save_channel_leakage():
    """The save-consent channel is owned by a running MCP server instance.
    Autouse cleanup guarantees one test can never leak its callback into a
    later test (regression: a server-construction test used to install a
    module-global channel that turned later save tests into 'denied')."""
    yield
    from peaksMCP.overrides import save as save_module

    save_module._set_approval_channel(None)
