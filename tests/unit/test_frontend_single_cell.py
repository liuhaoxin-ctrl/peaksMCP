"""Run the lightweight frontend regressions when Node dependencies are available."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


def test_single_cell_frontend_operations():
    root = Path(__file__).parents[2]
    node = shutil.which("node")
    compiler = root / "peaksMCP/extensions/jupyterlab/node_modules/typescript"
    if node is None or not compiler.is_dir():
        pytest.skip("Node.js and JupyterLab extension dev dependencies are required")
    result = subprocess.run(
        [node, "--test", "--test-concurrency=1", "tests/frontend/single_cell.test.cjs"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
