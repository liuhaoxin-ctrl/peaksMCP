"""Live-kernel real-data integration: the cut workflow renders an inline image.

This starts a real IPython kernel in the active environment and executes the
raw-PXT -> convert -> EF/offset -> k_convert -> figure session inside it with
the inline Matplotlib backend. An ``image/png`` ``display_data`` message proves
that the figure rendered rather than returning only a ``<Figure>`` repr.

Run explicitly (requires the L112 raw data folder too):

    PEAKSMCP_LIVE_KERNEL=1 python tools/test.py realdata

Without ``PEAKSMCP_LIVE_KERNEL=1`` (or without the raw data) the tests skip so
the default fast suite never boots a kernel.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.realdata,
    pytest.mark.live_kernel,
    pytest.mark.slow,
]

REPO_ROOT = Path(__file__).parents[2]
RAW_PXT_DIR = Path(
    os.environ.get("PEAKSMCP_REALDATA_PXT")
    or "/Users/haoxin/Documents/实验数据/BP260623/data"
)
CUT_STEM = "BP_0015"
EF_CORRECTION = 2.6591
THETA_OFFSET_DEG = 1.5

REQUIRES_LIVE_KERNEL = pytest.mark.skipif(
    os.environ.get("PEAKSMCP_LIVE_KERNEL") != "1",
    reason="set PEAKSMCP_LIVE_KERNEL=1 to boot a real kernel in this test",
)
REQUIRES_RAW = pytest.mark.skipif(
    not (RAW_PXT_DIR.is_dir() and (RAW_PXT_DIR / f"{CUT_STEM}.pxt").is_file()),
    reason=f"raw L112 PXT not found under {RAW_PXT_DIR} ({CUT_STEM}.pxt missing)",
)


def _kernel_code() -> str:
    raw = (RAW_PXT_DIR / f"{CUT_STEM}.pxt").as_posix()
    return f'''
import sys, shutil, tempfile
from pathlib import Path
sys.path.insert(0, {str(REPO_ROOT)!r})
import matplotlib
matplotlib.use("module://matplotlib_inline.backend_inline")
tmp = Path(tempfile.mkdtemp())
shutil.copy2({raw!r}, tmp / "{CUT_STEM}.pxt")
from peaksMCP.pxt_utils.converter import convert_pxt
item = convert_pxt(tmp / "{CUT_STEM}.pxt", tmp / "{CUT_STEM}.nc")
import peaks
from peaksMCP.pxt_utils.loader import _register_l112_loader
_register_l112_loader()
da = peaks.load(str(tmp / "{CUT_STEM}.nc"))
da.metadata.set_EF_correction({EF_CORRECTION})
shifted = da.assign_coords(theta_par=da.theta_par - {THETA_OFFSET_DEG})
kd = shifted.k_convert(quiet=True)
import matplotlib.pyplot as plt
from peaksMCP.plotting import plot_validation_pair
fig = plot_validation_pair(da, kd, shared_scale="auto")
plt.show()  # inline backend -> display_data image/png
print("TMPDIR=" + str(tmp))
print("KSPACE=" + ",".join(kd.dims) + ":" + str(kd.shape))
'''


@pytest.fixture(scope="module")
def kernel_client():
    """A real IPython kernel in the peaks env, CWD at the repo root."""
    from jupyter_client import KernelManager

    manager = KernelManager()
    manager.kernel_cmd = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    manager.start_kernel(cwd=str(REPO_ROOT))
    client = manager.client()
    client.start_channels()
    try:
        client.wait_for_ready(timeout=90)
    except Exception:
        manager.shutdown_kernel(now=True)
        raise
    yield client
    client.stop_channels()
    manager.shutdown_kernel(now=True)


def _execute(client, code: str) -> tuple[list, list]:
    """Run code in the kernel and return (outputs, errors) from IOPub."""
    outputs: list[tuple[str, dict]] = []
    errors: list[dict] = []
    msg_id = client.execute(code)
    while True:
        msg = client.get_iopub_msg(timeout=240)
        if msg["parent_header"].get("msg_id") != msg_id:
            continue
        msg_type = msg["msg_type"]
        content = msg["content"]
        if msg_type == "status" and content["execution_state"] == "idle":
            break
        if msg_type == "stream":
            outputs.append(("stream", {"text": content.get("text", "")}))
        elif msg_type in ("display_data", "execute_result"):
            outputs.append((msg_type, content))
        elif msg_type == "error":
            errors.append(content)
    return outputs, errors


@REQUIRES_LIVE_KERNEL
@REQUIRES_RAW
def test_kernel_renders_inline_cut_figure_image(kernel_client):
    """The figure really renders: an image/png display_data arrives in-kernel."""
    import re

    outputs, errors = _execute(kernel_client, _kernel_code())
    assert not errors, errors
    images = [
        content
        for kind, content in outputs
        if kind in ("display_data", "execute_result")
        and (content.get("data") or {}).get("image/png")
    ]
    assert images, "no image/png display_data: the figure did not render inline"
    assert len(images) >= 1

    text = "".join(c.get("text", "") for kind, c in outputs if kind == "stream")
    match = re.search(r"KSPACE=(.*):\((\d+), (\d+)\)", text)
    assert match, text
    assert match.group(1).split(",") == ["eV", "kx"]
    assert not list(Path(re.search(r"TMPDIR=(\S+)", text).group(1)).glob("*.png"))


@REQUIRES_LIVE_KERNEL
@REQUIRES_RAW
def test_kernel_session_writes_only_the_final_netcdf(kernel_client):
    """Inside the kernel, nothing but the conversion .nc hits the disk."""
    import re

    outputs, errors = _execute(kernel_client, _kernel_code())
    assert not errors, errors
    text = "".join(c.get("text", "") for kind, c in outputs if kind == "stream")
    tmp = Path(re.search(r"TMPDIR=(\S+)", text).group(1))
    names = sorted(p.name for p in tmp.iterdir())
    assert names == [f"{CUT_STEM}.nc", f"{CUT_STEM}.pxt"], names
