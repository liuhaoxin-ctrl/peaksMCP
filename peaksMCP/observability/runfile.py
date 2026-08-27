"""Atomic runfile used by CLI commands to find the supervisor."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def runfile_path() -> Path:
    root = Path(os.environ.get("PEAKSMCP_HOME", Path.home() / ".peaksMCP"))
    return root / "run.json"


def write_runfile(data: dict[str, Any]) -> Path:
    """Atomically publish supervisor discovery data.

    Parameters
    ----------
    data : dict
        Process ID, service URLs and authentication values for local CLI clients.

    Returns
    -------
    pathlib.Path
        Written runfile path.

    Notes
    -----
    The runfile contains the Jupyter authentication token, so it is written
    with ``0o600`` permissions (owner read/write only), mirroring the audit log.

    Examples
    --------
    >>> path = write_runfile({"pid": 1234})
    >>> path.name
    'run.json'
    """
    path = runfile_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return path


def read_runfile() -> dict[str, Any] | None:
    """Read supervisor discovery data and mark dead processes as stale.

    Returns
    -------
    dict or None
        Parsed run state, or ``None`` when no valid runfile exists.

    Examples
    --------
    >>> state = read_runfile()
    >>> state is None or "pid" in state
    True
    """
    try:
        data = json.loads(runfile_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    pid = int(data.get("pid") or 0)
    if pid:
        try:
            os.kill(pid, 0)
        except OSError:
            return {**data, "stale": True}
    return data


def remove_runfile() -> None:
    """Remove supervisor discovery data if it exists.

    Examples
    --------
    >>> remove_runfile()  # idempotent cleanup
    """
    try:
        runfile_path().unlink()
    except FileNotFoundError:
        pass
