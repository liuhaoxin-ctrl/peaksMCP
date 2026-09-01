"""Atomic runfile used by CLI commands to find the supervisor."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import psutil


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
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=".run-",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = -1
            stream.write(json.dumps(data, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return path


def _process_matches(data: dict[str, Any]) -> bool:
    """Verify that a runfile still identifies the peaksMCP supervisor process."""
    pid = int(data.get("pid") or 0)
    if pid <= 0:
        return False
    try:
        process = psutil.Process(pid)
        recorded_create_time = data.get("process_create_time")
        if recorded_create_time is not None:
            return abs(process.create_time() - float(recorded_create_time)) < 1.0
        # Backward compatibility for runfiles created before process identity
        # was recorded.  Never signal an arbitrary reused PID.
        command = " ".join(process.cmdline())
        return "peaksMCP" in command and "_serve" in command
    except (psutil.Error, OSError, TypeError, ValueError):
        return False


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
    if not _process_matches(data):
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
