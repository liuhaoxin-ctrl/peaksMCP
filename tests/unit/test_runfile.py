from __future__ import annotations

import json
import os

import psutil

from peaksMCP.observability.runfile import read_runfile, runfile_path, write_runfile


def test_runfile_is_private_atomic_and_checks_process_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("PEAKSMCP_HOME", str(tmp_path))
    process = psutil.Process(os.getpid())
    data = {
        "pid": os.getpid(),
        "process_create_time": process.create_time(),
        "token": "secret",
    }

    path = write_runfile(data)

    assert path == runfile_path()
    assert path.stat().st_mode & 0o777 == 0o600
    assert read_runfile() == data
    assert not list(tmp_path.glob(".run-*.tmp"))

    path.write_text(
        json.dumps({**data, "process_create_time": process.create_time() - 100}),
        encoding="utf-8",
    )
    assert read_runfile()["stale"] is True


def test_legacy_runfile_rejects_unrelated_reused_pid(tmp_path, monkeypatch):
    monkeypatch.setenv("PEAKSMCP_HOME", str(tmp_path))
    write_runfile({"pid": os.getpid()})
    assert read_runfile()["stale"] is True
