from __future__ import annotations

import json

import numpy as np
import pytest
import xarray as xr

from peaksMCP.overrides import load_data, report_dict, report_summary, save_result
from peaksMCP.overrides.models import Report


def _array() -> xr.DataArray:
    return xr.DataArray(
        np.arange(12, dtype=float).reshape(3, 4),
        dims=("eV", "kx"),
        coords={"eV": np.linspace(-1, 0, 3), "kx": np.linspace(-0.3, 0.3, 4)},
        attrs={"units": "counts"},
    )


# ---------- ② Report base ----------

def test_report_base_summary_and_dict():
    class Op(Report):
        pass

    report = Op(operation="op", status="partial", warnings=["edge"], partial=True)
    assert report_summary(report) == "op: partial (partial); 1 warning(s)"
    payload = report_dict(report)
    assert payload["operation"] == "op" and payload["status"] == "partial"


def test_batch_reports_inherit_report_base():
    from peaksMCP.batch.models import BatchItemResult, BatchResult

    assert issubclass(BatchItemResult, Report)
    assert issubclass(BatchResult, Report)
    result = BatchResult()
    assert result.to_dict()["completed"] == 0  # existing API unchanged


# ---------- ① load_data ----------

def test_load_data_rejects_missing_directory_and_suffix(tmp_path):
    with pytest.raises(ValueError, match="file not found"):
        load_data(str(tmp_path / "nope.nc"))
    with pytest.raises(ValueError, match="is a directory"):
        load_data(str(tmp_path))
    with pytest.raises(ValueError, match="unsupported file type"):
        unknown = tmp_path / "data.dat"
        unknown.write_bytes(b"x")
        load_data(str(unknown))


def test_load_data_pxt_happy_path(monkeypatch, tmp_path, capsys):
    from peaksMCP import pxt_utils

    target = tmp_path / "scan.pxt"
    target.write_bytes(b"fake")

    monkeypatch.setattr(
        pxt_utils.loader, "load_pxt", lambda path: _array().rename(path),
    )
    data = load_data(str(target))
    out = capsys.readouterr().out
    assert data.sizes == {"eV": 3, "kx": 4}
    assert "load_data:" in out and "dims" in out


@pytest.mark.parametrize("metadata_kind", ["path", "dict", "dataarray"])
def test_load_data_embeds_metadata(monkeypatch, tmp_path, metadata_kind):
    """Metadata must reach attrs["experiment_metadata_json"] in all three forms
    (regression: the reader was once imported under a name that no longer
    exists, and the ImportError was swallowed, silently embedding nothing)."""
    from peaksMCP import pxt_utils

    source = tmp_path / "scan.pxt"
    source.write_bytes(b"fake")
    document = {"records": {"5": {"experiment": {"data_format": "Au sweep"}}}}
    if metadata_kind == "path":
        metadata_path = tmp_path / "experiment_metadata.json"
        metadata_path.write_text(json.dumps(document), encoding="utf-8")
        metadata = metadata_path
    elif metadata_kind == "dict":
        metadata = document
    else:
        metadata = xr.DataArray([1], dims="eV", attrs={"experiment_metadata_json": json.dumps(document)})

    monkeypatch.setattr(
        pxt_utils.loader, "load_pxt", lambda _path: xr.DataArray(
            np.ones((2, 2)), dims=("eV", "theta_par"), attrs={"units": "counts"},
        ),
    )
    data = load_data(str(source), metadata=metadata)
    embedded = data.attrs["experiment_metadata_json"]
    assert embedded["records"]["5"]["experiment"]["data_format"] == "Au sweep"


# ---------- ⑤ save_result (preview -> approve) ----------

def test_save_result_default_is_preview_only(tmp_path, capsys):
    target = tmp_path / "out.nc"
    report = save_result(_array(), str(target))
    out = capsys.readouterr().out
    assert report.status == "awaiting_consent"
    assert not target.exists()
    assert "no write performed" in out


def test_save_result_approve_writes_atomically(tmp_path):
    target = tmp_path / "out.nc"
    report = save_result(_array(), str(target), approve=True)
    assert report.status == "saved" and target.exists()
    with xr.open_dataarray(target) as back:
        assert back.dims == ("eV", "kx")


def test_save_result_refuses_overwrite_without_flag(tmp_path):
    target = tmp_path / "out.nc"
    save_result(_array(), str(target), approve=True)
    report = save_result(_array(), str(target))
    assert report.status == "blocked"
    report_ok = save_result(_array(), str(target), overwrite=True, approve=True)
    assert report_ok.status == "saved"


def test_save_result_json(tmp_path):
    target = tmp_path / "summary.json"
    report = save_result({"idx": [1, 2]}, str(target), approve=True)
    assert report.status == "saved"
    assert json.loads(target.read_text(encoding="utf-8")) == {"idx": [1, 2]}


# ---------- ③ manifest file parses ----------

def test_override_manifest_v3_exists_and_registers_facades():
    from pathlib import Path

    import yaml

    raw = yaml.safe_load(
        Path("peaksMCP/config/override_manifest.yaml").read_text(encoding="utf-8")
    )
    assert raw["version"] == 3
    names = {entry["name"] for entry in raw["project"]}
    assert {"load_data", "save_result"} <= names


def test_native_catalog_v1_exists_and_holds_only_upstream_entries():
    from pathlib import Path

    import yaml

    raw = yaml.safe_load(
        Path("peaksMCP/config/native_catalog.yaml").read_text(encoding="utf-8")
    )
    assert raw["version"] == 1
    apis = raw["apis"]
    assert apis and len(apis) >= 25  # upstream presentation stays complete
    assert all(not (config or {}).get("project") for config in apis.values())
    assert "k_convert" in apis and "fit_gold" in apis
