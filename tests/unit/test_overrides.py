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

def test_load_data_rejects_missing_empty_directory_and_suffix(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        load_data(str(tmp_path / "nope.nc"))
    with pytest.raises(ValueError, match="no supported data files"):
        load_data(str(tmp_path))  # empty folder
    with pytest.raises(ValueError, match="unsupported file type"):
        unknown = tmp_path / "data.dat"
        unknown.write_bytes(b"x")
        load_data(str(unknown))
    with pytest.raises(ValueError, match="empty sequence"):
        load_data([])


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


def test_load_data_folder_returns_stem_mapping_with_datasheet(monkeypatch, tmp_path, capsys):
    """A whole experiment folder loads in one call; the sibling datasheet is
    translated and attached to the raw PXT arrays."""
    from peaksMCP import pxt_utils

    folder = tmp_path / "data"
    folder.mkdir()
    for stem in ("BP_0005", "BP_0020"):
        (folder / f"{stem}.pxt").write_bytes(b"fake")

    def fake_load_pxt(path):
        import os

        return xr.DataArray(
            np.ones((3, 4)), dims=("eV", "theta_par"),
            attrs={"units": "counts", "source_path": os.fspath(path)},
        )

    monkeypatch.setattr(pxt_utils.loader, "load_pxt", fake_load_pxt)
    (folder / "datasheet.csv").write_text(
        "Experiment title,,,,\n"
        "Index,Theta,Polarization,Temperture,Ei,Central Energy,Ef,slit,"
        "Pass E.,Data format,Comment\n"
        "5,430,S,9.4,2.2,,2.7,400,5,sweep,,\n"
        "20,430,S,9.4,2.2,,2.7,400,5,Au sweep,Au\n",
        encoding="utf-8",
    )
    loaded = load_data(str(folder))
    assert isinstance(loaded, dict)
    assert sorted(loaded) == ["BP_0005", "BP_0020"]
    assert loaded["BP_0020"].attrs["experiment_index"] == 20
    doc = loaded["BP_0005"].attrs["experiment_metadata_json"]
    assert doc["records"]["5"]["experiment"]["data_format"] == "sweep"
    assert doc["records"]["20"]["experiment"]["data_format"] == "Au sweep"
    out = capsys.readouterr().out
    assert "2 file(s) loaded" in out and "2 pxt" in out


def test_load_data_sequence_of_files(monkeypatch, tmp_path, capsys):
    from peaksMCP import pxt_utils

    a, b = tmp_path / "A_0001.pxt", tmp_path / "B_0002.pxt"
    a.write_bytes(b"x")
    b.write_bytes(b"x")
    monkeypatch.setattr(
        pxt_utils.loader, "load_pxt",
        lambda path: xr.DataArray(np.ones((2, 2)), dims=("eV", "theta_par")),
    )
    loaded = load_data([str(a), str(b)])
    assert sorted(loaded) == ["A_0001", "B_0002"]
    out = capsys.readouterr().out
    assert "2 file(s) loaded" in out


def test_load_data_folder_reports_failed_files(monkeypatch, tmp_path, capsys):
    from peaksMCP import pxt_utils

    folder = tmp_path / "data"
    folder.mkdir()
    (folder / "good.pxt").write_bytes(b"x")

    def flaky(path):
        if "good" in str(path):
            return xr.DataArray(np.ones((2, 2)), dims=("eV", "theta_par"))
        raise OSError("corrupt")

    (folder / "bad.pxt").write_bytes(b"y")
    monkeypatch.setattr(pxt_utils.loader, "load_pxt", flaky)
    loaded = load_data(str(folder))
    assert list(loaded) == ["good"]
    out = capsys.readouterr().out
    assert "skipped 1 file(s)" in out and "bad" in out
