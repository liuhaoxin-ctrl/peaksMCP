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


def test_load_data_folder_returns_index_with_datasheet(monkeypatch, tmp_path, capsys):
    """A whole experiment folder is indexed without reading data; the sibling
    datasheet feeds the decision metadata and the data layer loads on demand."""
    from peaksMCP import pxt_utils
    from peaksMCP.overrides import LoadedScans

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
    assert isinstance(loaded, LoadedScans)
    assert loaded.stems == ["BP_0005", "BP_0020"]
    # Decision metadata without touching data files.
    assert loaded.cuts == ["BP_0005"] and loaded.gold == ["BP_0020"]
    entry20 = loaded.entries[1]
    assert entry20.index == 20 and entry20.data_format == "Au sweep"
    assert entry20.scan_kind == "gold"
    # Data layer: loading one stem reads exactly that file and attaches the
    # translated document.
    data20 = loaded["BP_0020"]
    assert data20.attrs["experiment_index"] == 20
    doc = loaded["BP_0005"].attrs["experiment_metadata_json"]
    assert doc["records"]["5"]["experiment"]["data_format"] == "sweep"
    assert doc["records"]["20"]["experiment"]["data_format"] == "Au sweep"
    out = capsys.readouterr().out
    assert "2 file(s) indexed" in out and "gold=1, cuts=1" in out


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
    assert loaded.stems == ["A_0001", "B_0002"]
    out = capsys.readouterr().out
    assert "2 file(s) indexed" in out


def test_load_data_index_never_reads_but_data_layer_fails_per_file(monkeypatch, tmp_path):
    """Indexing a folder reads nothing; a corrupt file only fails when its
    data layer is accessed, and other stems keep working."""
    from peaksMCP import pxt_utils
    from peaksMCP.overrides import LoadedScans

    folder = tmp_path / "data"
    folder.mkdir()
    (folder / "good.pxt").write_bytes(b"x")
    (folder / "bad.pxt").write_bytes(b"y")

    def flaky(path):
        if "good" in str(path):
            return xr.DataArray(np.ones((2, 2)), dims=("eV", "theta_par"))
        raise OSError("corrupt")

    monkeypatch.setattr(pxt_utils.loader, "load_pxt", flaky)
    loaded = load_data(str(folder))
    assert isinstance(loaded, LoadedScans)
    assert loaded.stems == ["bad", "good"]  # both indexed, nothing read
    with pytest.raises(OSError, match="corrupt"):
        loaded["bad"]
    assert loaded["good"].dims == ("eV", "theta_par")


def test_load_data_index_uses_embedded_netcdf_metadata(monkeypatch, tmp_path, capsys):
    """A converted folder without a datasheet still gets decision metadata
    from each NetCDF's embedded experiment_metadata_json (header only)."""
    import json as _json

    from peaksMCP.overrides import LoadedScans
    from peaksMCP.overrides import load as load_module

    folder = tmp_path / "data_netcdf"
    folder.mkdir()
    for stem in ("BP_0015", "BP_0020"):
        (folder / f"{stem}.nc").write_bytes(b"fake-nc")

    def fake_single(path, lazy=True):
        import os

        index = load_module._index_from_stem(os.path.basename(os.fspath(path))[:-3])
        document = {
            "records": {
                "15": {"experiment": {"data_format": "sweep",
                                      "energy_start_eV": 2.2, "energy_stop_eV": 2.7}},
                "20": {"experiment": {"data_format": "Au sweep"}},
            }
        }
        return xr.DataArray(
            np.ones((3, 4)), dims=("eV", "theta_par"),
            attrs={
                "units": "counts",
                "experiment_index": index,
                "experiment_metadata_json": _json.dumps(document),
            },
        ), "NetCDF"

    monkeypatch.setattr(load_module, "_single", fake_single)
    loaded = load_data(str(folder))
    assert isinstance(loaded, LoadedScans)
    assert loaded.gold == ["BP_0020"] and loaded.cuts == ["BP_0015"]
    assert loaded.entries[0].energy_window_eV == (2.2, 2.7)
    assert loaded.entries[0].sizes == {"eV": 3, "theta_par": 4}
    assert loaded.entries[0].converted is True
    out = capsys.readouterr().out
    assert "metadata=embedded" in out
    # The data layer goes through the same single-file reader.
    data = loaded["BP_0015"]
    assert data.attrs["experiment_index"] == 15


def test_load_data_index_reports_pxt_dims_from_header(tmp_path, capsys):
    """A raw PXT folder indexes with per-file dims read from the wave header
    (no data materialised) - identical to what load_pxt reports."""
    from pathlib import Path as _Path

    from peaksMCP.overrides import LoadedScans, load_data
    from peaksMCP.pxt_utils.loader import load_pxt

    fixture = _Path(__file__).parents[1] / "fixtures" / "pxt" / "synthetic_2d_nested.pxt"
    folder = tmp_path / "data"
    folder.mkdir()
    (folder / "BP_0009.pxt").write_bytes(fixture.read_bytes())
    loaded = load_data(str(folder))
    assert isinstance(loaded, LoadedScans)
    entry = loaded.entries[0]
    expected = dict(load_pxt(fixture).sizes)
    assert entry.sizes == expected, (entry.sizes, expected)
    assert entry.file_kind == "pxt"
    assert loaded.entries[0].index == 9


def test_load_data_index_tags_processed_netcdf_entries(tmp_path, capsys):
    """*_processed.nc products index as their own processed entries, inherit
    the raw stem's datasheet identity, and stay out of the decision sets."""
    import json as _json

    from peaksMCP.overrides import LoadedScans, load_data
    from peaksMCP.overrides import load as load_module

    folder = tmp_path / "data_netcdf"
    folder.mkdir()
    (folder / "BP_0005.nc").write_bytes(b"fake")
    (folder / "BP_0005_processed.nc").write_bytes(b"fake")

    def fake_single(path, lazy=True):
        import os

        stem = os.path.basename(os.fspath(path))[:-3]
        processed = "_processed" in stem
        stem = stem.replace("_processed", "")
        index = load_module._index_from_stem(stem)
        document = {
            "records": {"5": {"experiment": {"data_format": "sweep",
                                             "energy_start_eV": 2.2,
                                             "energy_stop_eV": 2.7}}}
        }
        attrs = {"units": "counts", "experiment_index": index,
                 "experiment_metadata_json": _json.dumps(document)}
        if processed:
            dims, coords = ("eV", "kx"), {"eV": range(4), "kx": range(6)}
        else:
            dims, coords = ("eV", "theta_par"), {"eV": range(4), "theta_par": range(6)}
        return xr.DataArray(np.ones((4, 6)), dims=dims, coords=coords,
                            attrs=attrs), "NetCDF"

    monkeypatch = __import__("pytest").MonkeyPatch()
    try:
        monkeypatch.setattr(load_module, "_single", fake_single)
        loaded = load_data(str(folder))
    finally:
        monkeypatch.undo()
    assert isinstance(loaded, LoadedScans)
    stems = {e.stem: e for e in loaded.entries}
    assert stems["BP_0005"].processed is False
    proc = stems["BP_0005_processed"]
    assert proc.processed is True
    assert proc.index == 5 and proc.scan_kind == "cut"
    assert proc.sizes == {"eV": 4, "kx": 6}  # the product's own header dims
    assert loaded.cuts == ["BP_0005"]  # decision sets stay raw-only
    assert loaded.processed == ["BP_0005_processed"]
    out = capsys.readouterr().out
    assert "processed=1" in out
