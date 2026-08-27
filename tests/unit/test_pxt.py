from __future__ import annotations

import json

import numpy as np
import pytest
import xarray as xr

from peaksMCP.pxt_utils.converter import convert_path, convert_pxt, index_from_path
from peaksMCP.pxt_utils.csv_translator import translate_datasheet
from peaksMCP.pxt_utils.loader import load_pxt


def write_datasheet(path, rows):
    path.write_text("Experiment title,,,,\nIndex,Theta,Polarization,Temperture,Ei,Central Energy,Ef,slit,Pass E.,Comment,Unknown\n" + "\n".join(rows), encoding="utf-8")


def test_datasheet_translation_mapping_warning_and_atomic_write(tmp_path):
    source = tmp_path / "datasheet.csv"
    write_datasheet(source, ["5,43,S,9.4,2.2,,2.7,400,5,note,extra"])
    output = tmp_path / "metadata.json"
    document = translate_datasheet(source, output)
    record = document.records["5"]
    assert record.polarization_angle_deg == 43
    assert record.temperature == {"sample": 9.4, "unit": "K"}
    assert record.analyser["scan"]["pass_energy_eV"] == 5
    assert record.unmapped == {"Unknown": "extra"}
    assert json.loads(output.read_text())["records"]["5"]["photon"]["polarisation"] == "S"
    assert not output.with_suffix(".json.part").exists()


def test_duplicate_invalid_and_missing_index_rows(tmp_path):
    source = tmp_path / "datasheet.csv"
    write_datasheet(source, ["5,1,S,,,,,,,,", "5,2,P,,,,,,,,"])
    with pytest.raises(ValueError, match="duplicate"):
        translate_datasheet(source)
    write_datasheet(source, ["bad,1,S,,,,,,,,"])
    with pytest.raises(ValueError, match="invalid Index"):
        translate_datasheet(source)
    write_datasheet(source, [",1,S,,,,,,,,"])
    assert translate_datasheet(source).warnings


def test_loader_preserves_axes_units_and_float32(monkeypatch, tmp_path):
    source = tmp_path / "BP_0005.pxt"
    source.touch()
    monkeypatch.setattr("peaksMCP.pxt_utils.loader._extract_wave", lambda _path: (np.ones((3, 4)), np.array([3, 4]), np.array([0.1, 2.0]), np.array([-1.0, 10.0]), "[eV]ThetaX[deg]"))
    data = load_pxt(source)
    assert data.dims == ("eV", "theta_par")
    assert data.dtype == np.float32
    assert data.coords["eV"].attrs["units"] == "eV"
    assert data.coords["theta_par"].attrs["units"] == "deg"


def test_converter_embeds_matching_metadata_and_protects_output(monkeypatch, tmp_path):
    source = tmp_path / "BP_0005.pxt"
    source.touch()
    metadata = tmp_path / "metadata.json"
    write_datasheet(tmp_path / "datasheet.csv", ["5,43,S,9.4,2.2,,2.7,400,5,note,extra"])
    translate_datasheet(tmp_path / "datasheet.csv", metadata)
    monkeypatch.setattr("peaksMCP.pxt_utils.converter.load_pxt", lambda _path: xr.DataArray(np.ones((2, 3)), dims=("eV", "theta_par"), attrs={"units": "counts"}))
    target = tmp_path / "BP_0005.nc"
    first = convert_pxt(source, target, metadata_path=metadata)
    assert first.status == "converted"
    opened = xr.open_dataarray(target)
    assert opened.attrs["experiment_index"] == 5
    assert json.loads(opened.attrs["experiment_metadata_json"])["temperature"]["sample"] == 9.4
    opened.close()
    assert convert_pxt(source, target, metadata_path=metadata).status == "skipped"
    assert not target.with_suffix(".nc.part").exists()


@pytest.mark.parametrize(("name", "expected"), [("BP_0005.pxt", 5), ("scan_42.pxt", 42), ("scan.pxt", None)])
def test_index_from_filename(name, expected):
    assert index_from_path(name) == expected


def test_convert_path_single_file_to_directory(monkeypatch, tmp_path):
    """convert_path with a single file and a directory output must place the
    NetCDF inside the directory (stem + .nc) instead of treating the directory
    itself as the target (which previously produced a silent 'skipped')."""
    source = tmp_path / "BP_0003.pxt"
    source.touch()
    monkeypatch.setattr(
        "peaksMCP.pxt_utils.converter.load_pxt",
        lambda _path: xr.DataArray(np.ones((2, 3)), dims=("eV", "theta_par"), attrs={"units": "counts"}),
    )
    output_dir = tmp_path / "converted"
    output_dir.mkdir()
    report = convert_path(source, output_dir)
    item = report.items[0]
    assert item.status == "converted"
    assert item.output == str(output_dir / "BP_0003.nc")
    assert (output_dir / "BP_0003.nc").exists()
    # An explicit file path is still honoured as-is.
    explicit = tmp_path / "custom.nc"
    report = convert_path(source, explicit)
    assert report.items[0].status == "converted"
    assert report.items[0].output == str(explicit)
    # A not-yet-existing directory destination (no extension) is treated as a directory.
    fresh = tmp_path / "fresh_dir"
    report = convert_path(source, fresh)
    assert report.items[0].output == str(fresh / "BP_0003.nc")
