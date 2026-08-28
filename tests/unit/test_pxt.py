from __future__ import annotations

import json
from unittest.mock import Mock

import numpy as np
import pytest
import xarray as xr

from peaksMCP.pxt_utils.converter import (
    convert_path,
    convert_pxt,
    default_output_dir,
    index_from_path,
)
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
    assert not list(tmp_path.glob(f".{target.name}.*.part"))


def test_force_replaces_existing_output_without_unlinking_target(monkeypatch, tmp_path):
    source = tmp_path / "BP_0005.pxt"
    source.touch()
    target = tmp_path / "BP_0005.nc"
    target.write_bytes(b"previous-valid-output")
    monkeypatch.setattr(
        "peaksMCP.pxt_utils.converter.load_pxt",
        lambda _path: xr.DataArray(
            np.arange(6).reshape(2, 3),
            dims=("eV", "theta_par"),
            attrs={"units": "counts"},
        ),
    )
    original_unlink = type(target).unlink

    def reject_target_unlink(path, *args, **kwargs):
        if path == target:
            raise AssertionError("force conversion must not unlink the published target")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(type(target), "unlink", reject_target_unlink)
    result = convert_pxt(source, target, force=True)
    assert result.status == "converted"
    with xr.open_dataarray(target) as opened:
        assert opened.shape == (2, 3)
    assert not list(tmp_path.glob(f".{target.name}.*.part"))


def test_non_force_does_not_overwrite_concurrent_publisher(monkeypatch, tmp_path):
    source = tmp_path / "BP_0005.pxt"
    source.touch()
    target = tmp_path / "BP_0005.nc"

    def publish_competing_output(_path):
        target.write_bytes(b"competing-output")
        return xr.DataArray(
            np.ones((2, 3)),
            dims=("eV", "theta_par"),
            attrs={"units": "counts"},
        )

    monkeypatch.setattr("peaksMCP.pxt_utils.converter.load_pxt", publish_competing_output)
    result = convert_pxt(source, target, force=False)
    assert result.status == "skipped"
    assert result.warnings == ["output was created by another conversion"]
    assert target.read_bytes() == b"competing-output"
    assert not list(tmp_path.glob(f".{target.name}.*.part"))


@pytest.mark.parametrize("force", [False, True])
@pytest.mark.parametrize("target_kind", [
    "same_path", "relative_path", "symlink", "parent_symlink", "hardlink",
    "other_pxt", "other_pxt_symlink", "uppercase_pxt", "invalid_suffix",
    "input_named_nc", "directory",
])
def test_converter_rejects_unsafe_targets_before_loading(monkeypatch, tmp_path, target_kind, force):
    source = tmp_path / ("BP_0005.nc" if target_kind == "input_named_nc" else "BP_0005.pxt")
    raw = b"original raw PXT bytes must remain untouched"
    source.write_bytes(raw)
    other = tmp_path / "BP_0006.pxt"
    other.write_bytes(b"another original experiment")
    target = source
    if target_kind == "relative_path":
        monkeypatch.chdir(tmp_path)
        target = source.relative_to(tmp_path)
    elif target_kind in {"symlink", "other_pxt_symlink"}:
        target = tmp_path / "alias.nc"
        target.symlink_to(source if target_kind == "symlink" else other)
    elif target_kind == "parent_symlink":
        parent = tmp_path / "alias_dir"
        parent.symlink_to(tmp_path, target_is_directory=True)
        target = parent / source.name
    elif target_kind == "hardlink":
        target = tmp_path / "alias.nc"
        target.hardlink_to(source)
    elif target_kind == "other_pxt":
        target = other
    elif target_kind == "uppercase_pxt":
        target = tmp_path / "new.PXT"
    elif target_kind == "invalid_suffix":
        target = tmp_path / "experiment.json"
    elif target_kind == "directory":
        target = tmp_path / "directory.nc"
        target.mkdir()
    loader = Mock(side_effect=AssertionError("unsafe target must be rejected before loading"))
    monkeypatch.setattr("peaksMCP.pxt_utils.converter.load_pxt", loader)
    before = set(tmp_path.iterdir())

    result = convert_pxt(source, target, force=force)

    assert result.status == "failed"
    assert result.error_type == "ValueError"
    loader.assert_not_called()
    assert source.read_bytes() == raw
    assert other.read_bytes() == b"another original experiment"
    assert set(tmp_path.iterdir()) == before
    assert not list(tmp_path.glob(".*.part"))


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_converter_rechecks_source_alias_before_publication(monkeypatch, tmp_path, link_kind):
    source = tmp_path / "BP_0005.pxt"
    source.write_bytes(b"original raw data")
    target = tmp_path / "BP_0005.nc"

    def changed_target(_source):
        if link_kind == "symlink":
            target.symlink_to(source)
        else:
            target.hardlink_to(source)
        return xr.DataArray(np.ones((2, 3)), dims=("eV", "theta_par"))

    monkeypatch.setattr("peaksMCP.pxt_utils.converter.load_pxt", changed_target)
    result = convert_pxt(source, target, force=True)
    assert result.status == "failed"
    assert result.error_type == "ValueError"
    assert target.samefile(source)
    assert source.read_bytes() == b"original raw data"
    assert not list(tmp_path.glob(".*.part"))


def test_convert_path_force_cannot_overwrite_raw_input(monkeypatch, tmp_path):
    source = tmp_path / "BP_0005.pxt"
    source.write_bytes(b"original raw data")
    loader = Mock(side_effect=AssertionError("load must not run"))
    monkeypatch.setattr("peaksMCP.pxt_utils.converter.load_pxt", loader)
    report = convert_path(source, source, force=True)
    assert report.items[0].status == "failed"
    assert report.items[0].error_type == "ValueError"
    assert source.read_bytes() == b"original raw data"
    loader.assert_not_called()


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


def _fake_load_pxt(_path):
    """Module-level stand-in for ``load_pxt`` so batch workers (spawn) can pickle it."""
    return xr.DataArray(np.ones((2, 3)), dims=("eV", "theta_par"), attrs={"units": "counts"})


def test_default_output_dir_is_sibling_netcdf(tmp_path):
    """A folder conversion without an explicit output targets a sibling
    ``<folder>_netcdf/`` directory (pure-function check; the batched worker
    path itself needs a real PXT fixture)."""
    source = tmp_path / "raw"
    source.mkdir()
    (source / "BP_0001.pxt").touch()
    assert default_output_dir(source) == tmp_path / "raw_netcdf"
    assert default_output_dir(tmp_path / "another") == tmp_path / "another_netcdf"


def test_auto_datasheet_discovered_and_translated(tmp_path):
    """A datasheet.csv next to the data is auto-translated into
    <output>/experiment_metadata.json when no metadata path is supplied."""
    from peaksMCP.pxt_utils.converter import _auto_metadata, _find_datasheet

    data = tmp_path / "data"
    data.mkdir()
    csv = data / "datasheet.csv"
    write_datasheet(csv, ["5,43,S,9.4,2.2,,2.7,400,5,note,extra"])

    # Folder input: found in the folder itself; single-file input: in its parent.
    assert _find_datasheet(data) == csv
    assert _find_datasheet(data / "BP_0005.pxt") == csv

    out = tmp_path / "out"
    out.mkdir()
    meta = _auto_metadata(None, data, out)
    assert meta == str(out / "experiment_metadata.json")
    assert (out / "experiment_metadata.json").is_file()
    document = json.loads((out / "experiment_metadata.json").read_text())
    assert document["records"]["5"]["polarization_angle_deg"] == 43

    # An explicit metadata path wins over auto-discovery.
    assert _auto_metadata("/custom.json", data, out) == "/custom.json"


def test_auto_datasheet_malformed_is_ignored(tmp_path):
    """A datasheet that does not parse must not abort conversion; it is skipped."""
    from peaksMCP.pxt_utils.converter import _auto_metadata

    data = tmp_path / "data"
    data.mkdir()
    (data / "datasheet.csv").write_text("not a valid datasheet\n", encoding="utf-8-sig")
    assert _auto_metadata(None, data, data) is None
