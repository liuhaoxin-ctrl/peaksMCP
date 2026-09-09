"""Tests for the task-level facades (convert_experiment / inspect_experiment)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from peaksMCP.overrides import convert_experiment, inspect_experiment
from peaksMCP.overrides.inspection import ScanKind

# --------------------------------------------------------------------------- #
# convert_experiment
# --------------------------------------------------------------------------- #

def test_convert_experiment_validates_source_and_metadata(tmp_path):
    with pytest.raises(ValueError, match="source not found"):
        convert_experiment(tmp_path / "missing" / "BP.pxt")
    source = tmp_path / "BP_0001.pxt"
    source.write_bytes(b"fake")
    with pytest.raises(ValueError, match="metadata file not found"):
        convert_experiment(source, metadata=tmp_path / "nope.json")


def _fake_conversion(monkeypatch, array=None, doc=None):
    """Patch pure conversion + install an approval channel (active gateway)."""
    from peaksMCP.overrides import save as save_module

    monkeypatch.setattr("peaksMCP.pxt_utils.converter._converted_array",
                        lambda file, index, document: (array or _fa(), []))
    seen: dict = {}
    save_module._set_approval_channel(
        lambda payload: seen.update(payload=payload) or True
    )
    return seen


def _fa():
    return xr.DataArray(np.arange(12, dtype=float).reshape(3, 4),
                        dims=("eV", "theta_par"))


def test_convert_experiment_stages_and_publishes_after_approval(monkeypatch, tmp_path, capsys):
    """Pure conversion + consent: nothing is written until the card is
    approved; approved items are published atomically with output_exists;
    the facade prints nothing (ConversionReport is the outcome)."""
    source = tmp_path / "BP_0001.pxt"
    source.write_bytes(b"fake")
    seen = _fake_conversion(monkeypatch)
    result = convert_experiment(str(source))
    payload = seen["payload"]
    assert payload["operation"] == "convert_experiment"
    item_payload = payload["items"][0]
    assert item_payload["path"].endswith("BP_0001.nc")
    # The card manifest row carries the conversion boundary info: source ->
    # target plus dims/dtype from the staged structure.
    assert item_payload["structure"]["dims"] == ["eV", "theta_par"]
    out = capsys.readouterr().out
    assert out == ""
    target = tmp_path / "BP_0001.nc"
    assert target.exists()
    item = result.items[0]
    assert item.status == "converted" and item.output_exists is True
    assert item.output == str(target)


def test_convert_experiment_denied_writes_nothing(monkeypatch, tmp_path, capsys):
    from peaksMCP.overrides import save as save_module

    source = tmp_path / "BP_0001.pxt"
    source.write_bytes(b"fake")
    monkeypatch.setattr("peaksMCP.pxt_utils.converter._converted_array",
                        lambda file, index, document: (_fa(), []))
    save_module._set_approval_channel(lambda payload: False)
    result = convert_experiment(str(source))
    assert result.items[0].status == "denied"
    assert not (tmp_path / "BP_0001.nc").exists()
    assert not list(tmp_path.glob(".*.part-*"))
    assert capsys.readouterr().out == ""


def test_convert_experiment_without_channel_stays_awaiting(monkeypatch, tmp_path, capsys):
    from peaksMCP.overrides import save as save_module

    source = tmp_path / "BP_0001.pxt"
    source.write_bytes(b"fake")
    save_module._set_approval_channel(None)
    monkeypatch.setattr("peaksMCP.pxt_utils.converter._converted_array",
                        lambda file, index, document: (_fa(), []))
    result = convert_experiment(str(source))
    item = result.items[0]
    assert item.status == "failed"
    assert item.error and "no_consent_channel" in item.error
    assert not (tmp_path / "BP_0001.nc").exists()


def test_convert_experiment_directory_skips_existing(monkeypatch, tmp_path, capsys):
    from peaksMCP.overrides import save as save_module

    data = tmp_path / "data"
    data.mkdir()
    for stem in ("BP_0001", "BP_0002"):
        (data / f"{stem}.pxt").write_bytes(b"fake")
    (data.parent / "data_netcdf").mkdir()
    done = data.parent / "data_netcdf" / "BP_0001.nc"
    done.write_bytes(b"existing")
    monkeypatch.setattr("peaksMCP.pxt_utils.converter._converted_array",
                        lambda file, index, document: (_fa(), []))
    save_module._set_approval_channel(lambda payload: True)
    result = convert_experiment(str(data))
    by_input = {item.input.split("/")[-1]: item for item in result.items}
    assert by_input["BP_0001.pxt"].status == "skipped"  # idempotent skip
    assert done.read_bytes() == b"existing"
    assert by_input["BP_0002.pxt"].status == "converted"
    assert (data.parent / "data_netcdf" / "BP_0002.nc").exists()


def test_convert_experiment_directory_includes_metadata_json_item(monkeypatch, tmp_path):
    """A sibling datasheet adds experiment_metadata.json to the same ticket."""
    from peaksMCP.overrides import save as save_module

    data = tmp_path / "data"
    data.mkdir()
    (data / "BP_0005.pxt").write_bytes(b"fake")
    (data / "datasheet.csv").write_text(
        "Experiment title,,,,\n"
        "Index,Theta,Polarization,Temperture,Ei,Central Energy,Ef,slit,"
        "Pass E.,Data format,Comment\n"
        "5,430,S,9.4,2.2,,2.7,400,5,sweep,,\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("peaksMCP.pxt_utils.converter._converted_array",
                        lambda file, index, document: (_fa(), []))
    seen = {}
    save_module._set_approval_channel(
        lambda payload: seen.update(payload=payload) or True
    )
    result = convert_experiment(str(data))
    assert result.items[0].status == "converted"
    kinds = {item["kind"] for item in seen["payload"]["items"]}
    assert kinds == {"netcdf", "json"}
    json_path = data.parent / "data_netcdf" / "experiment_metadata.json"
    assert json_path.exists()


# --------------------------------------------------------------------------- #
# inspect_experiment
# --------------------------------------------------------------------------- #

def _document():
    return {
        "notes": ["Cut theta_offset=1.5"],
        "records": {
            "1": {"experiment": {"data_format": "Au sweep",
                                 "energy_start_eV": 25.0, "energy_stop_eV": 30.0}},
            "2": {"experiment": {"data_format": "sweep",
                                 "energy_start_eV": 2.2, "energy_stop_eV": 2.7}},
            "3": {"experiment": {"data_format": "mapping",
                                 "energy_start_eV": 2.2, "energy_stop_eV": 2.7}},
            "4": {"experiment": {"data_format": ""}},
            "5": {"experiment": {"data_format": "sweep"}},
            "6": {"experiment": {"data_format": "sweep"}},
            "7": {"experiment": {"data_format": "sweep"}},
        },
    }


def _scan(dims, size=4):
    shape = tuple([size] * len(dims))
    return xr.DataArray(np.zeros(shape), dims=dims)


def test_inspect_experiment_kind_inference_without_shapes():
    summary = inspect_experiment(_document())
    by_index = {row.index: row for row in summary.records}
    assert by_index[1].kind == ScanKind.GOLD
    assert by_index[2].kind == ScanKind.CUT
    assert by_index[3].kind == ScanKind.MAPPING
    assert by_index[4].kind == ScanKind.UNKNOWN
    assert summary.gold == [1]
    assert summary.energy_windows_eV == [(2.2, 2.7), (25.0, 30.0)]


def test_inspect_experiment_reports_classification_conflicts_from_shapes():
    scans = {
        # 3-D cube labelled "sweep": low-energy mapping mislabelled as a cut.
        2: _scan(("eV", "theta_par", "deflector_perp")),
        # 2-D labelled "mapping" keeps its mapping kind.
        3: _scan(("eV", "theta_par")),
        # hv dimension -> hv scan, not a plain mapping.
        5: _scan(("eV", "theta_par", "hv")),
        # 2-D spatial map.
        6: _scan(("eV", "x", "y")),
        # 1-D -> spectrum.
        7: _scan(("eV",)),
    }
    summary = inspect_experiment(_document(), scans=scans)
    by_index = {row.index: row for row in summary.records}
    assert by_index[2].kind == ScanKind.MAPPING
    assert by_index[3].kind == ScanKind.MAPPING  # 2-D keeps its mapping label
    assert by_index[5].kind == ScanKind.HV_SCAN
    assert by_index[6].kind == ScanKind.SPATIAL_MAP
    assert by_index[7].kind == ScanKind.SPECTRUM
    issues = {c.index: c.issue for c in summary.conflicts}
    assert 2 in issues and "mapping-shaped cube" in issues[2]
    assert summary.mappings == [2, 3]
    assert 5 not in summary.mappings and 6 not in summary.mappings
    # JSON-safe round trip.
    payload = summary.model_dump(mode="json")
    assert payload["conflicts"][0]["index"] == 2


def test_inspect_experiment_2d_mapping_without_spatial_axes_conflicts():
    document = {
        "records": {"9": {"experiment": {"data_format": "mapping"}}},
    }
    scans = {9: _scan(("eV", "theta_par"))}
    summary = inspect_experiment(document, scans=scans)
    assert any("no spatial x/y axes" in c.issue for c in summary.conflicts)


def test_inspect_experiment_loads_metadata_embedded_in_a_scan():
    import json

    meta = json.dumps(_document())
    array = xr.DataArray(np.zeros((2, 2)), dims=("eV", "theta_par"),
                         attrs={"experiment_metadata_json": meta})
    summary = inspect_experiment(array)
    assert {row.index for row in summary.records} == {1, 2, 3, 4, 5, 6, 7}


def test_inspect_experiment_accepts_a_loaded_scans_index():
    """The primary interface: scans = load_data(root); summary = inspect_experiment(scans).

    The index carries identity + provenance + header sizes only; inspection is
    the single classification owner: kinds, decision lists, dims from the
    entry sizes and shape conflicts - one row per experiment record (raw and
    processed entries sharing an index collapse onto the raw shape)."""
    from peaksMCP.overrides import LoadedScans, ScanEntry

    exp = LoadedScans(
        [
            ScanEntry(stem="BP_0007", path="/d/BP_0007.nc", representation="netcdf",
                      experiment_index=7, sizes={"eV": 215, "theta_par": 902, "deflector_perp": 31}),
            ScanEntry(stem="BP_0015", path="/d/BP_0015.nc", representation="netcdf",
                      experiment_index=15, sizes={"eV": 168, "theta_par": 902}),
            ScanEntry(stem="BP_0015_processed", path="/d/BP_0015_processed.nc",
                      representation="processed_netcdf", experiment_index=15,
                      sizes={"eV": 168, "kx": 902}),
            ScanEntry(stem="BP_0020", path="/d/BP_0020.nc", representation="netcdf",
                      experiment_index=20, sizes={"eV": 168, "theta_par": 902}),
            ScanEntry(stem="unindexed", path="/d/unindexed.pxt", representation="raw_pxt"),
        ],
        source="BP260623/data_netcdf",
        metadata_document={
            "notes": ["Cut theta_offset=1.5"],
            "records": {
                "7": {"experiment": {"data_format": "sweep",
                                     "energy_start_eV": 2.2, "energy_stop_eV": 2.7}},
                "15": {"experiment": {"data_format": "sweep"}},
                "20": {"experiment": {"data_format": "Au sweep"}},
            },
        },
        metadata_source="embedded",
    )
    summary = inspect_experiment(exp)
    by_index = {row.index: row for row in summary.records}
    assert set(by_index) == {7, 15, 20}  # unindexed files have no record row
    assert by_index[20].kind == ScanKind.GOLD
    assert by_index[15].kind == ScanKind.CUT
    assert by_index[15].dims == ["eV", "theta_par"]  # raw nc dims, not kx
    assert by_index[15].is_gold is False
    assert by_index[7].kind == ScanKind.MAPPING  # 3-D cube labelled sweep
    assert any(c.index == 7 and "mapping-shaped cube" in c.issue for c in summary.conflicts)
    assert set(summary.gold) == {20}
    assert set(summary.cuts) == {15}
    assert set(summary.mappings) == {7}
    assert summary.energy_windows_eV == [(2.2, 2.7)]
    assert summary.notes == ["Cut theta_offset=1.5"]
    payload = summary.model_dump(mode="json")
    assert payload["gold"] == [20]


def test_inspect_experiment_classifies_sizes_only_without_document():
    """Without any metadata document the shapes alone still classify; entries
    without a parseable experiment index are skipped (no record identity)."""
    from peaksMCP.overrides import LoadedScans, ScanEntry

    exp = LoadedScans(
        [
            ScanEntry(stem="BP_0001", path="/d/BP_0001.pxt", representation="raw_pxt",
                      experiment_index=1, sizes={"eV": 200}),
            ScanEntry(stem="BP_0002", path="/d/BP_0002.pxt", representation="raw_pxt",
                      experiment_index=2, sizes={"eV": 200, "theta_par": 100}),
            ScanEntry(stem="BP_0003", path="/d/BP_0003.pxt", representation="raw_pxt",
                      experiment_index=3, sizes={"eV": 60, "theta_par": 100, "deflector_perp": 10}),
            ScanEntry(stem="scrap", path="/d/scrap.pxt", representation="raw_pxt"),
        ],
        source="sequence",
    )
    summary = inspect_experiment(exp)
    by_index = {row.index: row for row in summary.records}
    assert by_index[1].kind == ScanKind.SPECTRUM
    assert by_index[2].kind == ScanKind.CUT
    assert by_index[3].kind == ScanKind.MAPPING
    assert summary.conflicts == []  # no declared format to disagree with
    assert summary.records[2].data_format == ""


# --------------------------------------------------------------------------- #
# fit_gold_reference
# --------------------------------------------------------------------------- #

def _gold(rows=8, cols=10):
    """Finite 2-D gold-like scan (eV, theta_par)."""
    values = np.abs(np.sin(np.linspace(0, 3, rows)))[:, None] * np.ones((1, cols))
    return xr.DataArray(
        values,
        dims=("eV", "theta_par"),
        coords={
            "eV": np.linspace(2.4, 2.9, rows),
            "theta_par": np.linspace(-9, 9, cols),
        },
        attrs={"units": "counts"},
        name="BP gold",
    )


def test_fit_gold_reference_validates_input():
    from peaksMCP.overrides.calibration import fit_gold_reference

    with pytest.raises(TypeError, match="must be an xarray.DataArray"):
        fit_gold_reference([[1, 2]])
    no_eV = _gold().rename({"eV": "energy"})
    with pytest.raises(ValueError, match="'eV' dimension"):
        fit_gold_reference(no_eV)
    with pytest.raises(ValueError, match="2D"):
        fit_gold_reference(_gold().expand_dims(polar=[1]))
    bad = _gold().copy()
    bad.values[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        fit_gold_reference(bad)


def test_fit_gold_reference_delegates_and_builds_calibration(monkeypatch):
    import peaksMCP.overrides.calibration as calibration

    seen: dict = {}

    def fake_fit(data, *, correction, outlier_exclusion, outlier_sigma, plot, show):
        seen.update(correction=correction, sigma=outlier_sigma, exclusion=outlier_exclusion,
                    plot=plot, show=show)
        result = xr.Dataset({"EF": ("theta_par", [2.65] * 3)})
        result.attrs["EF_correction"] = {"c0": 2.6591, "c1": 0.01}
        result.attrs["EF_poly4"] = 2.6591
        result.attrs["EF_quality"] = {"median": 2.66, "uniform": True}
        result.attrs["average_resolution_eV"] = 0.021
        result.attrs["accuracy_by_2nd_fitting_eV"] = 0.012
        if plot:
            result.attrs["figure"] = object()
        return result

    monkeypatch.setattr(calibration, "_fit_gold", fake_fit)
    gold = _gold()
    cal = calibration.fit_gold_reference(gold, plot=False)
    assert cal.status == "ok"
    assert cal.correction == {"c0": 2.6591, "c1": 0.01}
    assert cal.quality["uniform"] is True
    assert cal.average_resolution_eV == 0.021
    assert cal.accuracy_by_2nd_fitting_eV == 0.012
    assert cal.rendered is False
    assert seen == {"correction": "poly4", "sigma": 3.0, "exclusion": True, "plot": False, "show": False}
    assert cal.model_dump(mode="json")["correction"] == {"c0": 2.6591, "c1": 0.01}
    # plot=True marks rendered when peaks attached a figure.
    cal2 = calibration.fit_gold_reference(gold, correction="average", outlier_sigma=2.0)
    assert cal2.rendered is True and cal2.correction_type == "average"


def test_fit_gold_reference_fails_loudly_without_correction(monkeypatch):
    import peaksMCP.overrides.calibration as calibration

    monkeypatch.setattr(
        calibration, "_fit_gold",
        lambda *a, **k: xr.Dataset({"x": [1]}),  # attrs without EF_correction
    )
    with pytest.raises(ValueError, match="no EF_correction"):
        calibration.fit_gold_reference(_gold(), plot=False)


# --------------------------------------------------------------------------- #
# preprocess_cut / preprocess_mapping
# --------------------------------------------------------------------------- #

def _cut(rows=8, cols=10):
    return xr.DataArray(
        np.abs(np.linspace(-2, 0, rows))[:, None] * np.ones((1, cols)),
        dims=("eV", "theta_par"),
        coords={
            "eV": np.linspace(-2.0, 0.0, rows),
            "theta_par": np.linspace(-9, 9, cols),
        },
        attrs={"units": "counts"},
    )


def test_preprocess_cut_rejects_bad_inputs():
    from peaksMCP.overrides.preprocess import preprocess_cut

    with pytest.raises(TypeError, match="must be an xarray.DataArray"):
        preprocess_cut([[1]], calibration=1.0, theta_par_offset_deg=0.0)
    with pytest.raises(ValueError, match="2-D \\(eV, theta_par\\)"):
        preprocess_cut(_cut().expand_dims(polar=[1, 2, 3]), calibration=1.0, theta_par_offset_deg=0.0)
    bad = _cut().copy()
    bad.values[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        preprocess_cut(bad, calibration=1.0, theta_par_offset_deg=0.0)
    with pytest.raises(ValueError, match="calibration is required"):
        preprocess_cut(_cut(), calibration=None, theta_par_offset_deg=0.0)


def test_preprocess_cut_copies_input_and_applies_offset_and_ef(monkeypatch, capsys):
    import peaksMCP.overrides.preprocess as preprocess

    seen: dict = {}

    def fake_k_convert(data, *, ef, eV, kx, ky, quiet):
        seen["ef"] = ef
        seen["quiet"] = quiet
        # The theta_par coordinate on the work copy is shifted by the offset.
        seen["theta_start"] = float(data.theta_par.values[0])
        out = xr.DataArray(
            np.abs(np.arange(24).reshape(4, 6)),
            dims=("eV", "kx"),
            coords={"eV": data.eV.values[:4], "kx": np.linspace(-0.3, 0.3, 6)},
            attrs={"units": "counts"},
        )
        return out

    monkeypatch.setattr(preprocess, "_k_convert", fake_k_convert)
    cut = _cut()
    original = cut.copy(deep=True)
    result = preprocess.preprocess_cut(cut, calibration={"c0": 2.6591}, theta_par_offset_deg=1.5)
    assert seen["ef"] == {"c0": 2.6591}
    assert seen["quiet"] is True
    assert seen["theta_start"] == pytest.approx(-9.0 - 1.5)
    # Input untouched: values, coords, attrs identical.
    xr.testing.assert_identical(cut, original)
    assert result.data.dims == ("eV", "kx")
    assert result.report.operation == "preprocess_cut"
    assert result.report.dims_in == ["eV", "theta_par"]
    assert result.report.complete is True
    assert result.report.selection == {"theta_par_offset_deg": 1.5}
    out = capsys.readouterr().out
    assert "preprocess_cut: dims" in out
    assert result.to_dict()["shape_out"] == [4, 6]


def test_preprocess_cut_rejects_3d_with_mapping_hint():
    from peaksMCP.overrides.preprocess import preprocess_cut

    cube = _cut().expand_dims(deflector_perp=[1, 2, 3])
    with pytest.raises(ValueError, match="preprocess_mapping"):
        preprocess_cut(cube, calibration=2.6, theta_par_offset_deg=1.0)


def test_preprocess_mapping_full_cube_no_centre_slice(monkeypatch):
    import peaksMCP.overrides.preprocess as preprocess

    seen: dict = {}

    def fake_normal(data, normal_emission):
        seen["normal_emission"] = normal_emission
        return data

    def fake_k_convert(data, *, ef, eV, kx, ky, quiet):
        seen["ky_arg"] = ky
        seen["cube_shape"] = list(data.shape)
        return xr.DataArray(
            np.zeros((5, 4, 6)),
            dims=("ky", "eV", "kx"),  # real peaks order for type-II geometry
            coords={"eV": np.arange(4), "kx": np.arange(6), "ky": np.arange(5)},
        )

    monkeypatch.setattr(preprocess, "_set_normal_emission", fake_normal)
    monkeypatch.setattr(preprocess, "_k_convert", fake_k_convert)
    cube = _cut().expand_dims(deflector_perp=np.linspace(-15, 15, 7))
    result = preprocess.preprocess_mapping(
        cube,
        calibration=2.6591,
        normal_emission={"theta_par": 0.0, "polar": 0.0},
        ky=slice(-0.5, 0.5),
    )
    assert seen["normal_emission"] == {"theta_par": 0.0, "polar": 0.0}
    assert seen["cube_shape"] == list(cube.shape)  # full cube, not a slice
    assert result.report.selection == {}
    assert result.report.complete is True
    assert {"eV", "kx", "ky"} <= set(result.data.dims)


def test_preprocess_mapping_rejects_slices_and_missing_reference():
    from peaksMCP.overrides.preprocess import preprocess_mapping

    cut2d = _cut()
    with pytest.raises(ValueError, match="3-D mapping cube"):
        preprocess_mapping(cut2d, calibration=2.6, normal_emission={"theta_par": 0.0})
    cube = _cut().expand_dims(deflector_perp=[1, 2, 3])
    with pytest.raises(ValueError, match="normal_emission"):
        preprocess_mapping(cube, calibration=2.6, normal_emission={})


# --------------------------------------------------------------------------- #
# preprocess_batch (worker + validation, executor patched)
# --------------------------------------------------------------------------- #

def test_preprocess_batch_validates_inputs(tmp_path):
    from peaksMCP.overrides.batch_preprocess import BatchPreprocessItem, preprocess_batch

    with pytest.raises(ValueError, match="must not be empty"):
        preprocess_batch([], calibration=2.6, output_dir=tmp_path)
    missing = BatchPreprocessItem(index=1, source=str(tmp_path / "nope.nc"), kind="cut")
    with pytest.raises(ValueError, match="source not found"):
        preprocess_batch([missing], calibration=2.6, output_dir=tmp_path)


def test_preprocess_batch_run_item_stages_without_publishing(tmp_path, monkeypatch):
    """The worker only processes and stages: no target file is ever written,
    a skipped-existing item reports early, failures raise (executor maps)."""
    from peaksMCP.overrides.batch_preprocess import (
        BatchPreprocessItem,
        _run_item,
    )

    (tmp_path / "a.nc").write_bytes(b"src")
    (tmp_path / "b.nc").write_bytes(b"src")
    (tmp_path / "c.nc").write_bytes(b"src")
    existing = tmp_path / "a_processed.nc"
    existing.write_bytes(b"old")
    item = BatchPreprocessItem(index=1, source=str(tmp_path / "a.nc"), kind="cut",
                               output=str(existing))
    result = _run_item(item, 2.6)
    assert result == {"skipped_existing": str(existing)}
    assert existing.read_bytes() == b"old"

    # A load failure raises (executor turns it into a failed item).
    def boom(source):
        raise FileNotFoundError(source)

    import peaksMCP.overrides.batch_preprocess as batch

    monkeypatch.setattr(batch, "_load_array", boom)
    out = tmp_path / "b_processed.nc"
    item2 = BatchPreprocessItem(index=2, source=str(tmp_path / "b.nc"), kind="cut",
                                theta_par_offset_deg=1.5, output=str(out))
    with pytest.raises(FileNotFoundError):
        _run_item(item2, 2.6)
    assert not out.exists()

    # Successful processing stages hidden bytes next to the target: the
    # target itself does not exist until the gateway publishes it.
    monkeypatch.setattr(batch, "_load_array",
                        lambda source: xr.DataArray(np.ones((3, 4)),
                                                    dims=("eV", "theta_par")))
    import peaksMCP.overrides.preprocess as preprocess_module

    def fake_impl(cut, *, calibration, theta_par_offset_deg, eV, kx, quiet):
        from peaksMCP.overrides.preprocess import ProcessingReport, ProcessingResult

        report = ProcessingReport(operation="preprocess_cut", dims_in=["eV", "theta_par"],
                                  dims_out=["eV", "kx"])
        out_array = xr.DataArray(np.arange(12).reshape(3, 4), dims=("eV", "kx"),
                                 coords={"eV": [0, 1, 2], "kx": [0, 1, 2, 3]})
        return ProcessingResult(out_array, report)

    monkeypatch.setattr(preprocess_module, "_preprocess_cut_impl", fake_impl)
    item3 = BatchPreprocessItem(index=3, source=str(tmp_path / "c.nc"), kind="cut",
                                theta_par_offset_deg=1.0, output=str(tmp_path / "c_processed.nc"))
    staged = _run_item(item3, 2.6)
    assert "pending" in staged
    pending = staged["pending"]
    assert str(pending.path).endswith("c_processed.nc")
    assert not pending.path.exists()          # target untouched
    assert pending.tmp_path.exists()          # staged bytes ready
    pending.tmp_path.unlink(missing_ok=True)


def _fake_batch_executor(monkeypatch, run_fn):
    """Replace the process-pool executor with an inline runner for tests."""
    import peaksMCP.batch as batch_module

    class _FakeExecutor:
        def __init__(self, budget):
            self.budget = budget

        def run(self, function, items, progress=None):
            from peaksMCP.batch.models import BatchItemResult, BatchResult

            results = []
            for idx, item in enumerate(items):
                try:
                    output = function(item)
                    results.append(BatchItemResult(idx, item, "completed", output=output))
                except Exception as exc:
                    results.append(BatchItemResult(
                        idx, item, "failed", error_type=type(exc).__name__, error=str(exc)))
            return BatchResult(items=results, duration_s=0.0)

    monkeypatch.setattr(batch_module, "BatchExecutor", _FakeExecutor)


def test_preprocess_batch_approval_publishes_all(tmp_path, monkeypatch, capsys):
    """Processing results are staged under one ticket; approval publishes."""
    from peaksMCP.overrides import save as save_module
    from peaksMCP.overrides.batch_preprocess import (
        BatchPreprocessItem,
        preprocess_batch,
    )
    from peaksMCP.overrides.preprocess import ProcessingReport, ProcessingResult

    def fake_impl(cut, *, calibration, theta_par_offset_deg, eV, kx, quiet):
        return ProcessingResult(
            xr.DataArray(np.ones((2, 3)), dims=("eV", "kx")),
            ProcessingReport(operation="preprocess_cut", dims_in=["eV", "theta_par"],
                             dims_out=["eV", "kx"]),
        )

    import peaksMCP.overrides.batch_preprocess as batch_module
    import peaksMCP.overrides.preprocess as preprocess_module

    monkeypatch.setattr(preprocess_module, "_preprocess_cut_impl", fake_impl)
    monkeypatch.setattr(batch_module, "_load_array",
                        lambda source: xr.DataArray(np.ones((2, 3)),
                                                    dims=("eV", "theta_par")))
    _fake_batch_executor(monkeypatch, lambda *a: None)
    seen = {}
    save_module._set_approval_channel(
        lambda payload: seen.update(payload=payload) or True
    )

    (tmp_path / "a.nc").write_bytes(b"src")
    out_dir = tmp_path / "processed"
    request = BatchPreprocessItem(index=1, source=str(tmp_path / "a.nc"), kind="cut",
                                  theta_par_offset_deg=1.0)
    report = preprocess_batch([request], calibration=2.6, output_dir=out_dir)
    assert seen["payload"]["operation"] == "preprocess_batch"
    assert report.completed == 1
    item = report.items[0]
    assert item.status == "completed" and item.output_exists is True
    target = out_dir / "a_processed.nc"
    assert target.exists()
    assert item.output == str(target)
    out = capsys.readouterr().out
    assert "completed" in out


def test_preprocess_batch_denied_writes_nothing(tmp_path, monkeypatch, capsys):
    from peaksMCP.overrides import save as save_module
    from peaksMCP.overrides.batch_preprocess import (
        BatchPreprocessItem,
        preprocess_batch,
    )
    from peaksMCP.overrides.preprocess import ProcessingReport, ProcessingResult

    def fake_impl(cut, *, calibration, theta_par_offset_deg, eV, kx, quiet):
        return ProcessingResult(
            xr.DataArray(np.ones((2, 3)), dims=("eV", "kx")),
            ProcessingReport(operation="preprocess_cut", dims_in=["eV", "theta_par"],
                             dims_out=["eV", "kx"]),
        )

    import peaksMCP.overrides.batch_preprocess as batch_module
    import peaksMCP.overrides.preprocess as preprocess_module

    monkeypatch.setattr(preprocess_module, "_preprocess_cut_impl", fake_impl)
    monkeypatch.setattr(batch_module, "_load_array",
                        lambda source: xr.DataArray(np.ones((2, 3)),
                                                    dims=("eV", "theta_par")))
    _fake_batch_executor(monkeypatch, lambda *a: None)
    save_module._set_approval_channel(lambda payload: False)

    (tmp_path / "a.nc").write_bytes(b"src")
    out_dir = tmp_path / "processed2"
    request = BatchPreprocessItem(index=1, source=str(tmp_path / "a.nc"), kind="cut",
                                  theta_par_offset_deg=1.0)
    report = preprocess_batch([request], calibration=2.6, output_dir=out_dir)
    assert report.items[0].status == "denied"
    assert not (out_dir / "a_processed.nc").exists()
    assert not list(tmp_path.glob("**/.*.part-*"))


def test_inspect_experiment_accepts_datasheet_csv(tmp_path):
    """The standard experiment folder carries datasheet.csv next to the PXT:
    inspect_experiment translates it on the fly before summarizing."""
    csv_path = tmp_path / "datasheet.csv"
    csv_path.write_text(
        "Experiment title,,,,\n"
        "Index,Theta,Polarization,Temperture,Ei,Central Energy,Ef,slit,"
        "Pass E.,Data format,Comment,,AI请看的Note：Cut theta_offset=1.5\n"
        "5,430,S,9.4,2.2,,2.7,400,5,sweep,,,,\n"
        "20,430,S,9.4,2.2,,2.7,400,5,Au sweep,Au,,\n",
        encoding="utf-8",
    )
    summary = inspect_experiment(csv_path)
    by_index = {row.index: row for row in summary.records}
    assert by_index[5].kind == ScanKind.CUT
    assert by_index[20].kind == ScanKind.GOLD
    assert summary.gold == [20]
    assert summary.notes and any("theta_offset=1.5" in note for note in summary.notes)


def test_preprocess_batch_worker_is_pickleable_for_the_process_pool():
    """Regression: the pool worker used to be a local closure ("Can't get
    local object ... worker" under spawn).  The module-level _run_item behind
    a functools.partial must survive a pickle round trip exactly as the
    process pool serialises it."""
    import functools
    import pickle

    from peaksMCP.overrides.batch_preprocess import _run_item

    worker = functools.partial(_run_item, calibration=2.6591, force=False)
    restored = pickle.loads(pickle.dumps(worker))
    assert restored.func is _run_item
    assert restored.keywords == {"calibration": 2.6591, "force": False}
