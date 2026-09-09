from __future__ import annotations

import json

import numpy as np
import pytest
import xarray as xr

from peaksMCP.overrides import load_data
from peaksMCP.overrides.save import SaveReceipt, _save_result


def _array() -> xr.DataArray:
    return xr.DataArray(
        np.arange(12, dtype=float).reshape(3, 4),
        dims=("eV", "kx"),
        coords={"eV": np.linspace(-1, 0, 3), "kx": np.linspace(-0.3, 0.3, 4)},
        attrs={"units": "counts"},
    )


# ---------- ② generic Report layer abolished ----------

def test_batch_result_models_are_self_contained():
    """The generic Report layer is gone; batch results are independent
    dataclasses with their own to_dict (no Report vocabulary)."""
    from peaksMCP.batch.models import BatchResult

    result = BatchResult()
    assert result.to_dict()["completed"] == 0  # existing API unchanged
    assert not hasattr(result, "partial")


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


# ---------- ⑤ save gateway (_save_result, the save_with_consent internals) ----------

def _approval(approved: bool):
    """Install a fake approval channel for one test."""
    import pytest

    from peaksMCP.overrides import save as save_module

    monkeypatch = pytest.MonkeyPatch()
    seen: dict = {}

    def channel(payload):
        seen["payload"] = payload
        return approved

    monkeypatch.setattr(save_module, "_APPROVAL_CHANNEL", channel)
    return monkeypatch, seen


def _single_item(payload):
    """The card payload of a single-file ticket: {operation, summary, items}."""
    assert payload["items"] and len(payload["items"]) == 1
    return payload["items"][0]


def test_save_without_channel_stages_in_unified_area_and_waits(tmp_path, monkeypatch):
    """No frontend channel: the ticket waits in the unified staging area and
    nothing ever appears next to the target (no early .part, no target dir)."""
    from peaksMCP.overrides import save as save_module

    monkeypatch.setattr(save_module, "_APPROVAL_CHANNEL", None)  # no frontend
    target = tmp_path / "out.nc"
    receipt = _save_result(_array(), str(target))
    assert isinstance(receipt, SaveReceipt)
    assert receipt.status == "pending_consent"
    assert receipt.kind == "netcdf"
    assert receipt.ticket_id and receipt.sha256
    assert receipt.dims == {"eV": 3, "kx": 4}
    assert not target.exists()  # nothing written
    assert not list(tmp_path.iterdir())  # staging stays in the unified area
    save_module._discard_ticket(receipt.ticket_id)


def test_save_approval_channel_writes_exact_staged_bytes(tmp_path, monkeypatch):
    target = tmp_path / "out.nc"
    monkeypatch, seen = _approval(True)
    try:
        receipt = _save_result(_array(), str(target))
    finally:
        monkeypatch.undo()
    assert receipt.status == "saved" and target.exists()
    item = _single_item(seen["payload"])
    # The card shows the real content identity: path, sha256 of the staged
    # bytes, size and the array structure.
    assert item["path"].endswith("out.nc")
    assert item["sha256"] == receipt.sha256
    assert item["size_bytes"] == target.stat().st_size
    assert item["structure"]["dims"] == ["eV", "kx"]
    with xr.open_dataarray(target) as back:
        assert back.dims == ("eV", "kx")
    assert not list(tmp_path.glob(".*part*"))  # publish leaves no .part behind


def test_save_denied_writes_nothing_and_cleans_up(tmp_path, monkeypatch):
    target = tmp_path / "out.nc"
    monkeypatch, _ = _approval(False)
    try:
        receipt = _save_result(_array(), str(target))
    finally:
        monkeypatch.undo()
    assert receipt.status == "denied"
    assert not target.exists()
    assert not list(tmp_path.iterdir())


def test_save_refuses_overwrite_without_flag(tmp_path, monkeypatch):
    target = tmp_path / "out.nc"
    monkeypatch, _ = _approval(True)
    try:
        first = _save_result(_array(), str(target))
        assert first.status == "saved"
        blocked = _save_result(_array(), str(target))
        assert blocked.status == "blocked"
        assert not blocked.sha256  # nothing staged for a refused overwrite
        assert blocked.note and "exists" in blocked.note
        saved = _save_result(_array(), str(target), overwrite=True)
        assert saved.status == "saved"
    finally:
        monkeypatch.undo()


def test_save_json(tmp_path, monkeypatch):
    target = tmp_path / "summary.json"
    monkeypatch, seen = _approval(True)
    try:
        receipt = _save_result({"idx": [1, 2]}, str(target))
    finally:
        monkeypatch.undo()
    assert receipt.status == "saved" and receipt.kind == "json"
    assert json.loads(target.read_text(encoding="utf-8")) == {"idx": [1, 2]}
    assert "json_preview" in _single_item(seen["payload"])["structure"]


def test_ticket_is_one_time_and_gateway_requires_human_authorization(tmp_path, monkeypatch):
    """The gateway cannot write an unapproved ticket: notebook code cannot
    self-authorise (no approve flag exists anywhere in the flow)."""
    from peaksMCP.overrides import save as save_module

    monkeypatch.setattr(save_module, "_APPROVAL_CHANNEL", None)
    target = tmp_path / "out.nc"
    receipt = _save_result(_array(), str(target))  # no channel -> pending
    ticket = receipt.ticket_id
    assert not target.exists()
    # Direct gateway call on an unapproved ticket is refused.
    with pytest.raises(PermissionError, match="not authorized"):
        save_module._publish_batch(ticket)
    assert not target.exists()
    # Discard cleans the staging area without writing (safe without auth).
    save_module._discard_ticket(ticket)
    assert not list(tmp_path.iterdir())


def test_ticket_is_one_time_after_approval(tmp_path, monkeypatch):
    """Once the approval channel published the bytes, the ticket is spent."""
    from peaksMCP.overrides import save as save_module

    target = tmp_path / "out.nc"
    monkeypatch, _ = _approval(True)
    try:
        receipt = _save_result(_array(), str(target))
    finally:
        monkeypatch.undo()
    assert receipt.status == "saved" and receipt.ticket_id
    with pytest.raises(KeyError, match="already-used"):
        save_module._publish_batch(receipt.ticket_id)


def test_stage_never_accepts_a_code_level_approve():
    """The save implementation has no approve: **{'approve': True} is a
    TypeError, not a consent bypass."""
    target = "/tmp/save_result_approve_bypass_test.nc"
    try:
        with pytest.raises(TypeError):
            _save_result(_array(), target, **{"approve": True})
    finally:
        import os

        try:
            os.unlink(target)
        except FileNotFoundError:
            pass


# ---------- ③ manifest file parses ----------

def test_override_manifest_v4_exists_and_registers_public_facades():
    from pathlib import Path

    import yaml

    raw = yaml.safe_load(
        Path("peaksMCP/config/override_manifest.yaml").read_text(encoding="utf-8")
    )
    assert raw["version"] == 4
    # v4 is a single manifest: one row per public adapter, keyed by name,
    # each carrying the full structured contract.  No project-seeds block.
    assert "project" not in raw
    names = set(raw["apis"])
    assert {"load_data", "convert_experiment", "inspect_experiment"} <= names
    for name, entry in raw["apis"].items():
        assert entry["export"] == f"peaksMCP.overrides.{name}"
        assert entry["exposure"] == "facade"
        assert entry["summary"] and entry["inputs"] and entry["returns"]
        assert "docstring_note" not in entry
    # Not registered: the internal/legacy verbs are not model-facing.
    assert not ({"save_result", "read_meta"} & names)


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
    datasheet becomes the index's metadata document and classification stays
    with inspect_experiment."""
    from peaksMCP import pxt_utils
    from peaksMCP.overrides import LoadedScans
    from peaksMCP.overrides.inspection import inspect_experiment

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
    # Identity + provenance only; no classification at load time.
    assert loaded.metadata_source == "datasheet"
    entry20 = loaded.entries[1]
    assert entry20.experiment_index == 20
    assert entry20.representation == "raw_pxt"
    assert not hasattr(entry20, "scan_kind")
    # Classification is the inspect_experiment owner.
    summary = inspect_experiment(loaded)
    assert set(summary.gold) == {20}
    assert set(summary.cuts) == {5}
    assert summary.records[0].data_format == "Au sweep"  # str-sorted: "20" < "5"
    assert summary.conflicts == []
    # Data layer: loading one stem reads exactly that file and attaches the
    # translated document.
    data20 = loaded["BP_0020"]
    assert data20.attrs["experiment_index"] == 20
    doc = loaded["BP_0005"].attrs["experiment_metadata_json"]
    assert doc["records"]["5"]["experiment"]["data_format"] == "sweep"
    assert doc["records"]["20"]["experiment"]["data_format"] == "Au sweep"
    out = capsys.readouterr().out
    assert "2 file(s) indexed" in out and "metadata=datasheet" in out
    # The raw index never classifies: no gold/cuts on the object itself.
    assert not hasattr(loaded, "gold")


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
    """A converted folder without a datasheet still carries the metadata
    document embedded in each NetCDF (header only); classification stays with
    inspect_experiment, which sees the header sizes as real shapes."""
    import json as _json

    from peaksMCP.overrides import LoadedScans
    from peaksMCP.overrides import load as load_module
    from peaksMCP.overrides.inspection import inspect_experiment

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
    assert loaded.metadata_source == "embedded"
    entry0 = loaded.entries[0]
    assert entry0.experiment_index == 15
    assert entry0.representation == "netcdf"
    assert entry0.sizes == {"eV": 3, "theta_par": 4}
    assert entry0.converted is True
    # Classification from the embedded document + header dims (2-D sweeps).
    summary = inspect_experiment(loaded)
    assert set(summary.gold) == {20}
    assert set(summary.cuts) == {15}
    row15 = next(row for row in summary.records if row.index == 15)
    assert row15.dims == ["eV", "theta_par"]
    assert row15.energy_window_eV == (2.2, 2.7)
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
    assert entry.representation == "raw_pxt"
    assert loaded.entries[0].experiment_index == 9


def test_load_data_index_tags_processed_netcdf_entries(tmp_path, capsys):
    """*_processed.nc products index as processed_netcdf entries under their
    own stem; classification resolves the shared record index once, through
    the raw converted NetCDF (never the processed product's dims)."""
    import json as _json

    from peaksMCP.overrides import LoadedScans, load_data
    from peaksMCP.overrides import load as load_module
    from peaksMCP.overrides.inspection import inspect_experiment

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
    assert proc.representation == "processed_netcdf"
    assert proc.experiment_index == 5
    assert proc.sizes == {"eV": 4, "kx": 6}  # the product's own header dims
    assert loaded.processed == ["BP_0005_processed"]
    assert loaded.needs_conversion == []
    out = capsys.readouterr().out
    assert "processed=1" in out
    # One classification per experiment record; the raw NetCDF (eV/theta_par)
    # drives the shape, so the 2-D sweep stays a cut despite the processed
    # product carrying k-space dims.
    summary = inspect_experiment(loaded)
    assert set(summary.cuts) == {5}
    assert summary.gold == []
    row = summary.records[0]
    assert row.dims == ["eV", "theta_par"] and row.kind.value == "cut"


def test_load_data_accepts_experiment_root_with_subfolders(monkeypatch, tmp_path, capsys):
    """Pointing load_data at the experiment ROOT (data/ + data_netcdf/) works:
    subfolders are indexed with their own contexts, converted NetCDF wins over
    the raw PXT for the same stem, and inspect_experiment classifies."""
    import json as _json

    from peaksMCP.overrides import LoadedScans, load_data
    from peaksMCP.overrides import load as load_module
    from peaksMCP.overrides.inspection import inspect_experiment

    root = tmp_path / "BP260623"
    data_dir = root / "data"
    nc_dir = root / "data_netcdf"
    data_dir.mkdir(parents=True)
    nc_dir.mkdir()
    (data_dir / "BP_0005.pxt").write_bytes(b"x")
    (data_dir / "BP_0020.pxt").write_bytes(b"x")
    (nc_dir / "BP_0005.nc").write_bytes(b"x")
    (data_dir / "datasheet.csv").write_text(
        "Experiment title,,,,\n"
        "Index,Theta,Polarization,Temperture,Ei,Central Energy,Ef,slit,"
        "Pass E.,Data format,Comment\n"
        "5,430,S,9.4,2.2,,2.7,400,5,sweep,,\n"
        "20,430,S,9.4,2.2,,2.7,400,5,Au sweep,Au\n",
        encoding="utf-8",
    )

    def fake_pxt(path):
        return xr.DataArray(np.ones((3, 4)), dims=("eV", "theta_par"),
                            attrs={"units": "counts"})

    def fake_nc(path, lazy=True):
        import os

        stem = os.path.basename(os.fspath(path))[:-3]
        index = load_module._index_from_stem(stem)
        document = {"records": {"5": {"experiment": {"data_format": "sweep"}},
                                "20": {"experiment": {"data_format": "Au sweep"}}}}
        return xr.DataArray(
            np.ones((3, 4)), dims=("eV", "theta_par"),
            attrs={"experiment_index": index,
                   "experiment_metadata_json": _json.dumps(document)},
        ), "NetCDF"

    monkeypatch.setattr(load_module, "_single", fake_nc)  # nc header + data layer
    monkeypatch.setattr("peaksMCP.pxt_utils.loader.load_pxt", fake_pxt)
    monkeypatch.setattr(load_module, "_scan_pxt_header_sizes",
                        lambda path: {"eV": 3, "theta_par": 4})
    exp = load_data(str(root))
    assert isinstance(exp, LoadedScans)
    # Dedup: BP_0005 appears once (netcdf wins); BP_0020 stays raw pxt-only.
    assert exp.stems == ["BP_0005", "BP_0020"]
    assert exp.needs_conversion == ["BP_0020"]
    e5 = exp.entries[0]
    assert e5.file_kind == "netcdf" and e5.sizes == {"eV": 3, "theta_par": 4}
    summary = inspect_experiment(exp)
    assert set(summary.gold) == {20} and set(summary.cuts) == {5}
    out = capsys.readouterr().out
    assert "metadata=datasheet" in out and "data_netcdf" in out
    # repr is the one-line summary for print(exp).
    assert "file(s) indexed" in repr(exp)


def test_netcdf_safe_strips_unsafe_attrs_and_stringifies_units():
    """peaks-loaded arrays carry pint units and pydantic metadata models that
    raw to_netcdf rejects; the staged copy must survive serialisation while
    keeping values and coordinates intact."""
    from dataclasses import dataclass

    from peaksMCP.overrides.save import _netcdf_safe

    @dataclass
    class FakeModel:
        loc: str = "L112"

    import pint

    ureg = pint.UnitRegistry()
    value = xr.DataArray(
        np.arange(6, dtype=float).reshape(2, 3),
        dims=("eV", "theta_par"),
        coords={
            "eV": xr.DataArray([0.0, 1.0], dims="eV",
                               attrs={"units": ureg.electron_volt}),
            "theta_par": xr.DataArray([0.0, 1.0, 2.0], dims="theta_par",
                                      attrs={"units": "deg"}),
        },
        attrs={"units": "counts", "_scan": FakeModel(), "title": "BP"},
    )
    safe = _netcdf_safe(value)
    assert "_scan" not in safe.attrs and safe.attrs["title"] == "BP"
    assert isinstance(safe.coords["eV"].attrs["units"], str)
    assert safe.coords["theta_par"].attrs["units"] == "deg"
    assert np.array_equal(np.asarray(safe.values), np.asarray(value.values))
    # And the sanitised copy actually serialises.
    import io

    buffer = io.BytesIO()
    safe.to_netcdf(buffer)
    assert buffer.getvalue()


def test_batch_ticket_stages_many_and_publishes_on_approval(tmp_path, capsys):
    """A batch verb (convert/preprocess) stages N files under ONE ticket; the
    card lists every item and approval publishes all of them."""
    from peaksMCP.overrides import save as save_module

    targets = [tmp_path / f"BP_000{i}.nc" for i in (5, 6, 9)]
    requests = [(_array(), target, False) for target in targets]
    monkeypatch, seen = _approval(True)
    try:
        outcome = save_module._run_staged("convert_experiment", requests, "convert 3 files")
    finally:
        monkeypatch.undo()
    assert outcome["status"] == "saved"
    assert sorted(p["path"] for p in outcome["published"]) == sorted(
        str(t) for t in targets
    )
    payload = seen["payload"]
    assert payload["operation"] == "convert_experiment"
    assert len(payload["items"]) == 3
    assert all(item["path"].endswith(".nc") for item in payload["items"])
    assert all(target.exists() for target in targets)
    assert not list(tmp_path.glob(".*.part-*"))


def test_batch_ticket_denied_cleans_everything(tmp_path, capsys):
    from peaksMCP.overrides import save as save_module

    targets = [tmp_path / f"BP_00{i}.nc" for i in (5, 6)]
    monkeypatch, _ = _approval(False)
    try:
        outcome = save_module._run_staged(
            "preprocess_batch", [(_array(), t, False) for t in targets], "pre 2"
        )
    finally:
        monkeypatch.undo()
    assert outcome["status"] == "denied"
    assert not any(target.exists() for target in targets)
    assert not list(tmp_path.glob(".*.part-*"))


def test_batch_publish_skips_existing_targets_without_overwrite(tmp_path, capsys):
    """Idempotent verbs: an item whose target already exists at publish time
    is skipped (staging removed) unless overwrite=True."""
    from peaksMCP.overrides import save as save_module

    existing = tmp_path / "BP_0005.nc"
    existing.write_bytes(b"already-converted")
    fresh = tmp_path / "BP_0006.nc"
    outcome = save_module._run_staged(
        "convert_experiment",
        [(_array(), existing, False), (_array(), fresh, False)],
        "convert 2",
    )
    # No channel -> pending; gateway must not publish either file.
    assert outcome["status"] == "pending_consent"
    assert not fresh.exists()
    save_module._discard_ticket(outcome["ticket_id"])

    # With approval: fresh publishes, existing is skipped (kept untouched).
    monkeypatch, _ = _approval(True)
    try:
        outcome = save_module._run_staged(
            "convert_experiment",
            [(_array(), existing, False), (_array(), fresh, False)],
            "convert 2",
        )
    finally:
        monkeypatch.undo()
    assert outcome["status"] == "saved"
    assert [p["path"] for p in outcome["published"]] == [str(fresh)]
    assert existing.read_bytes() == b"already-converted"
    assert fresh.exists()
    assert not list(tmp_path.glob(".*.part-*"))
