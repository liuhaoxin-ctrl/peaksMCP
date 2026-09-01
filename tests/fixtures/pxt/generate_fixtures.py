"""Generate tiny, non-experimental Igor packed-experiment test fixtures.

This developer utility requires ``igorwriter`` but the generated fixtures and
the peaksMCP runtime do not.  Run it from this directory after installing
igorwriter when the binary fixtures need to be regenerated.
"""

from __future__ import annotations

import io
import struct
from pathlib import Path

import numpy as np
from igorwriter import IgorWave

HERE = Path(__file__).parent


def _wave_bytes(
    values: np.ndarray,
    name: str,
    axes: list[tuple[str, float, float, str, str]],
    *,
    note: str,
) -> bytes:
    wave = IgorWave(np.asarray(values, dtype=np.float32), name=name)
    for axis, start, step, unit, label in axes:
        wave.set_dimscale(axis, start, step, unit)
        wave.set_dimlabel("xyzt".index(axis), -1, label)
    wave.set_datascale("counts")
    wave.set_note(note)
    output = io.BytesIO()
    wave.save(output)
    return output.getvalue()


def _record(record_type: int, version: int, data: bytes = b"") -> bytes:
    return struct.pack("<Hhi", record_type, version, len(data)) + data


def _folder_start(name: str) -> bytes:
    return _record(9, 0, name.encode("utf-8") + b"\x00")


def _folder_end() -> bytes:
    return _record(10, 0)


def generate() -> None:
    """Write deterministic 2D and 3D PXT fixtures beside this script."""
    cut = _wave_bytes(
        np.arange(12, dtype=np.float32).reshape(4, 3),
        "ARPES2D",
        [
            ("x", 3.0, -0.25, "eV", "Kinetic Energy"),
            ("y", -10.0, 5.0, "deg", "ThetaX"),
        ],
        note="Synthetic ARPES 2D regression wave",
    )
    fixture_2d = (
        _folder_start("Experiment")
        + _folder_start("Nested")
        + _record(3, 5, cut)
        + _folder_end()
        + _folder_end()
    )
    (HERE / "synthetic_2d_nested.pxt").write_bytes(fixture_2d)

    mapping = _wave_bytes(
        np.arange(24, dtype=np.float32).reshape(4, 3, 2),
        "ARPES3D",
        [
            ("x", 3.0, -0.25, "eV", "Kinetic Energy"),
            ("y", -10.0, 5.0, "deg", "ThetaX"),
            ("z", 2.0, -2.0, "deg", "ThetaY"),
        ],
        note="Synthetic ARPES 3D regression wave",
    )
    fixture_3d = (
        _folder_start("Experiment")
        + _record(3, 5, mapping)
        + _folder_end()
    )
    (HERE / "synthetic_3d.PXT").write_bytes(fixture_3d)

    # Mirror the real Elettra VUV chunk layout: one data wave (chunkcube) plus
    # five per-axis helper waves under a DA_infoWaves folder.  The loader must
    # ignore the helpers and pick chunkcube as the single data wave.
    chunkcube = _wave_bytes(
        np.arange(24, dtype=np.float32).reshape(4, 3, 2),
        "chunkcube",
        [
            ("x", 1.92, 0.01, "eV", "Energy"),
            ("y", -18.736, 0.043221, "deg", "Thetax"),
            ("z", -15.0, 1.0, "deg", "Thetay"),
        ],
        note="Synthetic ARPES 3D chunk-cube regression wave",
    )
    info_waves = [
        ("chunkImage", np.zeros((2, 2), dtype=np.float32)),
        ("deltaInfoWave", np.array([0.01, 0.043221, 1.0], dtype=np.float32)),
        ("dimInfoWave", np.array([4, 3, 2], dtype=np.float32)),
        ("labelInfoWave", np.array([0, 0, 0], dtype=np.float32)),
        ("offsetInfoWave", np.array([1.92, -18.736, -15.0], dtype=np.float32)),
    ]
    info_bytes = b"".join(
        _record(3, 5, _wave_bytes(values, name, [], note=""))
        for name, values in info_waves
    )
    fixture_mapping = (
        _record(3, 5, chunkcube)
        + _folder_start("DA_infoWaves")
        + info_bytes
        + _folder_end()
    )
    (HERE / "synthetic_mapping_with_info_waves.pxt").write_bytes(fixture_mapping)


if __name__ == "__main__":
    generate()
