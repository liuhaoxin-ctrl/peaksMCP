"""Translate the L112 two-header-row datasheet into normalized metadata."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any

from .models import ExperimentMetadata, ExperimentRecord

_KNOWN_FIELDS = {
    "Index",
    "Theta",
    "Polarization",
    "Temperture",
    "Temperature",
    "Ei",
    "Central Energy",
    "Ef",
    "slit",
    "Pass E.",
    "L. Power",
    "N.S.",
    "Data format",
    "Comment",
}
_DISCARDED_FIELDS = {"L. Power", "N.S."}


def _number(value: str) -> float | None:
    text = (value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _unique_headers(headers: list[str]) -> list[str]:
    output: list[str] = []
    counts: dict[str, int] = {}
    for position, header in enumerate(headers):
        name = header.strip() or f"column_{position + 1}"
        counts[name] = counts.get(name, 0) + 1
        output.append(name if counts[name] == 1 else f"{name}_{counts[name]}")
    return output


def _record(index: int, row: dict[str, str], warnings: list[str]) -> ExperimentRecord:
    temperature = _number(row.get("Temperture") or row.get("Temperature") or "")
    central = _number(row.get("Central Energy", ""))
    pass_energy = _number(row.get("Pass E.", ""))
    slit = _number(row.get("slit", ""))
    start = _number(row.get("Ei", ""))
    stop = _number(row.get("Ef", ""))
    theta = _number(row.get("Theta", ""))
    if row.get("Theta", "").strip() and theta is None:
        warnings.append(f"Index {index}: Theta is not numeric")

    unmapped: dict[str, Any] = {}
    for key, value in row.items():
        text = (value or "").strip()
        if text and key not in _KNOWN_FIELDS and key not in _DISCARDED_FIELDS:
            unmapped[key] = text

    analyser: dict[str, Any] = {}
    if central is not None:
        analyser.setdefault("scan", {})["center_eV"] = central
    if pass_energy is not None:
        analyser.setdefault("scan", {})["pass_energy_eV"] = pass_energy
    if slit is not None:
        analyser.setdefault("slit", {})["width_um"] = slit

    experiment: dict[str, Any] = {}
    if start is not None:
        experiment["energy_start_eV"] = start
    if stop is not None:
        experiment["energy_stop_eV"] = stop
    if row.get("Data format", "").strip():
        experiment["data_format"] = row["Data format"].strip()
    if row.get("Comment", "").strip():
        experiment["comment"] = row["Comment"].strip()

    polarization = row.get("Polarization", "").strip()
    return ExperimentRecord(
        index=index,
        polarization_angle_deg=theta,
        photon={"polarisation": polarization} if polarization else {},
        temperature={"sample": temperature, "unit": "K"}
        if temperature is not None
        else {},
        analyser=analyser,
        experiment=experiment,
        unmapped=unmapped,
    )


def translate_datasheet(
    csv_path: str | Path,
    output: str | Path | None = None,
) -> ExperimentMetadata:
    """Translate a two-header-row L112 datasheet.

    Parameters
    ----------
    csv_path : path-like
        CSV whose first row contains the experiment title and whose second row contains fields.
    output : path-like, optional
        Destination JSON path. If omitted, no file is written.

    Returns
    -------
    ExperimentMetadata
        Validated experiment document with records keyed by ``Index``.

    Raises
    ------
    ValueError
        If the file has fewer than two rows, lacks ``Index`` or repeats an index.

    Examples
    --------
    >>> metadata = translate_datasheet("datasheet.csv", "experiment_metadata.json")
    >>> isinstance(metadata.records, dict)
    True
    """
    source = Path(csv_path).expanduser().resolve()
    raw = source.read_bytes()
    rows = list(csv.reader(raw.decode("utf-8-sig").splitlines()))
    if len(rows) < 2:
        raise ValueError("datasheet must contain a title row and a header row")
    title = next((cell.strip() for cell in rows[0] if cell.strip()), "")
    header_notes = [
        cell.strip()
        for cell in rows[1]
        if "note" in cell.lower() or "给agent" in cell.lower()
    ]
    headers = _unique_headers(rows[1])
    if "Index" not in headers:
        raise ValueError("datasheet header must contain Index")
    warnings: list[str] = []
    records: dict[str, ExperimentRecord] = {}
    for line_number, values in enumerate(rows[2:], start=3):
        values = [*values, *([""] * max(0, len(headers) - len(values)))]
        row = dict(zip(headers, values, strict=False))
        index_text = row.get("Index", "").strip()
        if not index_text:
            if any(value.strip() for value in values):
                warnings.append(f"line {line_number}: non-empty row without Index was skipped")
            continue
        try:
            numeric = float(index_text)
        except ValueError as exc:
            raise ValueError(f"line {line_number}: invalid Index {index_text!r}") from exc
        if not numeric.is_integer():
            raise ValueError(
                f"line {line_number}: Index {index_text!r} is not an integer "
                "(1.9 would silently truncate to 1 and attach the wrong record)"
            )
        index = int(numeric)
        key = str(index)
        if key in records:
            raise ValueError(f"line {line_number}: duplicate Index {index}")
        records[key] = _record(index, row, warnings)

    metadata = ExperimentMetadata(
        title=title,
        notes=header_notes,
        source_csv=str(source),
        source_sha256=hashlib.sha256(raw).hexdigest(),
        records=records,
        warnings=warnings,
    )
    if output is not None:
        metadata.write(output)
    return metadata
