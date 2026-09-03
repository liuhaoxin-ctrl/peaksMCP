"""Translate the L112 two-header-row datasheet into normalized metadata."""

from __future__ import annotations

import csv
import hashlib
import re
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


def _numeric_field(
    row: dict[str, str], field: str, index: int, warnings: list[str]
) -> float | None:
    raw = row.get(field, "")
    value = _number(raw)
    if raw.strip() and value is None:
        warnings.append(f"Index {index}: {field} is not numeric: {raw.strip()!r}")
    return value


def _note_kind(header: str) -> str:
    """Classify a datasheet note column as 'agent' (AI 请看) or 'human' (AI 别看).

    Headers that state the note must NOT be shown (``不要`` / ``别看`` / ``请勿``
    / ``hidden``) are AI-hidden; the rest that invite the AI (``请`` / ``agent``)
    are AI-visible; anything else (plain ``note``) is AI-hidden.
    """
    lowered = header.lower()
    if any(token in lowered for token in ("不要", "别看", "请勿", "勿看", "hidden")):
        return "human"
    if "请" in header or "agent" in lowered:
        return "agent"
    return "human"


#: Number following the ``theta_offset`` token in a note, e.g.
#: ``theta_offset：1.5`` / ``theta_offset= -0.5度``.
_THETA_OFFSET_RE = re.compile(
    r"theta[_ ]?offset\s*[:：=]?\s*([+-]?\d+(?:\.\d+)?)", re.IGNORECASE
)


def _theta_offset_deg(value: str) -> float | None:
    """Return the first ``theta_offset``-prefixed number in a note, or None."""
    match = _THETA_OFFSET_RE.search(value or "")
    if match is None:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _is_gold_format(data_format: str) -> bool:
    """True when ``Data format`` marks this index as a gold (Au) reference.

    The datasheet tags gold data (used for Fermi-edge fitting) with ``Au`` /
    ``gold`` / ``金`` in the ``Data format`` column, either alone or combined
    with the scan type in a comma/space separated list, e.g. ``Au``,
    ``Au sweep`` or ``sweep,Au`` (the real L112 datasheet uses the last form).
    """
    raw = (data_format or "").strip()
    lowered = (
        raw.lower()
        .replace("_", " ")
        .replace("-", " ")
        .replace(",", " ")
        .replace(";", " ")
        .split()
    )
    return "au" in lowered or "gold" in lowered or "金" in raw


def _agent_note_from_header(header: str) -> str | None:
    """Extract the AI-visible note text embedded in a note-column HEADER.

    The L112 datasheet stores the AI-facing note in the column header itself,
    e.g. ``AI请看的Note：Cut高对称点差不多在+1.5度``.  The part after the first
    full-width/half-width colon is the note content; nothing after the colon
    means the header carries no AI-visible note.
    """
    for sep in ("：", ":"):
        if sep in header:
            content = header.split(sep, 1)[1].strip()
            return content if content else None
    return None


def _is_note_marker(value: str) -> bool:
    """True when a cell is only a column-type marker with no real content.

    e.g. ``AI不要看的Note：`` (colon followed by nothing) is a label, not a note.
    """
    text = (value or "").strip()
    for sep in ("：", ":"):
        if sep in text:
            head, tail = text.split(sep, 1)
            if not tail.strip() and "note" in head.lower():
                return True
    return False



def _record(
    index: int,
    row: dict[str, str],
    warnings: list[str],
    note_headers: set[str],
) -> ExperimentRecord:
    temperature_field = "Temperture" if row.get("Temperture", "").strip() else "Temperature"
    temperature = _numeric_field(row, temperature_field, index, warnings)
    central = _numeric_field(row, "Central Energy", index, warnings)
    pass_energy = _numeric_field(row, "Pass E.", index, warnings)
    slit = _numeric_field(row, "slit", index, warnings)
    start = _numeric_field(row, "Ei", index, warnings)
    stop = _numeric_field(row, "Ef", index, warnings)
    theta = _numeric_field(row, "Theta", index, warnings)

    unmapped: dict[str, Any] = {}
    for key, value in row.items():
        text = (value or "").strip()
        if text and key not in _KNOWN_FIELDS and key not in _DISCARDED_FIELDS and key not in note_headers:
            unmapped[key] = text
            warnings.append(f"Index {index}: unmapped field {key!r} was preserved")

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
    data_format = row.get("Data format", "").strip()
    if data_format:
        experiment["data_format"] = data_format
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
        is_gold_reference=_is_gold_format(data_format),
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
    headers = _unique_headers(rows[1])
    note_headers = {
        header: _note_kind(header)
        for header in headers
        if "note" in header.lower() or "给agent" in header.lower()
    }
    if "Index" not in headers:
        raise ValueError("datasheet header must contain Index")
    warnings: list[str] = []
    notes: list[str] = []
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
        records[key] = _record(index, row, warnings, set(note_headers))
        # Only AI-visible ("agent") note-column values enter the metadata;
        # human-only columns (``AI不要看的Note`` / plain ``note``) are never
        # exposed to the agent.  The number following ``theta_offset`` in an
        # agent note becomes this record's structured ``theta_offset_deg``;
        # absent that token the field stays None (callers ask the user).
        for header, kind in note_headers.items():
            if kind != "agent":
                continue
            value = row.get(header, "").strip()
            if not value or _is_note_marker(value):
                continue
            notes.append(f"Index {index}: {value}")
            if records[key].theta_offset_deg is None:
                records[key].theta_offset_deg = _theta_offset_deg(value)

    # AI-visible notes embedded in note-column headers come first (agent reads
    # them before the per-Index notes).  The header may also carry the
    # experiment-wide high-symmetry offset (``AI请看的Note：Cut theta_offset=1.5``);
    # parse it and backfill any record that has no per-row offset of its own, so
    # ``process_cut`` finds ``theta_offset_deg`` on every record.
    agent_notes = []
    for header, kind in note_headers.items():
        if kind != "agent":
            continue
        content = _agent_note_from_header(header)
        if not content:
            continue
        label = header.split("：", 1)[0] if "：" in header else header.split(":", 1)[0]
        agent_notes.append(f"{label}：{content}")
        header_offset = _theta_offset_deg(content)
        if header_offset is not None:
            for record in records.values():
                if record.theta_offset_deg is None:
                    record.theta_offset_deg = header_offset
    metadata = ExperimentMetadata(
        title=title,
        notes=[*agent_notes, *notes],
        source_csv=str(source),
        source_sha256=hashlib.sha256(raw).hexdigest(),
        records=records,
        warnings=warnings,
    )
    if output is not None:
        metadata.write(output)
    return metadata
