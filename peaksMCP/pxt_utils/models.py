"""Data models for translated experiment metadata and conversion reports."""

from __future__ import annotations

import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ExperimentRecord(BaseModel):
    """Normalized metadata associated with one datasheet index."""

    index: int
    polarization_angle_deg: float | None = None
    photon: dict[str, Any] = Field(default_factory=dict)
    temperature: dict[str, Any] = Field(default_factory=dict)
    analyser: dict[str, Any] = Field(default_factory=dict)
    experiment: dict[str, Any] = Field(default_factory=dict)
    #: True when the datasheet marks this index as a gold (Au) reference in the
    #: ``Data format`` column — the record an agent fits a Fermi edge on to get
    #: ``EF_correction`` for the ordinary sweep data.
    is_gold_reference: bool = False
    #: High-symmetry angle offset (degrees) parsed from this index's AI-visible
    #: notes (``theta_offset...<number>``), when present.  None when the
    #: datasheet did not state it — callers then ask the user rather than invent.
    theta_offset_deg: float | None = None
    unmapped: dict[str, Any] = Field(default_factory=dict)


class ExperimentMetadata(BaseModel):
    """Experiment-level metadata document written next to converted data."""

    schema_version: int = 1
    title: str = ""
    notes: list[str] = Field(default_factory=list)
    source_csv: str
    source_sha256: str
    translated_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )
    records: dict[str, ExperimentRecord] = Field(default_factory=dict)
    discarded_fields: list[str] = Field(default_factory=lambda: ["L. Power", "N.S."])
    warnings: list[str] = Field(default_factory=list)

    def write(self, path: str | Path) -> Path:
        """Atomically write the metadata document as UTF-8 JSON.

        A unique temporary sibling (mkstemp) is used so concurrent writers to
        the same target cannot truncate each other's staging file; the rename
        then publishes exactly one complete document.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".part",
            dir=target.parent,
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        temporary.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(target)
        return target


class ConversionItem(BaseModel):
    """One PXT-to-NetCDF conversion result."""

    input: str
    output: str | None = None
    index: int | None = None
    status: str
    error_type: str | None = None
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)
    # Whether the output file actually exists on disk.  A ``skipped`` item may
    # mean "already on disk" (output exists) or "CPU budget wait timed out"
    # (output does NOT exist); ``cancelled`` items have no output either.  Only
    # items with output_exists=True can be loaded into the notebook.
    output_exists: bool = False


class ConversionReport(BaseModel):
    """Batch conversion result."""

    items: list[ConversionItem] = Field(default_factory=list)
    cpu: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    @property
    def converted(self) -> int:
        """Return the number of successfully converted files."""
        return sum(item.status == "converted" for item in self.items)

    @property
    def failed(self) -> int:
        """Return the number of failed files."""
        return sum(item.status == "failed" for item in self.items)


class ConversionTask(BaseModel):
    """Pickleable input passed to a conversion worker."""

    input_path: str
    output_path: str
    metadata_path: str | None = None
    force: bool = False
