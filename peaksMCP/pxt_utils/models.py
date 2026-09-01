"""Data models for translated experiment metadata and conversion reports."""

from __future__ import annotations

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
        """Atomically write the metadata document as UTF-8 JSON."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".part")
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
