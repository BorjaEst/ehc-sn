"""Report-run provenance model.

Defines the typed ``ReportRunProvenance`` contract for
``provenance.json`` written by the report-run manifest writer.

This module must remain free of ``lightning/``, ``eval/``, ``models/``,
``tasks/``, ``adapters/``, and benchmark imports.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ehc_sn.reporting.schema import CheckpointSpec

# ---------------------------------------------------------------------------
# Code provenance
# ---------------------------------------------------------------------------


class CodeProvenance(BaseModel, extra="forbid"):
    """Optional version-control metadata for the code that produced a report."""

    git_commit: str | None = Field(
        default=None,
        min_length=7,
        description="Full or abbreviated Git commit hash.",
    )
    git_dirty: bool | None = Field(
        default=None,
        description="Whether the working tree had uncommitted changes.",
    )


# ---------------------------------------------------------------------------
# ReportRunProvenance
# ---------------------------------------------------------------------------


class ReportRunProvenance(BaseModel, extra="forbid"):
    """Provenance record for one completed report run.

    All fields are populated at write time.  Once written, the record
    is immutable on disk (the containing directory is guarded by the
    ``_SUCCESS`` sentinel).

    The authoritative statement of what the report run contains is in
    :class:`ReportRunManifest.eval_artifacts`; provenance owns only
    identity and creation metadata.
    """

    schema_version: Literal["ehc_sn.reporting.provenance.v1"] = (
        "ehc_sn.reporting.provenance.v1"
    )
    model_family: str = Field(..., min_length=1)
    checkpoint: CheckpointSpec
    source_spec: Path | None = Field(
        default=None,
        description="Path to the ReportSpec file that produced this run.",
    )
    created_at: datetime = Field(
        ...,
        description="UTC creation timestamp.",
    )
    code: CodeProvenance | None = None
