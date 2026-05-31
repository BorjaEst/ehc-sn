"""Metric normalizer specification models.

Defines the typed declaration for converting a flat evaluator-artifact
``summary`` dict into normalized :class:`MetricRecord` rows.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ehc_sn.reporting.schema import RegimeKind


# ---------------------------------------------------------------------------
# Per-field specification
# ---------------------------------------------------------------------------


class MetricSummaryField(BaseModel, extra="forbid"):
    """One allowlisted metric entry, mapping a summary key to a report metric.

    Parameters
    ----------
    source_key:
        Key in the eval-artifact ``manifest.summary`` dict.
    metric:
        Normalized metric name in the output :class:`MetricRecord`.
    unit:
        Optional measurement unit (e.g. ``"ratio"``, ``"count"``, ``"steps"``).
    higher_is_better:
        Whether larger values are better for this metric.
        ``None`` means the direction is not meaningful (e.g. sample counts).
    required:
        Whether the metric must be present in the summary dict.
        Missing required fields raise ``KeyError`` during normalization.
        Missing optional fields are silently omitted.
    """

    source_key: str = Field(
        ..., min_length=1, description="Key in manifest.summary."
    )
    metric: str = Field(
        ..., min_length=1, description="Normalized MetricRecord metric name."
    )
    unit: str | None = None
    higher_is_better: bool | None = None
    required: bool = True


# ---------------------------------------------------------------------------
# Per-(task, regime_kind) normalizer spec
# ---------------------------------------------------------------------------


class MetricNormalizerSpec(BaseModel, extra="forbid"):
    """Declares which summary keys become :class:`MetricRecord` rows.

    One spec per (task, regime_kind) pair.  Fields are evaluated in
    declaration order; output rows follow the same order.
    """

    task: str = Field(..., min_length=1)
    regime_kind: RegimeKind
    fields: list[MetricSummaryField] = Field(
        ...,
        min_length=1,
        description="Non-empty allowlist of metric definitions.",
    )


# ---------------------------------------------------------------------------
__all__ = ["MetricSummaryField", "MetricNormalizerSpec"]
