"""MazeHard metric summary normalizer.

Converts the ``manifest["summary"]`` dict produced by the HRM v2
``aggregate_evaluation_case_metrics`` hook into :class:`MetricRecord` rows
for report consumption.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ehc_sn.metrics.values import validate_metric_value
from ehc_sn.reporting.metrics.protocol import MetricSummaryNormalizer
from ehc_sn.reporting.metrics.spec import (
    MetricNormalizerSpec,
    MetricSummaryField,
)
from ehc_sn.reporting.schema import EvalArtifactReference, MetricRecord

# ---------------------------------------------------------------------------
# Allowlist
# ---------------------------------------------------------------------------

_MAZEHARD_METRICS: list[MetricSummaryField] = [
    MetricSummaryField(
        source_key="token_accuracy",
        metric="token_accuracy",
        unit="ratio",
        higher_is_better=True,
        required=True,
    ),
    MetricSummaryField(
        source_key="sequence_accuracy",
        metric="sequence_accuracy",
        unit="ratio",
        higher_is_better=True,
        required=True,
    ),
    MetricSummaryField(
        source_key="sequence_exact",
        metric="sequence_exact",
        unit="ratio",
        higher_is_better=True,
        required=True,
    ),
    MetricSummaryField(
        source_key="n_token_correct",
        metric="n_token_correct",
        unit="count",
        higher_is_better=True,
        required=True,
    ),
    MetricSummaryField(
        source_key="n_token_total",
        metric="n_token_total",
        unit="count",
        higher_is_better=None,
        required=True,
    ),
    MetricSummaryField(
        source_key="n_sequence_completed",
        metric="n_sequence_completed",
        unit="count",
        higher_is_better=None,
        required=True,
    ),
    MetricSummaryField(
        source_key="n_sequence_eligible",
        metric="n_sequence_eligible",
        unit="count",
        higher_is_better=None,
        required=True,
    ),
]

# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------

_MAZEHARD_SPEC = MetricNormalizerSpec(
    task="mazehard",
    regime_kind="diagnostic",
    fields=_MAZEHARD_METRICS,
)


# ---------------------------------------------------------------------------
# Normalizer
# ---------------------------------------------------------------------------


class MazeHardMetricNormalizer:
    """Normalizes MazeHard diagnostic eval-artifact summaries to MetricRecord."""

    spec: MetricNormalizerSpec = _MAZEHARD_SPEC  # type: ignore[assignment]
    """Class-level spec used for registry resolution."""

    def normalize(
        self,
        *,
        artifact_ref: EvalArtifactReference,
        summary: Mapping[str, object],
    ) -> Sequence[MetricRecord]:
        """Normalize a MazeHard diagnostic summary into metric records."""
        records: list[MetricRecord] = []

        for field in self.spec.fields:
            value = summary.get(field.source_key)

            if value is None:
                if field.required:
                    raise KeyError(
                        f"Required metric {field.source_key!r} is missing "
                        f"from the summary dict."
                    )
                continue  # optional; silently omit

            validated = validate_metric_value(key=field.source_key, value=value)

            records.append(
                MetricRecord(
                    task=artifact_ref.task,
                    regime_id=artifact_ref.regime_id,
                    metric=field.metric,
                    value=validated,
                    unit=field.unit,
                    higher_is_better=field.higher_is_better,
                )
            )

        return records


# ---------------------------------------------------------------------------
# Singleton instance for registry registration
# ---------------------------------------------------------------------------

MAZEHARD_NORMALIZER = MazeHardMetricNormalizer()

# ---------------------------------------------------------------------------
__all__ = ["MazeHardMetricNormalizer", "MAZEHARD_NORMALIZER"]
