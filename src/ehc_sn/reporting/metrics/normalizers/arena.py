"""Arena metric summary normalizer.

Converts the ``manifest["summary"]`` dict produced by Arena evaluation
into :class:`MetricRecord` rows for report consumption.

The normalizer preserves pre-computed values from the summary. Accuracy
ratio fields use pathway-qualified names (``accuracy_post_*``,
``accuracy_recall_*``, ``accuracy_path_*``) as emitted by the
TEM adapter.  Count fields use model-agnostic names (``correct_all``,
``count_all``, etc.) from the task-level ``ArenaScoreReport``.
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
# The ``aggregate_evaluation_case_metrics`` hook on TEM-style models emits
# pathway-qualified ratio metrics. The model-agnostic count fields below
# are marked ``required=False`` so the normalizer accepts the reduced
# summary dict without raising.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Allowlist
# ---------------------------------------------------------------------------

_ARENA_METRICS: list[MetricSummaryField] = [
    MetricSummaryField(
        source_key="accuracy_post_all",
        metric="accuracy_post_all",
        unit="ratio",
        higher_is_better=True,
        required=False,
    ),
    MetricSummaryField(
        source_key="accuracy_post_revisit",
        metric="accuracy_post_revisit",
        unit="ratio",
        higher_is_better=True,
        required=False,
    ),
    MetricSummaryField(
        source_key="accuracy_recall_all",
        metric="accuracy_recall_all",
        unit="ratio",
        higher_is_better=True,
        required=False,
    ),
    MetricSummaryField(
        source_key="accuracy_recall_revisit",
        metric="accuracy_recall_revisit",
        unit="ratio",
        higher_is_better=True,
        required=False,
    ),
    MetricSummaryField(
        source_key="accuracy_path_all",
        metric="accuracy_path_all",
        unit="ratio",
        higher_is_better=True,
        required=False,
    ),
    MetricSummaryField(
        source_key="accuracy_path_revisit",
        metric="accuracy_path_revisit",
        unit="ratio",
        higher_is_better=True,
        required=False,
    ),
    MetricSummaryField(
        source_key="correct_all",
        metric="correct_all",
        unit="count",
        higher_is_better=True,
        required=False,
    ),
    MetricSummaryField(
        source_key="count_all",
        metric="count_all",
        unit="count",
        higher_is_better=None,
        required=False,
    ),
    MetricSummaryField(
        source_key="correct_revisit",
        metric="correct_revisit",
        unit="count",
        higher_is_better=True,
        required=False,
    ),
    MetricSummaryField(
        source_key="count_revisit",
        metric="count_revisit",
        unit="count",
        higher_is_better=None,
        required=False,
    ),
]

# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------

_ARENA_SPEC = MetricNormalizerSpec(
    task="arena",
    regime_kind="diagnostic",
    fields=_ARENA_METRICS,
)


# ---------------------------------------------------------------------------
# Normalizer
# ---------------------------------------------------------------------------


class ArenaMetricNormalizer:
    """Normalizes Arena diagnostic eval-artifact summaries to MetricRecord."""

    spec: MetricNormalizerSpec = _ARENA_SPEC  # type: ignore[assignment]
    """Class-level spec used for registry resolution."""

    def normalize(
        self,
        *,
        artifact_ref: EvalArtifactReference,
        summary: Mapping[str, object],
    ) -> Sequence[MetricRecord]:
        """Normalize an Arena diagnostic summary into metric records."""
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

ARENA_NORMALIZER = ArenaMetricNormalizer()

# ---------------------------------------------------------------------------
__all__ = ["ArenaMetricNormalizer", "ARENA_NORMALIZER"]
