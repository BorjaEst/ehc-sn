"""Metric summary normalizer protocol.

A normalizer converts a flat eval-artifact summary dict into a list
of :class:`MetricRecord` rows according to an allowlist.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from ehc_sn.reporting.metrics.spec import MetricNormalizerSpec
from ehc_sn.reporting.schema import EvalArtifactReference, MetricRecord


class MetricSummaryNormalizer(Protocol):
    """Structural protocol for metric summary normalizers.

    Implementations must expose a ``spec`` attribute and a ``normalize``
    method matching this signature.
    """

    spec: MetricNormalizerSpec

    def normalize(
        self,
        *,
        artifact_ref: EvalArtifactReference,
        summary: Mapping[str, object],
    ) -> Sequence[MetricRecord]:
        """Convert a flat summary dict into normalized metric records.

        Parameters
        ----------
        artifact_ref:
            Reference to the eval artifact being normalized (provides
            ``task`` and ``regime_id`` for the output records).
        summary:
            The ``manifest["summary"]`` dict — flat key-value scalars.

        Returns
        -------
        Sequence[MetricRecord]
            One record per present allowlisted field, in spec declaration
            order.
        """


# ---------------------------------------------------------------------------
__all__ = ["MetricSummaryNormalizer"]
