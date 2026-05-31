"""Report metric normalizers — convert eval-artifact summaries to MetricRecord.

This package implements positive-allowlist metric normalization for
report-ready :class:`MetricRecord` rows.  Each task/regime pair has an
explicit normalizer; unknown pairs fail with ``KeyError``.

Boundary rule:
    This package may import from ``ehc_sn.reporting.schema`` and
    ``ehc_sn.reporting.loaders`` only.  It must not import from
    ``tasks/``, ``models/``, ``adapters/``, ``lightning/``, or ``eval/``.
"""

from ehc_sn.reporting.metrics.normalizers.arena import ARENA_NORMALIZER
from ehc_sn.reporting.metrics.normalizers.mazehard import (
    MAZEHARD_NORMALIZER,
)
from ehc_sn.reporting.metrics.protocol import MetricSummaryNormalizer
from ehc_sn.reporting.metrics.registry import (
    MetricNormalizerKey,
    MetricNormalizerRegistry,
)
from ehc_sn.reporting.metrics.spec import (
    MetricNormalizerSpec,
    MetricSummaryField,
)

# ---------------------------------------------------------------------------
# Pre-built default registry with all known normalizers.
# ---------------------------------------------------------------------------

DEFAULT_REGISTRY = MetricNormalizerRegistry()
DEFAULT_REGISTRY.register(MAZEHARD_NORMALIZER)
DEFAULT_REGISTRY.register(ARENA_NORMALIZER)

# Backward-compatible alias (deprecated — prefer DEFAULT_REGISTRY).
DEFAULT_MAZEHARD_NORMALIZER = DEFAULT_REGISTRY


# ---------------------------------------------------------------------------
__all__ = [
    "DEFAULT_MAZEHARD_NORMALIZER",
    "DEFAULT_REGISTRY",
    "MetricNormalizerKey",
    "MetricNormalizerRegistry",
    "MetricNormalizerSpec",
    "MetricSummaryField",
    "MetricSummaryNormalizer",
]
