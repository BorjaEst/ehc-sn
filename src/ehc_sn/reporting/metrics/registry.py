"""Metric normalizer registry.

Maps (task, regime_kind) pairs to concrete :class:`MetricSummaryNormalizer`
implementations.  Resolve raises ``KeyError`` for unsupported pairs.
"""

from __future__ import annotations

from dataclasses import dataclass

from ehc_sn.reporting.metrics.protocol import MetricSummaryNormalizer

# ---------------------------------------------------------------------------
# Registry key
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricNormalizerKey:
    """Unique key for looking up a normalizer in the registry."""

    task: str
    regime_kind: str


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class MetricNormalizerRegistry:
    """Typed registry of (task, regime_kind) → MetricSummaryNormalizer.

    Usage::

        registry = MetricNormalizerRegistry()
        registry.register(my_normalizer)
        normalizer = registry.resolve(task="mazehard", regime_kind="diagnostic")
    """

    def __init__(self) -> None:
        self._entries: dict[MetricNormalizerKey, MetricSummaryNormalizer] = {}

    def register(self, normalizer: MetricSummaryNormalizer) -> None:
        """Register a normalizer by its ``spec.task`` and ``spec.regime_kind``.

        Raises ``ValueError`` if a normalizer is already registered for the
        same ``(task, regime_kind)`` pair.
        """
        key = MetricNormalizerKey(
            task=normalizer.spec.task,
            regime_kind=normalizer.spec.regime_kind,
        )
        if key in self._entries:
            raise ValueError(
                f"A normalizer is already registered for "
                f"(task={key.task!r}, regime_kind={key.regime_kind!r})."
            )
        self._entries[key] = normalizer

    def resolve(
        self,
        *,
        task: str,
        regime_kind: str,
    ) -> MetricSummaryNormalizer:
        """Return the normalizer registered for the given key.

        Raises ``KeyError`` if no normalizer is registered.
        """
        key = MetricNormalizerKey(task=task, regime_kind=regime_kind)
        if key not in self._entries:
            raise KeyError(
                f"No normalizer registered for "
                f"(task={task!r}, regime_kind={regime_kind!r}). "
                f"Known keys: {sorted(k for k in self._entries)}."
            )
        return self._entries[key]


# ---------------------------------------------------------------------------
__all__ = ["MetricNormalizerKey", "MetricNormalizerRegistry"]
