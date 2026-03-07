"""Mapping helpers for TorchMetrics collections."""

from __future__ import annotations

from typing import NamedTuple, Sequence

from torchmetrics import MetricCollection


# =================================================================================================
class Route(NamedTuple):
    """Mapping from a metric key to dotted attribute paths on a step-metrics object.

    Attributes:
        key: Base metric name (without collection prefix/postfix), e.g. ``"all/accuracy"``.
        num_path: Dotted attribute path to the numerator tensor on the step object.
        den_path: Dotted attribute path to the denominator tensor on the step object.
    """

    key: str
    num_path: str
    den_path: str


def _get(obj: object, path: str) -> object:
    """Resolve a dotted attribute path against obj."""
    for attr in path.split("."):
        obj = getattr(obj, attr)
    return obj


# =================================================================================================
def update_metrics_from_step(  # ------------------------------------------------------------------
    collection: MetricCollection, step: object, routes: Sequence[Route],
) -> None:  # fmt: skip
    """Update a metrics collection from aggregated step metrics.

    Routes each field of *step* to the corresponding :class:`~torchmetrics.Metric`
    inside *collection* using *routes*.  The collection prefix/postfix are
    stripped before lookup so the same logic works for both ``train_metrics``
    and ``val_metrics``.

    Args:
        collection: A :class:`~torchmetrics.MetricCollection` (optionally cloned
            with a ``prefix`` such as ``"train/"`` or ``"val/"``).
        step: Aggregated per-step metrics produced by a loss head.  Attribute
            paths declared in *routes* must be resolvable on this object.
        routes: Paradigm-specific routing table mapping metric keys to
            numerator/denominator attribute paths on *step*.
    """
    prefix = collection.prefix or ""
    postfix = collection.postfix or ""

    route_map = {r.key: r for r in routes}
    for full_key, metric in collection.items():
        base_key = full_key.removeprefix(prefix).removesuffix(postfix)
        route = route_map.get(base_key)
        if route is not None:
            metric.update(
                numerator_sum=_get(step, route.num_path),  # type: ignore[arg-type]
                denominator_sum=_get(step, route.den_path),  # type: ignore[arg-type]
            )


# =================================================================================================
__all__ = ["Route", "update_metrics_from_step"]
