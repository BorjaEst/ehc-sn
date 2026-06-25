"""Mapping helpers for TorchMetrics collections."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import NamedTuple

from torchmetrics import MetricCollection

# =============================================================================
# Sentinel for optional-default parameter.
_MISSING: object = object()


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


def _get(obj: object, path: str, default: object = _MISSING) -> object:
    """Resolve a dotted attribute or mapping path against ``obj``.

    When ``obj`` is a :class:`~collections.abc.Mapping` and a key is absent,
    returns *default* if provided; otherwise raises ``KeyError``.

    Args:
        obj: The object to resolve the path against.
        path: Dotted path such as ``"extras.my_key.numerator_sum"``.
        default: Optional fallback value for missing Mapping keys.

    Returns:
        The resolved value, or *default* when a Mapping key is absent.
    """
    for attr in path.split("."):
        if isinstance(obj, Mapping):
            if attr not in obj:
                if default is not _MISSING:
                    return default
                raise KeyError(attr)
            obj = obj[attr]
        else:
            obj = getattr(obj, attr)
    return obj


# =============================================================================
def update_metrics_from_step(  # ----------------------------------------------
    collection: MetricCollection,
    step: object,
    routes: Sequence[Route],
) -> None:
    """Update a metrics collection from aggregated step metrics.

    Routes each field of *step* to the corresponding :class:`~torchmetrics.Metric`
    inside *collection* using *routes*.  The collection prefix/postfix are
    stripped before lookup so the same logic works for both ``train_metrics``
    and ``val_metrics``.

    When a route references an attribute path that cannot be resolved — for
    example an extras key that the step objective did not populate — the
    metric update for that route is silently skipped.  This allows route
    tables to be shared across tasks with different output contracts
    without requiring every task to produce every extras key.

    Args:
        collection: A :class:`~torchmetrics.MetricCollection` (optionally cloned
            with a ``prefix`` such as ``"train/"`` or ``"val/"``).
        step: Aggregated per-step metrics produced by a rollout objective.  Attribute
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
            num = _get(step, route.num_path, default=None)
            den = _get(step, route.den_path, default=None)
            if num is not None and den is not None:
                metric.update(
                    numerator_sum=num,  # type: ignore[arg-type]
                    denominator_sum=den,  # type: ignore[arg-type]
                )


# =============================================================================
__all__ = ["Route", "update_metrics_from_step"]
