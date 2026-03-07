"""Mapping helpers for TorchMetrics collections."""

from __future__ import annotations

from torchmetrics import MetricCollection

from ehc_sn.metrics.types import StepMetrics

__all__ = ["update_metrics_from_step"]

# Routing table: base metric key → (numerator_field_path, denominator_field_path).
# Each entry maps a metric name (without collection prefix/postfix) to the two
# StepMetrics sub-fields that form its numerator and denominator.
# Evaluated lazily via _extract() rather than getattr chains to keep it readable.
_ROUTES: tuple[tuple[str, str, str], ...] = (
    # key                  numerator path                   denominator path
    ("all/accuracy",       "halted.accuracy_sum",           "halted.halted_count"),  # fmt: skip
    ("halted/rate",        "halted.halted_count",           "halted.eligible_count"),  # fmt: skip
    ("halted/avg_steps",   "halted.steps_sum",              "halted.halted_count"),  # fmt: skip
    ("tokens/accuracy",    "tokens.token_correct_sum",      "tokens.token_count_sum"),  # fmt: skip
    ("loss/lm",            "loss.lm_loss_sum",              "loss.batch_count"),  # fmt: skip
    ("loss/q_halt",        "loss.q_halt_loss_sum",          "loss.batch_count"),  # fmt: skip
    ("loss/q_continue",    "loss.q_continue_loss_sum",      "loss.batch_count"),  # fmt: skip
    ("loss/actor",         "loss.actor_loss_sum",           "loss.batch_count"),  # fmt: skip
    ("loss/critic",        "loss.critic_loss_sum",          "loss.batch_count"),  # fmt: skip
    ("loss/entropy",       "loss.entropy_loss_sum",         "loss.batch_count"),  # fmt: skip
    ("loss/q_value",         "loss.q_value_loss_sum",           "loss.batch_count"),  # fmt: skip
)


def _get(obj: object, path: str) -> object:
    """Resolve a dotted attribute path against obj."""
    for attr in path.split("."):
        obj = getattr(obj, attr)
    return obj


def update_metrics_from_step(  # ------------------------------------------------------------------
    collection: MetricCollection, step: StepMetrics,
) -> None:  # fmt: skip
    """Update a metrics collection from aggregated step metrics.

    Routes each field of *step* to the corresponding :class:`~torchmetrics.Metric`
    inside *collection* using :data:`_ROUTES`.  The collection prefix/postfix are
    stripped before lookup so the same routing table works for both
    ``train_metrics`` and ``val_metrics``.

    Args:
        collection: A :class:`~torchmetrics.MetricCollection` built by
            :func:`~ehc_sn.metrics.build_metrics` (optionally cloned with a
            ``prefix`` such as ``"train/"`` or ``"val/"``).
        step: Aggregated per-step metrics produced by a loss head
            (e.g. :class:`~ehc_sn.training.act_head.ACTLossHead` or
            :class:`~ehc_sn.training.rl_head.RLLossHead`).
    """
    prefix = collection.prefix or ""
    postfix = collection.postfix or ""

    for full_key, metric in collection.items():
        base_key = full_key.removeprefix(prefix).removesuffix(postfix)
        for key, num_path, den_path in _ROUTES:
            if base_key == key:
                metric.update(
                    numerator_sum=_get(step, num_path),  # type: ignore[arg-type]
                    denominator_sum=_get(step, den_path),  # type: ignore[arg-type]
                )
                break
