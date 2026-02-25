"""Mapping helpers for TorchMetrics collections."""

from __future__ import annotations

from torchmetrics import MetricCollection

from ehc_sn.metrics.types import StepMetrics

__all__ = ["update_metrics_from_step"]


def update_metrics_from_step(  # ------------------------------------------------------------------
    collection: MetricCollection, step: StepMetrics,
) -> None:  # fmt: skip
    """Update a metrics collection from aggregated step metrics."""
    collection["all/accuracy"].update(
        step.halted.accuracy_sum.detach(),
        step.halted.halted_count.detach(),
    )
    collection["halted/rate"].update(
        step.halted.halted_count.detach(),
        step.halted.eligible_count.detach(),
    )
    collection["halted/avg_steps"].update(
        step.halted.steps_sum.detach(),
        step.halted.halted_count.detach(),
    )
    collection["tokens/accuracy"].update(
        step.tokens.token_correct_sum.detach(),
        step.tokens.token_count_sum.detach(),
    )
    collection["loss/lm"].update(
        step.loss.lm_loss_sum.detach(),
        step.loss.batch_count.detach(),
    )
    collection["loss/q_halt"].update(
        step.loss.q_halt_loss_sum.detach(),
        step.loss.batch_count.detach(),
    )
    collection["loss/q_continue"].update(
        step.loss.q_continue_loss_sum.detach(),
        step.loss.batch_count.detach(),
    )
