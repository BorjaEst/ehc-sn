"""Shared variational-family head abstractions.

This module defines the family layer used by heads whose main public loss
contract is observation likelihood plus one or more named latent relations and
optional regularization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch import Tensor

import ehc_sn.loss.cross_entropy as cross_entropy_module
from ehc_sn.heads._base import BaseObjective
from ehc_sn.loss.consistency import LatentCode, LatentRelation
from ehc_sn.metrics import signals as S
from ehc_sn.rollouts import StepRecord
from ehc_sn.training.types import RatioStat, RolloutAgg, StepMetrics, TokenAgg, TransitionAgg
from ehc_sn.types import Batch
from ehc_sn.utils.detach import DetachMixin


# =================================================================================================
@dataclass(frozen=True)
class VariationalLosses(DetachMixin):
    """Shared parent loss structure for variational-family heads."""

    loss_obs_nll_sum: Tensor
    loss_reg_sum: Tensor

    @property
    def loss_latent_sum(self) -> Tensor:
        """Return the aggregate latent loss for the current step."""
        raise NotImplementedError

    @property
    def total(self) -> Tensor:
        """Return the total scalar loss for the current step."""
        return self.loss_obs_nll_sum + self.loss_latent_sum + self.loss_reg_sum


# =================================================================================================
@dataclass(frozen=True)
class VariationalLossStep:
    """A single rollout/loss step produced by a variational-family head."""

    losses: VariationalLosses
    metrics: StepMetrics
    outputs: Optional[Any] = None
    signals: Dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.signals is None:
            object.__setattr__(self, "signals", {})

    @property
    def loss(self) -> Tensor:
        """Scalar loss for this step, used for back-propagation."""
        return self.losses.total


# =================================================================================================
class VariationalLossHeadBase[ConfigT](BaseObjective[ConfigT]):  # fmt: skip
    """Base class for variational-family rollout heads.

    Family-level output contracts should expose named semantic latent relations.
    Each relation is binary and compares two semantic latent-code sides. Either
    side may itself be multi-block via :class:`LatentCode`.
    """

    @property
    def loss_fn(self) -> Any:
        """Return the configured observation loss primitive."""
        return getattr(cross_entropy_module, self.config.observation_loss)

    def evaluate_step(  # ------------------------------------------------------------------------
        self, record: StepRecord, **options: Any,
    ) -> Any:  # fmt: skip
        """Score one executed variational-family step."""
        carry, outputs = record.carry, record.outputs
        losses = self.compute_losses(outputs, carry, **options)
        metrics = build_variational_step_metrics(
            self._build_metric_ratios(losses, carry=carry, outputs=outputs, batch_size=int(carry.halted.shape[0])),
            batch_size=int(carry.halted.shape[0]),
            like=losses.total.detach(),
        )  # fmt: skip
        signals = self.compute_signals(record.batch, carry, outputs, losses)
        return self._build_step_output(losses, metrics, signals, outputs)

    def compute_losses(  # ------------------------------------------------------------------------
        self, outputs: Any, carry: Any, **options: Any,
    ) -> VariationalLosses:  # fmt: skip
        """Return variational-family losses for the current step."""
        raise NotImplementedError

    def _build_metric_ratios(  # ------------------------------------------------------------------
        self, losses: VariationalLosses, *, carry: Any, outputs: Any, batch_size: int,
    ) -> dict[str, RatioStat]:  # fmt: skip
        """Pack algorithm-specific ratio metrics for logging."""
        raise NotImplementedError

    def _build_step_output(  # --------------------------------------------------------------------
        self, losses: VariationalLosses, metrics: StepMetrics, signals: Dict[str, Any], outputs: Any,
    ) -> Any:  # fmt: skip
        """Wrap losses, metrics, and signals into the concrete step-output type."""
        raise NotImplementedError

    def compute_signals(  # -----------------------------------------------------------------------
        self, batch: Batch, carry: Any, outputs: Any, losses: VariationalLosses,
    ) -> Dict[str, Tensor]:  # fmt: skip
        """Return detached generic variational-family diagnostic signals."""
        return {
            S.STEPS_MEAN: carry.steps.float().mean().detach(),
            S.LOSS_TOTAL: losses.total.detach(),
            S.LOSS_OBS_NLL: losses.loss_obs_nll_sum.detach(),
            S.LOSS_LATENT: losses.loss_latent_sum.detach(),
            S.LOSS_REG: losses.loss_reg_sum.detach(),
        }

# =================================================================================================
def build_variational_step_metrics(  # -----------------------------------------------------------
    extras: Dict[str, RatioStat], *, batch_size: int, like: Tensor,
) -> StepMetrics:  # fmt: skip
    """Build generic metrics for a variational step.

    Variational heads do not currently report token or rollout accuracy, so
    those aggregates are zero-filled while loss ratios live in ``extras``.
    """
    zero = like.new_zeros(())
    batch_count = like.new_tensor(batch_size, dtype=torch.float32)
    return StepMetrics(
        episode=RolloutAgg(completed_count=zero, eligible_count=zero, accuracy_sum=zero, exact_sum=zero, steps_sum=zero),
        episode_tokens=TokenAgg(token_correct_sum=zero, token_count_sum=zero),
        step=TransitionAgg(evaluated_count=batch_count, eligible_count=batch_count, accuracy_sum=zero, exact_sum=zero, steps_sum=zero),
        step_tokens=TokenAgg(token_correct_sum=zero, token_count_sum=zero),
        extras=extras,
    )  # fmt: skip


# =================================================================================================
def require_latent_relation(  # ------------------------------------------------------------------
    latent_relations: dict[str, LatentRelation], name: str,
) -> LatentRelation:  # fmt: skip
    """Return a required latent relation by name with a clear error on absence."""
    try:
        return latent_relations[name]
    except KeyError as exc:
        available = ", ".join(sorted(latent_relations)) or "<none>"
        raise KeyError(f"Required latent relation '{name}' is missing. Available: {available}.") from exc


# =================================================================================================
def get_reg_term(  # -----------------------------------------------------------------------------
    reg_terms: dict[str, LatentCode] | None, name: str,
) -> LatentCode | None:  # fmt: skip
    """Return an optional named regularization term when present."""
    if reg_terms is None:
        return None
    return reg_terms.get(name)


# =================================================================================================
__all__ = [
    "VariationalLosses", "VariationalLossHeadBase", "VariationalLossStep",
    "build_variational_step_metrics", "get_reg_term", "require_latent_relation",
]  # fmt: skip
