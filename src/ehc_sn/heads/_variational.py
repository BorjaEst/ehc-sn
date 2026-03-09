"""Shared base types for variational rollout heads.

The generic variational layer owns aggregate observation, latent, and
regularization semantics. Model-specific latent decompositions, such as TEM's
grid/place split, belong in concrete subclasses.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Dict, Literal, Protocol

import torch
from pydantic import BaseModel, Field
from torch import Tensor

import ehc_sn.loss.cross_entropy as cross_entropy_module
from ehc_sn.heads._base import BaseLossHead, ControllerWithInitialState
from ehc_sn.loss.regularization import l1_penalty, l2_penalty
from ehc_sn.training.types import RatioStat, RolloutAgg, StepMetrics, TokenAgg, TransitionAgg
from ehc_sn.types import Batch
from ehc_sn.utils.detach import DetachMixin

LatentCode = Tensor | Sequence[Tensor]
RegularizationNorm = Literal["none", "l1", "l2"]


# =================================================================================================
def iter_latent_codes(  # ------------------------------------------------------------------------
    code: LatentCode,
) -> Sequence[Tensor]:  # fmt: skip
    """Return a sequence view over a flat or multi-part latent code."""
    if isinstance(code, Tensor):
        return (code,)
    return tuple(code)


# =================================================================================================
def sum_latent_terms(  # -------------------------------------------------------------------------
    loss_fn: Any, pred: LatentCode, target: LatentCode,
) -> Tensor:  # fmt: skip
    """Apply a flat-tensor loss over one or more latent-code blocks.

    Args:
        loss_fn: Callable returning a per-example tensor of shape ``(B,)``.
        pred: Predicted code tensor or sequence of tensors.
        target: Target code tensor or sequence of tensors.

    Returns:
        Per-example loss values of shape ``(B,)``.
    """
    pred_codes = iter_latent_codes(pred)
    target_codes = iter_latent_codes(target)
    if len(pred_codes) != len(target_codes):
        raise ValueError("Latent code groups must have the same number of blocks.")

    total: Tensor | None = None
    for pred_code, target_code in zip(pred_codes, target_codes, strict=True):
        term = loss_fn(pred_code, target_code)
        total = term if total is None else total + term

    if total is None:
        raise ValueError("Latent code groups must not be empty.")
    return total


# =================================================================================================
def sum_regularization_terms(  # -----------------------------------------------------------------
    code: LatentCode, norm: RegularizationNorm,
) -> Tensor:  # fmt: skip
    """Return per-example regularization over one or more latent-code blocks."""
    if norm == "none":
        codes = iter_latent_codes(code)
        if not codes:
            raise ValueError("Latent code groups must not be empty.")
        return codes[0].new_zeros((codes[0].shape[0],))

    penalty_fn = l1_penalty if norm == "l1" else l2_penalty
    total: Tensor | None = None
    for code_block in iter_latent_codes(code):
        term = penalty_fn(code_block)
        total = term if total is None else total + term

    if total is None:
        raise ValueError("Latent code groups must not be empty.")
    return total


# =================================================================================================
def mean_latent_norm(  # -------------------------------------------------------------------------
    code: LatentCode,
) -> Tensor:  # fmt: skip
    """Return the mean block-wise activation norm for diagnostics."""
    block_means = [block.detach().norm(dim=-1).mean() for block in iter_latent_codes(code)]
    return torch.stack(block_means).mean()


# =================================================================================================
def build_variational_step_metrics(  # ------------------------------------------------------------
    extras: Dict[str, RatioStat], *, batch_size: int, like: Tensor,
) -> StepMetrics:  # fmt: skip
    """Build generic metrics for a variational step.

    Variational heads do not currently report token or sequence accuracy, so the
    rollout/token aggregates are zero-filled while loss ratios live in
    ``extras``.
    """
    zero = like.new_zeros(())
    batch_count = like.new_tensor(batch_size, dtype=torch.float32)
    return StepMetrics(
        episode=RolloutAgg(
            completed_count=zero,
            eligible_count=zero,
            accuracy_sum=zero,
            exact_sum=zero,
            steps_sum=zero,
        ),
        episode_tokens=TokenAgg(token_correct_sum=zero, token_count_sum=zero),
        step=TransitionAgg(
            evaluated_count=batch_count,
            eligible_count=batch_count,
            accuracy_sum=zero,
            exact_sum=zero,
            steps_sum=zero,
        ),
        step_tokens=TokenAgg(token_correct_sum=zero, token_count_sum=zero),
        extras=extras,
    )


# =================================================================================================
class VariationalLossConfig(BaseModel, extra="forbid"):
    """Shared configuration for aggregate variational loss heads."""

    observation_loss: str = Field(
        default="softmax_cross_entropy",
        description="Observation negative log-likelihood primitive from ehc_sn.loss.cross_entropy.",
    )
    c_obs: float = Field(default=1.0, ge=0.0, description="Observation loss coefficient.")
    c_reg: float = Field(default=0.0, ge=0.0, description="Regularization loss coefficient.")
    reg_norm: RegularizationNorm = Field(
        default="none",
        description="Regularization norm applied to reg_latent when provided.",
    )


# =================================================================================================
@dataclass(frozen=True)
class VariationalLosses(DetachMixin):
    """Aggregate variational loss bundle.

    Concrete subclasses provide the latent term layout through
    :attr:`loss_latent_sum`.
    """

    loss_obs_nll_sum: Tensor
    loss_reg_sum: Tensor

    @property
    def loss_latent_sum(self) -> Tensor:
        """Aggregate latent loss term exposed by the concrete subclass."""
        raise NotImplementedError

    @property
    def total(self) -> Tensor:
        """Total scalar loss for the step."""
        return self.loss_obs_nll_sum + self.loss_latent_sum + self.loss_reg_sum


# =================================================================================================
@dataclass(frozen=True)
class VariationalLossStep:
    """A single rollout/loss step produced by a variational head."""

    losses: VariationalLosses
    metrics: StepMetrics
    outputs: Any = None
    signals: Dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.signals is None:
            object.__setattr__(self, "signals", {})

    @property
    def loss(self) -> Tensor:
        """Scalar loss for this step, used for back-propagation."""
        return self.losses.total


# =================================================================================================
class VariationalOutputLike(Protocol):
    """Semantic output contract expected by aggregate variational heads."""

    obs_logits: Tensor
    latent_post: LatentCode
    latent_prior: LatentCode
    reg_latent: LatentCode | None


# =================================================================================================
class VariationalLossHeadBase[ControllerT: ControllerWithInitialState, ConfigT: VariationalLossConfig](
    BaseLossHead[ControllerT, ConfigT]
):  # fmt: skip
    """Shared base for variational rollout heads.

    Concrete subclasses are responsible for their latent decomposition and any
    model-specific diagnostics.
    """

    @property
    def loss_fn(self) -> Any:
        """Return the configured observation loss primitive."""
        return getattr(cross_entropy_module, self.config.observation_loss)

    def forward(  # -------------------------------------------------------------------------------
        self, batch: Batch, carry: Any, **options: Any,
    ) -> tuple[Any, Any, bool]:  # fmt: skip
        """Run one controller step and compute aggregate variational losses."""
        carry, outputs = self.controller.step(carry, batch, **options)
        losses = self.compute_losses(outputs, batch, **options)
        metrics = build_variational_step_metrics(
            self._build_metric_ratios(losses, batch_size=int(carry.halted.shape[0])),
            batch_size=int(carry.halted.shape[0]),
            like=losses.total.detach(),
        )
        signals = self.compute_signals(carry, outputs, losses)
        step_output = self._build_step_output(losses, metrics, signals, outputs)
        return step_output, carry, bool(carry.halted.all())

    def compute_losses(  # ------------------------------------------------------------------------
        self, outputs: VariationalOutputLike, batch: Batch, **options: Any,
    ) -> VariationalLosses:  # fmt: skip
        """Return concrete variational losses for a step."""
        raise NotImplementedError

    def _build_metric_ratios(  # ------------------------------------------------------------------
        self, losses: VariationalLosses, *, batch_size: int,
    ) -> dict[str, RatioStat]:  # fmt: skip
        """Pack aggregate variational loss ratios for logging."""
        raise NotImplementedError

    def _build_step_output(  # --------------------------------------------------------------------
        self, losses: VariationalLosses, metrics: StepMetrics, signals: Dict[str, Any], outputs: Any,
    ) -> Any:  # fmt: skip
        """Wrap losses, metrics, and signals into the concrete step-output type."""
        raise NotImplementedError

    def compute_signals(  # -----------------------------------------------------------------------
        self, state: Any, outputs: VariationalOutputLike, losses: VariationalLosses,
    ) -> Dict[str, Tensor]:  # fmt: skip
        """Compute lightweight diagnostic signals for logging."""
        return {}


# =================================================================================================
__all__ = [
    "LatentCode", "RegularizationNorm", "VariationalLossConfig", "VariationalLossHeadBase",
    "VariationalLossStep", "VariationalLosses", "build_variational_step_metrics", "iter_latent_codes",
    "mean_latent_norm", "sum_latent_terms", "sum_regularization_terms",
]  # fmt: skip
