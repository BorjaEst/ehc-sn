"""Shared head bases and step-metric helpers.

Layer structure
---------------
:class:`BaseLossHead`
    Controller wiring only.  Stores the controller and config, exposes
    ``initial_carry``, and provides a generic forward skeleton (run step →
    delegate).  Has **no knowledge** of labels, LM logits, or loss functions.

:class:`TokenLossHeadBase`
    Specialization for heads that apply token-level supervised LM loss with
    rollout-completion metrics.  Owns ``compute_accuracy``, ``compute_lm_loss``,
    the token-supervision pipeline (``_run_token_step``), and generic
    ``StepMetrics`` assembly.  All current production heads (ACT, RL) inherit
    from here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch
from pydantic import BaseModel
from torch import Tensor, nn

import ehc_sn.loss.cross_entropy as cross_entropy_module
from ehc_sn.training.types import RatioStat, RolloutAgg, StepMetrics, TokenAgg, TransitionAgg
from ehc_sn.types import Batch

IGNORE_LABEL_ID: int = -100


# =================================================================================================
@dataclass(frozen=True)
class AccuracyStats:
    """Token-level correctness statistics shared by ACT and RL heads."""

    mask: Tensor
    is_correct: Tensor

    @property
    def loss_counts(self) -> Tensor:
        """Number of eligible tokens per sequence, shape ``(B,)``."""
        return self.mask.sum(-1)

    @property
    def loss_divisor(self) -> Tensor:
        """Safe divisor for per-sequence averages, shape ``(B, 1)``."""
        return self.loss_counts.clamp_min(1).unsqueeze(-1)

    @property
    def seq_is_correct(self) -> Tensor:
        """Whether every eligible token in a sequence was predicted correctly."""
        return self.is_correct.sum(-1) == self.loss_counts


# =================================================================================================
class ControllerWithInitialState(Protocol):
    """Controller protocol used by :class:`BaseLossHead`."""

    def initial_state(self, batch_sample: Batch) -> Any:
        """Build the initial rollout carry for a batch sample."""


# =================================================================================================
def compute_accuracy_stats(  # --------------------------------------------------------------------
    logits_lm: Tensor, labels: Tensor, *,
    ignore_label_id: int = IGNORE_LABEL_ID,
) -> AccuracyStats:  # fmt: skip
    """Compute masked token correctness statistics out of graph."""
    mask = labels != ignore_label_id
    is_correct = mask & (torch.argmax(logits_lm, dim=-1) == labels)
    return AccuracyStats(mask=mask, is_correct=is_correct)


# =================================================================================================
def compute_lm_loss_sum(  # -----------------------------------------------------------------------
    loss_fn: Any, logits_lm: Tensor, labels: Tensor, stats: AccuracyStats, *,
    ignore_label_id: int = IGNORE_LABEL_ID,
) -> Tensor:  # fmt: skip
    """Compute the summed supervised LM loss over the batch.

    Args:
        loss_fn: Token-level loss callable; signature
            ``(logits, labels, ignore_index=...) -> (B, S)``.
        logits_lm: Shape ``(B, S, V)``.
        labels: Shape ``(B, S)``.
        stats: Precomputed accuracy statistics; ``loss_counts`` used as divisor.
        ignore_label_id: Token id excluded from both loss and normalization.

    Returns:
        Scalar sum of per-sequence normalized losses.
    """
    loss_per_token = loss_fn(logits_lm, labels, ignore_index=ignore_label_id)
    loss_per_seq = loss_per_token.sum(-1) / stats.loss_counts.clamp_min(1)
    return loss_per_seq.sum()


# =================================================================================================
def build_rollout_token_aggs(  # -------------------------------------------------------------------
    *, steps: Tensor, completed: Tensor, stats: AccuracyStats,
) -> tuple[RolloutAgg, TokenAgg, TransitionAgg, TokenAgg]:  # fmt: skip
    """Build episode-level and step-level aggregate metrics for a step."""
    eligible_mask = stats.loss_counts > 0
    completed_mask = completed & eligible_mask
    completed_weights = completed_mask.to(torch.float32)
    eligible_weights = eligible_mask.to(torch.float32)

    token_correct_per_seq = stats.is_correct.to(torch.float32).sum(-1)
    token_count_per_seq = stats.loss_counts.clamp_min(1).to(torch.float32)
    seq_accuracy = token_correct_per_seq / token_count_per_seq

    rollout_agg = RolloutAgg(
        completed_count=completed_weights.sum(),
        eligible_count=eligible_mask.to(torch.float32).sum(),
        accuracy_sum=(seq_accuracy * completed_weights).sum(),
        exact_sum=(stats.seq_is_correct & completed_mask).to(torch.float32).sum(),
        steps_sum=(steps * completed_weights.to(steps.dtype)).sum(),
    )
    episode_token_agg = TokenAgg(
        token_correct_sum=(token_correct_per_seq * completed_weights).sum(),
        token_count_sum=(token_count_per_seq * completed_weights).sum(),
    )
    step_agg = TransitionAgg(
        evaluated_count=eligible_weights.sum(),
        eligible_count=eligible_weights.sum(),
        accuracy_sum=(seq_accuracy * eligible_weights).sum(),
        exact_sum=(stats.seq_is_correct & eligible_mask).to(torch.float32).sum(),
        steps_sum=(steps * eligible_weights.to(steps.dtype)).sum(),
    )
    step_token_agg = TokenAgg(
        token_correct_sum=(token_correct_per_seq * eligible_weights).sum(),
        token_count_sum=(token_count_per_seq * eligible_weights).sum(),
    )
    return rollout_agg, episode_token_agg, step_agg, step_token_agg


# =================================================================================================
class BaseLossHead[ControllerT: ControllerWithInitialState, ConfigT: BaseModel](
    nn.Module
): # fmt: skip
    """Minimal wiring base for all rollout-based loss heads.

    Owns exactly:

    * **Controller storage** — wraps the controller and exposes it via ``controller``.
    * **Config storage** — wraps the config and exposes it via ``config``.
    * **Carry initialization** — delegates to ``controller.initial_state`` via
      ``initial_carry``.

    Does **not** know about labels, loss functions, logits layout, or metrics.
    Those concerns belong to :class:`TokenLossHeadBase` or concrete subclasses.

    Output contract:
        Concrete ``forward`` methods must return
        ``(step_output, new_carry, all_halted)``.
    """

    def __init__(self, controller: ControllerT, config: ConfigT) -> None:
        super().__init__()
        self._controller = controller
        self._config = config

    @property
    def controller(self) -> ControllerT:
        """Return the wrapped controller."""
        return self._controller

    @property
    def config(self) -> ConfigT:
        """Return the head configuration."""
        return self._config

    def initial_carry(self, batch_sample: Batch) -> Any:
        """Initialize rollout carry/state from an example batch."""
        return self.controller.initial_state(batch_sample)


# =================================================================================================
class TokenLossHeadBase[ControllerT: ControllerWithInitialState, ConfigT: BaseModel](
    BaseLossHead[ControllerT, ConfigT]
):  # fmt: skip
    """Rollout head specialization for token-supervised LM loss with generic metrics.

    Adds the following on top of :class:`BaseLossHead`:

    * **Token-accuracy helper** — ``compute_accuracy(outputs, labels)`` reads
      ``outputs.lm_logits`` and returns :class:`AccuracyStats` detached from the
      compute graph.
    * **LM loss helper** — ``compute_lm_loss`` computes the summed supervised
        token loss using the function named by ``config.function``.
    * **Shared step pipeline** — ``_run_token_step`` covers labels lookup,
        accuracy stats, loss computation, rollout/token aggregation, and
        :class:`~ehc_sn.training.types.StepMetrics` assembly.
    * **Generic forward** — runs ``controller.step`` then ``_run_token_step``;
      override to add algorithm-specific options (e.g. ``is_warmup``).

    Config contract:
        ``self._config.function`` must name a callable in
        :mod:`ehc_sn.loss.cross_entropy`.

    Output contract:
        ``outputs`` passed to ``_run_token_step`` must expose
        ``outputs.lm_logits`` of shape ``(B, S, V)``.

    Subclasses **must** implement:

    * ``compute_losses`` — algorithm-specific loss terms.
    * ``_build_metric_ratios`` — pack algorithm-specific ratios into
        ``dict[str, RatioStat]``.
    * ``_build_step_output`` — wrap everything into the concrete step-output type.

        Subclasses **may** override ``compute_signals`` to add scalar diagnostic tensors.
    """

    @property
    def loss_fn(self) -> Any:
        """Return the configured token-level loss function."""
        return getattr(cross_entropy_module, self._config.function)

    def forward(  # -------------------------------------------------------------------------------
        self, batch: Any, carry: Any, **options: Any,
    ) -> tuple[Any, Any, bool]:  # fmt: skip
        """Run the controller and apply the token-supervision pipeline.

        Override to add algorithm-specific keyword arguments (e.g. ``is_warmup``);
        extract them before forwarding to ``_run_token_step``.

        Returns:
            ``(step_output, new_carry, all_halted)``.
        """
        carry, outputs = self.controller.step(carry, batch, **options)
        step_output = self._run_token_step(batch, carry, outputs)
        return step_output, carry, bool(carry.halted.all())

    def _run_token_step(  # -----------------------------------------------------------------------
        self, batch: Any, carry: Any, outputs: Any, **loss_options: Any,
    ) -> Any:  # fmt: skip
        """Execute the shared token-supervision pipeline once controller outputs exist.

        Pipeline:

        1. Extract ``labels = carry.data["labels"]``.
        2. Compute :class:`AccuracyStats` *out of graph* via ``compute_accuracy``.
        3. Call ``compute_losses(**loss_options)`` for algorithm losses.
        4. Assemble :class:`~ehc_sn.training.types.StepMetrics` via
           ``build_rollout_token_aggs`` + ``_build_metric_ratios``.
        5. Call ``compute_signals`` for diagnostic scalar dict.
        6. Return ``_build_step_output(losses, metrics, signals, outputs)``.

        Args:
            batch: The step batch (available to subclass overrides).
            carry: Current rollout carry; must expose ``.steps``, ``.halted``,
                and ``.data["labels"]``.
            outputs: Controller step outputs; must expose ``.lm_logits``.
            **loss_options: Forwarded verbatim to ``compute_losses``.
        """
        labels = carry.data["labels"]
        with torch.no_grad():
            stats = self.compute_accuracy(outputs, labels)
        losses = self.compute_losses(outputs, labels, stats, **loss_options)
        episode_agg, episode_token_agg, step_agg, step_token_agg = build_rollout_token_aggs(steps=carry.steps, completed=carry.halted, stats=stats)  # fmt: skip
        metric_ratios = self._build_metric_ratios(losses, batch_size=int(carry.halted.shape[0]))
        metrics = StepMetrics(episode=episode_agg, episode_tokens=episode_token_agg, step=step_agg, step_tokens=step_token_agg, extras=metric_ratios)  # fmt: skip
        signals = self.compute_signals(carry, outputs, losses)
        return self._build_step_output(losses, metrics, signals, outputs)

    def compute_accuracy(self, outputs: Any, labels: Tensor) -> AccuracyStats:
        """Compute masked token correctness statistics (out of graph).

        Args:
            outputs: Controller step output exposing ``lm_logits`` of shape ``(B, S, V)``.
            labels: Token labels of shape ``(B, S)``.
        """
        return compute_accuracy_stats(outputs.lm_logits, labels)

    def compute_lm_loss(self, logits_lm: Tensor, labels: Tensor, stats: AccuracyStats) -> Tensor:
        """Compute the summed supervised token loss for a step."""
        return compute_lm_loss_sum(self.loss_fn, logits_lm, labels, stats)

    # -- Abstract stubs (must be implemented by subclasses) -------------------------------------

    def compute_losses(  # ------------------------------------------------------------------------
        self, outputs: Any, labels: Tensor, stats: AccuracyStats, **options: Any,
    ) -> Any:  # fmt: skip
        """Return algorithm-specific loss terms.

        Args:
            outputs: Controller step outputs.
            labels: Token label tensor of shape ``(B, S)``.
            stats: Precomputed token-accuracy statistics.
            **options: Algorithm-specific keyword arguments (e.g. ``is_warmup``).

        Returns:
            A ``Losses`` dataclass defined by the concrete subclass.
        """
        raise NotImplementedError

    def _build_metric_ratios(  # ------------------------------------------------------------------
        self, losses: Any, *, batch_size: int,
    ) -> dict[str, RatioStat]:  # fmt: skip
        """Pack algorithm-specific ratio metrics for logging.

        Args:
            losses: The losses returned by ``compute_losses``.
            batch_size: Number of sequences in the current batch.

        Returns:
            Mapping from stable internal metric names to :class:`RatioStat`.
        """
        raise NotImplementedError

    def _build_step_output(  # --------------------------------------------------------------------
        self, losses: Any, metrics: Any, signals: dict, outputs: Any,
    ) -> Any:  # fmt: skip
        """Wrap losses, metrics, and signals into the concrete step-output type.

        Args:
            losses: Algorithm-specific losses (live, for ``backward``).
            metrics: :class:`~ehc_sn.training.types.StepMetrics` (detached).
            signals: Diagnostic signals dict (scalar tensors, detached).
            outputs: Raw controller outputs.

        Returns:
            Concrete step-output instance (e.g. ``ACTLossStep``, ``RLLossStep``).
        """
        raise NotImplementedError

    def compute_signals(  # -----------------------------------------------------------------------
        self, carry: Any, outputs: Any, losses: Any,
    ) -> dict:  # fmt: skip
        """Return a dict of scalar diagnostic tensors for logging.

        Default returns ``{}``.  Override to add algorithm-specific signals.
        """
        return {}


# =================================================================================================
__all__ = [
    "AccuracyStats", "BaseLossHead", "IGNORE_LABEL_ID", "TokenLossHeadBase",
    "build_rollout_token_aggs", "compute_accuracy_stats", "compute_lm_loss_sum",
]  # fmt: skip
