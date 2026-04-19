""" """

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, cast

import torch
from pydantic import BaseModel, Field
from torch import Tensor

from ehc_sn.controllers._base import BaseController, RolloutBackbone, RolloutState
from ehc_sn.types import Batch
from ehc_sn.utils.detach import DetachMixin


# =============================================================================
class ACTControllerConfig(BaseModel, extra="forbid"):
    """Configuration for :class:`ACTController`."""

    exploration_prob: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of flipping the greedy halt decision during exploration.",
    )
    max_steps: int = Field(
        ...,
        ge=1,
        description="Maximum deliberation steps per slot.",
    )
    done_action: int = Field(
        default=0,
        ge=0,
        description="Action index that terminates a slot when selected.",
    )


# =============================================================================
class ACTControlOutput(Protocol):
    """Control payload emitted by one ACT-consumable backbone step."""

    q_logits: Tensor


# =============================================================================
class ACTBackboneOutput(Protocol):
    """Named backbone output consumed by :class:`ACTController`."""

    task: object
    control: ACTControlOutput


# =============================================================================
class ACTRolloutBackbone[ModelState](RolloutBackbone[ModelState, ACTBackboneOutput], Protocol):
    """Backbone protocol expected by :class:`ACTController`."""


# =============================================================================
@dataclass
class ACTRolloutState[ModelState](RolloutState[ModelState]):
    """Controller carry/state for ACT rollouts."""


# =============================================================================
@dataclass(frozen=True)
class ACTHaltContinueScores:
    """Collapsed halt vs continue logits derived from q_logits.

    ``halt_logit``    = q_logits[..., done_action]
    ``continue_logit`` = max over all non-done action indices
    ``greedy_halt``   = halt_logit > continue_logit  (strict; ties continue)
    """

    halt_logit: Tensor
    continue_logit: Tensor

    @property
    def greedy_halt(self) -> Tensor:
        """Return bool tensor; True only when halt strictly beats every alternative."""
        return self.halt_logit > self.continue_logit


# =============================================================================
def collapse_act_halt_continue_logits(  # -------------------------------------
    q_logits: Tensor,
    *,
    done_action: int,
) -> ACTHaltContinueScores:
    """Collapse q_logits into halt vs continue scores.

    Args:
            q_logits: Shape ``(B, A)`` with ``A >= 2``.
            done_action: Index of the halt action; must be in ``[0, A)``.

    Returns:
            :class:`ACTHaltContinueScores` with ``halt_logit`` and ``continue_logit``.
    """
    if q_logits.ndim != 2:
        raise ValueError(f"collapse_act_halt_continue_logits expects shape (B, A), got {tuple(q_logits.shape)}.")
    n_actions = q_logits.shape[-1]
    if n_actions < 2:
        raise ValueError(f"collapse_act_halt_continue_logits requires at least 2 actions, got {n_actions}.")
    if done_action < 0 or done_action >= n_actions:
        raise ValueError(f"done_action={done_action} is out of range for {n_actions} actions.")

    non_done = [i for i in range(n_actions) if i != done_action]
    halt_logit = q_logits[..., done_action]
    continue_logit = q_logits[..., non_done].max(dim=-1).values
    return ACTHaltContinueScores(halt_logit=halt_logit, continue_logit=continue_logit)


# =============================================================================
def maybe_flip_halt_decision(  # ----------------------------------------------
    greedy_halt: Tensor,
    *,
    explore: bool,
    exploration_prob: float,
) -> Tensor:
    """Optionally flip the halt boolean with probability ``exploration_prob``.

    Args:
            greedy_halt: Bool tensor of shape ``(B,)``.
            explore: Whether exploration is active.
            exploration_prob: Per-slot probability of flipping the halt decision.

    Returns:
            Bool tensor of shape ``(B,)`` with some decisions flipped.
    """
    if not explore or exploration_prob <= 0.0:
        return greedy_halt

    flip = torch.rand(greedy_halt.shape, device=greedy_halt.device) < exploration_prob
    return greedy_halt ^ flip


# =============================================================================
@dataclass(frozen=True)
class ACTStepOutput(DetachMixin):
    """Raw execution output produced by a single ACT controller step."""

    backbone_output: ACTBackboneOutput


# =============================================================================
class ACTController[ModelState](BaseController[ModelState, ACTControllerConfig]):
    """One-step masked recurrent transition primitive for ACT rollouts."""

    def __init__(  # ----------------------------------------------------------
        self,
        backbone: ACTRolloutBackbone[ModelState],
        config: ACTControllerConfig,
    ) -> None:
        """Create an ACT controller."""
        super().__init__(backbone=cast(Any, backbone), config=config)

    @property
    def backbone(self) -> ACTRolloutBackbone[ModelState]:
        """Return the wrapped ACT backbone typed to the local protocol."""
        return cast(ACTRolloutBackbone[ModelState], super().backbone)

    def initial_state(  # -----------------------------------------------------
        self,
        batch_sample: Batch,
    ) -> ACTRolloutState[ModelState]:
        """Build an initial ACT state from a batch sample."""
        slots = self.initial_slots(batch_sample)
        return ACTRolloutState(
            model_state=slots.model_state,
            steps=slots.steps,
            halted=slots.halted,
            data=slots.data,
        )

    def step(  # ---------------------------------------------------------------
        self,
        state: ACTRolloutState[ModelState],
        batch: Batch,
        allow_halt: bool = True,
        explore: bool = True,
        **options: Any,
    ) -> tuple[ACTRolloutState[ModelState], ACTStepOutput]:
        """Advance the controller by one recurrent step.

        ``allow_halt=False`` disables learned halting for this step while still
        enforcing the hard ``config.max_steps`` budget. This is useful for
        fixed-budget evaluation over repeated sources, where early-halting rows
        would otherwise be refreshed immediately and never converge to a single
        batch-aligned stop event.
        """
        _ = options
        data = self.refresh_slot_data(batch, state)
        model_state = self.backbone.reset_state(state.halted, state.model_state)
        backbone_output, model_state = self.backbone(data, model_state)

        steps = self.advance_steps(state)
        done = self._compute_done(backbone_output, steps, allow_halt=allow_halt, explore=explore)

        next_state = ACTRolloutState(model_state=model_state, steps=steps, halted=done, data=data)
        output = ACTStepOutput(backbone_output=backbone_output)
        return next_state, output

    def _compute_done(  # -----------------------------------------------------
        self,
        backbone_output: ACTBackboneOutput,
        steps: Tensor,
        *,
        allow_halt: bool,
        explore: bool,
    ) -> Tensor:
        """Derive the halted mask from q_logits using the shared collapse contract."""
        q_logits = backbone_output.control.q_logits.detach()
        scores = collapse_act_halt_continue_logits(q_logits, done_action=self.config.done_action)

        if allow_halt:
            halt = maybe_flip_halt_decision(
                scores.greedy_halt,
                explore=explore,
                exploration_prob=self.config.exploration_prob,
            )
        else:
            halt = torch.zeros_like(scores.greedy_halt, dtype=torch.bool)

        return (halt | (steps >= self.config.max_steps)).to(dtype=torch.bool)


# =============================================================================
__all__ = [
    "ACTBackboneOutput",
    "ACTControlOutput",
    "ACTController",
    "ACTControllerConfig",
    "ACTHaltContinueScores",
    "ACTStepOutput",
    "ACTRolloutBackbone",
    "ACTRolloutState",
    "collapse_act_halt_continue_logits",
    "maybe_flip_halt_decision",
]
