"""ACT deliberation controller — canonical owner.

Canonical import path::

    from ehc_sn.controllers.deliberation.act import (
        ACTController, ACTControllerConfig, ACTRolloutState, ACTControllerStepOutput,
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, cast

import torch
from pydantic import BaseModel, Field
from torch import Tensor

from ehc_sn.controllers._base import (
    BaseController,
    RolloutBackbone,
    RolloutState,
)
from ehc_sn.types import Batch
from ehc_sn.utils.detach import DetachMixin


# =============================================================================
class ACTControllerConfig(BaseModel, extra="forbid"):
    """Configuration for :class:`ACTController`."""

    max_halt_steps: int = Field(
        ...,
        ge=1,
        description=(
            "Training-only per-slot step budget; slots are forced to halt once "
            "steps reach this value."
        ),
    )
    exploration_prob: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of delaying a greedy halt decision during "
        "exploration.",
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
class ACTRolloutBackbone[ModelState](
    RolloutBackbone[ModelState, ACTBackboneOutput], Protocol
):
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
        raise ValueError(
            "collapse_act_halt_continue_logits expects shape (B, A), "
            f"got {tuple(q_logits.shape)}."
        )
    n_actions = q_logits.shape[-1]
    if n_actions < 2:
        raise ValueError(
            "collapse_act_halt_continue_logits requires at least 2 actions, "
            f"got {n_actions}."
        )
    if done_action < 0 or done_action >= n_actions:
        raise ValueError(
            f"done_action={done_action} out of range for {n_actions} actions."
        )

    non_done = [i for i in range(n_actions) if i != done_action]
    halt_logit = q_logits[..., done_action]
    continue_logit = q_logits[..., non_done].max(dim=-1).values
    return ACTHaltContinueScores(
        halt_logit=halt_logit, continue_logit=continue_logit
    )


# =============================================================================
def maybe_flip_halt_decision(  # ----------------------------------------------
    greedy_halt: Tensor,
    *,
    steps: Tensor,
    explore: bool,
    exploration_prob: float,
    max_halt_steps: int,
) -> Tensor:
    """Optionally delay greedy halts until a sampled minimum step.

    Args:
        greedy_halt: Bool tensor of shape ``(B,)``.
        steps: Per-slot step counters of shape ``(B,)``.
        explore: Whether exploration is active.
        exploration_prob: Per-slot probability of flipping the halt decision.
        max_halt_steps: Forced halt step budget.

    Returns:
        Bool tensor of shape ``(B,)`` with greedy halts optionally delayed.
    """
    if not explore or exploration_prob <= 0.0 or max_halt_steps <= 1:
        return greedy_halt

    explore_mask = (
        torch.rand(greedy_halt.shape, device=greedy_halt.device)
        < exploration_prob
    )
    sampled_min = torch.randint(
        low=2,
        high=max_halt_steps + 1,
        size=steps.shape,
        device=steps.device,
        dtype=steps.dtype,
    )
    min_halt_steps = torch.where(
        explore_mask, sampled_min, torch.zeros_like(steps)
    )
    return greedy_halt & (steps >= min_halt_steps)


# =============================================================================
@dataclass(frozen=True)
class ACTControllerStepOutput(DetachMixin):
    """Raw execution output produced by a single ACT controller step."""

    backbone_output: ACTBackboneOutput
    done_action: int

    @property
    def task(self) -> object:
        """Return the task payload for objective-facing ACT contracts."""
        return self.backbone_output.task

    @property
    def q_logits(self) -> Tensor:
        """Return Q logits for objective-facing ACT contracts."""
        return self.backbone_output.control.q_logits


# =============================================================================
class ACTController[ModelState](
    BaseController[ModelState, ACTControllerConfig]
):
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

    def step(  # --------------------------------------------------------------
        self,
        state: ACTRolloutState[ModelState],
        batch: Batch,
        allow_halt: bool = True,
        explore: bool = True,
        **options: Any,
    ) -> tuple[ACTRolloutState[ModelState], ACTControllerStepOutput]:
        """Advance the controller by one recurrent step.

        ``allow_halt=False`` disables learned halting for this step.
        """
        _ = options
        data = self.refresh_slot_data(batch, state)
        model_state = self.backbone.reset_state(state.halted, state.model_state)
        backbone_output, model_state = self.backbone(data, model_state)

        steps = self.advance_steps(state)
        done = self._compute_done(
            backbone_output, steps, allow_halt=allow_halt, explore=explore
        )

        next_state = ACTRolloutState(
            model_state=model_state, steps=steps, halted=done, data=data
        )
        output = ACTControllerStepOutput(
            backbone_output=backbone_output,
            done_action=self.config.done_action,
        )
        return next_state, output

    def _compute_done(  # -----------------------------------------------------
        self,
        backbone_output: ACTBackboneOutput,
        steps: Tensor,
        *,
        allow_halt: bool,
        explore: bool,
    ) -> Tensor:
        """Derive the halted mask from q_logits using the shared collapse
        contract.
        """
        q_logits = backbone_output.control.q_logits.detach()
        scores = collapse_act_halt_continue_logits(
            q_logits, done_action=self.config.done_action
        )

        max_step_done = steps >= self.config.max_halt_steps

        if allow_halt:
            learned_done = maybe_flip_halt_decision(
                scores.greedy_halt,
                steps=steps,
                explore=explore,
                exploration_prob=self.config.exploration_prob,
                max_halt_steps=self.config.max_halt_steps,
            )
        else:
            learned_done = torch.zeros_like(max_step_done, dtype=torch.bool)

        halt = learned_done | max_step_done
        return halt.to(dtype=torch.bool)


# =============================================================================
__all__ = [
    "ACTBackboneOutput",
    "ACTControlOutput",
    "ACTController",
    "ACTControllerConfig",
    "ACTControllerStepOutput",
    "ACTHaltContinueScores",
    "ACTRolloutBackbone",
    "ACTRolloutState",
    "collapse_act_halt_continue_logits",
    "maybe_flip_halt_decision",
]
