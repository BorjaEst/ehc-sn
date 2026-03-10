"""ACT controller: halting decisions and recurrent state management.

This module implements the controller used by HRM v1 training.

The controller is responsible for:
    - refreshing per-slot buffers for halted slots (partial reset semantics)
    - resetting backbone recurrent state for halted slots
    - running the backbone forward pass
    - selecting a halt/continue action based on Q logits
    - optionally computing a TD(0) bootstrap target to supervise non-halt actions

The semantics of the actions are minimal:
    - ``done_action`` indicates the action index that terminates deliberation.
    - all other actions are treated as "continue" (potentially multiple).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Tuple

import torch
from pydantic import BaseModel, Field
from torch import Tensor

from ehc_sn.controllers._base import BaseController, RolloutBackbone, RolloutState
from ehc_sn.types import Batch
from ehc_sn.utils.detach import DetachMixin


# ==================================================================================================
class ACTControllerConfig(BaseModel, extra="forbid"):
    """Configuration for :class:`ACTController`.

    Attributes:
        exploration_prob: Probability of suppressing early halting during training.
        max_steps: Hard cap on steps before forced termination.
        done_action: Action index that signals termination when selected.
    """

    exploration_prob: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Exploration probability for deliberation.",
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


# =================================================================================================
class ACTRolloutBackbone[ModelState, ModelOutput](RolloutBackbone[ModelState, ModelOutput], Protocol):
    """Backbone protocol expected by :class:`ACTController`."""


# =================================================================================================
@dataclass
class ACTRolloutState[ModelState](RolloutState[ModelState]):
    """Controller carry/state for ACT rollouts."""


# =================================================================================================
@dataclass
class ACTOutput(DetachMixin):
    """Outputs produced by a controller step.

    Named property accessors (``lm_logits``, ``q_logits``) are the preferred
    read path for heads.  Direct positional reads via ``logits[N]`` are
    deprecated in head code and will be removed once all heads are migrated.
    """

    logits: Tuple[Tensor, ...]  # (lm_logits, q_logits, ...); prefer named properties in heads
    theta_cls: Tensor  # (B, D) — theta CLS features
    action: Tensor  # (B,) selected action indices for this step
    target_q: Tensor | None = None  # TD(0) bootstrap Q-target, shape: (B,). None outside training.

    @property
    def lm_logits(self) -> Tensor:
        """LM head logits, shape ``(B, S, V)``."""
        return self.logits[0]

    @property
    def q_logits(self) -> Tensor:
        """Q-value logits (halt/continue policy), shape ``(B, A)``."""
        return self.logits[1]


# =================================================================================================
class ACTController[ModelState](BaseController[ModelState, ACTControllerConfig]):
    """ACT controller for supervised HRM v1 deliberation.

    The controller:
        - runs the backbone forward pass each step
        - selects a halt/continue action via greedy argmax over Q-logits
        - applies probabilistic exploration gating to suppress premature halting
        - optionally computes a TD(0) bootstrap target to supervise continue actions

    Action semantics are minimal: ``done_action`` terminates deliberation; all
    other actions are treated as generic continue steps.
    """

    def __init__(  # ------------------------------------------------------------------------------
        self, backbone: ACTRolloutBackbone,  config: ACTControllerConfig,
    ) -> None:  # fmt: skip
        """Create a controller.

        Args:
            backbone: Model implementing :class:`ACTRolloutBackbone`.
            config: Controller configuration.
        """
        super().__init__(backbone=backbone, config=config)

    def initial_state(  # -------------------------------------------------------------------------
        self, batch_sample: Batch
    ) -> ACTRolloutState[ModelState]:  # fmt: skip
        """Build an initial ACT state from a batch sample.

        Args:
            batch_sample: Batch dict containing at least ``"inputs"`` of shape
                ``(B, ...)``.

        Returns:
            Initialized :class:`ACTRolloutState`.
        """
        slots = self.initial_slots(batch_sample)
        return ACTRolloutState(
            model_state=slots.model_state, steps=slots.steps, halted=slots.halted,
            data=slots.data,
        )  # fmt: skip

    def step(  # ----------------------------------------------------------------------------------
        self, state: ACTRolloutState[ModelState], batch: Batch,
        allow_halt: bool = True, explore: bool = True, td_target: bool = True,
    ) -> Tuple[ACTRolloutState[ModelState], ACTOutput]:  # fmt: skip
        """Advance the controller by one step.

        Args:
            state: Current rollout state.
            batch: Incoming batch used to refresh halted slots.
            allow_halt: If False, disables done-action halting.
            explore: If True, probabilistically suppresses early halting.
            td_target: If True, computes TD bootstrap targets (when applicable).

        Returns:
            ``(new_state, output)``.
        """
        data = self.refresh_slot_data(batch, state)
        model_state = self.backbone.reset_state(state.halted, state.model_state)
        model_state, logits, theta_cls = self.backbone(data, model_state)

        steps = self.advance_steps(state)
        action, done = self._select_action_and_done(logits, steps, allow_halt, explore)

        state = ACTRolloutState(model_state=model_state, steps=steps, halted=done, data=data)
        output = ACTOutput(logits=logits, theta_cls=theta_cls, action=action)

        # TD(0) bootstrap target for the Q-head.
        if td_target and self._config.max_steps > 1:
            output.target_q = self.compute_td_target(data, model_state, steps)

        return state, output

    def compute_td_target(  # ---------------------------------------------------------------------
        self, data: dict[str, Tensor], model_state: Any, steps: Tensor,
    ) -> Tensor:  # fmt: skip
        """Compute TD(0) bootstrap targets for the Q head.

        At the last allowed step, the target becomes the predicted Q for the done
        action; otherwise it is the max Q over actions.

        Args:
            data: Per-slot batch buffers; the full batch mapping is fed to the backbone.
            model_state: Current backbone recurrent state (used for next-step preview).
            steps: Per-slot step counters of shape ``(B,)``.

        Returns:
            Sigmoid-normalized TD target of shape ``(B,)``.
        """
        with torch.no_grad():
            _, logits, _ = self.backbone(data, model_state)
            next_q = logits[1]
        is_last_step = steps >= self.config.max_steps

        # At last step: forced done → target is Q(done_action). Otherwise: max over all actions.
        done_action = self._config.done_action
        target = torch.where(is_last_step, next_q[..., done_action], next_q.max(dim=-1).values)

        return torch.sigmoid(target)

    def _select_action_and_done(  # ---------------------------------------------------------------
        self, logits: list[Tensor], steps: Tensor, allow_halt: bool, explore: bool,
    ) -> Tuple[Tensor, Tensor]:  # fmt: skip
        """Select action and determine done flags for this step."""
        _logits_lm, logits_q, *_ = logits  # Unpack list of logits multiple heads
        config = self.config
        action = logits_q.detach().argmax(dim=-1)  # greedy over all actions (B,)
        done = steps >= config.max_steps

        if allow_halt:
            done = done | (action == config.done_action)

        if explore and (config.max_steps > 1):
            exploration_flag = torch.rand(steps.shape, device=steps.device) < config.exploration_prob
            min_steps = exploration_flag * torch.randint_like(steps, low=2, high=config.max_steps + 1)
            done = done & (steps >= min_steps)

        return action, done


# =================================================================================================
__all__ = ["ACTRolloutBackbone", "ACTControllerConfig", "ACTController", "ACTRolloutState", "ACTOutput"]
