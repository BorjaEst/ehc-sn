"""Generic rollout controller for latent-consistency models.

This controller owns rollout state and step cadence for models whose training
loss is expressed as observation likelihood plus a latent consistency term.
Unlike ACT and RL controllers, it does not interpret action logits or TD
targets. Termination is controlled only by ``max_steps``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Tuple

from pydantic import BaseModel, Field
from torch import Tensor

from ehc_sn.controllers._base import BaseController, RolloutBackbone, RolloutState
from ehc_sn.loss.consistency import LatentCode
from ehc_sn.types import Batch
from ehc_sn.utils.detach import DetachMixin


# ==================================================================================================
class VARControllerConfig(BaseModel, extra="forbid"):
    """Configuration for :class:`VARController`."""

    max_steps: int = Field(
        default=1,
        ge=1,
        description="Number of rollout steps to execute before halting a slot.",
    )


# =================================================================================================
class VARBackbone[BkState](RolloutBackbone[BkState], Protocol):
    """Backbone protocol expected by :class:`VARController`."""

    def __call__(  # ------------------------------------------------------------------------------
        self, inputs: Tensor, state: BkState | None = None,
    ) -> tuple[BkState, "VAROutput"]:  # fmt: skip
        ...


# =================================================================================================
@dataclass
class VARState[ModelState](RolloutState[ModelState]):
    """Controller carry/state for VAR rollouts."""


# =================================================================================================
@dataclass
class VAROutput(DetachMixin):
    """Semantic output contract for latent-consistency heads.

    Latent codes can be a single tensor or a sequence of tensors, depending on
    the model's design. The controller carries semantic fields so loss heads do
    not depend on positional tuple conventions.
    """

    obs_logits: Tensor  # (B, S, V) observation likelihood logits
    latent_post: LatentCode  # Latent code(s) produced by the inference path
    latent_prior: LatentCode  # Latent code(s) produced by the predictive path
    reg_latent: LatentCode | None = None  # Optional regularization code(s) for the latent space
    theta_cls: Tensor | None = None  # Optional (B, D) features for auxiliary classification losses


# =================================================================================================
class VARController[BkState](BaseController[BkState, VARControllerConfig]):
    """Rollout controller for generic latent-consistency objectives.
    
    The controller:
        - Executes a fixed number of steps per slot as defined by ``max_steps``.
        - Refreshes slot data from the incoming batch on each step.
        - Does not perform any action-based halting; termination is solely based on step count.
        - Expects the backbone to produce observation logits and named latent codes.
            
    This controller is suitable for training models where the loss is composed
    of an observation likelihood term and a latent consistency penalty.
    """

    def __init__(  # ------------------------------------------------------------------------------
        self, backbone: VARBackbone[BkState], config: VARControllerConfig,
    ) -> None:  # fmt: skip
        """Create a controller.

        Args:
            backbone: Model implementing :class:`VARBackbone`.
            config: Controller configuration.
        """
        super().__init__(backbone=backbone, config=config)

    def initial_state(  # -------------------------------------------------------------------------
        self, batch_sample: Batch
    ) -> VARState[BkState]:  # fmt: skip
        """Build an initial rollout state from a batch sample."""
        slots = self.initial_slots(batch_sample)
        return VARState(
            model_state=slots.model_state, steps=slots.steps, halted=slots.halted,
            data=slots.data,
        )  # fmt: skip

    def step(  # ----------------------------------------------------------------------------------
        self, state: VARState[BkState], batch: Batch,
    ) -> Tuple[VARState[BkState], VAROutput]:  # fmt: skip
        """Advance the controller by one variational step."""
        data = self.refresh_slot_data(batch, state)
        model_state = self.backbone.reset_state(state.halted, state.model_state)
        model_state, outputs = self.backbone(data["inputs"], model_state)

        steps = self.advance_steps(state)
        done = steps >= self.config.max_steps

        state = VARState(model_state=model_state, steps=steps, halted=done, data=data)
        return state, outputs

    def refresh_slot_data(  # ---------------------------------------------------------------------
        self, batch: Batch, state: VARState[BkState]
    ) -> dict[str, Tensor]:  # fmt: skip
        """Replace data for halted slots with incoming batch data."""
        return super().refresh_slot_data(batch, state)


# =================================================================================================
__all__ = ["LatentCode", "VARBackbone", "VARControllerConfig", "VARController", "VARState", "VAROutput"]
