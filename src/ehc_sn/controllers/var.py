""" """

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, Field
from torch import Tensor

from ehc_sn.controllers._base import BaseController, RolloutBackbone, RolloutState
from ehc_sn.loss.consistency import LatentCode, LatentRelation
from ehc_sn.types import Batch
from ehc_sn.utils.detach import DetachMixin

MAIN_LATENT_RELATION: str = "main"
"""Canonical key for the primary latent relation exposed by VAR outputs."""


# ==================================================================================================
class VARControllerConfig(BaseModel, extra="forbid"):
    """Configuration for :class:`VARController`."""

    max_steps: int = Field(
        default=1,
        ge=1,
        description="Number of rollout steps to execute before halting a slot.",
    )


# =================================================================================================
class VARRolloutBackbone[ModelState, ModelOutput](RolloutBackbone[ModelState, ModelOutput], Protocol):
    """Backbone protocol expected by :class:`VARController`."""


# =================================================================================================
@dataclass
class VARRolloutState[ModelState](RolloutState[ModelState]):
    """Controller carry/state for VAR rollouts."""


# =================================================================================================
@dataclass
class VARStepOutput(DetachMixin):
    """Semantic output contract for latent-consistency heads.

    Each semantic objective term stays explicit at the model/controller
    boundary. A single relation side may still span one or more tensor blocks
    via :class:`LatentCode`, but the role of the relation in the objective is
    named directly in ``latent_relations``.
    """

    obs_logits: Tensor  # Observation likelihood logits
    latent_relations: dict[str, LatentRelation]  # Named latent comparison terms for the objective
    reg_terms: dict[str, LatentCode] | None = None  # Optional named regularization targets
    theta_cls: Tensor | None = None  # Optional (B, D) features for auxiliary classification losses


# =================================================================================================
class VARController[ModelState](BaseController[ModelState, VARControllerConfig]):
    """Rollout controller for generic latent-consistency objectives.

    The controller:
        - Executes a fixed number of steps per slot as defined by ``max_steps``.
        - Refreshes slot data from the incoming batch on each step.
        - Does not perform any action-based halting; termination is solely based on step count.
        - Expects the backbone to produce observation logits and named latent terms.

    This controller is suitable for training models where the loss is composed
    of an observation likelihood term, a primary latent consistency penalty,
    and optional latent regularization.
    """

    def __init__(  # ------------------------------------------------------------------------------
        self, backbone: VARRolloutBackbone[ModelState], config: VARControllerConfig,
    ) -> None:  # fmt: skip
        """Create a controller.

        Args:
            backbone: Model implementing :class:`VARRolloutBackbone`.
            config: Controller configuration.
        """
        super().__init__(backbone=backbone, config=config)

    def initial_state(  # -------------------------------------------------------------------------
        self, batch_sample: Batch
    ) -> VARRolloutState[ModelState]:  # fmt: skip
        """Build an initial rollout state from a batch sample."""
        slots = self.initial_slots(batch_sample)
        return VARRolloutState(
            model_state=slots.model_state, steps=slots.steps, halted=slots.halted,
            data=slots.data,
        )  # fmt: skip

    def step(  # ----------------------------------------------------------------------------------
        self, state: VARRolloutState[ModelState], batch: Batch, *,
        allow_halt: bool = True, explore: bool = True, **_: Any,
    ) -> tuple[VARRolloutState[ModelState], VARStepOutput]:  # fmt: skip
        """Advance the controller by one variational step."""
        data = self.refresh_slot_data(batch, state)
        model_state = self.backbone.reset_state(state.halted, state.model_state)
        outputs, model_state = self.backbone(data, model_state)

        steps = self.advance_steps(state)
        done = steps >= self.config.max_steps

        state = VARRolloutState(model_state=model_state, steps=steps, halted=done, data=data)
        return state, outputs


# =================================================================================================
__all__ = [
    "LatentCode", "MAIN_LATENT_RELATION", 
    "VARRolloutBackbone", "VARController", "VARControllerConfig", "VARStepOutput", "VARRolloutState",
]  # fmt: skip
