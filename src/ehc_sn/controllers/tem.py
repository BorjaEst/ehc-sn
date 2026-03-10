"""Canonical TEM controller-facing contracts.

This module defines the rollout contract consumed by TEM loss heads. The
controller-facing output is intentionally semantic: loss heads read named TEM
properties rather than tuple positions or legacy model-internal structures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, Field
from torch import Tensor

from ehc_sn.controllers._base import BaseController, RolloutBackbone, RolloutState
from ehc_sn.loss.consistency import LatentCode, LatentRelation
from ehc_sn.types import Batch
from ehc_sn.utils.detach import DetachMixin

GRID_TRANSITION_RELATION: str = "grid_transition"
PLACE_TRANSITION_RELATION: str = "place_transition"
PLACE_SENSORY_RELATION: str = "place_sensory"
GRID_REG_TERM: str = "grid"
PLACE_REG_TERM: str = "place"


# ==================================================================================================
class TEMControllerConfig(BaseModel, extra="forbid"):
    """Configuration for :class:`TEMController`."""

    max_steps: int = Field(
        default=1,
        ge=1,
        description="Number of rollout steps to execute before halting a slot.",
    )


# =================================================================================================
class TEMRolloutBackbone[ModelState, ModelOutput](RolloutBackbone[ModelState, ModelOutput], Protocol):
    """Backbone protocol expected by :class:`TEMController`."""


# =================================================================================================
@dataclass
class TEMRolloutState[ModelState](RolloutState[ModelState]):
    """Controller carry/state for TEM rollouts."""


# =================================================================================================
@dataclass(frozen=True)
class TEMOutput(DetachMixin):
    """Canonical controller-facing TEM step output.

    Attributes:
        obs_logits:
            Three observation-logit tensors ordered as inference, retrieved,
            and ancestral pathways.
        latent_relations:
            Named TEM latent comparison terms. Expected keys are
            ``grid_transition``, ``place_transition``, and optional
            ``place_sensory``.
        reg_terms:
            Optional named regularization targets. Expected keys are ``grid``
            and ``place`` when present.
        theta_cls:
            Optional summary features for auxiliary diagnostics.

    Notes:
        Heads should consume the named relation map rather than legacy
        tuple-position semantics.
    """

    obs_logits: tuple[Tensor, Tensor, Tensor]

    @property
    def logits_inference(self) -> Tensor:
        """Return observation logits from the inference pathway."""
        return self.obs_logits[0]

    @property
    def logits_retrieved(self) -> Tensor:
        """Return observation logits from the retrieved pathway."""
        return self.obs_logits[1]

    @property
    def logits_ancestral(self) -> Tensor:
        """Return observation logits from the ancestral pathway."""
        return self.obs_logits[2]

    latent_relations: dict[str, LatentRelation]

    @property
    def grid_post(self) -> Tensor:
        """Return post-transition grid code."""
        return self.latent_relations[GRID_TRANSITION_RELATION].lhs

    @property
    def grid_prior(self) -> Tensor:
        """Return pre-transition grid code."""
        return self.latent_relations[GRID_TRANSITION_RELATION].rhs

    @property
    def place_post(self) -> Tensor:
        """Return post-transition place code."""
        return self.latent_relations[PLACE_TRANSITION_RELATION].lhs

    @property
    def place_prior(self) -> Tensor:
        """Return pre-transition place code."""
        return self.latent_relations[PLACE_TRANSITION_RELATION].rhs

    @property
    def place_sensory(self) -> Tensor | None:
        """Return sensory place code if present, else None."""
        relation = self.latent_relations.get(PLACE_SENSORY_RELATION)
        return relation.lhs if relation is not None else None

    reg_terms: dict[str, LatentCode] | None = None

    @property
    def reg_grid(self) -> LatentCode | None:
        """Return grid regularization target if present, else None."""
        return self.reg_terms.get(GRID_REG_TERM) if self.reg_terms is not None else None

    @property
    def reg_place(self) -> LatentCode | None:
        """Return place regularization target if present, else None."""
        return self.reg_terms.get(PLACE_REG_TERM) if self.reg_terms is not None else None

    theta_cls: Tensor | None = None


class TEMController[ModelState](BaseController[ModelState, TEMControllerConfig]):
    """ """

    def __init__(  # ------------------------------------------------------------------------------
        self, backbone: TEMRolloutBackbone[ModelState], config: TEMControllerConfig,
    ) -> None:  # fmt: skip
        """Create a controller.

        Args:
            backbone: Model implementing :class:`TEMRolloutBackbone`.
            config: Controller configuration.
        """
        super().__init__(backbone=backbone, config=config)

    def initial_state(  # -------------------------------------------------------------------------
        self, batch_sample: Batch
    ) -> TEMRolloutState[ModelState]:  # fmt: skip
        """Build an initial rollout state from a batch sample."""
        slots = self.initial_slots(batch_sample)
        return TEMRolloutState(
            model_state=slots.model_state, steps=slots.steps, halted=slots.halted,
            data=slots.data,
        )  # fmt: skip

    def step(  # ----------------------------------------------------------------------------------
        self, state: TEMRolloutState[ModelState], batch: Batch,
    ) -> tuple[TEMRolloutState[ModelState], TEMOutput]:  # fmt: skip
        """Advance the controller by one variational step."""
        data = self.refresh_slot_data(batch, state)
        model_state = self.backbone.reset_state(state.halted, state.model_state)
        model_state, obs_logits, _, grid, place = self.backbone(data, model_state)
        latent_relations = self._coerce_latent(grid, place)
        reg_terms = self._coerce_regularization(grid, place)

        steps = self.advance_steps(state)
        done = steps >= self.config.max_steps

        state = TEMRolloutState(model_state=model_state, steps=steps, halted=done, data=data)
        output = TEMOutput(obs_logits=obs_logits, latent_relations=latent_relations, reg_terms=reg_terms)

        return state, output

    def _coerce_latent(  # --------------------------------------------------------------
        self, grid: tuple[Tensor, Tensor], place: tuple[Tensor, Tensor, Tensor],
    ) -> dict[str, LatentRelation]:  # fmt: skip
        """ """
        grid_post, grid_prior = grid
        place_post, place_prior, place_sensory = place
        latent_relations = {
            GRID_TRANSITION_RELATION: LatentRelation(lhs=grid_post, rhs=grid_prior),
            PLACE_TRANSITION_RELATION: LatentRelation(lhs=place_post, rhs=place_prior),
        }
        if place_sensory is not None:
            latent_relations[PLACE_SENSORY_RELATION] = LatentRelation(lhs=place_post, rhs=place_sensory)

        return latent_relations

    def _coerce_regularization(  # --------------------------------------------------------------
        self, grid: tuple[Tensor, Tensor], place: tuple[Tensor, Tensor, Tensor],
    ) -> dict[str, LatentCode] | None:  # fmt: skip
        """ """
        reg_terms: dict[str, LatentCode] | None = None
        # TODO: add logic

        return reg_terms

    def refresh_slot_data(  # ---------------------------------------------------------------------
        self, batch: Batch, state: TEMRolloutState[ModelState]
    ) -> dict[str, Tensor]:  # fmt: skip
        """Replace data for halted slots with incoming batch data."""
        return super().refresh_slot_data(batch, state)


# =================================================================================================
__all__ = [
    "GRID_REG_TERM", "GRID_TRANSITION_RELATION", "PLACE_REG_TERM", "PLACE_SENSORY_RELATION",
    "PLACE_TRANSITION_RELATION",
    "TEMRolloutBackbone", "TEMController", "TEMControllerConfig", "TEMOutput", "TEMRolloutState",
]  # fmt: skip
