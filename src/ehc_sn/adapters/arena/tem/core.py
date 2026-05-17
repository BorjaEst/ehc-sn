"""Shared Arena TEM bridge family core.

Canonical home for shared config types, diagnostics, bridge output, and decode
helpers that are identical across TEM v1 and TEM v2. Version-specific modules
bind these shared types to model-native input, output, and state types.

The stable public family surface is the task-family barrel. This module owns
shared bridge types and helper implementations used by that barrel and the
version-specific bridge implementations.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field, model_validator
from torch import Tensor, nn

from ehc_sn.loss.consistency import LatentRelation
from ehc_sn.models.tem.core.tem_base import GridCodes, PlaceCodes, PredCodes
from ehc_sn.modules.autoencoder import TwoHotEncoder
from ehc_sn.objectives.tem import (
    GRID_TRANSITION_RELATION,
    PLACE_SENSORY_RELATION,
    PLACE_TRANSITION_RELATION,
)
from ehc_sn.tasks.arena.contracts import ArenaTaskOutput
from ehc_sn.types import Batch, MultiScaleCode
from ehc_sn.utils.detach import DetachMixin


# =============================================================================
# Config types — shared across all Arena TEM versions
# =============================================================================
class ArenaEncoderConfig(BaseModel, extra="forbid"):
    """Encoder strategy config for the arena-to-TEM sensory pathway.

    Only ``kind='two_hot'`` with ``layout='replicated'`` is implemented.
    """

    kind: Literal["two_hot"] = Field(
        default="two_hot",
        description="Observation encoder family. Only 'two_hot' is supported.",
    )
    layout: Literal["replicated"] = Field(
        default="replicated",
        description="Band-packing strategy. Only 'replicated' is supported with 'two_hot'.",
    )


class ArenaDecoderConfig(BaseModel, extra="forbid"):
    """Decoder strategy config for the TEM-to-arena output pathway."""

    kind: Literal["single_scale", "multi_scale"] = Field(
        default="single_scale",
        description="Decoder input policy.",
    )
    prediction_freq: int = Field(
        default=0,
        ge=0,
        description="HPC band index for single_scale decoding. Must be 0 when kind='multi_scale'.",
    )

    @model_validator(mode="after")
    def _check_concat_has_no_freq(self) -> "ArenaDecoderConfig":
        if self.kind == "multi_scale" and self.prediction_freq != 0:
            raise ValueError(
                "ArenaDecoderConfig: prediction_freq must be 0 (unused) when "
                f"kind='multi_scale', got prediction_freq={self.prediction_freq}."
            )
        return self


class ArenaTEMAdapterSettings(BaseModel, extra="forbid"):
    """Task-side Arena settings required to bind any Arena TEM version.

    Both TEM v1 and TEM v2 use identical adapter settings.
    """

    observation_dim: int = Field(
        ...,
        description="Dimensionality of arena task observations.",
    )
    action_count: int = Field(
        ...,
        ge=1,
        description="Number of discrete actions.",
    )
    encoder: ArenaEncoderConfig = Field(
        default_factory=ArenaEncoderConfig,
        description="Observation encoder strategy.",
    )
    decoder: ArenaDecoderConfig = Field(
        default_factory=ArenaDecoderConfig,
        description="Observation decoder strategy.",
    )


# =============================================================================
# Shared bridge types — diagnostics and output
# =============================================================================
@dataclass(frozen=True)
class ArenaTEMDiagnostics(DetachMixin):
    """TEM-family diagnostic surface for controller and objective consumption.

    Satisfies :class:`~ehc_sn.objectives.tem.TEMStepOutput` via computed
    properties that map bridge-native fields to the protocol's attribute names.

    Used by both TEM v1 and TEM v2 bridge adapters.
    """

    obs_logits: tuple[Tensor, Tensor, Tensor]
    grid_codes: GridCodes  # (prior, post)
    place_codes: PlaceCodes  # (posterior, prior, retrieved, sensory)
    pred_codes: PredCodes  # (inference, retrieved, ancestral)

    # -- TEMStepOutput protocol surface -----------------------------------------

    @property
    def logits_inference(self) -> Tensor:
        """Observation logits from the HPC inference (posterior) pathway."""
        return self.obs_logits[0]

    @property
    def logits_retrieved(self) -> Tensor:
        """Observation logits from the HPC retrieved (corrected-grid) pathway."""
        return self.obs_logits[1]

    @property
    def logits_ancestral(self) -> Tensor:
        """Observation logits from the HPC ancestral (structural prior) pathway."""
        return self.obs_logits[2]

    @property
    def latent_relations(self) -> dict[str, LatentRelation]:
        """Named latent consistency relations expected by :class:`~ehc_sn.objectives.tem.TEMObjective`."""
        relations: dict[str, LatentRelation] = {
            GRID_TRANSITION_RELATION: LatentRelation(
                lhs=self.grid_codes.posterior, rhs=self.grid_codes.prior
            ),
            PLACE_TRANSITION_RELATION: LatentRelation(
                lhs=self.place_codes.posterior, rhs=self.place_codes.retrieved
            ),
        }
        if self.place_codes.sensory is not None:
            relations[PLACE_SENSORY_RELATION] = LatentRelation(
                lhs=self.place_codes.posterior, rhs=self.place_codes.sensory
            )
        return relations

    @property
    def reg_terms(self) -> None:
        """No regularization-code overrides; TEMObjective falls back to relation codes."""
        return None


@dataclass(frozen=True)
class ArenaTEMBridgeOutput(DetachMixin):
    """Split bridge output: canonical task surface plus TEM-family diagnostics.

    Used by both TEM v1 and TEM v2 bridge adapters.
    """

    task: ArenaTaskOutput
    tem: ArenaTEMDiagnostics

    @property
    def obs_logits(self) -> tuple[Tensor, Tensor, Tensor]:
        """Return pathway logits on the bridge output's public surface."""
        return self.tem.obs_logits

    @property
    def logits_inference(self) -> Tensor:
        """Return posterior-path observation logits."""
        return self.tem.logits_inference

    @property
    def logits_retrieved(self) -> Tensor:
        """Return sensory-recall-path observation logits."""
        return self.tem.logits_retrieved

    @property
    def logits_ancestral(self) -> Tensor:
        """Return structural-prior-path observation logits."""
        return self.tem.logits_ancestral

    @property
    def latent_relations(self) -> dict[str, LatentRelation]:
        """Expose TEM latent-consistency relations on the bridge output."""
        return self.tem.latent_relations

    @property
    def reg_terms(self) -> None:
        """Expose optional TEM regularization-code overrides on the bridge output."""
        return self.tem.reg_terms


# =============================================================================
# Shared encoder base
# =============================================================================
class ArenaTwoHotEncoder(nn.Module):
    """Encodes arena step data into a multiscale sensory code.

    Materialises the one-hot observation from the raw ``observation_id`` integer,
    applies a :class:`~ehc_sn.modules.autoencoder.TwoHotEncoder`, and replicates
    the code once per HPC frequency band.  The resulting list is passed to the
    model-native input constructor in the versioned encoder subclass.
    """

    def __init__(  # ----------------------------------------------------------
        self,
        observation_dim: int,
        feature_dim: int,
        n_freq: int,
    ) -> None:
        super().__init__()
        self.encoder = TwoHotEncoder(observation_dim, feature_dim)
        self._obs_dim = observation_dim
        self._n_freq = n_freq

    @staticmethod
    def _normalize_landmark_id(landmark_id: Tensor | None) -> Tensor | None:
        """Map Arena's absent-cue sentinel to the model-facing no-cue value."""
        if landmark_id is None or not torch.any(landmark_id < 0):
            return landmark_id
        return torch.where(
            landmark_id < 0, torch.zeros_like(landmark_id), landmark_id
        )

    def forward(  # -----------------------------------------------------------
        self,
        batch: Batch,
    ) -> tuple[MultiScaleCode, Tensor, object, object]:
        """Return ``(observation_embedding, previous_action, episode_start, landmark_id)``."""
        obs_id = batch["observation_id"].view(-1).long()  # (B,)
        observation = F.one_hot(
            obs_id, num_classes=self._obs_dim
        ).float()  # (B, obs_dim)
        code = self.encoder(observation)
        observation_embedding: MultiScaleCode = [
            code.clone() for _ in range(self._n_freq)
        ]
        landmark_id = self._normalize_landmark_id(batch.get("landmark_id"))
        return (
            observation_embedding,
            batch["previous_action"],
            batch.get("episode_start"),
            landmark_id,
        )


# =============================================================================
__all__ = [
    "ArenaDecoderConfig",
    "ArenaEncoderConfig",
    "ArenaTEMAdapterSettings",
    "ArenaTEMBridgeOutput",
    "ArenaTEMDiagnostics",
    "ArenaTwoHotEncoder",
    "ArenaTaskOutput",
]
