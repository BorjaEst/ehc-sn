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

from ehc_sn.loss.consistency import LatentCode, LatentRelation
from ehc_sn.models.tem.core.tem_base import GridCodes, PlaceCodes, PredCodes
from ehc_sn.modules.autoencoder import TwoHotEncoder
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

    kind: Literal["two_hot", "learned"] = Field(
        default="two_hot",
        description="Observation encoder family.",
    )
    layout: Literal["replicated"] = Field(
        default="replicated",
        description="Band-packing strategy. Only 'replicated' is supported.",
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
class TEMLearningState(DetachMixin):
    """TEM-family diagnostic surface for controller and objective consumption.

    Carries typed latent-relation fields (grid transition, place transition,
    place sensory) and regularization-code overrides.  ``TEMObjective`` reads
    these fields directly instead of looking up string keys in a dictionary.

    Used by both TEM v1 and TEM v2 bridge adapters.
    """

    obs_logits: tuple[Tensor, Tensor, Tensor]
    grid_codes: GridCodes  # (prior, post)
    place_codes: PlaceCodes  # (posterior, prior, retrieved, sensory)
    pred_codes: PredCodes  # (inference, retrieved, ancestral)

    # Typed latent relations — consumed directly by TEMObjective.
    grid_transition: LatentRelation
    place_transition: LatentRelation
    place_sensory: LatentRelation | None

    # Typed regularization codes — consumed directly by TEMObjective.
    grid_reg_code: LatentCode
    place_reg_code: LatentCode

    # -- TEMStepOutput protocol surface -----------------------------------------

    @property
    def logits_post(self) -> Tensor:
        """Observation logits from the HPC inference (posterior) pathway."""
        return self.obs_logits[0]

    @property
    def logits_recall(self) -> Tensor:
        """Observation logits from the HPC retrieved (corrected-grid) pathway."""
        return self.obs_logits[1]

    @property
    def logits_path(self) -> Tensor:
        """Observation logits from the HPC ancestral (structural prior) pathway."""
        return self.obs_logits[2]


@dataclass(frozen=True)
class ArenaTEMBridgeOutput(DetachMixin):
    """Split bridge output: canonical task surface plus TEM-family diagnostics.

    Used by both TEM v1 and TEM v2 bridge adapters.
    """

    task: ArenaTaskOutput
    tem: TEMLearningState

    @property
    def obs_logits(self) -> tuple[Tensor, Tensor, Tensor]:
        """Return pathway logits on the bridge output's public surface."""
        return self.tem.obs_logits

    @property
    def logits_post(self) -> Tensor:
        """Return posterior-path observation logits."""
        return self.tem.logits_post

    @property
    def logits_recall(self) -> Tensor:
        """Return sensory-recall-path observation logits."""
        return self.tem.logits_recall

    @property
    def logits_path(self) -> Tensor:
        """Return structural-prior-path observation logits."""
        return self.tem.logits_path

    @property
    def grid_transition(self) -> LatentRelation:
        """Return grid transition relation."""
        return self.tem.grid_transition

    @property
    def place_transition(self) -> LatentRelation:
        """Return place transition relation."""
        return self.tem.place_transition

    @property
    def place_sensory(self) -> LatentRelation | None:
        """Return place sensory relation."""
        return self.tem.place_sensory

    @property
    def grid_reg_code(self) -> LatentCode:
        """Return grid regularization code."""
        return self.tem.grid_reg_code

    @property
    def place_reg_code(self) -> LatentCode:
        """Return place regularization code."""
        return self.tem.place_reg_code


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
        landmark_id = _normalize_landmark_id(batch.get("landmark_id"))
        return (
            observation_embedding,
            batch["previous_action"],
            batch.get("episode_start"),
            landmark_id,
        )


# =============================================================================
def _normalize_landmark_id(landmark_id: Tensor | None) -> Tensor | None:
    """Map Arena's absent-cue sentinel to the model-facing no-cue value."""
    if landmark_id is None or not torch.any(landmark_id < 0):
        return landmark_id
    return torch.where(
        landmark_id < 0, torch.zeros_like(landmark_id), landmark_id
    )


# =============================================================================
class ArenaLearnedEncoder(nn.Module):
    """Learned dense embedding encoder for arena observation IDs.

    Maps each ``observation_id`` integer to a dense learned vector via
    ``nn.Embedding`` and replicates the vector across all LEC frequency bands
    (identical copy per band, preserving the TEM assumption that bands differ
    only in temporal dynamics, not sensory content).

    This is an adapter-owned alternative to :class:`ArenaTwoHotEncoder` with
    the same output contract: ``forward(batch) -> (MultiScaleCode, action, ...)``.
    """

    def __init__(  # ----------------------------------------------------------
        self,
        observation_dim: int,
        feature_dim: int,
        n_freq: int,
        *,
        device=None,
        dtype=None,
    ) -> None:
        """Initialize the learned embedding encoder."""
        super().__init__()
        if observation_dim < 1:
            raise ValueError(
                f"observation_dim must be positive, got {observation_dim}."
            )
        if feature_dim < 1:
            raise ValueError(
                f"feature_dim must be positive, got {feature_dim}."
            )
        if n_freq < 1:
            raise ValueError(f"n_freq must be positive, got {n_freq}.")
        self.embed = nn.Embedding(
            observation_dim, feature_dim, device=device, dtype=dtype
        )
        self._obs_dim = observation_dim
        self._n_freq = n_freq

    def forward(  # ------------------------------------------------------------
        self,
        batch: Batch,
    ) -> tuple[MultiScaleCode, Tensor, object, object]:
        """Return ``(observation_embedding, previous_action, episode_start, landmark_id)``."""
        obs_id = batch["observation_id"].view(-1).long()  # (B,)
        code = self.embed(obs_id)  # (B, feature_dim)
        observation_embedding: MultiScaleCode = [
            code.clone() for _ in range(self._n_freq)
        ]
        landmark_id = _normalize_landmark_id(batch.get("landmark_id"))
        return (
            observation_embedding,
            batch["previous_action"],
            batch.get("episode_start"),
            landmark_id,
        )


# =============================================================================
def build_arena_observation_encoder(  # ---------------------------------------
    config: ArenaEncoderConfig,
    observation_dim: int,
    feature_dim: int,
    n_freq: int,
    *,
    device=None,
    dtype=None,
) -> ArenaTwoHotEncoder | ArenaLearnedEncoder:
    """Construct the concrete arena observation encoder from config."""
    match config.kind:
        case "two_hot":
            return ArenaTwoHotEncoder(observation_dim, feature_dim, n_freq)
        case "learned":
            return ArenaLearnedEncoder(
                observation_dim,
                feature_dim,
                n_freq,
                device=device,
                dtype=dtype,
            )
        case _:
            raise ValueError(
                f"Unsupported arena encoder kind: {config.kind!r}."
            )


# =============================================================================
__all__ = [
    "ArenaDecoderConfig",
    "ArenaEncoderConfig",
    "ArenaLearnedEncoder",
    "ArenaTEMAdapterSettings",
    "ArenaTEMBridgeOutput",
    "ArenaTEMDiagnostics",
    "ArenaTwoHotEncoder",
    "ArenaTaskOutput",
    "build_arena_observation_encoder",
]
