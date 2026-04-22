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
from pydantic import BaseModel, Field, model_validator
from torch import Tensor, nn

from ehc_sn.loss.consistency import LatentRelation
from ehc_sn.models.tem.core.tem_base import GridCodes, PlaceCodes
from ehc_sn.modules.autoencoder import MLPDecoder, TwoHotEncoder
from ehc_sn.objectives.tem import GRID_TRANSITION_RELATION, PLACE_SENSORY_RELATION, PLACE_TRANSITION_RELATION
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

    kind: Literal["single_scale", "concat_all_scales"] = Field(
        default="single_scale",
        description="Decoder input policy.",
    )
    prediction_freq: int = Field(
        default=0,
        ge=0,
        description="HPC band index for single_scale decoding. Must be 0 when kind='concat_all_scales'.",
    )

    @model_validator(mode="after")
    def _check_concat_has_no_freq(self) -> "ArenaDecoderConfig":
        if self.kind == "concat_all_scales" and self.prediction_freq != 0:
            raise ValueError(
                "ArenaDecoderConfig: prediction_freq must be 0 (unused) when "
                f"kind='concat_all_scales', got prediction_freq={self.prediction_freq}."
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

    Satisfies :class:`~ehc_sn.objectives.tem.TEMStepOutputs` via computed
    properties that map bridge-native fields to the protocol's attribute names.

    Used by both TEM v1 and TEM v2 bridge adapters.
    """

    obs_logits: tuple[Tensor, Tensor, Tensor]
    grid_codes: GridCodes
    place_codes: PlaceCodes

    # -- TEMStepOutputs protocol surface -----------------------------------------

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
        """Named latent consistency relations expected by :class:`~ehc_sn.objectives.tem.TEMLossHead`."""
        relations: dict[str, LatentRelation] = {
            GRID_TRANSITION_RELATION: LatentRelation(lhs=self.grid_codes.post, rhs=self.grid_codes.prior),
            PLACE_TRANSITION_RELATION: LatentRelation(lhs=self.place_codes.inference, rhs=self.place_codes.ancestral),
        }
        if self.place_codes.retrieved is not None and self.place_codes.sensory is not None:
            relations[PLACE_SENSORY_RELATION] = LatentRelation(lhs=self.place_codes.retrieved, rhs=self.place_codes.sensory)
        return relations

    @property
    def reg_terms(self) -> None:
        """No regularization-code overrides; TEMLossHead falls back to relation codes."""
        return None


@dataclass(frozen=True)
class ArenaTEMBridgeOutput(DetachMixin):
    """Split bridge output: canonical task surface plus TEM-family diagnostics.

    Used by both TEM v1 and TEM v2 bridge adapters.
    """

    task: ArenaTaskOutput
    tem: ArenaTEMDiagnostics


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

    def __init__(
        self,
        observation_dim: int,
        feature_dim: int,
        n_freq: int,
    ) -> None:
        super().__init__()
        self.encoder = TwoHotEncoder(observation_dim, feature_dim)
        self._obs_dim = observation_dim
        self._n_freq = n_freq

    def encode(self, batch: Batch) -> tuple[MultiScaleCode, Tensor, object, object]:
        """Return ``(sensory_codes, previous_action, episode_start, landmark_id)``."""
        import torch.nn.functional as F

        obs_id = batch["observation_id"].view(-1).long()  # (B,)
        observation = F.one_hot(obs_id, num_classes=self._obs_dim).float()  # (B, obs_dim)
        code = self.encoder(observation)
        sensory_codes: MultiScaleCode = [code.clone() for _ in range(self._n_freq)]
        return (
            sensory_codes,
            batch["previous_action"],
            batch.get("episode_start"),
            batch.get("landmark_id"),
        )


# =============================================================================
# Shared decode helper
# =============================================================================
def decode_observation_pathways(
    pc: PlaceCodes,
    gc: GridCodes,
    decoder: MLPDecoder,
    obs_dim: int,
    single_freq: int | None,
) -> ArenaTEMBridgeOutput:
    """Decode all three HPC place pathways into an :class:`ArenaTEMBridgeOutput`.

    Shared by :class:`ArenaOutputsDecoderV1` and :class:`ArenaOutputsDecoderV2`.
    """
    obs_inference = decoder(_select_code(pc.inference, single_freq))
    obs_retrieved = (
        decoder(_select_code(pc.retrieved, single_freq))
        if pc.retrieved is not None
        else obs_inference.new_zeros(obs_inference.shape[0], obs_dim)
    )
    obs_ancestral = decoder(_select_code(pc.ancestral, single_freq))
    ol = (obs_inference, obs_retrieved, obs_ancestral)
    task = ArenaTaskOutput(obs_logits=obs_inference)
    tem = ArenaTEMDiagnostics(obs_logits=ol, grid_codes=gc, place_codes=pc)
    return ArenaTEMBridgeOutput(task=task, tem=tem)


def _select_code(code: Tensor | Sequence[Tensor], single_freq: int | None) -> Tensor:
    """Return a ``(B, D)`` tensor for decoder use."""
    if isinstance(code, Tensor):
        if single_freq is not None:
            raise ValueError("single_scale decoder received a pre-concatenated flat Tensor; " "expected a sequence of per-band tensors.")
        return code
    if len(code) == 0:
        raise ValueError("TEM latent code sequences must not be empty.")
    if single_freq is not None:
        return code[single_freq]
    return torch.cat(tuple(code), dim=1)


# =============================================================================
__all__ = [
    "ArenaDecoderConfig",
    "ArenaEncoderConfig",
    "ArenaTEMAdapterSettings",
    "ArenaTEMBridgeOutput",
    "ArenaTEMDiagnostics",
    "ArenaTwoHotEncoder",
    "decode_observation_pathways",
]
