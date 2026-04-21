"""Arena plus TEM v2 bridge implementation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field, model_validator
from torch import Tensor, nn

from ehc_sn.models.tem.core.tem_base import GridCodes, PlaceCodes
from ehc_sn.models.tem.tem_v2 import TEMInputV2, TEMModelV2, TEMOutputV2, TEMStateV2
from ehc_sn.modules.autoencoder import MLPDecoder, TwoHotEncoder
from ehc_sn.tasks.arena.contracts import ArenaTaskOutput
from ehc_sn.types import Batch, MultiScaleCode
from ehc_sn.utils.detach import DetachMixin


# =============================================================================
class ArenaEncoderConfig(BaseModel, extra="forbid"):
    """Encoder strategy config for the arena-to-TEM sensory pathway."""

    kind: Literal["two_hot", "mlp", "identity"] = Field(
        default="two_hot",
        description="Observation encoder family.",
    )
    layout: Literal["replicated", "per_band"] = Field(
        default="replicated",
        description="Band-packing strategy: 'replicated' or 'per_band'.",
    )


# =============================================================================
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


# =============================================================================
class ArenaTEMV2AdapterSettings(BaseModel, extra="forbid"):
    """Task-side Arena settings required to bind the TEM v2 core."""

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
@dataclass(frozen=True)
class ArenaTEMV2Diagnostics(DetachMixin):
    """TEM-family diagnostic surface for controller and objective consumption."""

    obs_logits: tuple[Tensor, Tensor, Tensor]
    grid_codes: GridCodes
    place_codes: PlaceCodes


# =============================================================================
@dataclass(frozen=True)
class ArenaTEMV2BridgeOutput(DetachMixin):
    """Split bridge output: canonical task surface plus TEM-family diagnostics."""

    task: ArenaTaskOutput
    tem: ArenaTEMV2Diagnostics


# =============================================================================
class ArenaInputsEncoderV2(nn.Module):
    """Encodes arena step data into a :class:`TEMInputV2` payload."""

    def __init__(
        self,
        observation_dim: int,
        feature_dim: int,
        n_freq: int,
        encoder_config: ArenaEncoderConfig,
    ) -> None:
        super().__init__()
        if encoder_config.kind != "two_hot":
            raise ValueError(
                f"ArenaInputsEncoderV2: encoder kind {encoder_config.kind!r} is not yet "
                "implemented. Only 'two_hot' is supported in this release."
            )
        if encoder_config.layout != "replicated":
            raise ValueError(
                f"ArenaInputsEncoderV2: layout {encoder_config.layout!r} is not supported " "with kind='two_hot'. Use layout='replicated'."
            )
        self.encoder = TwoHotEncoder(observation_dim, feature_dim)
        self._obs_dim = observation_dim
        self._n_freq = n_freq

    def forward(self, batch: dict[str, Tensor]) -> TEMInputV2:
        """Encode a pre-extracted arena step payload into a TEM v2 input."""
        obs_id = batch["observation_id"].view(-1).long()  # (B,)
        observation = F.one_hot(obs_id, num_classes=self._obs_dim).float()  # (B, obs_dim)
        code = self.encoder(observation)
        sensory_codes: MultiScaleCode = [code.clone() for _ in range(self._n_freq)]

        prev_action = batch["previous_action"]
        episode_start = batch.get("episode_start")
        landmark_id = batch.get("landmark_id")

        return TEMInputV2(
            sensory_codes=sensory_codes,
            previous_action=prev_action,
            episode_start=episode_start,
            landmark_id=landmark_id,
        )


# =============================================================================
class ArenaOutputsDecoderV2(nn.Module):
    """Decodes a :class:`TEMOutputV2` into an :class:`ArenaTEMV2BridgeOutput`."""

    def __init__(
        self,
        observation_dim: int,
        latent_dim: int,
        *,
        single_freq: int | None = None,
    ) -> None:
        super().__init__()
        self.decoder = MLPDecoder(latent_dim, observation_dim)
        self._obs_dim = observation_dim
        self._single_freq = single_freq

    def forward(self, model_output: TEMOutputV2) -> ArenaTEMV2BridgeOutput:
        """Decode all three place pathways and return the split task + TEM surfaces."""
        pc = model_output.place_codes
        gc = model_output.grid_codes

        obs_inference = self.decoder(_select_code(pc.inference, self._single_freq))
        obs_retrieved = (
            self.decoder(_select_code(pc.retrieved, self._single_freq))
            if pc.retrieved is not None
            else obs_inference.new_zeros(obs_inference.shape[0], self._obs_dim)
        )
        obs_ancestral = self.decoder(_select_code(pc.ancestral, self._single_freq))
        ol = (obs_inference, obs_retrieved, obs_ancestral)

        task = ArenaTaskOutput(obs_logits=obs_inference)
        tem = ArenaTEMV2Diagnostics(obs_logits=ol, grid_codes=gc, place_codes=pc)
        return ArenaTEMV2BridgeOutput(task=task, tem=tem)


# =============================================================================
class ArenaTEMV2BridgeAdapter(nn.Module):
    """Arena plus TEM v2 bridge adapter.

    Implements the :class:`~ehc_sn.controllers._base.RolloutBackbone` protocol.
    """

    def __init__(self, model: TEMModelV2, config: ArenaTEMV2AdapterSettings) -> None:
        """Initialize the arena plus TEM v2 bridge adapter."""
        super().__init__()
        self._config = config
        self.model = model
        self.encoder = _build_encoder_v2(model, config)
        self.decoder = _build_decoder_v2(model, config)

    @property
    def config(self) -> ArenaTEMV2AdapterSettings:
        """The arena plus TEM v2 bridge adapter settings."""
        return self._config

    def init_state(self, batch_size: int, *, device: Optional[torch.device] = None) -> TEMStateV2:
        """Create a fresh TEM recurrent state for one rollout batch."""
        return self.model.init_state(batch_size, device=device)

    def reset_state(self, reset_flag: Tensor, state: TEMStateV2) -> TEMStateV2:
        """Reset halted rows of the TEM recurrent state."""
        return self.model.reset_state(reset_flag, state)

    def prepare_inputs(self, batch: Batch) -> TEMInputV2:
        """Encode a pre-extracted arena step dict into a model-native :class:`TEMInputV2`."""
        return self.encoder(batch)

    def prepare_outputs(self, model_output: TEMOutputV2) -> ArenaTEMV2BridgeOutput:
        """Decode a TEM model output into the split task + TEM bridge surfaces."""
        return self.decoder(model_output)

    def forward(
        self,
        batch: Batch,
        state: TEMStateV2 | None = None,
    ) -> tuple[ArenaTEMV2BridgeOutput, TEMStateV2]:
        """Run a forward pass of the TEM v2 bridge adapter on an arena step payload."""
        inputs = self.prepare_inputs(batch)
        model_output, next_state = self.model(inputs, state=state)
        bridge_output = self.prepare_outputs(model_output)
        return bridge_output, next_state


# =============================================================================
def _build_encoder_v2(model: TEMModelV2, config: ArenaTEMV2AdapterSettings) -> ArenaInputsEncoderV2:
    return ArenaInputsEncoderV2(
        observation_dim=config.observation_dim,
        feature_dim=model.config.lec.feature_dim,
        n_freq=model.lec.n_freq,
        encoder_config=config.encoder,
    )


def _build_decoder_v2(model: TEMModelV2, config: ArenaTEMV2AdapterSettings) -> ArenaOutputsDecoderV2:
    hpc_shape = model.config.hpc.shape
    n_freq = len(hpc_shape)
    if config.decoder.kind == "single_scale":
        freq = config.decoder.prediction_freq
        if not (0 <= freq < n_freq):
            raise ValueError(f"prediction_freq={freq} is out of range for hpc.shape with {n_freq} bands " f"(valid: 0..{n_freq - 1}).")
        return ArenaOutputsDecoderV2(
            observation_dim=config.observation_dim,
            latent_dim=hpc_shape[freq],
            single_freq=freq,
        )
    return ArenaOutputsDecoderV2(
        observation_dim=config.observation_dim,
        latent_dim=sum(hpc_shape),
    )


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
    "ArenaInputsEncoderV2",
    "ArenaOutputsDecoderV2",
    "ArenaTEMV2Diagnostics",
    "ArenaTEMV2AdapterSettings",
    "ArenaTEMV2BridgeAdapter",
    "ArenaTEMV2BridgeOutput",
]
