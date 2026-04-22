"""Arena plus TEM v2 bridge implementation.

Binds model-native TEM v2 types to the shared Arena TEM family core.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from ehc_sn.adapters.arena.tem.core import (
    ArenaDecoderConfig,
    ArenaEncoderConfig,
    ArenaTEMAdapterSettings,
    ArenaTEMBridgeOutput,
    ArenaTwoHotEncoder,
    decode_observation_pathways,
)
from ehc_sn.models.tem.tem_v2 import TEMInputV2, TEMModelV2, TEMOutputV2, TEMStateV2
from ehc_sn.modules.autoencoder import MLPDecoder
from ehc_sn.types import Batch


# =============================================================================
class ArenaInputsEncoderV2(nn.Module):
    """Encodes arena step data into a :class:`TEMInputV2` payload."""

    def __init__(self, observation_dim: int, feature_dim: int, n_freq: int) -> None:
        super().__init__()
        self._core = ArenaTwoHotEncoder(observation_dim, feature_dim, n_freq)

    def forward(self, batch: Batch) -> TEMInputV2:
        """Encode a pre-extracted arena step payload into a TEM v2 input."""
        sensory_codes, prev_action, episode_start, landmark_id = self._core.encode(batch)
        return TEMInputV2(
            sensory_codes=sensory_codes,
            previous_action=prev_action,
            episode_start=episode_start,
            landmark_id=landmark_id,
        )


# =============================================================================
class ArenaOutputsDecoderV2(nn.Module):
    """Decodes a :class:`TEMOutputV2` into an :class:`ArenaTEMBridgeOutput`."""

    def __init__(self, observation_dim: int, latent_dim: int, *, single_freq: int | None = None) -> None:
        super().__init__()
        self.decoder = MLPDecoder(latent_dim, observation_dim)
        self._obs_dim = observation_dim
        self._single_freq = single_freq

    def forward(self, model_output: TEMOutputV2) -> ArenaTEMBridgeOutput:
        """Decode all three place pathways and return the split task + TEM surfaces."""
        return decode_observation_pathways(
            model_output.place_codes,
            model_output.grid_codes,
            self.decoder,
            self._obs_dim,
            self._single_freq,
        )


# =============================================================================
class ArenaTEMV2BridgeAdapter(nn.Module):
    """Arena plus TEM v2 bridge adapter implementing RolloutBackbone."""

    def __init__(self, model: TEMModelV2, config: ArenaTEMAdapterSettings) -> None:
        super().__init__()
        self._config = config
        self.model = model
        self._encoder = _build_encoder_v2(model, config)
        self._decoder = _build_decoder_v2(model, config)

    @property
    def config(self) -> ArenaTEMAdapterSettings:
        return self._config

    def init_state(self, batch_size: int, *, device: Optional[torch.device] = None) -> TEMStateV2:
        return self.model.init_state(batch_size, device=device)

    def reset_state(self, reset_flag: Tensor, state: TEMStateV2) -> TEMStateV2:
        return self.model.reset_state(reset_flag, state)

    def prepare_inputs(self, batch: Batch) -> TEMInputV2:
        return self._encoder(batch)

    def postprocess(self, model_output: TEMOutputV2) -> ArenaTEMBridgeOutput:
        return self._decoder(model_output)

    def forward(
        self,
        batch: Batch,
        state: TEMStateV2 | None = None,
    ) -> tuple[ArenaTEMBridgeOutput, TEMStateV2]:
        inputs = self.prepare_inputs(batch)
        model_output, next_state = self.model(inputs, state=state)
        bridge_output = self.postprocess(model_output)
        return bridge_output, next_state


# =============================================================================
def _build_encoder_v2(model: TEMModelV2, config: ArenaTEMAdapterSettings) -> ArenaInputsEncoderV2:
    return ArenaInputsEncoderV2(
        observation_dim=config.observation_dim,
        feature_dim=model.config.lec.feature_dim,
        n_freq=model.lec.n_freq,
    )


def _build_decoder_v2(model: TEMModelV2, config: ArenaTEMAdapterSettings) -> ArenaOutputsDecoderV2:
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


# =============================================================================
__all__ = [
    "ArenaTEMV2BridgeAdapter",
    "ArenaInputsEncoderV2",
    "ArenaOutputsDecoderV2",
    "ArenaTEMAdapterSettings",
    "ArenaTEMBridgeOutput",
    "ArenaDecoderConfig",
    "ArenaEncoderConfig",
]
