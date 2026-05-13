"""Arena plus EHC v1 bridge implementation.

Binds model-native EHC v1 types to the shared Arena EHC family core.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from ehc_sn.adapters.arena.ehc import core as _core
from ehc_sn.adapters.arena.ehc.core import (
    ArenaDecoderConfig,
    ArenaEHCAdapterSettings,
    ArenaEHCBridgeOutput,
    ArenaEHCDiagnostics,
    ArenaEncoderConfig,
    ArenaTwoHotEncoder,
)
from ehc_sn.models.ehc.ehc_v1 import EHCInputV1, EHCModelV1, EHCOutputV1, EHCStateV1
from ehc_sn.modules.autoencoder import MLPDecoder
from ehc_sn.types import Batch


# =============================================================================
class ArenaInputsEncoderV1(nn.Module):
    """Encodes arena step data into a :class:`EHCInputV1` payload."""

    def __init__(
        self,
        observation_dim: int,
        feature_dim: int,
        n_freq: int,
    ) -> None:
        super().__init__()
        self._core = ArenaTwoHotEncoder(observation_dim, feature_dim, n_freq)

    def forward(
        self,
        batch: Batch,
    ) -> EHCInputV1:
        """Encode a pre-extracted arena step payload into a EHC v1 input."""
        observation_embedding, prev_action, episode_start, landmark_id = self._core.encode(batch)
        return EHCInputV1(
            observation_embedding=observation_embedding,
            previous_action=prev_action,
            episode_start=episode_start,
            landmark_id=landmark_id,
        )


# =============================================================================
class ArenaOutputsDecoderV1(nn.Module):
    """Decodes a :class:`EHCOutputV1` into an :class:`ArenaEHCBridgeOutput`."""

    def __init__(
        self,
        observation_dim: int,
        latent_dim: int,
        *,
        single_freq: int,
    ) -> None:
        super().__init__()
        self._obs_dim = observation_dim
        self._single_freq = single_freq
        self.w_x = torch.nn.Parameter(torch.tensor(1.0))
        self.b_x = torch.nn.Parameter(torch.zeros(latent_dim))
        self.decoder = MLPDecoder(latent_dim, observation_dim)

    def _decode(self, pred_code: list) -> Tensor:
        """Select the configured frequency band, apply affine, then MLP decode."""
        recon = self.w_x * pred_code[self._single_freq] + self.b_x
        return self.decoder(recon)

    def forward(self, model_output: EHCOutputV1) -> ArenaEHCBridgeOutput:
        """Decode all three pred_code pathways and return the split task + EHC surfaces."""
        pc = model_output.place_codes
        gc = model_output.grid_codes
        xc = model_output.pred_codes

        obs_inference = self._decode(xc.inference)
        obs_retrieved = self._decode(xc.retrieved) if xc.retrieved is not None else obs_inference.new_zeros(obs_inference.shape[0], self._obs_dim)  # fmt: skip
        obs_ancestral = self._decode(xc.ancestral)

        ol = (obs_inference, obs_retrieved, obs_ancestral)
        task = _core.ArenaTaskOutput(obs_logits=obs_inference)
        ehc = ArenaEHCDiagnostics(
            obs_logits=ol,
            grid_codes=gc,
            place_codes=pc,
            pred_codes=xc,
            theta_cls=model_output.control.theta_summary,
        )
        return ArenaEHCBridgeOutput(task=task, ehc=ehc)


# =============================================================================
class ArenaEHCV1BridgeAdapter(nn.Module):
    """Arena plus EHC v1 bridge adapter implementing RolloutBackbone."""

    def __init__(self, model: EHCModelV1, config: ArenaEHCAdapterSettings) -> None:
        super().__init__()
        self._config = config
        self.model = model
        self._encoder = _build_encoder_v1(model, config)
        self._decoder = _build_decoder_v1(model, config)

    @property
    def config(self) -> ArenaEHCAdapterSettings:
        return self._config

    def init_state(self, batch_size: int, *, device: Optional[torch.device] = None) -> EHCStateV1:
        return self.model.init_state(batch_size, device=device)

    def reset_state(self, reset_flag: Tensor, state: EHCStateV1) -> EHCStateV1:
        return self.model.reset_state(reset_flag, state)

    def prepare_inputs(self, batch: Batch) -> EHCInputV1:
        return self._encoder(batch)

    def postprocess(self, model_output: EHCOutputV1) -> ArenaEHCBridgeOutput:
        return self._decoder(model_output)

    def forward(
        self,
        batch: Batch,
        state: EHCStateV1 | None = None,
    ) -> tuple[ArenaEHCBridgeOutput, EHCStateV1]:
        inputs = self.prepare_inputs(batch)
        model_output, next_state = self.model(inputs, state=state)
        bridge_output = self.postprocess(model_output)
        return bridge_output, next_state


# =============================================================================
def _build_encoder_v1(model: EHCModelV1, config: ArenaEHCAdapterSettings) -> ArenaInputsEncoderV1:
    return ArenaInputsEncoderV1(
        observation_dim=config.observation_dim,
        feature_dim=model.config.lec.feature_dim,
        n_freq=model.lec.n_freq,
    )


def _build_decoder_v1(model: EHCModelV1, config: ArenaEHCAdapterSettings) -> ArenaOutputsDecoderV1:
    feature_dim = model.config.lec.feature_dim
    n_freq = len(model.config.hpc.shape)
    if config.decoder.kind == "single_scale":
        freq = config.decoder.prediction_freq
        if not (0 <= freq < n_freq):
            raise ValueError(f"prediction_freq={freq} is out of range for hpc.shape with {n_freq} bands " f"(valid: 0..{n_freq - 1}).")
        return ArenaOutputsDecoderV1(
            observation_dim=config.observation_dim,
            latent_dim=feature_dim,
            single_freq=freq,
        )
    raise NotImplementedError("Multi-scale decoding is not implemented for ArenaEHCV1BridgeAdapter.")


# =============================================================================
__all__ = [
    "ArenaEHCV1BridgeAdapter",
    "ArenaInputsEncoderV1",
    "ArenaOutputsDecoderV1",
    "ArenaEHCAdapterSettings",
    "ArenaEHCBridgeOutput",
    "ArenaDecoderConfig",
    "ArenaEncoderConfig",
]
