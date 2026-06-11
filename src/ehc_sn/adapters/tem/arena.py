"""Arena plus TEM v2 bridge implementation.

Binds model-native TEM v2 types to the shared Arena TEM family core.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from ehc_sn.adapters.tem import _base
from ehc_sn.adapters.tem._base import (
    ArenaDecoderConfig,
    ArenaEncoderConfig,
    ArenaTEMAdapterSettings,
    ArenaTEMBridgeOutput,
)
from ehc_sn.models.tem.tem_v1 import (
    TEMInputV1,
    TEMModelV1,
    TEMOutputV1,
    TEMStateV1,
)
from ehc_sn.models.tem.tem_v2 import (
    TEMInputV2,
    TEMModelV2,
    TEMOutputV2,
    TEMStateV2,
)
from ehc_sn.modules.autoencoder import MLPDecoder
from ehc_sn.types import Batch


# =============================================================================
class ArenaInputsEncoderV1(nn.Module):
    """Encodes arena step data into a :class:`TEMInputV1` payload."""

    def __init__(  # ----------------------------------------------------------
        self,
        observation_dim: int,
        feature_dim: int,
        n_freq: int,
    ) -> None:
        """Initializes the encoder with the given dimensions and frequency count."""
        super().__init__()
        self.encoder = _base.ArenaTwoHotEncoder(
            observation_dim, feature_dim, n_freq
        )

    def forward(  # -----------------------------------------------------------
        self,
        batch: Batch,
    ) -> TEMInputV1:
        """Encode a pre-extracted arena step payload into a TEM v1 input."""
        observation_embedding, prev_action, episode_start, landmark_id = (
            self.encoder(batch)
        )
        return TEMInputV1(
            observation_embedding=observation_embedding,
            previous_action=prev_action,
            episode_start=episode_start,
            landmark_id=landmark_id,
        )


# =============================================================================
class ArenaOutputsDecoderV1(nn.Module):
    """Decodes a :class:`TEMOutputV1` into an :class:`ArenaTEMBridgeOutput`."""

    def __init__(  # ----------------------------------------------------------
        self,
        observation_dim: int,
        latent_dim: int,
        *,
        single_freq: int | None = None,
    ) -> None:
        super().__init__()
        self._obs_dim = observation_dim
        self._single_freq = single_freq
        self.w_x = torch.nn.Parameter(
            torch.tensor(1.0)
        )  # For reconstructing c from x
        self.b_x = torch.nn.Parameter(
            torch.zeros(latent_dim)
        )  # Bias for reconstructing c from x
        self.decoder = MLPDecoder(latent_dim, observation_dim)

    def _decode(  # -----------------------------------------------------------
        self,
        pred_code: list,
    ) -> Tensor:
        """Decode a single prediction code into an observation reconstruction."""
        recon = self.w_x * pred_code[self._single_freq] + self.b_x
        return self.decoder(recon)

    def forward(  # -----------------------------------------------------------
        self,
        model_output: TEMOutputV1,
    ) -> ArenaTEMBridgeOutput:
        """Decode all three place pathways and return the split task + TEM surfaces."""
        obs_inference = self._decode(model_output.pred_codes.inference)
        obs_retrieved = (
            self._decode(model_output.pred_codes.retrieved)
            if model_output.pred_codes.retrieved is not None
            else obs_inference.new_zeros(obs_inference.shape[0], self._obs_dim)
        )
        obs_ancestral = self._decode(model_output.pred_codes.ancestral)

        task = _base.ArenaTaskOutput(obs_logits=obs_inference)
        tem = _base.ArenaTEMDiagnostics(
            obs_logits=(obs_inference, obs_retrieved, obs_ancestral),
            grid_codes=model_output.grid_codes,
            place_codes=model_output.place_codes,
            pred_codes=model_output.pred_codes,
        )
        return _base.ArenaTEMBridgeOutput(task=task, tem=tem)


# =============================================================================
class ArenaTEMV1BridgeAdapter(nn.Module):
    """Arena plus TEM v1 bridge adapter implementing RolloutBackbone."""

    def __init__(  # ----------------------------------------------------------
        self,
        model: TEMModelV1,
        config: ArenaTEMAdapterSettings,
    ) -> None:
        """Initializes the adapter with the given TEM v1 model and arena adapter settings."""
        super().__init__()
        self._config = config
        self.model = model
        self._encoder = _build_encoder_v1(model, config)
        self._decoder = _build_decoder_v1(model, config)

    @property
    def config(self) -> ArenaTEMAdapterSettings:
        """Returns the adapter's configuration settings."""
        return self._config

    def init_state(  # --------------------------------------------------------
        self,
        batch_size: int,
        *,
        device: Optional[torch.device] = None,
    ) -> TEMStateV1:
        """Initializes the TEM state for a new episode."""
        return self.model.init_state(batch_size, device=device)

    def reset_state(  # -------------------------------------------------------
        self,
        reset_flag: Tensor,
        state: TEMStateV1,
    ) -> TEMStateV1:
        """Resets the TEM state for episodes indicated by the reset_flag."""
        return self.model.reset_state(reset_flag, state)

    def prepare_inputs(  # ----------------------------------------------------
        self,
        batch: Batch,
    ) -> TEMInputV1:
        """Prepares the TEM v1 input from the arena step batch."""
        return self._encoder(batch)

    def postprocess(  # -------------------------------------------------------
        self,
        model_output: TEMOutputV1,
    ) -> ArenaTEMBridgeOutput:
        """Postprocesses the TEM v1 output into the arena bridge output."""
        return self._decoder(model_output)

    def forward(  # -----------------------------------------------------------
        self,
        batch: Batch,
        state: TEMStateV1 | None = None,
    ) -> tuple[ArenaTEMBridgeOutput, TEMStateV1]:
        inputs = self.prepare_inputs(batch)
        model_output, next_state = self.model(inputs, state=state)
        bridge_output = self.postprocess(model_output)
        return bridge_output, next_state


# =============================================================================
def _build_encoder_v1(  # -----------------------------------------------------
    model: TEMModelV1,
    config: ArenaTEMAdapterSettings,
) -> ArenaInputsEncoderV1:
    """Build the TEM-to-arena inputs encoder according to the requested config."""
    return ArenaInputsEncoderV1(
        observation_dim=config.observation_dim,
        feature_dim=model.config.lec.feature_dim,
        n_freq=model.lec.n_freq,
    )


# =============================================================================
def _build_decoder_v1(  # -----------------------------------------------------
    model: TEMModelV1,
    config: ArenaTEMAdapterSettings,
) -> ArenaOutputsDecoderV1:
    """Build the TEM-to-arena outputs decoder according to the requested config."""
    feature_dim = model.config.lec.feature_dim
    n_freq = len(model.config.hpc.shape)
    if config.decoder.kind == "single_scale":
        freq = config.decoder.prediction_freq
        if not (0 <= freq < n_freq):
            raise ValueError(
                f"prediction_freq={freq} is out of range for hpc.shape with {n_freq} bands "
                f"(valid: 0..{n_freq - 1})."
            )
        return ArenaOutputsDecoderV1(
            observation_dim=config.observation_dim,
            latent_dim=feature_dim,
            single_freq=freq,
        )
    if config.decoder.kind == "multi_scale":
        raise NotImplementedError(
            "Multi-scale decoding is not yet implemented for ArenaTEMV1BridgeAdapter."
        )
    raise ValueError(
        f"Unsupported decoder kind: {config.decoder.kind}",
    )


# =============================================================================
class ArenaInputsEncoderV2(nn.Module):
    """Encodes arena step data into a :class:`TEMInputV2` payload."""

    def __init__(  # ----------------------------------------------------------
        self,
        observation_dim: int,
        feature_dim: int,
        n_freq: int,
    ) -> None:
        super().__init__()
        self.encoder = _base.ArenaTwoHotEncoder(
            observation_dim, feature_dim, n_freq
        )

    def forward(  # -----------------------------------------------------------
        self,
        batch: Batch,
    ) -> TEMInputV2:
        """Encode a pre-extracted arena step payload into a TEM v2 input."""
        observation_embedding, prev_action, episode_start, landmark_id = (
            self.encoder(batch)
        )
        return TEMInputV2(
            observation_embedding=observation_embedding,
            previous_action=prev_action,
            episode_start=episode_start,
            landmark_id=landmark_id,
        )


# =============================================================================
class ArenaOutputsDecoderV2(nn.Module):
    """Decodes a :class:`TEMOutputV2` into an :class:`ArenaTEMBridgeOutput`."""

    def __init__(  # ----------------------------------------------------------
        self,
        observation_dim: int,
        latent_dim: int,
        *,
        single_freq: int | None = None,
    ) -> None:
        super().__init__()
        self._obs_dim = observation_dim
        self._single_freq = single_freq
        self.w_x = torch.nn.Parameter(torch.tensor(1.0))
        self.b_x = torch.nn.Parameter(torch.zeros(latent_dim))
        self.decoder = MLPDecoder(latent_dim, observation_dim)

    def _decode(  # -----------------------------------------------------------
        self,
        pred_code: list,
    ) -> Tensor:
        """Decode a single prediction code into an observation reconstruction."""
        recon = self.w_x * pred_code[self._single_freq] + self.b_x
        return self.decoder(recon)

    def forward(  # -----------------------------------------------------------
        self,
        model_output: TEMOutputV2,
    ) -> ArenaTEMBridgeOutput:
        """Decode all three place pathways and return the split task + TEM surfaces."""
        obs_inference = self._decode(model_output.pred_codes.inference)
        obs_retrieved = (
            self._decode(model_output.pred_codes.retrieved)
            if model_output.pred_codes.retrieved is not None
            else obs_inference.new_zeros(obs_inference.shape[0], self._obs_dim)
        )
        obs_ancestral = self._decode(model_output.pred_codes.ancestral)

        task = _base.ArenaTaskOutput(obs_logits=obs_inference)
        tem = _base.ArenaTEMDiagnostics(
            obs_logits=(obs_inference, obs_retrieved, obs_ancestral),
            grid_codes=model_output.grid_codes,
            place_codes=model_output.place_codes,
            pred_codes=model_output.pred_codes,
        )
        return _base.ArenaTEMBridgeOutput(task=task, tem=tem)


# =============================================================================
class ArenaTEMV2BridgeAdapter(nn.Module):
    """Arena plus TEM v2 bridge adapter implementing RolloutBackbone."""

    def __init__(  # ----------------------------------------------------------
        self,
        model: TEMModelV2,
        config: ArenaTEMAdapterSettings,
    ) -> None:
        super().__init__()
        self._config = config
        self.model = model
        self._encoder = _build_encoder_v2(model, config)
        self._decoder = _build_decoder_v2(model, config)

    @property
    def config(self) -> ArenaTEMAdapterSettings:
        return self._config

    def init_state(
        self, batch_size: int, *, device: Optional[torch.device] = None
    ) -> TEMStateV2:
        return self.model.init_state(batch_size, device=device)

    def reset_state(self, reset_flag: Tensor, state: TEMStateV2) -> TEMStateV2:
        return self.model.reset_state(reset_flag, state)

    def prepare_inputs(self, batch: Batch) -> TEMInputV2:
        return self._encoder(batch)

    def postprocess(self, model_output: TEMOutputV2) -> ArenaTEMBridgeOutput:
        return self._decoder(model_output)

    def forward(  # -----------------------------------------------------------
        self,
        batch: Batch,
        state: TEMStateV2 | None = None,
    ) -> tuple[ArenaTEMBridgeOutput, TEMStateV2]:
        inputs = self.prepare_inputs(batch)
        model_output, next_state = self.model(inputs, state=state)
        bridge_output = self.postprocess(model_output)
        return bridge_output, next_state


# =============================================================================
def _build_encoder_v2(  # -----------------------------------------------------
    model: TEMModelV2,
    config: ArenaTEMAdapterSettings,
) -> ArenaInputsEncoderV2:
    return ArenaInputsEncoderV2(
        observation_dim=config.observation_dim,
        feature_dim=model.config.lec.feature_dim,
        n_freq=model.lec.n_freq,
    )


# =============================================================================
def _build_decoder_v2(  # -----------------------------------------------------
    model: TEMModelV2,
    config: ArenaTEMAdapterSettings,
) -> ArenaOutputsDecoderV2:
    feature_dim = model.config.lec.feature_dim
    n_freq = len(model.config.hpc.shape)
    if config.decoder.kind == "single_scale":
        freq = config.decoder.prediction_freq
        if not (0 <= freq < n_freq):
            raise ValueError(
                f"prediction_freq={freq} is out of range for hpc.shape with {n_freq} bands "
                f"(valid: 0..{n_freq - 1})."
            )
        return ArenaOutputsDecoderV2(
            observation_dim=config.observation_dim,
            latent_dim=feature_dim,
            single_freq=freq,
        )
    if config.decoder.kind == "multi_scale":
        raise NotImplementedError(
            "multi_scale decoding is not yet implemented for TEM v2."
        )
    raise ValueError(f"Unsupported decoder kind: {config.decoder.kind}")


# =============================================================================
__all__ = [
    "ArenaTEMV1BridgeAdapter",
    "ArenaInputsEncoderV1",
    "ArenaOutputsDecoderV1",
    "ArenaTEMAdapterSettings",
    "ArenaTEMBridgeOutput",
    "ArenaDecoderConfig",
    "ArenaEncoderConfig",
    "ArenaTEMV2BridgeAdapter",
    "ArenaInputsEncoderV2",
    "ArenaOutputsDecoderV2",
    "ArenaTEMAdapterSettings",
    "ArenaTEMBridgeOutput",
    "ArenaDecoderConfig",
    "ArenaEncoderConfig",
]
