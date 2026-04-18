"""Navigation plus TEM v2 bridge implementation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Optional

import torch
from pydantic import BaseModel, Field
from torch import Tensor, nn

from ehc_sn.models.tem.core.tem_base import GridCodes, PlaceCodes
from ehc_sn.models.tem.tem_v2 import TEMInputV2, TEMModelV2, TEMOutputV2, TEMStateV2
from ehc_sn.modules.autoencoder import MLPDecoder, TwoHotEncoder
from ehc_sn.tasks.navigation.contracts import NavigationTaskOutput
from ehc_sn.tasks.navigation.runtime import coerce_navigation_step_input
from ehc_sn.types import Batch
from ehc_sn.utils.detach import DetachMixin


# =============================================================================
class NavigationTEMV2AdapterSettings(BaseModel, extra="forbid"):
    """Task-side Navigation settings required to bind the TEM v2 core."""

    observation_dim: int = Field(
        ...,
        description="Dimensionality of the navigation task observations (must match environment.observation_dim).",
    )
    action_count: int = Field(
        ...,
        ge=1,
        description="Number of discrete actions (must match environment.action_count).",
    )
    decoder_kind: Literal["single_scale", "concat_all_scales"] = Field(
        default="single_scale",
        description="Decoder input policy. 'single_scale' uses one HPC frequency band (paper-fidelity); 'concat_all_scales' concatenates all bands.",
    )
    prediction_freq: int = Field(
        default=0,
        ge=0,
        description="HPC band index used by the single_scale decoder (paper-fidelity stream 1 / prediction_freq=0). Ignored when decoder_kind='concat_all_scales'.",
    )


# =============================================================================
@dataclass(frozen=True)
class NavigationTEMV2Diagnostics(DetachMixin):
    """TEM-family diagnostic surface for controller and objective consumption.

    Carries all three observation-logit pathways and the raw latent bundles
    needed by the TEM objective. Kept separate from the canonical task surface
    so the controller can remain model-agnostic.

    Attributes:
        obs_logits: Three observation-logit tensors ``(inference, retrieved, ancestral)``,
            each of shape ``(B, obs_dim)``.
        grid_codes: MEC grid codes ``(g_post, g_prior)`` for grid-transition latent loss.
        place_codes: Full named TEM place-code bundle for latent-relation assembly.
    """

    obs_logits: tuple[Tensor, Tensor, Tensor]
    grid_codes: GridCodes
    place_codes: PlaceCodes


# =============================================================================
@dataclass(frozen=True)
class NavigationTEMV2BridgeOutput(DetachMixin):
    """Split bridge output: canonical task surface plus TEM-family diagnostics.

    Attributes:
        task: Canonical :class:`~ehc_sn.tasks.navigation.contracts.NavigationTaskOutput`
            carrying the inference-pathway observation logits.
        tem: TEM-family diagnostic bundle consumed by the controller/objective.
    """

    task: NavigationTaskOutput
    tem: NavigationTEMV2Diagnostics


# =============================================================================
class NavigationInputsEncoder(nn.Module):
    """Encodes navigation step data into a :class:`TEMInputV2` payload."""

    def __init__(  # ----------------------------------------------------------
        self,
        observation_dim: int,
        feature_dim: int,
    ) -> None:
        super().__init__()
        self.encoder = TwoHotEncoder(observation_dim, feature_dim)

    def forward(  # -----------------------------------------------------------
        self,
        batch: Batch,
    ) -> TEMInputV2:
        """Encode pre-extracted navigation step data into a TEM v2 input payload."""
        task_input = coerce_navigation_step_input(batch)
        return TEMInputV2(
            obs_embedding=self.encoder(task_input.observation),
            previous_action=task_input.previous_action,
            episode_start=task_input.episode_start,
            landmark_id=task_input.landmark_id,
        )


# =============================================================================
class NavigationOutputsDecoder(nn.Module):
    """Decodes a :class:`TEMOutputV2` into a :class:`NavigationTEMV2BridgeOutput`.

    When ``single_freq`` is an int, uses only that HPC band (paper-fidelity
    single-scale decoder). When ``None``, concatenates all bands.
    """

    def __init__(  # ----------------------------------------------------------
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

    def forward(  # -----------------------------------------------------------
        self,
        model_output: TEMOutputV2,
    ) -> NavigationTEMV2BridgeOutput:
        """Decode all three place pathways and return the split task + TEM surfaces."""
        pc = model_output.place_codes
        gc = model_output.grid_codes

        obs_inference = self.decoder(_select_code(pc.inference, self._single_freq))
        obs_retrieved = self.decoder(_select_code(pc.retrieved, self._single_freq)) if pc.retrieved is not None else obs_inference.new_zeros(obs_inference.shape[0], self._obs_dim)  # fmt: skip
        obs_ancestral = self.decoder(_select_code(pc.ancestral, self._single_freq))
        ol = (obs_inference, obs_retrieved, obs_ancestral)

        task = NavigationTaskOutput(obs_logits=obs_inference)
        tem = NavigationTEMV2Diagnostics(obs_logits=ol, grid_codes=gc, place_codes=pc)
        return NavigationTEMV2BridgeOutput(task=task, tem=tem)


# =============================================================================
class NavigationTEMV2BridgeAdapter(nn.Module):
    """Navigation plus TEM v2 bridge adapter.

    Explicit task-to-model transformations follow the adapter-interface spec:
    :meth:`prepare_inputs` encodes a navigation step batch into a model-native
    payload; :meth:`prepare_outputs` decodes a model output into the split
    task + TEM surfaces.
    """

    def __init__(  # ----------------------------------------------------------
        self,
        model: TEMModelV2,
        config: NavigationTEMV2AdapterSettings,
    ) -> None:
        """Initialize the navigation plus TEM v2 bridge adapter."""
        super().__init__()
        self._config = config
        self.model = model
        self.encoder = _build_encoder(model, config)
        self.decoder = _build_decoder(model, config)

    @property
    def config(self) -> NavigationTEMV2AdapterSettings:
        """The navigation plus TEM v2 bridge adapter settings."""
        return self._config

    def init_state(  # --------------------------------------------------------
        self,
        batch_size: int,
        *,
        device: Optional[torch.device] = None,
    ) -> TEMStateV2:
        """Create a fresh TEM recurrent state for one rollout batch."""
        return self.model.init_state(batch_size, device=device)

    def reset_state(  # -------------------------------------------------------
        self,
        reset_flag: Tensor,
        state: TEMStateV2,
    ) -> TEMStateV2:
        """Reset halted rows of the TEM recurrent state."""
        return self.model.reset_state(reset_flag, state)

    def prepare_inputs(  # ----------------------------------------------------
        self,
        batch: dict[str, Tensor],
    ) -> TEMInputV2:
        """Encode a pre-extracted navigation step dict into a model-native :class:`TEMInputV2`."""
        return self.encoder(batch)

    def prepare_outputs(  # ---------------------------------------------------
        self,
        model_output: TEMOutputV2,
    ) -> NavigationTEMV2BridgeOutput:
        """Decode a TEM model output into the split task + TEM bridge surfaces."""
        return self.decoder(model_output)

    def forward(  # -----------------------------------------------------------
        self,
        batch: Batch,
        state: TEMStateV2 | None = None,
    ) -> tuple[TEMStateV2, NavigationTEMV2BridgeOutput]:
        """Run a forward pass of the TEM v2 bridge adapter on a Navigation task batch."""
        inputs = self.prepare_inputs(batch)
        model_output, next_state = self.model(inputs, state=state)
        bridge_output = self.prepare_outputs(model_output)
        return next_state, bridge_output


# =============================================================================
def _build_encoder(  # --------------------------------------------------------
    model: TEMModelV2,
    config: NavigationTEMV2AdapterSettings,
) -> NavigationInputsEncoder:
    """Construct the observation encoder front-end for the v2 bridge."""
    return NavigationInputsEncoder(
        observation_dim=config.observation_dim,
        feature_dim=model.config.lec.feature_dim,
    )


def _build_decoder(  # --------------------------------------------------------
    model: TEMModelV2,
    config: NavigationTEMV2AdapterSettings,
) -> NavigationOutputsDecoder:
    """Construct the observation decoder back-end for the v2 bridge.

    Policy is adapter-owned: ``single_scale`` uses one HPC band (paper-fidelity
    stream 1 / prediction_freq=0); ``concat_all_scales`` concatenates all bands.
    """
    hpc_shape = model.config.hpc.shape
    n_freq = len(hpc_shape)
    if config.decoder_kind == "single_scale":
        freq = config.prediction_freq
        if not (0 <= freq < n_freq):
            raise ValueError(f"prediction_freq={freq} is out of range for hpc.shape with {n_freq} bands (valid: 0..{n_freq - 1}).")
        return NavigationOutputsDecoder(
            observation_dim=config.observation_dim,
            latent_dim=hpc_shape[freq],
            single_freq=freq,
        )
    # concat_all_scales
    return NavigationOutputsDecoder(
        observation_dim=config.observation_dim,
        latent_dim=sum(hpc_shape),
    )


def _select_code(  # ----------------------------------------------------------
    code: Tensor | Sequence[Tensor],
    single_freq: int | None,
) -> Tensor:
    """Return a ``(B, D)`` tensor for decoder use.

    When ``single_freq`` is an int, returns the single band at that index
    (paper-fidelity stream 1 / prediction_freq=0 decoder). When ``None``,
    concatenates all bands.

    Raises ``ValueError`` if ``single_freq`` is set but ``code`` is already a
    flat ``Tensor``; band selection on a pre-concatenated tensor is ambiguous.
    """
    if isinstance(code, Tensor):
        if single_freq is not None:
            raise ValueError(
                "single_scale decoder received a pre-concatenated flat Tensor; "
                "expected a sequence of per-band tensors so that band "
                f"prediction_freq={single_freq} can be selected unambiguously."
            )
        return code
    if len(code) == 0:
        raise ValueError("TEM latent code sequences must not be empty.")
    if single_freq is not None:
        return code[single_freq]
    return torch.cat(tuple(code), dim=1)


# =============================================================================
__all__ = [
    "NavigationInputsEncoder",
    "NavigationOutputsDecoder",
    "NavigationTEMV2Diagnostics",
    "NavigationTEMV2AdapterSettings",
    "NavigationTEMV2BridgeAdapter",
    "NavigationTEMV2BridgeOutput",
]
