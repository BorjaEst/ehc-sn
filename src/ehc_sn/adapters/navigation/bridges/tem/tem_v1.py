"""Navigation plus TEM v1 bridge implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from pydantic import BaseModel, Field
from torch import Tensor, nn

from ehc_sn.models.tem.core.tem_base import GridCodes, TEMPlaceCodes
from ehc_sn.models.tem.tem_v1 import TEMInputV1, TEMModelV1, TEMOutputV1, TEMStateV1
from ehc_sn.modules.autoencoder import MLPDecoder, TwoHotEncoder
from ehc_sn.tasks.navigation.contracts import NavigationTaskOutput
from ehc_sn.tasks.navigation.runtime import coerce_navigation_step_input
from ehc_sn.types import Batch
from ehc_sn.utils.detach import DetachMixin


# =============================================================================
class NavigationTEMV1AdapterSettings(BaseModel, extra="forbid"):
    """Task-side Navigation settings required to bind the TEM v1 core."""

    observation_dim: int = Field(
        ...,
        description="Dimensionality of the navigation task observations (must match environment.observation_dim).",
    )
    action_count: int = Field(
        ...,
        ge=1,
        description="Number of discrete actions (must match environment.action_count).",
    )


# =============================================================================
@dataclass(frozen=True)
class NavigationTEMV1Diagnostics(DetachMixin):
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
    place_codes: TEMPlaceCodes


# =============================================================================
@dataclass(frozen=True)
class NavigationTEMV1BridgeOutput(DetachMixin):
    """Split bridge output: canonical task surface plus TEM-family diagnostics.

    Attributes:
        task: Canonical :class:`~ehc_sn.tasks.navigation.contracts.NavigationTaskOutput`
            carrying the inference-pathway observation logits.
        tem: TEM-family diagnostic bundle consumed by the controller/objective.
    """

    task: NavigationTaskOutput
    tem: NavigationTEMV1Diagnostics


# =============================================================================
class NavigationInputsEncoder(nn.Module):
    """Encodes navigation step data into a :class:`TEMInputV1` payload."""

    def __init__(  # ----------------------------------------------------------
        self,
        observation_dim: int,
        feature_dim: int,
    ) -> None:
        super().__init__()
        self.encoder = TwoHotEncoder(observation_dim, feature_dim)

    def forward(  # -----------------------------------------------------------
        self,
        batch: dict[str, Tensor],
    ) -> TEMInputV1:
        """Encode pre-extracted navigation step data into a TEM v1 input payload."""
        task_input = coerce_navigation_step_input(batch)
        return TEMInputV1(
            obs_embedding=self.encoder(task_input.observation),
            previous_action=task_input.previous_action,
            episode_start=task_input.episode_start,
            landmark_id=task_input.landmark_id,
        )


# =============================================================================
class NavigationOutputsDecoder(nn.Module):
    """Decodes a :class:`TEMOutputV1` into a :class:`NavigationTEMV1BridgeOutput`."""

    def __init__(  # ----------------------------------------------------------
        self,
        observation_dim: int,
        latent_dim: int,
    ) -> None:
        super().__init__()
        self.decoder = MLPDecoder(latent_dim, observation_dim)
        self._obs_dim = observation_dim

    def forward(  # -----------------------------------------------------------
        self,
        model_output: TEMOutputV1,
    ) -> NavigationTEMV1BridgeOutput:
        """Decode all three place pathways and return the split task + TEM surfaces."""
        gc = model_output.grid_codes
        pc = model_output.place_codes

        obs_inference = self.decoder(pc.inference)
        obs_retrieved = self.decoder(pc.retrieved) if pc.retrieved is not None else obs_inference.new_zeros(obs_inference.shape[0], self._obs_dim)  # fmt: skip
        obs_ancestral = self.decoder(pc.ancestral)
        ol = (obs_inference, obs_retrieved, obs_ancestral)

        task = NavigationTaskOutput(obs_logits=obs_inference)
        tem = NavigationTEMV1Diagnostics(obs_logits=ol, grid_codes=gc, place_codes=pc)
        return NavigationTEMV1BridgeOutput(task=task, tem=tem)


# =============================================================================
class NavigationTEMV1BridgeAdapter(nn.Module):
    """Navigation plus TEM v1 bridge adapter.

    Explicit task-to-model transformations follow the adapter-interface spec:
    :meth:`prepare_inputs` encodes a navigation step batch into a model-native
    payload; :meth:`prepare_outputs` decodes a model output into the split
    task + TEM surfaces.
    """

    def __init__(  # ----------------------------------------------------------
        self,
        model: TEMModelV1,
        config: NavigationTEMV1AdapterSettings,
    ) -> None:
        """Initialize the navigation plus TEM v1 bridge adapter."""
        super().__init__()
        self._config = config
        self.model = model
        self.encoder = _build_encoder(model, config)
        self.decoder = _build_decoder(model, config)

    @property
    def config(self) -> NavigationTEMV1AdapterSettings:
        """The navigation plus TEM v1 bridge adapter settings."""
        return self._config

    def init_state(  # --------------------------------------------------------
        self,
        batch_size: int,
        *,
        device: Optional[torch.device] = None,
    ) -> TEMStateV1:
        """Create a fresh TEM recurrent state for one rollout batch."""
        return self.model.init_state(batch_size, device=device)

    def reset_state(  # -------------------------------------------------------
        self,
        reset_flag: Tensor,
        state: TEMStateV1,
    ) -> TEMStateV1:
        """Reset halted rows of the TEM recurrent state."""
        return self.model.reset_state(reset_flag, state)

    def prepare_inputs(  # ----------------------------------------------------
        self,
        batch: Batch,
    ) -> TEMInputV1:
        """Encode a pre-extracted navigation step dict into a model-native :class:`TEMInputV1`."""
        return self.encoder(batch)

    def prepare_outputs(  # ---------------------------------------------------
        self,
        model_output: TEMOutputV1,
    ) -> NavigationTEMV1BridgeOutput:
        """Decode a TEM model output into the split task + TEM bridge surfaces."""
        return self.decoder(model_output)

    def forward(  # -----------------------------------------------------------
        self,
        batch: Batch,
        state: TEMStateV1 | None = None,
    ) -> tuple[TEMStateV1, NavigationTEMV1BridgeOutput]:
        """Run a forward pass of the TEM v1 bridge adapter on a Navigation task batch."""
        inputs = self.prepare_inputs(batch)
        model_output, next_state = self.model(inputs, state=state)
        bridge_output = self.prepare_outputs(model_output)
        return next_state, bridge_output


# =============================================================================
def _build_encoder(
    model: TEMModelV1,
    config: NavigationTEMV1AdapterSettings,
) -> NavigationInputsEncoder:
    """Construct the observation encoder front-end for the v1 bridge."""
    n_freq = len(model.config.hpc.shape)
    feature_dim = n_freq * model.config.lec.feature_dim
    return NavigationInputsEncoder(
        observation_dim=config.observation_dim,
        feature_dim=feature_dim,
    )


def _build_decoder(
    model: TEMModelV1,
    config: NavigationTEMV1AdapterSettings,
) -> NavigationOutputsDecoder:
    """Construct the observation decoder back-end for the v1 bridge."""
    latent_dim = sum(model.config.hpc.shape)
    return NavigationOutputsDecoder(
        observation_dim=config.observation_dim,
        latent_dim=latent_dim,
    )


# =============================================================================
__all__ = [
    "NavigationInputsEncoder",
    "NavigationOutputsDecoder",
    "NavigationTEMV1Diagnostics",
    "NavigationTEMV1AdapterSettings",
    "NavigationTEMV1BridgeAdapter",
    "NavigationTEMV1BridgeOutput",
]
