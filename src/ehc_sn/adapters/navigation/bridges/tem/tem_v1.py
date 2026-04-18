"""Navigation plus TEM v1 bridge implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import torch
from pydantic import BaseModel, Field
from torch import Tensor
from torch import device as Device
from torch import dtype as Dtype
from torch import nn

from ehc_sn.adapters.navigation.decoders import NavigationDecoder
from ehc_sn.adapters.navigation.encoders import NavigationEncoder
from ehc_sn.models.tem.tem_v1 import TEMInputV1, TEMModelV1, TEMOutputV1, TEMStateV1
from ehc_sn.modules.autoencoder import MLPDecoder, TwoHotEncoder
from ehc_sn.tasks.navigation.contracts import NavigationTaskInput, NavigationTaskOutput
from ehc_sn.tasks.navigation.runtime import extract_navigation_task_input
from ehc_sn.types import Batch


# =============================================================================
class NavigationTEMV1AdapterSettings(BaseModel, extra="forbid"):
    """Task-side Navigation settings required to bind the TEM v1 core."""

    observation_dim: int = Field(
        ...,
        description="Dimensionality of the navigation task observations.",
    )


# =============================================================================
@dataclass(frozen=True)
class NavigationTEMV1ControlOutput:
    """ """


# =============================================================================
@dataclass(frozen=True)
class NavigationTEMV1BridgeOutput:
    """ """


# =============================================================================
class NavigationInputsEncoder(nn.Module, NavigationEncoder):
    """ """

    def __init__(  # ----------------------------------------------------------
        self,
        observation_dim: int,
        feature_dim: int,
        action_count: int,
        *,
        device: Device | None = None,
        dtype: Dtype | None = None,
    ) -> None:
        """Initialize the navigation decoder heads over model-facing features."""
        super().__init__()

        # The action logits are computed directly from the encoded features.
        self.task_action_head = nn.LazyLinear(action_count)
        self.encoder = TwoHotEncoder(observation_dim, feature_dim, device=device, dtype=dtype)

    def forward(  # -----------------------------------------------------------
        self,
        batch: NavigationTaskInput,
    ) -> TEMInputV1:
        """Encode the navigation task input into a TEM batch."""

        # Encode the observations into model-facing features, then compute action logits.
        features = self.encoder(batch.observations)
        action_logits = self.task_action_head(features)

        return TEMInputV1(
            features=features,
            action_logits=action_logits,
        )


# =============================================================================
class NavigationOutputsDecoder(nn.Module, NavigationDecoder):
    """ """

    def __init__(  # ----------------------------------------------------------
        self,
        observation_dim: int,
        latent_dim: int,
        *,
        device: Device | None = None,
        dtype: Dtype | None = None,
    ) -> None:
        """Initialize the navigation decoder heads over model-facing features."""
        super().__init__()

        # The predicted observations are decoded from the model-facing features via an MLP.
        self.decoder = MLPDecoder(latent_dim, observation_dim, device=device, dtype=dtype)

    def forward(  # -----------------------------------------------------------
        self,
        model_output: TEMOutputV1,
    ) -> NavigationTaskOutput:
        """Decode the TEM batch output into a navigation task output."""

        # Decode the predicted observations from the model-facing features.
        return NavigationTaskOutput(
            predicted_observations=self.decoder(model_output.features),
            action_logits=model_output.action_logits,
        )


# =============================================================================
class NavigationTEMV1BridgeAdapter(nn.Module):
    """Thin navigation plus TEM v1 binding that preserves the TEM rollout surface."""

    def __init__(  # ----------------------------------------------------------
        self,
        model: TEMModelV1,
        config: NavigationTEMV1AdapterSettings,
    ) -> None:
        """Initialize the navigation plus TEM v1 bridge adapter with its component modules."""
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
    ) -> TEMStateV1:
        """Create a fresh TEM recurrent state for one rollout batch."""
        return self.model.init_state(batch_size)

    def reset_state(  # --------------------------------------------------------
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
        """Prepare the model-facing core payload and task-side decoder context."""
        task_input = extract_navigation_task_input(batch)
        return self.encoder(task_input)

    def prepare_outputs(  # ---------------------------------------------------
        self,
        logits: TEMOutputV1,
    ) -> NavigationTaskOutput:
        """Decode one task-owned Navigation output from one TEM core output."""
        return NavigationTEMV1BridgeOutput(
            task=self.decoder(logits),
            control=NavigationTEMV1ControlOutput(...),
        )

    def forward(  # -----------------------------------------------------------
        self,
        batch: Batch,
        state: TEMStateV1 | None = None,
    ) -> tuple[TEMStateV1, NavigationTEMV1BridgeOutput]:
        """Run a forward pass of the TEM v1 bridge adapter on a Navigation task batch."""
        inputs = self.prepare_inputs(batch)
        next_state, outputs = self.model(inputs, state=state)
        outputs = self.prepare_outputs(outputs)
        return next_state, outputs


# =============================================================================
__all__ = [
    "NavigationInputsEncoder",
    "NavigationOutputsDecoder",
    "NavigationTEMV1BridgeAdapter",
]
