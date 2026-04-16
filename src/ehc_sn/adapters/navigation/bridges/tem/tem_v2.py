"""Navigation plus TEM v2 bridge implementation."""

from __future__ import annotations

import torch
from torch import Tensor
from torch import device as Device
from torch import dtype as Dtype
from torch import nn

from ehc_sn.adapters.navigation.decoders import NavigationDecoder
from ehc_sn.adapters.navigation.encoders import NavigationEncoder
from ehc_sn.models.tem.tem_v2 import TEMInputV2, TEMModelV2, TEMOutputV2, TEMStateV2
from ehc_sn.modules.autoencoder import MLPDecoder, TwoHotEncoder
from ehc_sn.tasks.navigation.contracts import NavigationTaskInput, NavigationTaskOutput


# =============================================================================
class NavigationInputsEncoder(nn.Module, NavigationEncoder):

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
    ) -> TEMInputV2:
        """Encode the navigation task input into a TEM batch."""

        # Encode the observations into model-facing features, then compute action logits.
        features = self.encoder(batch.observations)
        action_logits = self.task_action_head(features)

        return TEMInputV2(
            features=features,
            action_logits=action_logits,
        )


# =============================================================================
class NavigationOutputsDecoder(nn.Module, NavigationDecoder):

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
        model_output: TEMOutputV2,
    ) -> NavigationTaskOutput:
        """Decode the TEM batch output into a navigation task output."""

        # Decode the predicted observations from the model-facing features.
        return NavigationTaskOutput(
            predicted_observations=self.decoder(model_output.features),
            action_logits=model_output.action_logits,
        )


# =============================================================================
class NavigationTEMV2BridgeAdapter(nn.Module):
    """Thin navigation plus TEM v2 binding that preserves the TEM rollout surface."""

    def __init__(  # ----------------------------------------------------------
        self,
        model: TEMModelV2,
        encoder: NavigationEncoder,
        decoder: NavigationDecoder,
    ) -> None:
        """Initialize the navigation plus TEM v2 bridge adapter with its component modules."""
        super().__init__()
        self._model = model
        self.encoder = encoder
        self.decoder = decoder

    def prepare_inputs(  # ----------------------------------------------------
        self,
        batch: NavigationTaskInput,
    ) -> TEMInputV2:
        """Prepare the model-facing core payload and task-side decoder context."""
        return self.encoder(batch)

    def prepare_outputs(  # ---------------------------------------------------
        self,
        logits: TEMOutputV2,
    ) -> NavigationTaskOutput:
        """Decode one task-owned Navigation output from one TEM core output."""
        return self.decoder(logits)

    def forward(  # -----------------------------------------------------------
        self,
        batch: NavigationTaskInput,
        state: TEMStateV2 | None = None,
    ) -> tuple[TEMStateV2, NavigationTaskOutput]:
        """Run a forward pass of the TEM v2 bridge adapter on a Navigation task batch."""
        inputs = self.prepare_inputs(batch)
        next_state, logits = self.model(inputs, state=state)
        outputs = self.decode(logits)
        return next_state, outputs


# =============================================================================
__all__ = [
    "NavigationInputsEncoder",
    "NavigationOutputsDecoder",
    "NavigationTEMV2BridgeAdapter",
]
