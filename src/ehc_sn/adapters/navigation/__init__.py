"""Navigation adapter heads over the task-agnostic EHC v3 backbone."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch import device as Device
from torch import dtype as Dtype
from torch import nn

from ehc_sn.modules.autoencoder import Autoencoder, AutoencoderSettings
from ehc_sn.tasks.navigation import NavigationTaskOutput
from ehc_sn.types import Batch


# =============================================================================
class NavigationAdapter(nn.Module):
    """Project backbone latents into task-owned navigation action logits."""

    def __init__(  # ----------------------------------------------------------
        self,
        observation_dim: int,
        feature_dim: int,
        action_count: int,
        autoencoder_settings: AutoencoderSettings,
    ) -> None:
        """Initialize the navigation adapter head with a simple MLP over the backbone summary."""
        super().__init__()
        self.task_action_head = nn.LazyLinear(action_count)
        self.autoencoder = Autoencoder(observation_dim, feature_dim, autoencoder_settings)

    def forward(  # -----------------------------------------------------------
        self,
        input: NavigationTaskInput,
        # TODO: add arguments for the backbone output needed to decode obs logits
    ) -> NavigationTaskOutput:
        """Return task-owned navigation outputs decoded from one backbone step."""
        raise NotImplementedError("NavigationAdapter.forward is signature-only for now.")


# =============================================================================
class NavigationTEMV1BridgeAdapter(nn.Module):
    """ """


# =============================================================================
class NavigationTEMV2BridgeAdapter(nn.Module):
    """ """


# =============================================================================
__all__ = [
    "NavigationAdapter",
    "NavigationTEMV1BridgeAdapter",
    "NavigationTEMV2BridgeAdapter",
]
