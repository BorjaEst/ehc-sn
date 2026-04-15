"""Navigation adapter heads over the task-agnostic EHC v3 backbone."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch import device as Device
from torch import dtype as Dtype
from torch import nn

from ehc_sn.modules.autoencoder import Autoencoder, AutoencoderSettings
from ehc_sn.types import Batch


# =============================================================================
class NavigationAdapter(nn.Module):
    """Project backbone latents into task-owned navigation action logits."""

    def __init__(  # ----------------------------------------------------------
        self,
        hidden_size: int,
        action_count: int,
    ) -> None:
        """Initialize the navigation adapter head with a simple MLP over the backbone summary."""
        super().__init__()
        self.task_action_head = nn.Linear(hidden_size, action_count)
        self.autoencoder = Autoencoder(observation_dim, feature_dim, autoencoder)

    def build_backbone_input(  # ----------------------------------------------
        self,
        batch: Batch,
    ):  # TODO:
        """Encode raw observations and pack the task-agnostic input."""
        return

    def decode_obs_logits(  # -------------------------------------------------
        self,
        # TODO: add arguments for the backbone output needed to decode obs logits
    ):  # TODO:
        """Decode raw observation logits from backbone place codes."""
        return

    def decode_from_place(  # ------------------------------------------------
        self,
        # TODO: add arguments for the backbone output needed to decode obs logits
    ) -> Tensor:
        """Decode one place-like code back into raw observation logits."""
        return

    def forward(  # -----------------------------------------------------------
        self,
        # TODO: add arguments for the backbone output needed to decode obs logits
    ):  # TODO: who defines the output?
        """Return task-owned navigation ...???."""
        return


# =============================================================================
class NavigationTEMV1BridgeAdapter(nn.Module):
    """ """


# =============================================================================
class NavigationTEMV2BridgeAdapter(nn.Module):
    """ """


# =============================================================================
__all__ = [
    "NavigationAdapter",
    "NavigationAdapterHead",
]
