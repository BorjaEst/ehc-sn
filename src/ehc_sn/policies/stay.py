"""Deterministic stay / no-op action policy."""

from __future__ import annotations

from typing import Literal

import torch
from pydantic import BaseModel, Field

from ehc_sn.policies._base import PolicyDecision, PolicyInput


# =================================================================================================
class StayPolicyConfig(BaseModel, extra="forbid"):
    """Configuration for deterministic stay / no-op policies."""

    kind: Literal["stay"] = Field(
        default="stay",
        description="Select the environment no-op action.",
    )
    stay_action: int = Field(
        default=0,
        ge=0,
        description="Action index emitted by this policy. Defaults to 0 (STAY in canonical navigation ontology).",
    )


# =================================================================================================
class StayPolicy:
    """Always emit the configured stay action."""

    def __init__(  # ------------------------------------------------------------------------------
        self, *, action: int,
    ) -> None:  # fmt: skip
        self._action = int(action)

    def __call__(  # ------------------------------------------------------------------------------
        self, policy_input: PolicyInput, *, explore: bool = True,
    ) -> PolicyDecision:  # fmt: skip
        """Return the stay action for every batch row."""
        _ = explore
        batch_shape = policy_input.valid_action_mask.shape[:-1]
        action = torch.full(
            (*batch_shape, 1),
            self._action,
            dtype=torch.int64,
            device=policy_input.valid_action_mask.device,
        )
        return PolicyDecision(action=action)


# =================================================================================================
__all__ = ["StayPolicyConfig", "StayPolicy"]
