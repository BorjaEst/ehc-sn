"""Reusable action-selection contracts.

Policies consume typed rollout-state views rather than controller internals.
The only required field today is a valid-action mask with shape ``(B, A)``,
but the contract leaves room for richer learned or scripted policies later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field
from torch import Tensor


# =================================================================================================
class ScriptedPolicyConfig(BaseModel, extra="forbid"):
    """Configuration for reusable scripted action policies."""

    kind: Literal["stay", "random_walk"] = Field(
        default="random_walk",
        description="Scripted policy used to select rollout actions.",
    )
    seed: int | None = Field(
        default=None,
        description=(
            "Optional RNG seed for deterministic policy sampling. The seed is applied when "
            "the policy instance is created; RNG state is not checkpointed."
        ),
    )
    stay_action: int = Field(
        default=0,
        ge=0,
        description="Action index treated as the environment no-op / stay action.",
    )


# =================================================================================================
@dataclass(frozen=True)
class PolicyInput:
    """Typed rollout-state view consumed by action policies.

    Attributes:
        valid_action_mask:
            Boolean mask of legal actions with shape ``(B, A)``.
        location_id:
            Optional location identifier with shape ``(B, 1)``.
        region_id:
            Optional region identifier with shape ``(B, 1)``.
        step_count:
            Optional per-slot step counter with shape ``(B, 1)``.
        logits:
            Optional policy logits for future learned-policy reuse.
        metadata:
            Optional rollout metadata for future policy variants.
    """

    valid_action_mask: Tensor
    location_id: Tensor | None = None
    region_id: Tensor | None = None
    step_count: Tensor | None = None
    logits: Tensor | None = None
    metadata: dict[str, Any] | None = None


# =================================================================================================
@dataclass(frozen=True)
class PolicyDecision:
    """Action-selection result returned by a policy.

    Attributes:
        action:
            Selected action tensor with shape ``(B, 1)`` or ``(B,)``.
        log_prob:
            Optional action log probability for learned policies.
        entropy:
            Optional policy entropy for learned policies.
        diagnostics:
            Optional detached policy-side diagnostics.
    """

    action: Tensor
    log_prob: Tensor | None = None
    entropy: Tensor | None = None
    diagnostics: dict[str, Any] | None = None


# =================================================================================================
class ActionPolicy(Protocol):
    """Public protocol for reusable action-selection policies."""

    def __call__(  # ------------------------------------------------------------------------------
        self, policy_input: PolicyInput, *, explore: bool = True,
    ) -> PolicyDecision:  # fmt: skip
        """Select an action from a typed rollout-state view."""


# =================================================================================================
__all__ = ["ActionPolicy", "PolicyDecision", "PolicyInput", "ScriptedPolicyConfig"]
