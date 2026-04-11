"""Uniform random walk policy over valid actions."""

from __future__ import annotations

from typing import Literal

import torch
from pydantic import BaseModel, Field

from ehc_sn.policies._base import PolicyDecision, PolicyInput


# =================================================================================================
class RandomWalkPolicyConfig(BaseModel, extra="forbid"):
    """Configuration for uniform random walk policies."""

    kind: Literal["random_walk"] = Field(
        default="random_walk",
        description="Uniformly sample from currently legal actions.",
    )
    seed: int | None = Field(
        default=None,
        description=(
            "Optional RNG seed for deterministic policy sampling. The seed is applied when "
            "the policy instance is created; RNG state is not checkpointed."
        ),
    )


# =================================================================================================
class RandomWalkPolicy:
    """Uniformly sample one valid action per slot."""

    def __init__(  # ------------------------------------------------------------------------------
        self, *, seed: int | None = None,
    ) -> None:  # fmt: skip
        self._generator = torch.Generator(device="cpu")
        self._seed: int | None = None
        self.set_seed(seed)

    @property
    def seed(self) -> int | None:
        """Return the current RNG seed, if one has been set explicitly."""
        return self._seed

    def set_seed(self, seed: int | None) -> None:
        """Reset policy-local randomness to an explicit seed or clear it."""
        self._seed = None if seed is None else int(seed)
        if self._seed is not None:
            self._generator.manual_seed(self._seed)

    def __call__(  # ------------------------------------------------------------------------------
        self, policy_input: PolicyInput, *, explore: bool = True,
    ) -> PolicyDecision:  # fmt: skip
        """Sample one valid action per row from the legal-action mask."""
        if not explore and self._seed is None:
            raise ValueError("RandomWalkPolicy evaluation requires an explicit seed.")
        valid_action_mask = policy_input.valid_action_mask.to(torch.bool)
        if valid_action_mask.ndim != 2:
            raise ValueError("RandomWalkPolicy expects valid_action_mask with shape (B, A).")

        actions = []
        for row in valid_action_mask:
            valid = row.nonzero(as_tuple=False).flatten()
            if valid.numel() == 0:
                raise ValueError("RandomWalkPolicy requires at least one legal action per row.")
            index = torch.randint(0, int(valid.numel()), (1,), generator=self._generator, device="cpu")
            action = valid[index.to(valid.device)].to(torch.int64)
            actions.append(action)

        return PolicyDecision(action=torch.stack(actions, dim=0).view(-1, 1))


# =================================================================================================
__all__ = ["RandomWalkPolicyConfig", "RandomWalkPolicy"]
