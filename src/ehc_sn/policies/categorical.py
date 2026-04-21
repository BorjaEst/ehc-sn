"""Categorical action policy over logits and valid-action masks."""

from __future__ import annotations

import torch
from pydantic import BaseModel, Field
from torch import Tensor
from torch.distributions import Categorical

from ehc_sn.policies._base import PolicyDecision, PolicyInput


# =================================================================================================
class CategoricalPolicyConfig(BaseModel, extra="forbid"):
    """Configuration for categorical learned action selection."""

    exploration_prob: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Probability of replacing the sampled action with a uniform valid action during exploration.",
    )
    fallback_action: int = Field(
        default=0,
        ge=0,
        description="Action index used if a row has no valid actions.",
    )
    seed: int | None = Field(
        default=None,
        description="Optional RNG seed for deterministic policy sampling.",
    )


# =================================================================================================
class CategoricalPolicy:
    """Sample actions from logits while respecting an explicit valid-action mask."""

    def __init__(  # ------------------------------------------------------------------------------
        self, config: CategoricalPolicyConfig | None = None,
    ) -> None:  # fmt: skip
        self._config = config or CategoricalPolicyConfig()
        self._generator = torch.Generator(device="cpu")
        if self._config.seed is not None:
            self._generator.manual_seed(self._config.seed)

    def __call__(  # ------------------------------------------------------------------------------
        self, policy_input: PolicyInput, *, explore: bool = True,
    ) -> PolicyDecision:  # fmt: skip
        """Sample an action from logits with optional uniform-valid exploration override."""
        if policy_input.logits is None:
            raise ValueError("CategoricalPolicy requires policy_input.logits.")

        valid_action_mask = policy_input.valid_action_mask.to(torch.bool)
        logits = policy_input.logits.detach()
        if logits.ndim != 2:
            raise ValueError("CategoricalPolicy expects logits with shape (B, A).")
        if valid_action_mask.shape != logits.shape:
            raise ValueError("CategoricalPolicy expects valid_action_mask to match logits shape.")

        masked_logits = logits.masked_fill(~valid_action_mask, torch.finfo(logits.dtype).min)
        empty_rows = ~valid_action_mask.any(dim=-1)
        if torch.any(empty_rows):
            masked_logits = masked_logits.clone()
            masked_logits[empty_rows, self._config.fallback_action] = 0.0

        dist = Categorical(logits=masked_logits)
        action = dist.sample()
        if self._config.exploration_prob is not None and explore:
            explore_flag = torch.rand(action.shape, generator=self._generator, device="cpu")
            explore_flag = explore_flag.to(action.device) < self._config.exploration_prob
            random_action = self._sample_uniform_valid(valid_action_mask)
            action = torch.where(explore_flag, random_action, action)

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return PolicyDecision(action=action, log_prob=log_prob, entropy=entropy)

    def _sample_uniform_valid(  # -----------------------------------------------------------------
        self, valid_action_mask: Tensor,
    ) -> Tensor:  # fmt: skip
        """Sample one uniformly valid action per row."""
        actions = []
        for row in valid_action_mask:
            valid = row.nonzero(as_tuple=False).flatten()
            if valid.numel() == 0:
                action = torch.tensor(self._config.fallback_action, dtype=torch.int64, device=row.device)
            else:
                index = torch.randint(0, int(valid.numel()), (1,), generator=self._generator, device="cpu")
                action = valid[index.to(valid.device)].to(torch.int64).squeeze(0)
            actions.append(action)

        return torch.stack(actions, dim=0)


# =================================================================================================
__all__ = ["CategoricalPolicy", "CategoricalPolicyConfig", "PolicyInput", "PolicyDecision"]
