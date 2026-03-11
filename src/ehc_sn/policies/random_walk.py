"""Uniform random walk policy over valid actions."""

from __future__ import annotations

import torch

from ehc_sn.policies._base import PolicyDecision, PolicyInput


# =================================================================================================
class RandomWalkPolicy:
    """Uniformly sample one valid action per slot."""

    def __init__(  # ------------------------------------------------------------------------------
        self, *, stay_action: int = 0, seed: int | None = None,
    ) -> None:  # fmt: skip
        self._stay_action = int(stay_action)
        self._generator = torch.Generator(device="cpu")
        if seed is not None:
            self._generator.manual_seed(seed)

    def __call__(  # ------------------------------------------------------------------------------
        self, policy_input: PolicyInput, *, explore: bool = True,
    ) -> PolicyDecision:  # fmt: skip
        """Sample one valid action per row, falling back to stay when none are valid."""
        _ = explore
        valid_action_mask = policy_input.valid_action_mask.to(torch.bool)
        if valid_action_mask.ndim != 2:
            raise ValueError("RandomWalkPolicy expects valid_action_mask with shape (B, A).")

        actions = []
        for row in valid_action_mask:
            valid = row.nonzero(as_tuple=False).flatten()
            if valid.numel() == 0:
                action = torch.tensor(self._stay_action, dtype=torch.int64, device=row.device)
            else:
                index = torch.randint(0, int(valid.numel()), (1,), generator=self._generator, device="cpu")
                action = valid[index.to(valid.device)].to(torch.int64)
            actions.append(action)

        return PolicyDecision(action=torch.stack(actions, dim=0).view(-1, 1))


# =================================================================================================
__all__ = ["RandomWalkPolicy"]
