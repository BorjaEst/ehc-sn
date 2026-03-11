"""Deterministic stay / no-op action policy."""

from __future__ import annotations

import torch

from ehc_sn.policies._base import PolicyDecision, PolicyInput


# =================================================================================================
class StayPolicy:
    """Always emit the configured stay action."""

    def __init__(  # ------------------------------------------------------------------------------
        self, *, stay_action: int = 0,
    ) -> None:  # fmt: skip
        self._stay_action = int(stay_action)

    def __call__(  # ------------------------------------------------------------------------------
        self, policy_input: PolicyInput, *, explore: bool = True,
    ) -> PolicyDecision:  # fmt: skip
        """Return the stay action for every batch row."""
        _ = explore
        batch_shape = policy_input.valid_action_mask.shape[:-1]
        action = torch.full(
            (*batch_shape, 1),
            self._stay_action,
            dtype=torch.int64,
            device=policy_input.valid_action_mask.device,
        )
        return PolicyDecision(action=action)


# =================================================================================================
__all__ = ["StayPolicy"]
