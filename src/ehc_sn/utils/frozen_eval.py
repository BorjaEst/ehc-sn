"""Frozen-weight helpers for B2 and B3 benchmark evaluation."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


# =================================================================================================
class FrozenEvalMutationError(RuntimeError):
    """Raised when frozen-weight benchmark evaluation mutates model state."""

    def __init__(self, message: str, *, seed: int, mutated_key: str) -> None:
        super().__init__(message)
        self.seed = seed
        self.mutated_key = mutated_key


# =================================================================================================
def snapshot_state_dict(model: Any) -> dict[str, Tensor]:
    """Clone floating tensors from ``model.state_dict()`` for mutation checks."""
    snapshot: dict[str, Tensor] = {}
    for key, value in model.state_dict().items():
        if torch.is_floating_point(value):
            snapshot[key] = value.detach().clone().cpu()
    return snapshot


# =================================================================================================
def snapshot_optimizer_steps(optimizer: Any | None) -> tuple[int, ...] | None:
    """Capture optimizer step counters when an optimizer object is available."""
    if optimizer is None:
        return None
    steps: list[int] = []
    for state in optimizer.state.values():
        step_value = state.get("step") if isinstance(state, dict) else None
        if step_value is None:
            steps.append(0)
        elif isinstance(step_value, Tensor):
            steps.append(int(step_value.item()))
        else:
            steps.append(int(step_value))
    return tuple(steps)


# =================================================================================================
__all__ = [
    "FrozenEvalMutationError", "snapshot_optimizer_steps", "snapshot_state_dict",
]  # fmt: skip
