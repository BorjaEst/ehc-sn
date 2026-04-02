"""Narrow rollout capability contracts used by benchmark evaluators."""

from __future__ import annotations

from typing import Any, Protocol


class ResetState(Protocol):
    """Allocate a fresh benchmark-time state object."""

    def reset_state(self, *, batch_size: int) -> Any:
        """Return a fresh state object for the given batch size."""


class Acts(Protocol):
    """Advance a benchmark-time policy one step."""

    def act(self, obs: Any, state: Any) -> tuple[Any, Any]:
        """Return ``(action, next_state)`` for the current observation."""


class RolloutAgent(ResetState, Acts, Protocol):
    """Composite rollout capability used by navigation-style benchmarks."""


__all__ = ["Acts", "ResetState", "RolloutAgent"]
