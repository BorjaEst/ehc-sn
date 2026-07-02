"""Shared online environment rollout helpers for environment-stepping controllers.

Contains the :class:`EnvRolloutEnvironment` protocol, :func:`initial_env_reset`,
and :func:`reset_halted_slots` — used by :mod:`ehp_sn.controllers.online.actor_critic`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import torch
from tensordict import TensorDict, TensorDictBase
from torch import Tensor

from ehc_sn.types import Batch


# =============================================================================
class EnvRolloutEnvironment(Protocol):
    """ """

    def reset(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Reset the environment and return the initial state TensorDict."""
        ...

    def step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Step the environment and return a TensorDict with a ``"next"`` key."""
        ...

    def reset_slots(
        self,
        reset_mask: Tensor,
        tensordict: TensorDictBase,
        state: TensorDictBase,
    ) -> TensorDictBase:
        """Reset only the slots indicated by *reset_mask* (``True`` = reset).

        Args:
            reset_mask: Boolean mask of shape ``(B,)``; ``True`` means reset.
            tensordict: TensorDict containing the new static reset tensors for
                halted slots (already merged with active-slot values).
            state: Current full-batch environment state.

        Returns:
            Updated environment state TensorDict.
        """
        ...

    def _set_seed(self, seed: int | None) -> None:
        """Seed the environment's internal random state."""
        ...


# =============================================================================
def initial_env_reset(
    batch: Batch,
    build_reset_td: Callable[[Batch], TensorDict],
    environment: EnvRolloutEnvironment,
) -> tuple[TensorDict, TensorDictBase]:
    """Perform the initial environment reset from a batch sample.

    Converts *batch* to a reset TensorDict via *build_reset_td*, then calls
    ``environment.reset()`` to obtain the initial environment state.

    Returns:
        ``(reset_td, env_td)`` — the static reset payload and the initial
        environment state TensorDict.
    """
    reset_td = build_reset_td(batch)
    env_td = environment.reset(reset_td)
    return reset_td, env_td


# =============================================================================
def reset_halted_slots(
    halted: Tensor,
    new_reset_td: TensorDict,
    old_static: dict[str, Tensor],
    env_td: TensorDictBase,
    visit_counts: Tensor,
    *,
    environment: EnvRolloutEnvironment,
) -> tuple[dict[str, Tensor], TensorDictBase, Tensor]:
    """Reset halted environment slots from fresh static data.

    Merges new static data for halted slots with existing data for active
    slots, calls ``environment.reset_slots``, and resets visit counts for
    the halted slots.  Owns all partial-reset env orchestration so that task
    runtimes do not need to import controller-layer types.

    Args:
        halted: Boolean mask of shape ``(B,)``; ``True`` for slots to reset.
        new_reset_td: TensorDict with fresh static tensors (from the next
            incoming batch row), covering all keys in *old_static*.
        old_static: Existing static data dict for currently active slots.
        env_td: Current full-batch environment TensorDict state.
        visit_counts: Per-slot visit counters of shape ``(B, N_locs)`` int32.
        environment: Environment supporting ``reset_slots``.

    Returns:
        ``(next_static, next_env_td, next_visit_counts)`` triple:

        - *next_static*: merged static dict; halted slots hold new values,
          active slots hold old values.
        - *next_env_td*: environment state after ``reset_slots``.
        - *next_visit_counts*: visit counters with halted-slot rows zeroed.

    Raises:
        KeyError: If the static-data key set has changed (schema drift).
    """
    new_keys = frozenset(new_reset_td.keys())
    old_keys = frozenset(old_static.keys())
    if new_keys != old_keys:
        added = sorted(new_keys - old_keys)
        removed = sorted(old_keys - new_keys)
        parts: list[str] = []
        if added:
            parts.append(f"unexpected new keys: {added}")
        if removed:
            parts.append(f"missing previously present keys: {removed}")
        raise KeyError(
            f"Env partial-reset schema drift detected — {'; '.join(parts)}."
        )

    if not torch.any(halted):
        return old_static, env_td, visit_counts

    # Merge: halted slots get new_reset_td values; active slots keep old_static.
    next_static = {
        key: torch.where(
            halted.view((-1,) + (1,) * (value.ndim - 1)), value, old_static[key]
        )
        for key, value in new_reset_td.items()
    }
    merged_td = TensorDict(
        next_static, batch_size=env_td.batch_size, device=env_td.device
    )
    next_env_td = environment.reset_slots(halted, merged_td, env_td)

    # Zero visit counters for halted slots.
    next_visit_counts = visit_counts.clone()
    next_visit_counts[halted] = 0

    return next_static, next_env_td, next_visit_counts


# =============================================================================
__all__ = ["EnvRolloutEnvironment", "initial_env_reset", "reset_halted_slots"]
