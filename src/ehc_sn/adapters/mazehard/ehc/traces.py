"""MazeHard+EHC RL trace extras.

Owns trace field definitions and trace_meta helpers for the EHC family's
hybrid RL reason_pretrain path. Mirrors the HRM family pattern but reads
from the actor-critic record surface, not from an ACT backbone output.

Usage
-----
    from ehc_sn.adapters.mazehard.ehc.traces import (
        MAZE_HARD_EHC_ACTOR_CRITIC_TRACE_FIELDS,
        build_mazehard_ehc_trace_meta,
    )
    self.trace_spec = build_trace_spec("rl", extra_fields=MAZE_HARD_EHC_ACTOR_CRITIC_TRACE_FIELDS)
"""

from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor

from ehc_sn.tasks.mazehard.runtime import PATH_ID as _O_ID
from ehc_sn.traces import TraceField, TraceValue
from ehc_sn.types import Batch

TARGET_SOLUTION_OVERLAY_META_KEY = "target/solution_overlay"


# =============================================================================
# Minimal typed context for MazeHard+EHC actor-critic trace getters
# =============================================================================


class _MazeHardTaskLogits(Protocol):
    task_logits: Tensor


class _MazeHardEHCActorCriticTraceOutputs(Protocol):
    task_output: _MazeHardTaskLogits


class _MazeHardEHCActorCriticTraceContext(Protocol):
    outputs: _MazeHardEHCActorCriticTraceOutputs


# =============================================================================
# Getter functions
# =============================================================================


def _solution_overlay_from_task_logits(task_logits: Tensor) -> TraceValue:
    pred = torch.argmax(task_logits.detach(), dim=-1)  # (B, S)
    return (pred == _O_ID).to(torch.uint8)


def _get_maze_hard_solution_overlay_actor_critic(
    ctx: _MazeHardEHCActorCriticTraceContext,
) -> TraceValue:
    return _solution_overlay_from_task_logits(
        ctx.outputs.task_output.task_logits
    )


def build_mazehard_ehc_trace_meta(batch: Batch) -> dict[str, object]:
    """Return out-of-band trace metadata required by MazeHard+EHC figures."""
    root_key, leaf_key = TARGET_SOLUTION_OVERLAY_META_KEY.split("/", maxsplit=1)
    return {
        root_key: {
            leaf_key: (batch["labels"] == _O_ID).to(torch.uint8),
        },
    }


# =============================================================================
# Named field objects
# =============================================================================

_MAZE_HARD_EHC_TRACE_SOLUTION_OVERLAY_ACTOR_CRITIC = TraceField(
    name="pred/solution_overlay",
    get=_get_maze_hard_solution_overlay_actor_critic,
)

MAZE_HARD_EHC_ACTOR_CRITIC_TRACE_FIELDS: tuple[TraceField, ...] = (
    _MAZE_HARD_EHC_TRACE_SOLUTION_OVERLAY_ACTOR_CRITIC,
)


# =============================================================================
__all__ = [
    "build_mazehard_ehc_trace_meta",
    "MAZE_HARD_EHC_ACTOR_CRITIC_TRACE_FIELDS",
    "TARGET_SOLUTION_OVERLAY_META_KEY",
]
