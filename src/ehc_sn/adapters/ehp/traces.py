"""TODO"""

from __future__ import annotations

from typing import Iterable, Mapping, Protocol

import torch
from torch import Tensor

from ehc_sn.tasks.arena.evaluation import coerce_observation_ids
from ehc_sn.tasks.mazehard.runtime import PATH_ID as _O_ID
from ehc_sn.traces import TraceField, TraceValue
from ehc_sn.types import Batch

TARGET_SOLUTION_OVERLAY_META_KEY = "target/solution_overlay"


# =============================================================================
# Minimal typed context for MazeHard+EHP actor-critic trace getters
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
    """Return out-of-band trace metadata required by MazeHard+EHP figures."""
    root_key, leaf_key = TARGET_SOLUTION_OVERLAY_META_KEY.split("/", maxsplit=1)
    return {
        root_key: {
            leaf_key: (batch["labels"] == _O_ID).to(torch.uint8),
        },
    }


# =============================================================================
# Named field objects
# =============================================================================

_MAZE_HARD_EHP_TRACE_SOLUTION_OVERLAY_ACTOR_CRITIC = TraceField(
    name="pred/solution_overlay",
    get=_get_maze_hard_solution_overlay_actor_critic,
)

MAZE_HARD_EHP_ACTOR_CRITIC_TRACE_FIELDS: tuple[TraceField, ...] = (
    _MAZE_HARD_EHP_TRACE_SOLUTION_OVERLAY_ACTOR_CRITIC,
)


# =============================================================================
class _ArenaEHCCarryData(Protocol):
    def __getitem__(self, key: str) -> Tensor: ...
    def get(self, key: str, default: Tensor | None = None) -> Tensor | None: ...


class _ArenaEHCCarry(Protocol):
    data: _ArenaEHCCarryData


class _ArenaEHCBackboneOutputs(Protocol):
    obs_logits: tuple[Tensor, Tensor, Tensor]


class _ArenaEHCOutputs(Protocol):
    backbone_output: _ArenaEHCBackboneOutputs


class _ArenaEHCTraceContext(Protocol):
    carry: _ArenaEHCCarry
    outputs: _ArenaEHCOutputs
    batch: Mapping[str, Tensor]


# =============================================================================
def _get_world_observation_id(ctx: _ArenaEHCTraceContext) -> TraceValue:
    obs_id: Tensor = ctx.batch["observation_id"]
    return coerce_observation_ids(obs_id).detach()


def _get_is_revisit(ctx: _ArenaEHCTraceContext) -> TraceValue:
    is_revisit: Tensor | None = ctx.batch.get("is_revisit")
    if is_revisit is None:
        return None
    return is_revisit.view(-1).bool().detach()


def _get_pred_obs_id_post(ctx: _ArenaEHCTraceContext) -> TraceValue:
    return ctx.outputs.backbone_output.obs_logits[0].detach().argmax(dim=-1)


def _get_pred_obs_id_recall(ctx: _ArenaEHCTraceContext) -> TraceValue:
    return ctx.outputs.backbone_output.obs_logits[1].detach().argmax(dim=-1)


def _get_pred_obs_id_path(ctx: _ArenaEHCTraceContext) -> TraceValue:
    return ctx.outputs.backbone_output.obs_logits[2].detach().argmax(dim=-1)


# =============================================================================
ARENA_EHP_TRACE_WORLD_OBS_ID = TraceField(
    name="world_step/observation_id",
    get=_get_world_observation_id,
)

ARENA_EHP_TRACE_IS_REVISIT = TraceField(
    name="protocol/is_revisit",
    get=_get_is_revisit,
)

ARENA_EHP_TRACE_PRED_POST = TraceField(
    name="pred/observation_id/post",
    get=_get_pred_obs_id_post,
)

ARENA_EHP_TRACE_PRED_RECALL = TraceField(
    name="pred/observation_id/recall",
    get=_get_pred_obs_id_recall,
)

ARENA_EHP_TRACE_PRED_PATH = TraceField(
    name="pred/observation_id/path",
    get=_get_pred_obs_id_path,
)

ARENA_EHP_TRACE_FIELDS: tuple[TraceField, ...] = (
    ARENA_EHP_TRACE_WORLD_OBS_ID,
    ARENA_EHP_TRACE_IS_REVISIT,
    ARENA_EHP_TRACE_PRED_POST,
    ARENA_EHP_TRACE_PRED_RECALL,
    ARENA_EHP_TRACE_PRED_PATH,
)
"""All Arena EHP trace fields in canonical order."""


# =============================================================================
def select_arena_ehc_trace_fields(
    include_keys: Iterable[str] | None,
) -> tuple[TraceField, ...]:
    """Return the Arena EHP trace fields matching a requested key set."""
    if include_keys is None:
        return ARENA_EHP_TRACE_FIELDS
    requested = set(include_keys)
    return tuple(f for f in ARENA_EHP_TRACE_FIELDS if f.name in requested)


# =============================================================================
__all__ = [
    "build_mazehard_ehc_trace_meta",
    "MAZE_HARD_EHP_ACTOR_CRITIC_TRACE_FIELDS",
    "TARGET_SOLUTION_OVERLAY_META_KEY",
    "ARENA_EHP_TRACE_FIELDS",
    "ARENA_EHP_TRACE_WORLD_OBS_ID",
    "ARENA_EHP_TRACE_IS_REVISIT",
    "ARENA_EHP_TRACE_PRED_POST",
    "ARENA_EHP_TRACE_PRED_RECALL",
    "ARENA_EHP_TRACE_PRED_PATH",
    "select_arena_ehc_trace_fields",
]
