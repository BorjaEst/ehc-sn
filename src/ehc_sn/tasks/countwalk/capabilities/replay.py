"""Countwalk replay capability.

Satisfies the :class:`~ehc_sn.controllers.replay.trajectory.ReplayTrajectoryRuntime`
protocol for Countwalk task-corpus replay.  Wires Countwalk task-corpus tensors
into the replay trajectory controller.

Per-step extraction produces only task-safe payloads: ``previous_action``,
``cue_visible``, ``cue_surface_id``, ``cue_tokens``, ``cue_token_mask``,
``step_count``, ``episode_start``, and ``is_query``.

``trajectory_value`` is NOT extracted — the latent integer must not be exposed
to the model through the replay payload.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from ehc_sn.types import Batch

from ..builder import (
    CHANNEL_CUE_SURFACE_ID,
    CHANNEL_QUERY_MASK,
    CHANNEL_TRAJECTORY_ANCHOR_VISIBLE,
    CHANNEL_TRAJECTORY_CUE_MASK,
    CHANNEL_TRAJECTORY_CUE_TOKENS,
    CHANNEL_TRAJECTORY_EPISODE_START,
    CHANNEL_TRAJECTORY_LENGTH,
    CHANNEL_TRAJECTORY_PREVIOUS_ACTION,
)


# =============================================================================
class CountwalkReplayCapability:
    """Countwalk replay capability satisfying the ReplayTrajectoryRuntime protocol.

    Provides per-slot cursor-indexed step extraction from batch-major Countwalk
    replay tensors.  No mutable instance state is kept; all per-step carry is
    returned through the ``task_state`` dict.
    """

    def trajectory_lengths(self, batch: Batch) -> Tensor:
        """Return per-slot trajectory lengths from ``batch[trajectory_length]``."""
        return batch[CHANNEL_TRAJECTORY_LENGTH]

    def initial_task_state(self, batch: Batch, *, device: Any) -> dict[str, Tensor]:
        """Return an empty initial task state (Countwalk needs no carry state)."""
        return {}

    def extract_step_per_slot(
        self,
        batch: Batch,
        cursor: Tensor,
        task_state: dict[str, Tensor],
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Extract current-step tensors at per-slot cursor positions.

        Returns ``(step_data, new_task_state)``.  The new task state is always
        empty for Countwalk V1.

        Extracted keys:
        - ``previous_action``: ``(B, 1)`` int64
        - ``cue_visible``: ``(B,)`` bool
        - ``cue_surface_id``: ``(B, 1)`` int64 (episode-level, broadcast from scalar)
        - ``cue_tokens``: ``(B, W)`` int64
        - ``cue_token_mask``: ``(B, W)`` bool
        - ``step_count``: ``(B, 1)`` int32
        - ``episode_start``: ``(B,)`` bool
        - ``is_query``: ``(B,)`` bool

        NOT extracted (must not be exposed to model):
        - ``trajectory_value`` — latent integer
        - ``valid_action_mask`` — boundary structure
        """
        B = int(batch[CHANNEL_TRAJECTORY_PREVIOUS_ACTION].shape[0])
        device = cursor.device
        t = cursor.to(device=device, dtype=torch.int64)
        arange_b = torch.arange(B, device=device)

        prev_action = batch[CHANNEL_TRAJECTORY_PREVIOUS_ACTION][arange_b, t].long()
        anchor_visible = batch[CHANNEL_TRAJECTORY_ANCHOR_VISIBLE][arange_b, t].bool()
        cue_tokens = batch[CHANNEL_TRAJECTORY_CUE_TOKENS][arange_b, t].long()
        cue_token_mask = batch[CHANNEL_TRAJECTORY_CUE_MASK][arange_b, t].bool()
        episode_start = batch[CHANNEL_TRAJECTORY_EPISODE_START][arange_b, t].bool()
        is_query = batch[CHANNEL_QUERY_MASK][arange_b, t].bool()

        # cue_surface_id is episode-level (scalar per episode), broadcast to (B, 1)
        cue_surface_id = batch[CHANNEL_CUE_SURFACE_ID].long().unsqueeze(-1)

        step_data: dict[str, Tensor] = {
            "previous_action": prev_action.unsqueeze(-1),
            "cue_visible": anchor_visible,
            "cue_surface_id": cue_surface_id,
            "cue_tokens": cue_tokens,
            "cue_token_mask": cue_token_mask,
            "step_count": t.to(dtype=torch.int32).unsqueeze(-1),
            "episode_start": episode_start,
            "is_query": is_query,
        }

        return step_data, {}


# =============================================================================
__all__ = [
    "CountwalkReplayCapability",
]
