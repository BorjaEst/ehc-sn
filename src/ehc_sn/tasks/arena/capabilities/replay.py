"""Arena replay capability object.

Canonical owner of the :class:`ArenaReplayCapability` that wires arena task
semantics into the replay trajectory controller.  This is an *execution binding*,
not a task-identity definition.  Arena semantics (score, evaluation, contracts)
live in the parent task package.

Batch schema constants and helpers live in :mod:`ehc_sn.tasks.arena.runtime`.
"""

from __future__ import annotations

from typing import Any, Final

import torch
import torch.nn.functional as F
from torch import Tensor

from ehc_sn.tasks._movement import _ACTION_DELTAS
from ehc_sn.tasks.arena.contracts import ArenaTargets, ArenaTaskInput
from ehc_sn.types import Batch


# =============================================================================
def _compute_valid_action_mask(
    topology: Tensor,
    row: Tensor,
    col: Tensor,
) -> Tensor:
    B, H, W = topology.shape
    device = topology.device
    arange_b = torch.arange(B, device=device)

    masks: list[Tensor] = []
    for dr, dc in _ACTION_DELTAS:
        new_r = row + dr
        new_c = col + dc
        in_bounds = (new_r >= 0) & (new_r < H) & (new_c >= 0) & (new_c < W)
        clamped_r = new_r.clamp(0, H - 1)
        clamped_c = new_c.clamp(0, W - 1)
        passable = topology[arange_b, clamped_r, clamped_c]
        masks.append(in_bounds & passable)

    return torch.stack(masks, dim=-1)


# =============================================================================
def _new_arena_visit_counts(
    batch: Batch,
    *,
    device: Any,
) -> Tensor:
    """Allocate per-slot location visit counters from the batch topology."""
    topology = batch["topology"]
    B = int(topology.shape[0])
    n_locations = int(topology.shape[-2] * topology.shape[-1])
    return torch.zeros((B, n_locations), dtype=torch.int32, device=device)


# =============================================================================
def _record_arena_visit(
    visit_counts: Tensor,
    location_id: Tensor,
) -> Tensor:
    """Return updated visit counters after recording visits at ``location_id``."""
    updated = visit_counts.clone()
    index = location_id.to(device=updated.device, dtype=torch.int64)
    if index.ndim == 1:
        index = index.unsqueeze(-1)
    increments = torch.ones_like(index, dtype=updated.dtype)
    updated.scatter_add_(1, index, increments)
    return updated


# =============================================================================
def _annotate_arena_revisit_state(
    payload: dict[str, Tensor],
    location_id: Tensor,
    visit_counts: Tensor,
) -> dict[str, Tensor]:
    """Attach a revisit annotation to a current-step payload dict."""
    idx = location_id.to(device=visit_counts.device, dtype=torch.int64)
    if idx.ndim == 1:
        idx = idx.unsqueeze(-1)
    prior_counts = visit_counts.gather(dim=1, index=idx).squeeze(-1)
    payload["is_revisit"] = prior_counts > 0
    return payload


# =============================================================================
class ArenaReplayCapability:
    """Arena-family replay capability satisfying the ReplayTrajectoryRuntime protocol.

    Provides per-slot cursor-indexed step extraction from batch-major arena
    replay tensors.  Visit counts are threaded explicitly through the
    controller's ``task_state`` carry — no mutable instance state is kept.
    """

    def __init__(self, observation_dim: int) -> None:
        self._obs_dim = observation_dim

    def trajectory_lengths(self, batch: Batch) -> Tensor:
        """Return per-slot trajectory lengths from ``batch["trajectory_length"]``."""
        return batch["trajectory_length"]

    def initial_task_state(self, batch: Batch, *, device: Any) -> dict[str, Tensor]:
        """Allocate zeroed visit counters for a fresh episode."""
        return {"visit_counts": _new_arena_visit_counts(batch, device=device)}

    def extract_step_per_slot(
        self,
        batch: Batch,
        cursor: Tensor,
        task_state: dict[str, Tensor],
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Extract current-step tensors at per-slot cursor positions.

        Returns ``(step_data, new_task_state)`` with updated visit counts.
        """
        topology = batch["topology"]  # (B, H, W) bool
        B, _H, W = topology.shape
        device = topology.device
        arange_b = torch.arange(B, device=device)
        t = cursor.to(device=device, dtype=torch.int64)

        row = batch["trajectory_row"][arange_b, t].long()  # (B,)
        col = batch["trajectory_col"][arange_b, t].long()  # (B,)
        prev_action = batch["trajectory_previous_action"][arange_b, t]  # (B,)
        episode_start = batch["trajectory_episode_start"][arange_b, t]  # (B,) bool

        location_id = row * W + col  # (B,)
        observation_id = batch["observations"][arange_b, row, col]  # (B,)
        valid_action_mask = _compute_valid_action_mask(topology, row, col)  # (B, A)
        observation = F.one_hot(observation_id.long(), num_classes=self._obs_dim).float()  # (B, obs_dim)

        result: dict[str, Tensor] = {
            "observation": observation,
            "observation_id": observation_id.unsqueeze(-1),
            "previous_action": prev_action.unsqueeze(-1),
            "location_id": location_id.unsqueeze(-1),
            "valid_action_mask": valid_action_mask,
            "step_count": t.to(dtype=torch.int32).unsqueeze(-1),
            "episode_start": episode_start,
        }

        if "regions" in batch:
            result["region_id"] = batch["regions"][arange_b, row, col].unsqueeze(-1)
        if "landmarks" in batch:
            result["landmark_id"] = batch["landmarks"][arange_b, row, col].unsqueeze(-1)

        visit_counts = task_state["visit_counts"]

        # Reset visit counts on episode boundaries.
        reset_mask = episode_start.view(-1)
        if reset_mask.any():
            fresh = torch.zeros_like(visit_counts)
            visit_counts = torch.where(reset_mask.unsqueeze(-1), fresh, visit_counts)

        result = _annotate_arena_revisit_state(result, location_id, visit_counts)
        new_visit_counts = _record_arena_visit(visit_counts, location_id)

        return result, {"visit_counts": new_visit_counts}


# =============================================================================
__all__ = [
    "ArenaReplayCapability",
]
