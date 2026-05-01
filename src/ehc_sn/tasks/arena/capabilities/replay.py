"""Arena replay capability object — v1 (topology-free, pure sequence slicer).

Canonical owner of the :class:`ArenaReplayCapability` that wires arena task
semantics into the replay trajectory controller.

Arena replay v1 contract:
- No topology is present in the batch.
- Revisit flags are precomputed at build time and sliced at runtime.
- ``valid_action_mask`` is not computed.
- ``location_id`` is not exposed.
- Initial task state is empty — no carry state is maintained.
- Halt logic uses ``trajectory_length`` only.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from ehc_sn.tasks.arena.contracts import ArenaTargets, ArenaTaskInput
from ehc_sn.types import Batch


# =============================================================================
class ArenaReplayCapability:
    """Arena-family replay capability satisfying the ReplayTrajectoryRuntime protocol.

    Pure sequence slicer: slices precomputed arrays at cursor ``t`` and emits
    typed step data.  No topology is required.  No carry state is maintained.
    """

    def trajectory_lengths(self, batch: Batch) -> Tensor:
        """Return per-slot trajectory lengths from ``batch["trajectory_length"]``."""
        return batch["trajectory_length"]

    def initial_task_state(self, batch: Batch, *, device: Any) -> dict[str, Tensor]:
        """Return empty initial task state — Arena replay v1 is stateless."""
        return {}

    def extract_step_per_slot(
        self,
        batch: Batch,
        cursor: Tensor,
        task_state: dict[str, Tensor],
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Extract current-step tensors at per-slot cursor positions.

        Slices precomputed trajectory arrays at per-slot cursor ``t``.  No
        topology is accessed; no carry state is updated.

        Returns:
            ``(step_data, new_task_state)`` where ``new_task_state`` is empty.
        """
        B = batch["trajectory_row"].shape[0]
        device = batch["trajectory_row"].device
        arange_b = torch.arange(B, device=device)
        t = cursor.to(device=device, dtype=torch.int64)

        observation_id = batch["trajectory_observation_id"][arange_b, t]  # (B,)
        previous_action = batch["trajectory_previous_action"][arange_b, t]  # (B,)
        landmark_id = batch["trajectory_landmark_id"][arange_b, t]  # (B,)
        episode_start = batch["trajectory_episode_start"][arange_b, t]  # (B,) bool
        is_revisit = batch["trajectory_is_revisit"][arange_b, t]  # (B,) bool

        result: dict[str, Tensor] = {
            "observation_id": observation_id.unsqueeze(-1),
            "previous_action": previous_action.unsqueeze(-1),
            "landmark_id": landmark_id.unsqueeze(-1),
            "step_count": t.to(dtype=torch.int32).unsqueeze(-1),
            "episode_start": episode_start,
            "is_revisit": is_revisit,
        }

        return result, {}


# =============================================================================
__all__ = [
    "ArenaReplayCapability",
]
