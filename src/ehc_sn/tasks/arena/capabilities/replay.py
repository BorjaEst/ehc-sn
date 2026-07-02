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

After the slot-authoritative refactor the ``extract_step_per_slot`` method
receives *resident* — the carry-owned trajectory arrays for the admitted row,
not the raw source batch.  The slicing logic is identical; only the data
source changes.
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

    def initial_task_state(
        self, batch: Batch, *, device: Any
    ) -> dict[str, Tensor]:
        """Return empty initial task state — Arena replay v1 is stateless."""
        return {}

    def extract_step_per_slot(
        self,
        resident: Batch,
        cursor: Tensor,
        task_state: dict[str, Tensor],
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Extract current-step tensors at per-slot cursor positions.

        Slices precomputed trajectory arrays at per-slot cursor ``t``.  No
        topology is accessed; no carry state is updated.

        Args:
            resident: Carry-owned trajectory arrays for the admitted row (NOT the
                raw source batch).  Contains the same trajectory_* columns as the
                original batch but is indexed by the admitted slot, not the latest
                incoming batch row.
            cursor: Per-slot step cursor ``(B,)`` int64.
            task_state: Unused; Arena replay v1 is stateless.

        Returns:
            ``(step_data, new_task_state)`` where ``new_task_state`` is empty.
        """
        B = resident["trajectory_row"].shape[0]
        device = resident["trajectory_row"].device
        arange_b = torch.arange(B, device=device)
        t = cursor.to(device=device, dtype=torch.int64)

        observation_id = resident["trajectory_observation_id"][
            arange_b, t
        ]  # (B,)
        previous_action = resident["trajectory_previous_action"][
            arange_b, t
        ]  # (B,)
        landmark_id = resident["trajectory_landmark_id"][arange_b, t]  # (B,)
        episode_start = resident["trajectory_episode_start"][
            arange_b, t
        ]  # (B,) bool
        is_revisit = resident["trajectory_is_revisit"][arange_b, t]  # (B,) bool

        # Derive a spatial location_id.  When trajectory row/col are available
        # compute row-major index; otherwise fall back to observation_id as a
        # spatial proxy (each observation maps to exactly one location).
        traj_row = resident.get("trajectory_row")
        traj_col = resident.get("trajectory_col")
        if traj_row is not None and traj_col is not None:
            row = traj_row[arange_b, t].to(dtype=torch.int64)
            col = traj_col[arange_b, t].to(dtype=torch.int64)
            max_col = traj_col.max().item() + 1
            location_id = row * max_col + col
        else:
            location_id = observation_id.to(dtype=torch.int64)

        result: dict[str, Tensor] = {
            "observation_id": observation_id.unsqueeze(-1),
            "location_id": location_id.unsqueeze(-1),
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
