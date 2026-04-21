"""Arena replay runtime helpers.

These helpers operate on batch-major replay tensors (Phase 1 contract) and
derive current-step payloads from the compact trajectory arrays.  The runtime
does NOT call ``EnvBase.step()`` and does NOT own any recurrent carry.
Cursor management belongs to the replay controller (Phase 3).

Step ``t`` inputs depend only on replay batch tensors, the cursor position
``t``, and visit-count carry — no future observations leak through this layer
(SCI-001).
"""

from __future__ import annotations

from typing import Any, Final, Mapping

import torch
from torch import Tensor

from ehc_sn.tasks._movement import _ACTION_DELTAS
from ehc_sn.types import Batch

from .contracts import ArenaTargets, ArenaTaskInput

# =============================================================================
ARENA_REPLAY_REQUIRED_KEYS: Final[tuple[str, ...]] = (
    "topology",
    "observations",
    "mask_valid",
    "trajectory_row",
    "trajectory_col",
    "trajectory_previous_action",
    "trajectory_episode_start",
    "trajectory_valid_step",
    "trajectory_length",
)

ARENA_REPLAY_OPTIONAL_KEYS: Final[tuple[str, ...]] = (
    "regions",
    "landmarks",
    "start",
    "goals",
)

ARENA_STEP_KEYS: Final[tuple[str, ...]] = (
    "observation",
    "observation_id",
    "previous_action",
    "location_id",
    "region_id",
    "landmark_id",
    "valid_action_mask",
    "step_count",
    "episode_start",
    "is_revisit",
)


# =============================================================================
def extract_arena_step_tensors(
    batch: Batch,
    t: int,
) -> dict[str, Tensor]:
    """Extract current-step tensors from a batch-major arena replay batch at step ``t``.

    Reads trajectory arrays at step index ``t`` and derives ``location_id``
    and ``valid_action_mask`` from the static grid tensors.  Does NOT produce
    ``observation`` (one-hot / encoded); that is the adapter's responsibility.

    Args:
        batch: Arena replay batch containing all required trajectory and grid
            tensors as defined by the Phase 1 contract.
        t: Current replay step index (0-based, must be < T).

    Returns:
        Dict with ``observation_id``, ``previous_action``, ``location_id``,
        ``valid_action_mask``, ``step_count``, ``episode_start``, and
        optional ``region_id`` / ``landmark_id`` when present in ``batch``.
    """
    topology = batch["topology"]  # (B, H, W) bool
    observations = batch["observations"]  # (B, H, W) int64
    row = batch["trajectory_row"][:, t].long()  # (B,)
    col = batch["trajectory_col"][:, t].long()  # (B,)
    prev_action = batch["trajectory_previous_action"][:, t]  # (B,)
    episode_start = batch["trajectory_episode_start"][:, t]  # (B,) bool

    B, _H, W = topology.shape
    arange_b = torch.arange(B, device=topology.device)

    location_id = row * W + col  # (B,)
    observation_id = observations[arange_b, row, col]  # (B,)
    valid_action_mask = _compute_valid_action_mask(topology, row, col)  # (B, A)
    step_count = topology.new_full((B,), t, dtype=torch.int32)  # (B,)

    result: dict[str, Tensor] = {
        "observation_id": observation_id.unsqueeze(-1),
        "previous_action": prev_action.unsqueeze(-1),
        "location_id": location_id.unsqueeze(-1),
        "valid_action_mask": valid_action_mask,
        "step_count": step_count.unsqueeze(-1),
        "episode_start": episode_start,
    }

    if "regions" in batch:
        result["region_id"] = batch["regions"][arange_b, row, col].unsqueeze(-1)
    if "landmarks" in batch:
        result["landmark_id"] = batch["landmarks"][arange_b, row, col].unsqueeze(-1)

    return result


# =============================================================================
def _compute_valid_action_mask(
    topology: Tensor,
    row: Tensor,
    col: Tensor,
) -> Tensor:
    """Return a valid-action mask derived from the topology grid.

    Args:
        topology: Passable-cell grid, shape ``(B, H, W)`` bool.
        row: Current row per batch element, shape ``(B,)`` int64.
        col: Current column per batch element, shape ``(B,)`` int64.

    Returns:
        Boolean mask of shape ``(B, A)``.
    """
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
def coerce_arena_step_input(
    data: Mapping[str, Tensor],
) -> ArenaTaskInput:
    """Convert a step payload dict to a typed :class:`ArenaTaskInput`.

    Callers must enrich ``data`` with an ``observation`` key (adapter-encoded
    sensory vector) before calling this function.

    Args:
        data: Step payload dict, must contain all required arena step fields.

    Returns:
        Typed :class:`ArenaTaskInput`.

    Raises:
        KeyError: If any mandatory step field is absent.
    """
    required = ("observation", "observation_id", "previous_action", "location_id")
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"Arena step payload is missing required fields: {', '.join(missing)}.")
    return ArenaTaskInput(
        observation=data["observation"],
        observation_id=data["observation_id"],
        previous_action=data["previous_action"],
        location_id=data["location_id"],
        valid_action_mask=data.get(
            "valid_action_mask",
            data["observation"].new_zeros(data["observation"].shape[0]),
        ),
        step_count=data.get(
            "step_count",
            data["observation"].new_zeros(data["observation"].shape[0], 1),
        ),
        region_id=data.get("region_id"),
        landmark_id=data.get("landmark_id"),
        episode_start=data.get("episode_start"),
        is_revisit=data.get("is_revisit"),
    )


# =============================================================================
def coerce_arena_targets(
    data: Mapping[str, Tensor],
) -> ArenaTargets:
    """Extract :class:`ArenaTargets` from a carry data mapping.

    Args:
        data: Carry data dict, must contain ``observation_id``.

    Returns:
        :class:`ArenaTargets` with ``observation_id`` and optional
        ``is_revisit``.

    Raises:
        KeyError: If ``observation_id`` is absent.
    """
    if "observation_id" not in data:
        raise KeyError("Arena carry data must provide 'observation_id' to build ArenaTargets.")
    return ArenaTargets(
        observation_id=data["observation_id"],
        is_revisit=data.get("is_revisit"),
    )


# =============================================================================
def new_arena_visit_counts(
    batch: Batch,
    *,
    device: Any,
) -> Tensor:
    """Allocate per-slot location visit counters from the batch topology.

    Args:
        batch: Arena replay batch containing ``topology`` of shape
            ``(B, H, W)``.
        device: Target device for the counter tensor.

    Returns:
        Zero-initialised int32 tensor of shape ``(B, H * W)``.
    """
    topology = batch["topology"]
    B = int(topology.shape[0])
    n_locations = int(topology.shape[-2] * topology.shape[-1])
    return torch.zeros((B, n_locations), dtype=torch.int32, device=device)


# =============================================================================
def record_arena_visit(
    visit_counts: Tensor,
    location_id: Tensor,
) -> Tensor:
    """Return updated visit counters after recording visits at ``location_id``.

    Args:
        visit_counts: Per-slot visit counter of shape ``(B, N_locs)`` int32.
        location_id: Current-step locations, shape ``(B, 1)`` or ``(B,)``
            int64.

    Returns:
        Cloned and updated visit-count tensor.
    """
    updated = visit_counts.clone()
    index = location_id.to(device=updated.device, dtype=torch.int64)
    if index.ndim == 1:
        index = index.unsqueeze(-1)
    increments = torch.ones_like(index, dtype=updated.dtype)
    updated.scatter_add_(1, index, increments)
    return updated


# =============================================================================
def annotate_arena_revisit_state(
    payload: dict[str, Tensor],
    location_id: Tensor,
    visit_counts: Tensor,
) -> dict[str, Tensor]:
    """Attach a revisit annotation to a current-step payload dict.

    Args:
        payload: Current-step tensor dict; modified in-place.
        location_id: Current-step location ids, shape ``(B, 1)`` or ``(B,)``
            int64.
        visit_counts: Per-slot visit counter of shape ``(B, N_locs)`` int32.

    Returns:
        The input ``payload`` dict with ``is_revisit`` added.
    """
    idx = location_id.to(device=visit_counts.device, dtype=torch.int64)
    if idx.ndim == 1:
        idx = idx.unsqueeze(-1)
    prior_counts = visit_counts.gather(dim=1, index=idx).squeeze(-1)
    payload["is_revisit"] = prior_counts > 0
    return payload


# =============================================================================
def batch_size_from_arena_batch(batch: Batch) -> int:
    """Return the leading batch dimension from an arena replay batch."""
    return int(batch[ARENA_REPLAY_REQUIRED_KEYS[0]].shape[0])


# =============================================================================
def infer_arena_replay_batch_keys(batch: Batch) -> tuple[str, ...]:
    """Return the replay batch keys present in ``batch``, validating required keys.

    Raises:
        KeyError: If any required key is absent.
    """
    missing = [key for key in ARENA_REPLAY_REQUIRED_KEYS if key not in batch]
    if missing:
        raise KeyError("Arena replay batch is missing required keys: " + ", ".join(missing) + ".")
    return ARENA_REPLAY_REQUIRED_KEYS + tuple(k for k in ARENA_REPLAY_OPTIONAL_KEYS if k in batch)


# =============================================================================
class ArenaReplayTrajectoryRuntime:
    """Arena-family replay runtime satisfying the :class:`ReplayTrajectoryRuntime` protocol.

    Provides per-slot cursor-indexed step extraction from batch-major arena
    replay tensors.  Maintains per-slot visit counters internally so that
    ``is_revisit`` annotations are available to objectives and trace fields.

    Visit counters are reset for any batch slot whose extracted step has
    ``episode_start=True``, which is guaranteed to be True at cursor=0 of
    each fresh trajectory.

    This class is stateful across sequential ``extract_step_per_slot`` calls:
    visit counts accumulate within each episode and reset at episode boundaries.
    """

    def __init__(self) -> None:
        self._visit_counts: Tensor | None = None

    def trajectory_lengths(self, batch: Batch) -> Tensor:
        """Return per-slot trajectory lengths from ``batch["trajectory_length"]``."""
        return batch["trajectory_length"]

    def extract_step_per_slot(self, batch: Batch, cursor: Tensor) -> dict[str, Tensor]:
        """Extract current-step tensors at per-slot cursor positions.

        Does NOT produce ``observation`` (encoded sensory vector); that is
        the adapter encoder's responsibility (SCI-001 / adapter enrichment
        pattern).

        Args:
            batch: Full arena replay batch with ``(B, T, ...)`` trajectory axes.
            cursor: Per-slot current step indices, shape ``(B,)`` int64.

        Returns:
            Dict containing ``observation_id``, ``previous_action``,
            ``location_id``, ``valid_action_mask``, ``step_count``,
            ``episode_start``, optional ``region_id`` / ``landmark_id``,
            and ``is_revisit``.
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

        result: dict[str, Tensor] = {
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

        # Visit bookkeeping: allocate lazily, reset on episode boundaries.
        if self._visit_counts is None:
            self._visit_counts = new_arena_visit_counts(batch, device=device)

        reset_mask = episode_start.view(-1)  # (B,)
        if reset_mask.any():
            fresh = torch.zeros_like(self._visit_counts)
            self._visit_counts = torch.where(reset_mask.unsqueeze(-1), fresh, self._visit_counts)

        result = annotate_arena_revisit_state(result, location_id, self._visit_counts)
        self._visit_counts = record_arena_visit(self._visit_counts, location_id)

        return result


# =============================================================================
__all__ = [
    "ARENA_REPLAY_OPTIONAL_KEYS",
    "ARENA_REPLAY_REQUIRED_KEYS",
    "ARENA_STEP_KEYS",
    "annotate_arena_revisit_state",
    "batch_size_from_arena_batch",
    "coerce_arena_step_input",
    "coerce_arena_targets",
    "extract_arena_step_tensors",
    "infer_arena_replay_batch_keys",
    "new_arena_visit_counts",
    "record_arena_visit",
]
