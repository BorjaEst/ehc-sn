"""Dungeon task runtime helpers for online env-facing control.

These helpers support the live DungeonWalk-based execution pattern: building
reset TensorDicts, extracting current-step payloads, and managing visit
counters.

The partial-reset slot helper (``refresh_halted_slots``) is NOT included here.
It requires controller-layer integration (``environment.reset_slots``), which
creates an upward dependency from tasks/ to controllers/.  That integration
lives in ``controllers/_env_rollout.reset_halted_slots``; callers that need
partial-reset should use ``DungeonControllerRuntime``.
"""

from __future__ import annotations

from typing import Any, Final, Mapping

import torch
from tensordict import TensorDict, TensorDictBase
from torch import Tensor

from ehc_sn.policies import PolicyInput
from ehc_sn.types import Batch

from .contracts import DungeonTaskInput

# =============================================================================
DUNGEON_RESET_REQUIRED_KEYS: Final[tuple[str, ...]] = (
    "topology",
    "observations",
    "mask_valid",
)
DUNGEON_RESET_OPTIONAL_KEYS: Final[tuple[str, ...]] = (
    "regions",
    "start",
    "goals",
    "landmarks",
)
DUNGEON_STEP_KEYS: Final[tuple[str, ...]] = (
    "observation",
    "observation_id",
    "previous_action",
    "location_id",
    "region_id",
    "landmark_id",
    "valid_action_mask",
    "step_count",
)


# =============================================================================
def build_dungeon_reset_td(batch: Batch) -> TensorDict:
    """Return the static maze tensors required by the dungeon environment reset.

    Args:
        batch: Batch mapping containing at least the required reset keys.

    Returns:
        :class:`TensorDict` with static maze tensors, ready for env reset.

    Raises:
        KeyError: If any required reset key is absent.
    """
    keys = infer_dungeon_static_batch_keys(batch)
    reset_data = {key: batch[key] for key in keys}
    batch_size = int(next(iter(reset_data.values())).shape[0])
    device = next(iter(reset_data.values())).device
    return TensorDict(reset_data, batch_size=[batch_size], device=device)


# =============================================================================
def extract_dungeon_step_data(
    env_td: TensorDictBase,
    *,
    trace_metadata: Mapping[str, Tensor] | None = None,
) -> dict[str, Tensor]:
    """Extract the current-step task payload from a dungeon environment state.

    Args:
        env_td: TorchRL environment TensorDict produced after a step.
        trace_metadata: Optional extra tensors to merge into the payload.

    Returns:
        Dict with the current-step dungeon tensors, plus ``episode_start``
        derived from ``step_count`` when present.
    """
    payload = {key: env_td[key] for key in DUNGEON_STEP_KEYS if key in env_td.keys()}
    if "step_count" in payload:
        payload["episode_start"] = payload["step_count"].squeeze(-1).to(torch.int32) == 0
    if trace_metadata is not None:
        payload.update(trace_metadata)
    return payload


# =============================================================================
def coerce_dungeon_step_input(
    data: Mapping[str, Tensor],
) -> DungeonTaskInput:
    """Convert a step payload dict to a typed :class:`DungeonTaskInput`.

    Args:
        data: Step payload dict, must contain all required dungeon step fields.

    Returns:
        Typed :class:`DungeonTaskInput`.

    Raises:
        KeyError: If any mandatory step field is absent.
    """
    required = ("observation", "observation_id", "previous_action", "location_id")
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"Dungeon step payload is missing required fields: {', '.join(missing)}.")
    return DungeonTaskInput(
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
def make_dungeon_policy_input(env_td: TensorDictBase) -> PolicyInput:
    """Build a :class:`PolicyInput` from a dungeon environment state."""
    return PolicyInput(
        valid_action_mask=env_td["valid_action_mask"],
        location_id=env_td["location_id"] if "location_id" in env_td.keys() else None,
        region_id=env_td["region_id"] if "region_id" in env_td.keys() else None,
        step_count=env_td["step_count"] if "step_count" in env_td.keys() else None,
    )


# =============================================================================
def new_dungeon_visit_counts(
    reset_td: TensorDictBase,
    *,
    device: Any,
) -> Tensor:
    """Allocate per-slot location visit counters from a dungeon reset TensorDict.

    Args:
        reset_td: TensorDict containing ``topology`` of shape ``(B, H, W)``.
        device: Target device for the counter tensor.

    Returns:
        Zero-initialised int32 tensor of shape ``(B, H * W)``.
    """
    topology = reset_td["topology"]
    batch_size = int(topology.shape[0])
    n_locations = int(topology.shape[-2] * topology.shape[-1])
    return torch.zeros((batch_size, n_locations), dtype=torch.int32, device=device)


# =============================================================================
def record_dungeon_visit(
    visit_counts: Tensor,
    location_id: Tensor,
) -> Tensor:
    """Return updated visit counters after recording visits at ``location_id``.

    Args:
        visit_counts: Per-slot visit counter of shape ``(B, N_locs)`` int32.
        location_id: Current-step locations, shape ``(B, 1)`` int64.

    Returns:
        Cloned and updated visit-count tensor.
    """
    updated = visit_counts.clone()
    index = location_id.to(device=updated.device, dtype=torch.int64)
    increments = torch.ones_like(index, dtype=updated.dtype)
    updated.scatter_add_(1, index, increments)
    return updated


# =============================================================================
def annotate_dungeon_revisit_state(
    payload: dict[str, Tensor],
    env_td: TensorDictBase,
    visit_counts: Tensor,
) -> dict[str, Tensor]:
    """Attach revisit annotation to a current-step dungeon payload.

    Args:
        payload: Current-step tensor dict; modified in-place.
        env_td: Environment TensorDict providing ``location_id``.
        visit_counts: Per-slot visit counter of shape ``(B, N_locs)`` int32.

    Returns:
        The input ``payload`` dict with ``is_revisit`` added.
    """
    location_id = env_td["location_id"].to(device=visit_counts.device, dtype=torch.int64)
    prior_counts = visit_counts.gather(dim=1, index=location_id).squeeze(-1)
    payload["is_revisit"] = prior_counts > 0
    return payload


# =============================================================================
def batch_size_from_dungeon_batch(batch: Batch) -> int:
    """Return the leading batch dimension from a dungeon static maze batch."""
    return int(batch[DUNGEON_RESET_REQUIRED_KEYS[0]].shape[0])


# =============================================================================
def infer_dungeon_static_batch_keys(batch: Batch) -> tuple[str, ...]:
    """Return the static dungeon batch keys present in ``batch``.

    Raises:
        KeyError: If any required reset key is absent.
    """
    missing = [key for key in DUNGEON_RESET_REQUIRED_KEYS if key not in batch]
    if missing:
        raise KeyError("Dungeon batch is missing required reset keys: " + ", ".join(missing) + ".")
    return DUNGEON_RESET_REQUIRED_KEYS + tuple(k for k in DUNGEON_RESET_OPTIONAL_KEYS if k in batch)


# =============================================================================
class DungeonControllerRuntime:
    """Controller-facing dungeon rollout runtime.

    Satisfies the :class:`~ehc_sn.controllers.tem.TEMTaskRuntime` structural
    protocol for online environment-rollout controllers without importing from
    the controllers layer.  Use this in Lightning training surfaces instead of
    Use this in Lightning training surfaces that require online environment rollouts.
    """

    def build_reset_td(self, batch: Batch) -> TensorDict:
        """Return the dungeon environment reset TensorDict."""
        return build_dungeon_reset_td(batch)

    def extract_step_data(
        self,
        env_td: TensorDictBase,
        *,
        trace_metadata: Mapping[str, Tensor] | None = None,
    ) -> dict[str, Tensor]:
        """Return the model-facing dungeon payload for the current step."""
        return extract_dungeon_step_data(env_td, trace_metadata=trace_metadata)

    def annotate_revisit_state(
        self,
        payload: dict[str, Tensor],
        env_td: TensorDictBase,
        visit_counts: Tensor,
    ) -> dict[str, Tensor]:
        """Attach revisit annotations aligned with the current dungeon step."""
        return annotate_dungeon_revisit_state(payload, env_td, visit_counts)

    def new_visit_counts(
        self,
        reset_td: TensorDictBase,
        *,
        device: Any,
    ) -> Tensor:
        """Allocate per-slot dungeon visit counters."""
        return new_dungeon_visit_counts(reset_td, device=device)

    def record_visit(self, visit_counts: Tensor, location_id: Tensor) -> Tensor:
        """Increment visit counters for the current dungeon locations."""
        return record_dungeon_visit(visit_counts, location_id)

    def make_policy_input(self, env_td: TensorDictBase) -> PolicyInput:
        """Adapt dungeon environment state to the policy input contract."""
        return make_dungeon_policy_input(env_td)


# =============================================================================
__all__ = [
    "DUNGEON_RESET_OPTIONAL_KEYS",
    "DUNGEON_RESET_REQUIRED_KEYS",
    "DUNGEON_STEP_KEYS",
    "DungeonControllerRuntime",
    "annotate_dungeon_revisit_state",
    "batch_size_from_dungeon_batch",
    "build_dungeon_reset_td",
    "coerce_dungeon_step_input",
    "extract_dungeon_step_data",
    "infer_dungeon_static_batch_keys",
    "make_dungeon_policy_input",
    "new_dungeon_visit_counts",
    "record_dungeon_visit",
]
