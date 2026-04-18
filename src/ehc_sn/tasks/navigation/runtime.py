"""Navigation task-local runtime helpers.

These helpers hold navigation-owned reset contracts, current-step payload
shaping, and revisit bookkeeping so controller code remains task-agnostic.
"""

from __future__ import annotations

from typing import Any, Final, Mapping

import torch
from tensordict import TensorDict, TensorDictBase
from torch import Tensor

from ehc_sn.controllers.tem import TEMEnvironment
from ehc_sn.policies import PolicyInput
from ehc_sn.tasks.navigation.contracts import NavigationTargets, NavigationTaskInput
from ehc_sn.types import Batch

NAVIGATION_RESET_REQUIRED_KEYS: Final[tuple[str, ...]] = ("topology", "observations", "mask_valid")
NAVIGATION_RESET_OPTIONAL_KEYS: Final[tuple[str, ...]] = ("regions", "start", "goals", "landmarks")
NAVIGATION_STEP_KEYS: Final[tuple[str, ...]] = (
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
class NavigationControllerRuntime:
    """Controller-facing navigation rollout runtime."""

    def build_reset_td(  # ----------------------------------------------------
        self,
        batch: Batch,
    ) -> TensorDict:
        """Return the navigation environment reset TensorDict."""
        return build_navigation_reset_td(batch)

    def extract_step_data(  # -------------------------------------------------
        self,
        env_td: TensorDictBase,
        *,
        trace_metadata: Mapping[str, Tensor] | None = None,
    ) -> dict[str, Tensor]:
        """Return the model-facing navigation payload for the current step."""
        return extract_navigation_task_input(env_td, trace_metadata=trace_metadata)

    def annotate_revisit_state(  # --------------------------------------------
        self,
        payload: dict[str, Tensor],
        env_td: TensorDictBase,
        visit_counts: Tensor,
    ) -> dict[str, Tensor]:
        """Attach revisit annotations aligned with the current navigation step."""
        return annotate_navigation_revisit_state(payload, env_td, visit_counts)

    def refresh_halted_slots(  # ----------------------------------------------
        self,
        batch: Batch,
        halted: Tensor,
        static_data: dict[str, Tensor],
        env_td: TensorDictBase,
        visit_counts: Tensor,
        *,
        environment: TEMEnvironment,
    ) -> tuple[dict[str, Tensor], TensorDictBase, Tensor]:
        """Reset halted navigation slots from the next incoming static maze batch."""
        return refresh_navigation_halted_slots(
            batch,
            halted,
            static_data,
            env_td,
            visit_counts,
            environment=environment,
        )

    def new_visit_counts(  # --------------------------------------------------
        self,
        reset_td: TensorDictBase,
        *,
        device: Any,
    ) -> Tensor:
        """Allocate per-slot navigation visit counters."""
        return new_navigation_visit_counts(reset_td, device=device)

    def record_visit(  # ------------------------------------------------------
        self,
        visit_counts: Tensor,
        location_id: Tensor,
    ) -> Tensor:
        """Increment visit counters for the current navigation locations."""
        return record_navigation_visit(visit_counts, location_id)

    def make_policy_input(  # -------------------------------------------------
        self,
        env_td: TensorDictBase,
    ) -> PolicyInput:
        """Adapt navigation environment state to the policy input contract."""
        return make_navigation_policy_input(env_td)


# =============================================================================
def make_navigation_policy_input(  # ------------------------------------------
    env_td: TensorDictBase,
) -> PolicyInput:
    """Build a :class:`PolicyInput` from a navigation environment state."""
    return PolicyInput(
        valid_action_mask=env_td["valid_action_mask"],
        location_id=env_td["location_id"] if "location_id" in env_td.keys() else None,
        region_id=env_td["region_id"] if "region_id" in env_td.keys() else None,
        step_count=env_td["step_count"] if "step_count" in env_td.keys() else None,
    )


# =============================================================================
def coerce_navigation_step_input(  # ------------------------------------------
    data: Mapping[str, Tensor],
) -> NavigationTaskInput:
    """Convert a generic navigation step mapping into a typed :class:`NavigationTaskInput`.

    Validates that the required step fields are present and populates optional
    fields from the mapping when available.  This is the canonical task-owned
    coercion point so adapter encoders can operate on typed inputs rather than
    raw dicts.

    Args:
        data: Generic navigation step dict as produced by
            :func:`extract_navigation_task_input`.

    Returns:
        Fully typed :class:`NavigationTaskInput`.

    Raises:
        KeyError: If any mandatory step field is absent.
    """
    required = ("observation", "observation_id", "previous_action", "location_id")
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"Navigation step payload is missing required fields: {', '.join(missing)}.")
    return NavigationTaskInput(
        observation=data["observation"],
        observation_id=data["observation_id"],
        previous_action=data["previous_action"],
        location_id=data["location_id"],
        valid_action_mask=data.get("valid_action_mask", data["observation"].new_zeros(data["observation"].shape[0])),
        step_count=data.get("step_count", data["observation"].new_zeros(data["observation"].shape[0], 1)),
        region_id=data.get("region_id"),
        landmark_id=data.get("landmark_id"),
        episode_start=data.get("episode_start"),
        is_revisit=data.get("is_revisit"),
    )


# =============================================================================
def coerce_navigation_targets(  # ---------------------------------------------
    data: Mapping[str, Tensor],
) -> NavigationTargets:
    """Extract :class:`NavigationTargets` from a navigation carry data mapping.

    This is the canonical task-owned extraction point.  It requires
    ``observation_id`` to be present and does **not** fall back to a
    ``labels`` key — that legacy path is not part of the navigation task
    contract.

    Args:
        data: Controller carry data dict (e.g., ``carry.data``).

    Returns:
        :class:`NavigationTargets` with ``observation_id`` and optional
        ``is_revisit``.

    Raises:
        KeyError: If ``observation_id`` is absent from ``data``.
    """
    if "observation_id" not in data:
        raise KeyError(
            "Navigation carry data must provide 'observation_id' to build NavigationTargets; " "'labels' fallback is not supported."
        )
    return NavigationTargets(
        observation_id=data["observation_id"],
        is_revisit=data.get("is_revisit"),
    )


# =============================================================================
def infer_navigation_static_batch_keys(  # ------------------------------------
    batch: Batch,
) -> tuple[str, ...]:
    """Return the static navigation batch keys present in the current batch."""
    missing = [key for key in NAVIGATION_RESET_REQUIRED_KEYS if key not in batch]
    if missing:
        raise KeyError("Navigation batch is missing required reset keys: " + ", ".join(missing) + ".")
    return NAVIGATION_RESET_REQUIRED_KEYS + tuple(key for key in NAVIGATION_RESET_OPTIONAL_KEYS if key in batch)


# =============================================================================
def build_navigation_reset_td(  # ---------------------------------------------
    batch: Batch,
) -> TensorDict:
    """Return the static maze tensors required by the environment reset."""
    keys = infer_navigation_static_batch_keys(batch)
    reset_data = {key: batch[key] for key in keys}
    batch_size = int(next(iter(reset_data.values())).shape[0])
    device = next(iter(reset_data.values())).device
    return TensorDict(reset_data, batch_size=[batch_size], device=device)


# =============================================================================
def extract_navigation_task_input(  # -----------------------------------------
    env_td: TensorDictBase,
    *,
    trace_metadata: Mapping[str, Tensor] | None = None,
) -> dict[str, Tensor]:
    """Extract the current-step TEM payload from one environment state."""
    payload = {key: env_td[key] for key in NAVIGATION_STEP_KEYS if key in env_td.keys()}
    if "step_count" in payload:
        payload["episode_start"] = payload["step_count"].squeeze(-1).to(torch.int32) == 0
    if trace_metadata is not None:
        payload.update(trace_metadata)
    return payload


# =============================================================================
def annotate_navigation_revisit_state(  # -------------------------------------
    payload: dict[str, Tensor],
    env_td: TensorDictBase,
    visit_counts: Tensor,
) -> dict[str, Tensor]:
    """Attach revisit eligibility aligned with the current-step payload."""
    location_id = env_td["location_id"].to(device=visit_counts.device, dtype=torch.int64)
    prior_counts = visit_counts.gather(dim=1, index=location_id).squeeze(-1)
    payload["is_revisit"] = prior_counts > 0
    return payload


# =============================================================================
def refresh_navigation_halted_slots(  # ---------------------------------------
    batch: Batch,
    halted: Tensor,
    static_data: dict[str, Tensor],
    env_td: TensorDictBase,
    visit_counts: Tensor,
    *,
    environment: TEMEnvironment,
) -> tuple[dict[str, Tensor], TensorDictBase, Tensor]:
    """Reset halted navigation slots from the next incoming static maze batch."""
    if not torch.any(halted):
        return static_data, env_td, visit_counts

    new_static = build_navigation_reset_td(batch)
    new_keys = frozenset(new_static.keys())
    frozen_keys = frozenset(static_data.keys())
    if new_keys != frozen_keys:
        added = sorted(new_keys - frozen_keys)
        removed = sorted(frozen_keys - new_keys)
        parts = []
        if added:
            parts.append(f"unexpected new keys: {added}")
        if removed:
            parts.append(f"missing previously present keys: {removed}")
        raise KeyError(f"Navigation partial-reset schema drift detected — {'; '.join(parts)}.")
    next_static = {
        key: torch.where(halted.view((-1,) + (1,) * (value.ndim - 1)), value, static_data[key]) for key, value in new_static.items()
    }
    reset_td = TensorDict(next_static, batch_size=env_td.batch_size, device=env_td.device)
    next_env_td = environment.reset_slots(halted, reset_td, env_td)
    next_visit_counts = visit_counts.clone()
    next_visit_counts[halted] = 0
    return next_static, next_env_td, next_visit_counts


# =============================================================================
def batch_size_from_navigation_batch(  # --------------------------------------
    batch: Batch,
) -> int:
    """Return the leading batch dimension from a navigation static maze batch."""
    return int(batch[NAVIGATION_RESET_REQUIRED_KEYS[0]].shape[0])


# =============================================================================
def new_navigation_visit_counts(  # -------------------------------------------
    reset_td: TensorDictBase,
    *,
    device: Any,
) -> Tensor:
    """Allocate per-slot location visit counters for the current maze shape."""
    topology = reset_td["topology"]
    batch_size = int(topology.shape[0])
    n_locations = int(topology.shape[-2] * topology.shape[-1])
    return torch.zeros((batch_size, n_locations), dtype=torch.int32, device=device)


# =============================================================================
def record_navigation_visit(  # -----------------------------------------------
    visit_counts: Tensor,
    location_id: Tensor,
) -> Tensor:
    """Increment visit counters for the current navigation locations."""
    updated = visit_counts.clone()
    index = location_id.to(device=updated.device, dtype=torch.int64)
    increments = torch.ones_like(index, dtype=updated.dtype, device=updated.device)
    updated.scatter_add_(1, index, increments)
    return updated


# =============================================================================
__all__ = [
    "NAVIGATION_RESET_OPTIONAL_KEYS",
    "NAVIGATION_RESET_REQUIRED_KEYS",
    "NavigationControllerRuntime",
    "annotate_navigation_revisit_state",
    "batch_size_from_navigation_batch",
    "build_navigation_reset_td",
    "coerce_navigation_step_input",
    "coerce_navigation_targets",
    "extract_navigation_task_input",
    "infer_navigation_static_batch_keys",
    "make_navigation_policy_input",
    "new_navigation_visit_counts",
    "record_navigation_visit",
    "refresh_navigation_halted_slots",
]
