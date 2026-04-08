"""Policy-driven dungeon walk environment for TEM rollouts.

The environment consumes static maze tensors on reset and emits compact,
current-state step payloads for TEM. Policy selection is external: the caller
must pass a concrete action to :meth:`step`.

TensorDict contract:
    reset input:
        "topology"     : (H, W) bool   — passable cells
        "observations" : (H, W) int64  — observation id per cell
        "mask_valid"   : (H, W) bool   — valid cells allowed for starts/walks
        optional:
            "regions"  : (H, W) int64  — region id per cell
            "start"    : (H, W) bool   — preferred reset locations

    observation_spec / state_spec:
        "inputs"            : (O,)   float32 — one-hot observation encoding
        "observation_target": (1,)   int64   — categorical observation id
        "previous_action"   : (1,)   int64   — action that produced current state
        "location_id"       : (1,)   int64   — flattened current cell index
        "region_id"         : (1,)   int64   — current region id (0 if absent)
        "landmark_id"       : (1,)   int64   — current landmark id / shiny cue (0 if absent)
        "valid_action_mask" : (A,)   bool    — legal movement actions
        "step_count"        : (1,)   int32   — transitions taken in episode

    action_spec:
        "action" : (1,) int64 — movement action chosen by the controller

    reward_spec:
        "reward" : (1,) float32 — always zero in the first TEM rollout version
"""

from __future__ import annotations

from typing import Final

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field
from tensordict import TensorDict, TensorDictBase
from torch import Tensor
from torchrl.data import Categorical, Composite, Unbounded
from torchrl.envs import EnvBase

from ehc_sn.types import Device

ACTION_STAY: Final[int] = 0
ACTION_UP: Final[int] = 1
ACTION_RIGHT: Final[int] = 2
ACTION_DOWN: Final[int] = 3
ACTION_LEFT: Final[int] = 4
DEFAULT_ACTION_COUNT: Final[int] = 5


# =================================================================================================
class EnvConfig(BaseModel, extra="forbid"):
    """Configuration for :class:`DungeonWalk`."""

    max_steps: int = Field(
        default=32,
        ge=1,
        description="Maximum steps before truncation.",
    )
    observation_dim: int = Field(
        ...,
        ge=1,
        description="Encoded observation feature dimension.",
    )
    action_count: int = Field(
        default=DEFAULT_ACTION_COUNT,
        ge=DEFAULT_ACTION_COUNT,
        le=DEFAULT_ACTION_COUNT,
        description="Fixed movement action count: stay, up, right, down, left.",
    )
    use_start_channel: bool = Field(
        default=True,
        description="If True, prefer the optional start channel when resetting slots.",
    )


# =================================================================================================
class DungeonWalk(EnvBase):
    """TorchRL environment for policy-driven dungeon walks.

    The environment caches static maze tensors per batch slot and emits a compact
    current-state representation suitable for TEM. The controller owns policy
    invocation and passes the selected action into :meth:`step`.
    """

    batch_locked = True

    def __init__(  # ------------------------------------------------------------------------------
        self, config: EnvConfig, batch_size: int, device: Device | str | None = None,
    ) -> None:  # fmt: skip
        super().__init__(batch_size=[batch_size], device=device)
        self._config = config
        self._topology: Tensor | None = None
        self._observations: Tensor | None = None
        self._mask_valid: Tensor | None = None
        self._regions: Tensor | None = None
        self._landmarks: Tensor | None = None
        self._height = 0
        self._width = 0
        self._action_deltas = torch.tensor([[0, 0], [-1, 0], [0, 1], [1, 0], [0, -1]], dtype=torch.int64)
        self._generator_seed: int | None = None
        self._generator = torch.Generator(device="cpu")
        self._make_specs()

    @property
    def config(self) -> EnvConfig:
        """Return the environment configuration."""
        return self._config

    def _make_specs(self) -> None:
        """Build TorchRL specs for current-state walk payloads."""
        bs = self.batch_size
        action_count = self._config.action_count
        observation_dim = self._config.observation_dim
        step_spec = Composite(
            inputs=Unbounded(shape=(*bs, observation_dim), dtype=torch.float32),
            observation_target=Unbounded(shape=(*bs, 1), dtype=torch.int64),
            previous_action=Unbounded(shape=(*bs, 1), dtype=torch.int64),
            location_id=Unbounded(shape=(*bs, 1), dtype=torch.int64),
            region_id=Unbounded(shape=(*bs, 1), dtype=torch.int64),
            landmark_id=Unbounded(shape=(*bs, 1), dtype=torch.int64),
            valid_action_mask=Unbounded(shape=(*bs, action_count), dtype=torch.bool),
            step_count=Unbounded(shape=(*bs, 1), dtype=torch.int32),
            shape=bs,
        )
        self.observation_spec = step_spec.clone()
        self.state_spec = step_spec.clone()
        self.action_spec = Composite(
            action=Categorical(n=action_count, shape=(*bs, 1), dtype=torch.int64),
            shape=bs,
        )
        self.reward_spec = Unbounded(shape=(*bs, 1), dtype=torch.float32)

    def _reset(  # -------------------------------------------------------------------------------
        self, tensordict: TensorDictBase | None,
    ) -> TensorDictBase:  # fmt: skip
        """Reset all slots from a static maze batch."""
        if tensordict is None or tensordict.is_empty():
            raise ValueError("DungeonWalk.reset requires a maze batch with topology, observations, and mask_valid.")  # fmt: skip

        self._cache_static_maps(tensordict)
        runtime_device = self._runtime_device()
        location_id = self._sample_start_locations(tensordict)
        previous_action = torch.zeros((*self.batch_size, 1), dtype=torch.int64, device=runtime_device)
        step_count = torch.zeros((*self.batch_size, 1), dtype=torch.int32, device=runtime_device)
        return self._build_state(location_id=location_id, previous_action=previous_action, step_count=step_count)  # fmt: skip

    @torch.no_grad()
    def _step(  # --------------------------------------------------------------------------------
        self, tensordict: TensorDictBase,
    ) -> TensorDictBase:  # fmt: skip
        """Apply a controller-selected action and emit the next current-state payload."""
        self._require_static_maps()
        runtime_device = self._runtime_device()
        self._ensure_runtime_device(runtime_device)

        action = tensordict["action"].to(device=runtime_device, dtype=torch.int64)
        if action.shape != (*self.batch_size, 1):
            raise ValueError(f"DungeonWalk.step expected action shape {(*self.batch_size, 1)}, got {tuple(action.shape)}.")  # fmt: skip

        valid_action_mask = tensordict["valid_action_mask"].to(device=runtime_device)
        invalid = ~valid_action_mask.gather(dim=-1, index=action)
        if torch.any(invalid):
            bad_rows = invalid.squeeze(-1).nonzero(as_tuple=False).flatten().tolist()
            raise ValueError(f"DungeonWalk.step received invalid actions for rows {bad_rows}.")

        location_id = tensordict["location_id"].to(device=runtime_device, dtype=torch.int64)
        step_count = tensordict["step_count"].to(device=runtime_device, dtype=torch.int32) + 1
        rows, cols = self._unflatten_location(location_id.squeeze(-1))
        next_rows = rows + self._action_deltas[action.squeeze(-1), 0]
        next_cols = cols + self._action_deltas[action.squeeze(-1), 1]
        next_location_id = self._flatten_location(next_rows, next_cols).unsqueeze(-1)

        next_state = self._build_state(
            location_id=next_location_id,
            previous_action=action,
            step_count=step_count,
        )

        truncated = step_count >= self._config.max_steps
        terminated = torch.zeros_like(truncated, dtype=torch.bool)
        done = terminated | truncated
        reward = torch.zeros((*self.batch_size, 1), dtype=torch.float32, device=runtime_device)

        return TensorDict(
            {
                **next_state.to_dict(),
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "done": done,
            },
            batch_size=self.batch_size,
            device=runtime_device,
        )

    def reset_slots(  # --------------------------------------------------------------------------
        self, reset_mask: Tensor, tensordict: TensorDictBase, state: TensorDictBase,
    ) -> TensorDictBase:  # fmt: skip
        """Reset a subset of batch slots while preserving active ones.

        This is a controller-side helper for partial-reset batching. It is not
        part of the standard TorchRL env API but avoids rebuilding active slot
        state outside the environment.
        """
        if reset_mask.shape != self.batch_size:
            raise ValueError(f"reset_mask must have shape {tuple(self.batch_size)}, got {tuple(reset_mask.shape)}.")
        if not torch.any(reset_mask):
            return state

        self._cache_static_maps(tensordict, reset_mask=reset_mask)
        runtime_device = self._runtime_device()
        reset_location_id = self._sample_start_locations(tensordict, reset_mask=reset_mask)
        reset_previous_action = torch.zeros((int(reset_mask.sum().item()), 1), dtype=torch.int64, device=runtime_device)
        reset_step_count = torch.zeros((int(reset_mask.sum().item()), 1), dtype=torch.int32, device=runtime_device)
        reset_state = self._build_state(
            location_id=reset_location_id,
            previous_action=reset_previous_action,
            step_count=reset_step_count,
        )

        next_state = state.clone()
        for key in self.observation_spec.keys(True, True):
            next_state[key][reset_mask] = reset_state[key]
        for key in ("done", "terminated", "truncated"):
            if key in next_state.keys():
                next_state[key][reset_mask] = False
        if "reward" in next_state.keys():
            next_state["reward"][reset_mask] = 0.0
        return next_state

    def _cache_static_maps(  # -------------------------------------------------------------------
        self, tensordict: TensorDictBase, reset_mask: Tensor | None = None,
    ) -> None:  # fmt: skip
        """Cache static maze tensors for all or selected slots."""
        topology = self._require_key(tensordict, "topology").to(dtype=torch.bool)
        runtime_device = topology.device
        observations = self._require_key(tensordict, "observations").to(device=runtime_device, dtype=torch.int64)  # fmt: skip
        mask_valid = self._require_key(tensordict, "mask_valid").to(device=runtime_device, dtype=torch.bool)
        regions = tensordict.get("regions")
        if regions is not None:
            regions = regions.to(device=runtime_device, dtype=torch.int64)
        landmarks = tensordict.get("landmarks")
        if landmarks is not None:
            landmarks = landmarks.to(device=runtime_device, dtype=torch.int64)

        if topology.shape != observations.shape or topology.shape != mask_valid.shape:
            raise ValueError("DungeonWalk reset tensors must share the same shape for topology, observations, and mask_valid.")
        if topology.shape[0] != self.batch_size[0]:
            raise ValueError(f"DungeonWalk expected batch dimension {self.batch_size[0]}, got {topology.shape[0]}.")

        incoming_height, incoming_width = int(topology.shape[-2]), int(topology.shape[-1])
        if reset_mask is None or self._topology is None:
            self._height, self._width = incoming_height, incoming_width
            self._topology = topology
            self._observations = observations
            self._mask_valid = mask_valid
            self._regions = regions if regions is not None else torch.zeros_like(observations, dtype=torch.int64)  # fmt: skip
            self._landmarks = landmarks if landmarks is not None else torch.zeros_like(observations, dtype=torch.int64)
            return

        incoming_spatial = (incoming_height, incoming_width)
        cached_spatial = (self._height, self._width)
        if incoming_spatial != cached_spatial:
            raise ValueError(
                "DungeonWalk partial reset received maze with spatial shape "
                f"{incoming_spatial}, but cached maze has shape {cached_spatial}. "
                "Spatial dimensions must match for reset_slots(). Call reset() instead to initialize a new maze."
            )

        self._topology[reset_mask] = topology[reset_mask]
        self._observations[reset_mask] = observations[reset_mask]
        self._mask_valid[reset_mask] = mask_valid[reset_mask]
        if self._regions is None:
            self._regions = torch.zeros_like(observations, dtype=torch.int64)
        if regions is not None:
            self._regions[reset_mask] = regions[reset_mask]
        else:
            self._regions[reset_mask] = 0
        if self._landmarks is None:
            self._landmarks = torch.zeros_like(observations, dtype=torch.int64)
        if landmarks is not None:
            self._landmarks[reset_mask] = landmarks[reset_mask]
        else:
            self._landmarks[reset_mask] = 0

    def _sample_start_locations(  # -------------------------------------------------------------
        self, tensordict: TensorDictBase, reset_mask: Tensor | None = None,
    ) -> Tensor:  # fmt: skip
        """Sample one valid reset location per selected slot."""
        self._require_static_maps()
        runtime_device = self._runtime_device()
        self._ensure_runtime_device(runtime_device)
        start = tensordict.get("start")
        if start is not None:
            start = start.to(device=runtime_device, dtype=torch.bool)

        slot_mask = reset_mask if reset_mask is not None else torch.ones(self.batch_size, dtype=torch.bool, device=runtime_device)
        slot_ids = slot_mask.nonzero(as_tuple=False).flatten()
        location_ids: list[Tensor] = []
        valid_cells = self._topology & self._mask_valid

        for slot_id in slot_ids.tolist():
            candidates = valid_cells[slot_id]
            if self._config.use_start_channel and start is not None:
                preferred = candidates & start[slot_id]
                if torch.any(preferred):
                    candidates = preferred
            flat_candidates = candidates.view(-1).nonzero(as_tuple=False).flatten()
            if flat_candidates.numel() == 0:
                raise ValueError(f"DungeonWalk slot {slot_id} has no valid reset locations.")
            choice = torch.randint(
                low=0,
                high=int(flat_candidates.numel()),
                size=(1,),
                generator=self._generator,
                device=flat_candidates.device,
            )
            location_ids.append(flat_candidates[choice])

        return torch.stack(location_ids, dim=0).view(-1, 1).to(device=runtime_device, dtype=torch.int64)

    def _build_state(  # -------------------------------------------------------------------------
        self, *, location_id: Tensor, previous_action: Tensor, step_count: Tensor,
    ) -> TensorDict:  # fmt: skip
        """Construct a current-state step payload from location ids."""
        self._require_static_maps()
        runtime_device = self._runtime_device()

        flat_location = location_id.squeeze(-1)
        rows, cols = self._unflatten_location(flat_location)
        batch_index = torch.arange(flat_location.shape[0], device=runtime_device)
        observation_target = self._observations[batch_index, rows, cols].view(-1, 1)
        self._validate_observation_ids(observation_target)
        inputs = F.one_hot(observation_target.squeeze(-1), num_classes=self._config.observation_dim).to(torch.float32)
        region_id = (
            self._regions[batch_index, rows, cols].view(-1, 1) if self._regions is not None else torch.zeros_like(observation_target)
        )
        landmark_id = (
            self._landmarks[batch_index, rows, cols].view(-1, 1) if self._landmarks is not None else torch.zeros_like(observation_target)
        )
        valid_action_mask = self._compute_valid_action_mask(rows, cols)

        return TensorDict(
            {
                "inputs": inputs,
                "observation_target": observation_target.to(torch.int64),
                "previous_action": previous_action.to(torch.int64),
                "location_id": location_id.to(torch.int64),
                "region_id": region_id.to(torch.int64),
                "landmark_id": landmark_id.to(torch.int64),
                "valid_action_mask": valid_action_mask,
                "step_count": step_count.to(torch.int32),
            },
            batch_size=[flat_location.shape[0]],
            device=runtime_device,
        )

    def _compute_valid_action_mask(self, rows: Tensor, cols: Tensor) -> Tensor:
        """Return legal movement actions for the current locations."""
        self._require_static_maps()
        self._ensure_runtime_device(rows.device)

        next_rows = rows.unsqueeze(-1) + self._action_deltas[:, 0]
        next_cols = cols.unsqueeze(-1) + self._action_deltas[:, 1]
        in_bounds = (0 <= next_rows) & (next_rows < self._height) & (0 <= next_cols) & (next_cols < self._width)
        safe_rows = next_rows.clamp(0, self._height - 1)
        safe_cols = next_cols.clamp(0, self._width - 1)
        batch_index = torch.arange(rows.shape[0], device=rows.device).unsqueeze(-1).expand_as(safe_rows)
        passable = self._topology[batch_index, safe_rows, safe_cols] & self._mask_valid[batch_index, safe_rows, safe_cols]
        return in_bounds & passable

    def _flatten_location(self, rows: Tensor, cols: Tensor) -> Tensor:
        """Return flattened cell indices from grid coordinates."""
        return rows * self._width + cols

    def _unflatten_location(self, location_id: Tensor) -> tuple[Tensor, Tensor]:
        """Return grid coordinates from flattened cell indices."""
        return torch.div(location_id, self._width, rounding_mode="floor"), location_id % self._width

    def _validate_observation_ids(self, observation_target: Tensor) -> None:
        """Fail fast if cached observation ids exceed the configured encoding dimension."""
        min_id = int(observation_target.min().item())
        max_id = int(observation_target.max().item())
        if min_id < 0 or max_id >= self._config.observation_dim:
            raise ValueError(
                "DungeonWalk observation ids must lie within " f"[0, {self._config.observation_dim - 1}], got min={min_id}, max={max_id}."
            )

    def _require_static_maps(self) -> None:
        """Ensure the environment has been reset before stepping."""
        if self._topology is None or self._observations is None or self._mask_valid is None:
            raise RuntimeError("DungeonWalk static maze tensors are not initialized. Call reset() first.")

    def _runtime_device(self) -> torch.device:
        """Return the device hosting the cached maze tensors."""
        if self._topology is not None:
            return self._topology.device
        return torch.device(self.device) if self.device is not None else torch.device("cpu")

    def _ensure_runtime_device(self, device: Device | str | None) -> None:
        """Move cached runtime tensors and RNG to the active execution device."""
        target = torch.device(device) if device is not None else torch.device("cpu")
        if self._action_deltas.device != target:
            self._action_deltas = self._action_deltas.to(target)

        generator_device = torch.device(self._generator.device)
        if generator_device.type == target.type:
            return

        seed = self._generator_seed if self._generator_seed is not None else self._generator.initial_seed()
        self._generator = torch.Generator(device=target)
        self._generator.manual_seed(seed)

    @staticmethod
    def _require_key(tensordict: TensorDictBase, key: str) -> Tensor:
        """Return a required reset key or raise a clear error."""
        if key not in tensordict.keys():
            raise KeyError(f"DungeonWalk reset requires '{key}'.")
        return tensordict[key]

    def _set_seed(self, seed: int | None) -> None:
        """Seed the internal reset sampler."""
        if seed is not None:
            self._generator_seed = seed
            self._generator.manual_seed(seed)


# =================================================================================================
__all__ = [
    "ACTION_DOWN", "ACTION_LEFT", "ACTION_RIGHT", "ACTION_STAY", "ACTION_UP",
    "DungeonWalk", "EnvConfig",
]  # fmt: skip
