"""Replay trajectory controller — canonical owner.

Canonical import path::

    from ehc_sn.controllers.replay.trajectory import (
        ReplayTrajectoryController, ReplayTrajectoryControllerConfig,
        ReplayTrajectoryRuntime, ReplayRolloutState, ReplayStepOutput,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

import torch
from pydantic import BaseModel, Field
from torch import Tensor

from ehc_sn.controllers._base import BaseController, RolloutBackbone, RolloutState, batch_anchor_tensor
from ehc_sn.types import Batch
from ehc_sn.utils.detach import DetachMixin

# Key prefix that identifies trajectory-array columns in source batches.
# Resident payload stores only these columns (minus trajectory_length, which
# lives as a typed carry field).
_TRAJECTORY_PREFIX = "trajectory_"
_TRAJECTORY_IDENTITY_KEY = "__trajectory_id__"


# =============================================================================
class ReplayTrajectoryRuntime(Protocol):
    """Task-owned per-step extraction interface for replay trajectory controllers."""

    def extract_step_per_slot(
        self,
        resident: Batch,
        cursor: Tensor,
        task_state: dict[str, Tensor],
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Extract current-step tensors from *resident* carry data.

        Args:
            resident: Per-slot resident trajectory arrays (carry-owned, not the
                incoming source batch).  Contains trajectory columns for every slot.
            cursor: Per-slot step cursor ``(B,)`` int64.
            task_state: Mutable task-local state threaded through carry.

        Returns:
            ``(step_data, new_task_state)`` — step tensors and updated carry state.
        """
        ...

    def initial_task_state(self, batch: Batch, *, device: Any) -> dict[str, Tensor]:
        """Allocate the initial (zeroed) task-local state for a fresh episode."""
        ...

    def trajectory_lengths(self, batch: Batch) -> Tensor:
        """Return the per-slot effective trajectory length from a source batch."""
        ...


# =============================================================================
class ReplayTrajectoryControllerConfig(BaseModel, extra="forbid"):
    """Configuration for :class:`ReplayTrajectoryController`."""

    window_size: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "Maximum steps to replay per window.  None = full-episode replay "
            "(canonical benchmark mode).  A positive integer enables fixed "
            "contiguous windows for training optimisation."
        ),
    )


# =============================================================================
@dataclass
class ReplayRolloutState[ModelState](RolloutState[ModelState]):
    """Controller carry for replay trajectory controllers.

    Carry is the **sole** continuity authority.  Active slots never consult
    the incoming source batch for trajectory data — they read exclusively from
    *resident_payload* which was atomically set at the last admission boundary.

    Attributes:
        cursor: Per-slot step index ``(B,)`` int64.  0 = first step of admitted episode.
        trajectory_length: Per-slot effective length ``(B,)`` int64.  Set at admission only.
        trajectory_id: Stable fit-path identity ``(B,)`` int64 derived from the
            dataset index of the admitted row.  -1 = slot not yet admitted.
            Changes only on halted→admitted transition.  Never passed to the model.
        resident_payload: Carry-owned trajectory arrays ``{key: (B, T, ...)}``.
            Populated at admission from the source batch; active slots read from
            here, never from the incoming batch.  Excluded from rollout snapshots
            so large trajectory tensors are not cloned into per-step records.
        task_state: Task-owned per-slot runtime state threaded explicitly through carry.
    """

    cursor: Tensor  # (B,) int64 — current step index per slot
    trajectory_length: Tensor  # (B,) int64 — effective trajectory length per slot
    trajectory_id: Tensor  # (B,) int64 — stable admitted-trajectory identity; -1 = empty
    resident_payload: dict[str, Tensor]  # carry-owned trajectory arrays; NOT in snapshot
    task_state: dict[str, Tensor] = field(default_factory=dict)
    """Task-owned replay-local state (e.g. visit counts) threaded explicitly through carry."""


# =============================================================================
@dataclass(frozen=True)
class ReplayStepOutput(DetachMixin):
    """Output produced by one replay controller step."""

    backbone_output: Any
    cursor: Tensor  # (B,) int64


# =============================================================================
def _build_resident_payload(
    old_resident: dict[str, Tensor],
    batch: Batch,
    admission: Tensor,
) -> dict[str, Tensor]:
    """Return an updated resident payload, mixing admitted and active rows.

    Admitted slots (``admission=True``) receive a copy of the corresponding
    row from *batch*.  Active slots (``admission=False``) keep their existing
    row from *old_resident*.

    Only trajectory-column keys (prefix ``"trajectory_"``, excluding
    ``"trajectory_length"``) are carried in the resident payload.

    Args:
        old_resident: Current carry-owned trajectory arrays ``{key: (B, T, ...)}``.
            May be empty ``{}`` on the very first step when all slots are halted.
        batch: Source batch providing new trajectory arrays for admitted slots.
        admission: Boolean mask ``(B,)``; ``True`` = slot is being admitted.

    Returns:
        Updated resident payload ``{key: (B, T, ...)}`` on the same device as *batch*.
    """
    resident_keys = [k for k in batch if k.startswith(_TRAJECTORY_PREFIX) and k != "trajectory_length"]

    if not old_resident:
        # First call: all slots are halted (initial state), copy everything from batch.
        return {k: batch[k].clone() for k in resident_keys if k in batch}

    new_resident: dict[str, Tensor] = {}
    for k in resident_keys:
        if k not in batch:
            if k in old_resident:
                new_resident[k] = old_resident[k]
            continue
        new_val = batch[k]  # (B, T, ...)
        if k not in old_resident:
            new_resident[k] = new_val.clone()
            continue
        old_val = old_resident[k]  # (B, T, ...)
        # Broadcast admission mask across time and feature dimensions.
        expand_shape = (admission.shape[0],) + (1,) * (new_val.ndim - 1)
        mask = admission.view(expand_shape).expand_as(new_val)
        new_resident[k] = torch.where(mask, new_val, old_val)
    return new_resident


# =============================================================================
class ReplayTrajectoryController[ModelState](BaseController[ModelState, ReplayTrajectoryControllerConfig]):
    """Generic stepwise replay controller for arena-family recurrent replay.

    Slot continuity contract
    ------------------------
    *Carry is the only continuity authority.*  Active slots advance by reading
    exclusively from ``state.resident_payload`` (trajectory arrays admitted at
    the last halted→active boundary), never from the incoming source batch.

    Lifecycle per slot per step:

    1. If ``state.halted[b]`` is True → *admission*: atomically copy trajectory
       arrays from *batch* into ``resident_payload[b]``, reset cursor to 0,
       update trajectory_length, update trajectory_id, reset model state.
    2. If ``state.halted[b]`` is False → *active*: advance cursor by 1, read
       from ``resident_payload[b]``, continue model state.
    3. Extract step tensors from the updated ``resident_payload`` at the updated
       cursor.  The incoming *batch* is **not** consulted for active slots.
    4. After the backbone forward, compute the new ``halted`` flag from carry
       state only (cursor vs trajectory_length).
    """

    def __init__(
        self,
        backbone: RolloutBackbone[ModelState],
        config: ReplayTrajectoryControllerConfig,
        runtime: ReplayTrajectoryRuntime,
    ) -> None:
        """Create a replay trajectory controller."""
        super().__init__(backbone=backbone, config=config)
        self._runtime = runtime

    @property
    def runtime(self) -> ReplayTrajectoryRuntime:
        """Return the injected task-owned replay runtime."""
        return self._runtime

    def initial_state(
        self,
        batch_sample: Batch,
    ) -> ReplayRolloutState[ModelState]:
        """Build the initial all-halted carry from a sample batch.

        All slots start halted (``halted=True``), so the very first call to
        :meth:`step` will admit every slot from the incoming batch.

        Args:
            batch_sample: Any source batch; only used to infer batch size and device.
                Trajectory arrays are NOT read here — they will be read at the
                first admission boundary in :meth:`step`.
        """
        anchor = batch_anchor_tensor(batch_sample)
        B = int(anchor.shape[0])
        device = anchor.device

        task_state = self._runtime.initial_task_state(batch_sample, device=device)

        return ReplayRolloutState(
            model_state=self.backbone.init_state(B, device=device),
            steps=torch.zeros((B,), dtype=torch.int32, device=device),
            halted=torch.ones((B,), dtype=torch.bool, device=device),
            data={},
            cursor=torch.zeros((B,), dtype=torch.int64, device=device),
            trajectory_length=torch.zeros((B,), dtype=torch.int64, device=device),
            trajectory_id=torch.full((B,), -1, dtype=torch.int64, device=device),
            resident_payload={},
            task_state=task_state,
        )

    def step(
        self,
        state: ReplayRolloutState[ModelState],
        batch: Batch,
        *,
        allow_halt: bool = True,
        **_: Any,
    ) -> tuple[ReplayRolloutState[ModelState], ReplayStepOutput]:
        """Advance the replay controller by one step.

        Active slots read exclusively from ``state.resident_payload``; the
        incoming *batch* is only consulted for newly admitted (halted) slots.

        Args:
            state: Current per-slot carry state.
            batch: Source batch.  Used ONLY for halted slots being admitted.
                Active slots ignore it entirely.
            allow_halt: If False, suppress window-size halting (used during
                validation where full episodes are replayed).
        """
        admission = state.halted  # (B,) — slots being admitted this step

        # ── 1. Admission: update resident payload from batch for admitted slots ──────
        new_resident = _build_resident_payload(state.resident_payload, batch, admission)

        # ── 2. Cursor: admitted slots reset to 0; active slots advance by 1 ─────────
        cursor = torch.where(admission, torch.zeros_like(state.cursor), state.cursor + 1)

        # ── 3. Trajectory length: admitted slots read from batch; active keep carry ──
        new_traj_len = self._runtime.trajectory_lengths(batch).to(device=cursor.device, dtype=torch.int64)
        traj_len = torch.where(admission, new_traj_len, state.trajectory_length)

        # ── 4. Trajectory identity: changes ONLY at halted→admitted boundary ─────────
        if _TRAJECTORY_IDENTITY_KEY in batch:
            batch_traj_id = batch[_TRAJECTORY_IDENTITY_KEY].to(device=cursor.device, dtype=torch.int64)
        else:
            batch_traj_id = state.trajectory_id  # no identity in batch → keep carry
        trajectory_id = torch.where(admission, batch_traj_id, state.trajectory_id)

        # ── 5. Model state: reset admitted slots, continue active slots ───────────────
        model_state = self.backbone.reset_state(admission, state.model_state)

        # ── 6. Extract step data from resident payload (not from batch!) ──────────────
        # Active slots read from carry-owned trajectory arrays; admitted slots also
        # read from resident_payload which was just updated from batch above.
        current_data, new_task_state = self._runtime.extract_step_per_slot(new_resident, cursor, state.task_state)

        # ── 7. Backbone forward pass ──────────────────────────────────────────────────
        backbone_output, model_state = self.backbone(current_data, model_state)

        # ── 8. Advance per-slot step counters ────────────────────────────────────────
        steps = self.advance_steps(state)

        # ── 9. Compute new halted from carry state only ───────────────────────────────
        new_halted: Tensor = cursor >= (traj_len - 1)
        if self.config.window_size is not None and allow_halt:
            new_halted = new_halted | (cursor >= self.config.window_size - 1)

        new_state = ReplayRolloutState(
            model_state=model_state,
            steps=steps,
            halted=new_halted,
            data=current_data,
            cursor=cursor,
            trajectory_length=traj_len,
            trajectory_id=trajectory_id,
            resident_payload=new_resident,
            task_state=new_task_state,
        )
        output = ReplayStepOutput(backbone_output=backbone_output, cursor=cursor)
        return new_state, output


# =============================================================================
__all__ = [
    "ReplayRolloutState",
    "ReplayStepOutput",
    "ReplayTrajectoryController",
    "ReplayTrajectoryControllerConfig",
    "ReplayTrajectoryRuntime",
]
