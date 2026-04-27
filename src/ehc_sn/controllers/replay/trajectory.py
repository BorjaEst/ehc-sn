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


# =============================================================================
class ReplayTrajectoryRuntime(Protocol):
    """Task-owned per-step extraction interface for replay trajectory controllers."""

    def extract_step_per_slot(
        self,
        batch: Batch,
        cursor: Tensor,
        task_state: dict[str, Tensor],
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Extract current-step tensors and return the updated task state.

        Args:
            batch: Current replay batch.
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
        """Return the per-slot effective trajectory length."""
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
    """Controller carry for replay trajectory controllers."""

    cursor: Tensor  # (B,) int64 — current step index per slot
    trajectory_length: Tensor  # (B,) int64 — effective trajectory length per slot
    task_state: dict[str, Tensor] = field(default_factory=dict)
    """Task-owned replay-local state (e.g. visit counts) threaded explicitly through carry."""


# =============================================================================
@dataclass(frozen=True)
class ReplayStepOutput(DetachMixin):
    """Output produced by one replay controller step."""

    backbone_output: Any
    cursor: Tensor  # (B,) int64


# =============================================================================
class ReplayTrajectoryController[ModelState](BaseController[ModelState, ReplayTrajectoryControllerConfig]):
    """Generic stepwise replay controller for arena-family recurrent replay."""

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
        """Build the initial carry from the first source batch."""
        anchor = batch_anchor_tensor(batch_sample)
        B = int(anchor.shape[0])
        device = anchor.device

        traj_len = self._runtime.trajectory_lengths(batch_sample).to(device=device, dtype=torch.int64)
        task_state = self._runtime.initial_task_state(batch_sample, device=device)

        return ReplayRolloutState(
            model_state=self.backbone.init_state(B),
            steps=torch.zeros((B,), dtype=torch.int32, device=device),
            halted=torch.ones((B,), dtype=torch.bool, device=device),
            data={},
            cursor=torch.zeros((B,), dtype=torch.int64, device=device),
            trajectory_length=traj_len,
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
        """Advance the replay controller by one step."""
        cursor = torch.where(state.halted, torch.zeros_like(state.cursor), state.cursor + 1)

        new_traj_len = self._runtime.trajectory_lengths(batch).to(device=cursor.device, dtype=torch.int64)
        traj_len = torch.where(state.halted, new_traj_len, state.trajectory_length)

        model_state = self.backbone.reset_state(state.halted, state.model_state)
        current_data, new_task_state = self._runtime.extract_step_per_slot(batch, cursor, state.task_state)
        backbone_output, model_state = self.backbone(current_data, model_state)

        steps = self.advance_steps(state)

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
