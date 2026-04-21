"""Replay trajectory controller for stepwise recurrent arena replay.

The replay controller advances a cursor through source-provided batch-major
trajectory tensors one step at a time.  It is generic and task-agnostic: all
step-data extraction logic lives in the injected
:class:`ReplayTrajectoryRuntime`.

Carry safety:
- :class:`ReplayRolloutState` stores only recurrent state, per-slot cursor,
  per-slot trajectory length, halted mask, step counter, and the current-step
  payload.
- Full ``(B, T, ...)`` trajectory tensors are never stored in carry or carry
  data; they live entirely in the source batch.

Works with :class:`~ehc_sn.rollouts.sources.RepeatSource` (single-batch
full-episode replay) and
:class:`~ehc_sn.rollouts.sources.PartialResetSource` (multi-episode recurrent
training with variable trajectory lengths).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol

import torch
from pydantic import BaseModel, Field
from torch import Tensor

from ehc_sn.controllers._base import BaseController, RolloutBackbone, RolloutState, batch_anchor_tensor
from ehc_sn.types import Batch
from ehc_sn.utils.detach import DetachMixin


# =============================================================================
class ReplayTrajectoryRuntime(Protocol):
    """Task-owned per-step extraction interface for replay trajectory controllers.

    Implementations must be stateless helpers: all trajectory state lives in
    the source batch.  The controller provides the current per-slot cursor to
    index into the batch.
    """

    def extract_step_per_slot(self, batch: Batch, cursor: Tensor) -> dict[str, Tensor]:
        """Extract current-step tensors independently for each batch slot.

        Args:
            batch: Full replay batch with trajectory axes of shape ``(B, T, ...)``.
            cursor: Per-slot current step indices, shape ``(B,)`` int64.

        Returns:
            Dict of current-step tensors with shape ``(B, ...)``.  Must not
            contain full ``(B, T, ...)`` tensors.
        """
        ...

    def trajectory_lengths(self, batch: Batch) -> Tensor:
        """Return the per-slot effective trajectory length.

        Args:
            batch: Full replay batch containing trajectory length metadata.

        Returns:
            Tensor of shape ``(B,)`` int64 with per-slot valid step counts.
        """
        ...


# =============================================================================
class ReplayTrajectoryControllerConfig(BaseModel, extra="forbid"):
    """Configuration for :class:`ReplayTrajectoryController`.

    Attributes:
        window_size: Optional step cap per replay window.  ``None`` means
            replay the full trajectory until ``trajectory_length`` is reached.
            Set to a positive integer to use fixed contiguous windows, e.g.,
            for truncated BPTT.
    """

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

    Carry invariants:
    - ``data`` contains the **current-step payload** extracted from the source
      batch at ``cursor``.  It does NOT contain full ``(B, T, ...)`` trajectory
      tensors.
    - ``cursor`` is the 0-indexed position of the step currently being
      processed.  It is reset to 0 for halted slots on the next ``step`` call.
    - ``trajectory_length`` is the per-slot effective trajectory length.  It is
      updated from the source batch when a slot is halted and receives a fresh
      trajectory.
    - ``halted`` is ``True`` when the slot has processed its last valid step
      and needs a fresh trajectory on the next call.

    Attributes:
        cursor: Per-slot current step index, shape ``(B,)`` int64.
        trajectory_length: Per-slot effective trajectory length, shape
            ``(B,)`` int64.
    """

    cursor: Tensor  # (B,) int64 — current step index per slot
    trajectory_length: Tensor  # (B,) int64 — effective trajectory length per slot


# =============================================================================
@dataclass(frozen=True)
class ReplayStepOutput(DetachMixin):
    """Output produced by one replay controller step.

    Attributes:
        backbone_output: Model output for the current step.
        cursor: Per-slot cursor positions after this step, shape ``(B,)``
            int64.  Consumers can use this to index back into the source batch
            when computing objectives or traces.
    """

    backbone_output: Any
    cursor: Tensor  # (B,) int64


# =============================================================================
class ReplayTrajectoryController[ModelState](BaseController[ModelState, ReplayTrajectoryControllerConfig]):
    """Generic stepwise replay controller for arena-family recurrent replay.

    Steps a backbone through source-provided replay trajectories one position
    at a time.  Halts per-slot when the cursor reaches the end of the
    trajectory (or the configured window limit).

    The controller is task-agnostic: step-data extraction is delegated to
    the injected :class:`ReplayTrajectoryRuntime`.  Carry is safe: no full
    ``(B, T, ...)`` trajectory tensors are stored.

    Cursor semantics:
    - After ``initial_state``: all slots halted; cursor = 0.
    - First ``step`` call: halted=True resets cursor to 0; processes step 0.
    - Subsequent calls: cursor advances by 1 for active slots.
    - Halt when ``cursor >= trajectory_length - 1`` (current step is the last
      valid step; next would be out of bounds).  With ``window_size``, also
      halt when ``cursor >= window_size - 1``.
    """

    def __init__(  # -----------------------------------------------------------------------
        self,
        backbone: RolloutBackbone[ModelState],
        config: ReplayTrajectoryControllerConfig,
        runtime: ReplayTrajectoryRuntime,
    ) -> None:  # fmt: skip
        """Create a replay trajectory controller.

        Args:
            backbone: Model implementing the :class:`RolloutBackbone` protocol.
            config: Controller configuration.
            runtime: Task-owned step-extraction and trajectory-length runtime.
        """
        super().__init__(backbone=backbone, config=config)
        self._runtime = runtime

    @property
    def runtime(self) -> ReplayTrajectoryRuntime:
        """Return the injected task-owned replay runtime."""
        return self._runtime

    def initial_state(  # ------------------------------------------------------------------
        self, batch_sample: Batch,
    ) -> ReplayRolloutState[ModelState]:  # fmt: skip
        """Build the initial carry from the first source batch.

        All slots start as halted so the first ``step`` call will process step 0
        for every slot.

        Args:
            batch_sample: First batch from the replay source.

        Returns:
            :class:`ReplayRolloutState` with all slots halted, cursor=0, and
            trajectory lengths extracted from the batch.
        """
        anchor = batch_anchor_tensor(batch_sample)
        B = int(anchor.shape[0])
        device = anchor.device

        traj_len = self._runtime.trajectory_lengths(batch_sample).to(device=device, dtype=torch.int64)

        return ReplayRolloutState(
            model_state=self.backbone.init_state(B),
            steps=torch.zeros((B,), dtype=torch.int32, device=device),
            halted=torch.ones((B,), dtype=torch.bool, device=device),
            data={},  # filled in first step call
            cursor=torch.zeros((B,), dtype=torch.int64, device=device),
            trajectory_length=traj_len,
        )

    def step(  # ---------------------------------------------------------------------------
        self,
        state: ReplayRolloutState[ModelState],
        batch: Batch,
        *,
        allow_halt: bool = True,
        **_: Any,
    ) -> tuple[ReplayRolloutState[ModelState], ReplayStepOutput]:  # fmt: skip
        """Advance the replay controller by one step.

        For halted slots, the cursor resets to 0 and trajectory length is
        refreshed from the incoming batch.  For active slots, the cursor
        advances by 1.  The backbone is run at the new cursor position and the
        slot is marked halted if the cursor has reached the last valid position.

        Args:
            state: Current carry.
            batch: Current source batch (may contain fresh trajectories for
                halted slots when using :class:`PartialResetSource`).
            allow_halt: When ``False``, the controller-side halt condition
                (window / trajectory end) is suppressed.  Used for burn-in runs
                that must not emit early halts.

        Returns:
            ``(new_state, output)``.
        """
        # --- Advance cursor: reset for halted slots, increment for active. ---
        cursor = torch.where(state.halted, torch.zeros_like(state.cursor), state.cursor + 1)

        # --- Refresh trajectory lengths for newly-reset slots. ---------------
        new_traj_len = self._runtime.trajectory_lengths(batch).to(device=cursor.device, dtype=torch.int64)
        traj_len = torch.where(state.halted, new_traj_len, state.trajectory_length)

        # --- Reset backbone state for halted slots. --------------------------
        model_state = self.backbone.reset_state(state.halted, state.model_state)

        # --- Extract current-step data at per-slot cursor positions. ---------
        current_data = self._runtime.extract_step_per_slot(batch, cursor)

        # --- Forward pass. ---------------------------------------------------
        backbone_output, model_state = self.backbone(current_data, model_state)

        # --- Advance step counter (resets for halted, increments for active). -
        steps = self.advance_steps(state)

        # --- Compute halt condition. -----------------------------------------
        # Halt when the current step is the last valid one (next would be OOB).
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
