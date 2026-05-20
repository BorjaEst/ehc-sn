"""MazeHard HRM-family task bindings for ACT and hybrid RL objectives.

The ACT binding reads from the objective-owned ACT step output protocol
(``ACTStepOutput.task``).
The hybrid RL binding reads task-specific fields from
:class:`~ehc_sn.controllers.contracts.value_control.ValueControlInteractionRecord` for
the value-control path.

Mirrors the pattern used by
:class:`~ehc_sn.adapters.arena.tem.objectives.ArenaTEMTaskBinding`
for the Arena+TEM family.
"""

from __future__ import annotations

from typing import Any, Protocol, cast

import torch
from torch import Tensor

from ehc_sn.controllers.contracts.value_control import (
    ValueControlInteractionRecord,
)
from ehc_sn.objectives._token import AccuracyStats, compute_accuracy_stats
from ehc_sn.objectives.act import ACTObjectiveBinding
from ehc_sn.objectives.hybrid_rl import HybridValueTaskBinding
from ehc_sn.rollouts.runtime import CarrySnapshot
from ehc_sn.tasks.mazehard.contracts import (
    MAZE_HARD_IGNORE_LABEL_ID,
    MazeHardTargets,
)
from ehc_sn.tasks.mazehard.runtime import PATH_ID
from ehc_sn.types import Batch


# =============================================================================
class _HasTaskLogits(Protocol):
    """Capability protocol for task payloads that expose supervised logits."""

    task_logits: Tensor


class _HasTaskPayload(Protocol):
    """Capability protocol for ACT step outputs with task logits."""

    task: _HasTaskLogits


# =============================================================================
class MazeHardHRMV1ACTTaskBinding(ACTObjectiveBinding[MazeHardTargets]):
    """ACT task binding for MazeHard token prediction via the HRM family.

    Extracts supervised logits from ``step_output.task.task_logits``
    and constructs targets from the canonical MazeHard batch ``"labels"`` key.
    """

    def extract_logits(  # ----------------------------------------------------
        self,
        executed_batch: Batch,
        snapshot: CarrySnapshot,
        step_output: Any,
    ) -> Tensor:
        """Return token logits from the MazeHard task payload."""
        _ = executed_batch, snapshot
        return _extract_act_task_logits(cast(_HasTaskPayload, step_output))

    def extract_targets(  # ---------------------------------------------------
        self,
        executed_batch: Batch,
        snapshot: CarrySnapshot,
        step_output: Any,
    ) -> MazeHardTargets:
        """Return MazeHard token supervision targets from the executed batch."""
        _ = snapshot, step_output
        if "labels" not in executed_batch:
            raise RuntimeError(
                "MazeHardHRMV1ACTTaskBinding: 'labels' missing from executed batch. "
                "Ensure StepRecord.executed_frame (record.batch) includes labels."
            )
        return MazeHardTargets(labels=executed_batch["labels"])

    def evaluate_sequences(  # ------------------------------------------------
        self, logits: Tensor, targets: MazeHardTargets
    ) -> AccuracyStats:
        """Return masked sequence-correctness statistics for MazeHard token prediction."""
        return compute_accuracy_stats(
            logits, targets.labels, ignore_label_id=MAZE_HARD_IGNORE_LABEL_ID
        )

    def build_token_weights(self, labels: Tensor) -> Tensor:
        """Return per-token Token weights that emphasize MazeHard PATH labels."""
        return _build_mazehard_token_weights(labels)


# =============================================================================
class MazeHardHRMV2HybridTaskBinding:
    """MazeHard-specific extraction for the hybrid RL value-control batch path.

    Implements :class:`~ehc_sn.objectives.hybrid_rl.HybridValueTaskBinding`
    for the HRM v2 + MazeHard pairing.  Extracts token logits from the
    task output on the interaction record and supervision labels from the
    observation dict used for the decision.

    Injected into :class:`~ehc_sn.training.actor_critic.TD0ActorCriticBatchBuilder`
    and :class:`~ehc_sn.training.actor_critic.ZeroBootstrapActorCriticValidationScorer`
    at wiring time in the Lightning module.
    """

    def extract_task_logits(  # -----------------------------------------------
        self, record: ValueControlInteractionRecord
    ) -> Tensor:
        """Return token-prediction logits from ``record.task_output.task_logits``."""
        return _extract_record_task_logits(record)

    def extract_labels(  # -------------------------------------------------------------
        self,
        record: ValueControlInteractionRecord,
    ) -> Tensor:
        """Return supervision labels from ``record.observation_used_for_decision``."""
        if "labels" not in record.observation_used_for_decision:
            raise RuntimeError(
                "MazeHardHRMV2HybridTaskBinding: 'labels' key missing from "
                "observation_used_for_decision. The batch must include supervision labels."
            )
        return record.observation_used_for_decision["labels"]

    def extract_token_weights(
        self, record: ValueControlInteractionRecord
    ) -> Tensor:
        """Return per-token Token weights that emphasize MazeHard PATH labels."""
        labels = self.extract_labels(record)
        return _build_mazehard_token_weights(labels)


# make the type-checker confirm the protocol is satisfied
_: HybridValueTaskBinding = MazeHardHRMV2HybridTaskBinding()


# =============================================================================
def _extract_act_task_logits(  # ----------------------------------------------
    step_output: _HasTaskPayload,
) -> Tensor:
    """Return task logits from any ACT-compatible step output."""
    return step_output.task.task_logits


# =============================================================================
def _extract_record_task_logits(  # -------------------------------------------
    record: ValueControlInteractionRecord,
) -> Tensor:
    """Return task logits from any value-control record with task payload."""
    task_output = cast(_HasTaskLogits | None, record.task_output)
    if task_output is None:
        raise RuntimeError(
            "MazeHardHRMV2HybridTaskBinding: task_output is None. "
            "The controller must attach a task payload with task_logits."
        )
    return task_output.task_logits


# =============================================================================
def _build_mazehard_token_weights(  # -----------------------------------------
    labels: Tensor,
) -> Tensor:
    """Build per-token weights that upweight PATH labels during Token loss."""
    weights = torch.ones_like(labels, dtype=torch.float32)
    weights = torch.where(
        labels == PATH_ID,
        torch.full_like(weights, 2.0),
        weights,
    )
    weights = torch.where(
        labels == MAZE_HARD_IGNORE_LABEL_ID,
        torch.zeros_like(weights),
        weights,
    )
    return weights


# =============================================================================
__all__ = [
    "MazeHardHRMV1ACTTaskBinding",
    "MazeHardHRMV2HybridTaskBinding",
]
