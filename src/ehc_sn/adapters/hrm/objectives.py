"""MazeHard HRM-family task bindings for ACT and hybrid RL objectives.

The ACT binding reads from the objective-owned ACT step output protocol
(``ACTStepOutput.task``).
The hybrid RL binding reads task-specific fields from
:class:`~ehp_sn.controllers.contracts.actor_critic.QHaltingInteractionRecord` for
the value-control path.

Mirrors the pattern used by
:class:`~ehp_sn.adapters.arena.tem.objectives.ArenaTEMTaskBinding`
for the Arena+TEM family.
"""

from __future__ import annotations

from typing import Any, Protocol, cast

import torch
from torch import Tensor

from ehc_sn.controllers.contracts.actor_critic import (
    QHaltingInteractionRecord,
)
from ehc_sn.objectives._token import AccuracyStats, compute_accuracy_stats
from ehc_sn.objectives.act import ACTObjectiveBinding
from ehc_sn.objectives.continuous_field import ContinuousFieldObjectiveBinding
from ehc_sn.objectives.hybrid_rl import HybridValueObjectiveBinding
from ehc_sn.rollouts.runtime import CarrySnapshot
from ehc_sn.tasks.mazehard.contracts import (
    MAZE_HARD_IGNORE_LABEL_ID,
    MazeHardTargets,
)
from ehc_sn.tasks.mazehard.runtime import PATH_ID
from ehc_sn.tasks.seqmaze.contracts import (
    SEQMAZE_IGNORE_LABEL_ID,
    SeqMazeTargets,
)
from ehc_sn.tasks.seqmaze.evaluation import _canonicalize
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
        self,
        logits: Tensor,
        targets: MazeHardTargets,
    ) -> AccuracyStats:
        """Return masked sequence-correctness statistics for MazeHard token prediction."""
        return compute_accuracy_stats(
            logits, targets.labels, ignore_label_id=MAZE_HARD_IGNORE_LABEL_ID
        )

    def extract_loss_labels(  # -----------------------------------------------
        self,
        targets: MazeHardTargets,
    ) -> Tensor:
        """Return MazeHard labels directly — already loss-compatible."""
        return targets.labels

    def build_token_weights(  # -----------------------------------------------
        self,
        labels: Tensor,
    ) -> Tensor:
        """Return per-token Token weights that emphasize MazeHard PATH labels."""
        return _build_mazehard_token_weights(labels)


# =============================================================================
class MazeHardHRMV2HybridTaskBinding:
    """MazeHard-specific extraction for the hybrid RL value-control batch path.

    Implements :class:`~ehp_sn.objectives.hybrid_rl.HybridValueObjectiveBinding`
    for the HRM v2 + MazeHard pairing.  Extracts token logits from the
    task output on the interaction record and supervision labels from the
    observation dict used for the decision.

    Injected into :class:`~ehp_sn.training.q_halting.TD0ActorCriticBatchBuilder`
    and :class:`~ehp_sn.training.q_halting.ZeroBootstrapActorCriticValidationScorer`
    at wiring time in the Lightning module.
    """

    def extract_task_logits(  # -----------------------------------------------
        self,
        record: QHaltingInteractionRecord,
    ) -> Tensor:
        """Return token-prediction logits from ``record.task_output.task_logits``."""
        return _extract_record_task_logits(record)

    def extract_labels(  # ----------------------------------------------------
        self,
        record: QHaltingInteractionRecord,
    ) -> Tensor:
        """Return supervision labels from ``record.observation_used_for_decision``."""
        if "labels" not in record.observation_used_for_decision:
            raise RuntimeError(
                "MazeHardHRMV2HybridTaskBinding: 'labels' key missing from "
                "observation_used_for_decision. The batch must include supervision labels."
            )
        return record.observation_used_for_decision["labels"]

    def extract_token_weights(  # ---------------------------------------------
        self,
        record: QHaltingInteractionRecord,
    ) -> Tensor:
        """Return per-token Token weights that emphasize MazeHard PATH labels."""
        labels = self.extract_labels(record)
        return _build_mazehard_token_weights(labels)


# make the type-checker confirm the protocol is satisfied
_: HybridValueObjectiveBinding = MazeHardHRMV2HybridTaskBinding()


# =============================================================================
def _extract_act_task_logits(  # ----------------------------------------------
    step_output: _HasTaskPayload,
) -> Tensor:
    """Return task logits from any ACT-compatible step output."""
    return step_output.task.task_logits


# =============================================================================
def _extract_record_task_logits(  # -------------------------------------------
    record: QHaltingInteractionRecord,
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
class _HasSeqMazeTaskPayload(Protocol):
    """Capability protocol for task payloads that expose path logits."""

    path_logits: Tensor


# =============================================================================
class _HasSeqMazeStepOutput(Protocol):
    """Capability protocol for ACT step outputs with seqmaze task logits."""

    task: _HasSeqMazeTaskPayload


# =============================================================================
def _extract_seqmaze_act_logits(  # -------------------------------------------
    step_output: _HasSeqMazeStepOutput,
) -> Tensor:
    """Return path logits from a seqmaze ACT-compatible step output."""
    return step_output.task.path_logits


# =============================================================================
class SeqMazeHRMV1ACTTaskBinding(ACTObjectiveBinding[SeqMazeTargets]):
    """ACT task binding for SeqMaze path prediction via the HRM v1 family.

    Extracts constrained path logits from ``step_output.task.path_logits``,
    constructs targets from canonical SeqMaze batch fields, and evaluates
    sequence correctness after EOS canonicalization.
    """

    def __init__(  # ----------------------------------------------------------
        self,
        n_max: int = 32,
    ) -> None:
        self._eos_id = n_max
        self._pad_id = n_max + 1
        self._n_max = n_max

    def extract_logits(  # ----------------------------------------------------
        self,
        executed_batch: Batch,
        snapshot: CarrySnapshot,
        step_output: Any,
    ) -> Tensor:
        """Return constrained path logits from the seqmaze task payload."""
        _ = executed_batch, snapshot
        return _extract_seqmaze_act_logits(
            cast(_HasSeqMazeStepOutput, step_output)
        )

    def extract_targets(  # ---------------------------------------------------
        self,
        executed_batch: Batch,
        snapshot: CarrySnapshot,
        step_output: Any,
    ) -> SeqMazeTargets:
        """Return SeqMaze path supervision targets from the executed batch."""
        _ = snapshot, step_output
        for key in ("target_path", "path_mask", "path_length"):
            if key not in executed_batch:
                raise RuntimeError(
                    f"SeqMazeHRMV1ACTTaskBinding: '{key}' missing from executed batch. "
                    "Ensure StepRecord.executed_frame (record.batch) includes it."
                )
        return SeqMazeTargets(
            path_index=executed_batch["target_path"].to(dtype=torch.int64),
            path_mask=executed_batch["path_mask"].to(dtype=torch.bool),
            path_length=executed_batch["path_length"].to(dtype=torch.int64),
        )

    def evaluate_sequences(  # ------------------------------------------------
        self,
        logits: Tensor,
        targets: SeqMazeTargets,
    ) -> AccuracyStats:
        """Evaluate sequence correctness after EOS canonicalization.

        Canonicalizes predictions (truncates after first EOS), then compares
        against the canonical target under the target's path_mask.
        """
        pred = logits.argmax(dim=-1)  # (B, T)
        pred = _canonicalize(pred, self._eos_id, self._pad_id)

        mask = targets.path_mask  # (B, T)
        is_correct = mask & pred.eq(targets.path_index)

        return AccuracyStats(mask=mask, is_correct=is_correct)

    def extract_loss_labels(  # -----------------------------------------------
        self,
        targets: SeqMazeTargets,
    ) -> Tensor:
        """Convert structured target into loss-compatible label tensor.

        PAD positions (path_mask=False) are replaced with the ignore label
        so the loss function skips them.
        """
        return torch.where(
            targets.path_mask,
            targets.path_index,
            torch.full_like(
                targets.path_index, SEQMAZE_IGNORE_LABEL_ID, dtype=torch.int64
            ),
        )


# =============================================================================
class _HasSeqMazePathLogits(Protocol):
    """Capability protocol for task payloads that expose path logits."""

    path_logits: Tensor


# =============================================================================
class SeqMazeHRMV2HybridTaskBinding:
    """SeqMaze-specific extraction for the hybrid RL value-control batch path.

    Implements :class:`~ehp_sn.objectives.hybrid_rl.HybridValueObjectiveBinding`
    for the HRM v2 + SeqMaze pairing.  Extracts path logits from the
    task output on the interaction record and supervision labels from the
    observation dict used for the decision.

    Injected into :class:`~ehp_sn.training.q_halting.TD0ActorCriticBatchBuilder`
    and :class:`~ehp_sn.training.q_halting.ZeroBootstrapActorCriticValidationScorer`
    at wiring time in the Lightning module.
    """

    def __init__(self, n_max: int = 32) -> None:
        self._n_max = n_max
        self._pad_id = n_max + 1

    def extract_task_logits(
        self,
        record: QHaltingInteractionRecord,
    ) -> Tensor:
        """Return path-prediction logits from ``record.task_output.path_logits``."""
        task_output = record.task_output
        if task_output is None:
            raise RuntimeError(
                "SeqMazeHRMV2HybridTaskBinding: task_output is None. "
                "The controller must attach a task payload with path_logits."
            )
        if not hasattr(task_output, "path_logits"):
            raise RuntimeError(
                f"SeqMazeHRMV2HybridTaskBinding: task_output has no path_logits. "
                f"Got {type(task_output).__name__}."
            )
        return cast(_HasSeqMazePathLogits, task_output).path_logits

    def extract_labels(
        self,
        record: QHaltingInteractionRecord,
    ) -> Tensor:
        """Return supervision labels from ``record.observation_used_for_decision``.

        Converts PAD positions (N_max+1) to ``SEQMAZE_IGNORE_LABEL_ID`` so
        the cross-entropy loss skips them.
        """
        obs = record.observation_used_for_decision
        if "target_path" not in obs:
            raise RuntimeError(
                "SeqMazeHRMV2HybridTaskBinding: 'target_path' missing from "
                "observation_used_for_decision."
            )
        labels = obs["target_path"].to(dtype=torch.int64)
        # Replace PAD (n_max+1) with ignore label
        labels = torch.where(
            labels == self._pad_id,
            torch.full_like(labels, SEQMAZE_IGNORE_LABEL_ID),
            labels,
        )
        return labels

    def extract_token_weights(  # ------------------------------------------
        self,
        record: QHaltingInteractionRecord,
    ) -> Tensor:
        """Return per-token loss weights for SeqMaze hybrid RL.

        Uniform-1.0 weights for all non-ignore positions, zero for
        ``SEQMAZE_IGNORE_LABEL_ID`` positions.  Unlike MazeHard, SeqMaze
        does not upweight any token class.
        """
        labels = self.extract_labels(record)
        return torch.where(
            labels == SEQMAZE_IGNORE_LABEL_ID,
            torch.zeros_like(labels, dtype=torch.float32),
            torch.ones_like(labels, dtype=torch.float32),
        )


# =============================================================================
class GoaltraceHRMV1ACTTaskBinding(ContinuousFieldObjectiveBinding):
    """ACT task binding for Goaltrace field prediction via HRM v1.

    Implements :class:`~ehp_sn.objectives.continuous_field.ContinuousFieldObjectiveBinding`
    by reading ``step_output.task.firing_field`` and target data from the
    executed batch.

    The ACT ``extract_logits`` / ``extract_targets`` / ``evaluate_sequences`` /
    ``extract_loss_labels`` surface is not used by the goaltrace field
    objective and is not implemented here.
    """

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # ContinuousFieldObjectiveBinding implementation
    # ------------------------------------------------------------------

    def extract_field(
        self,
        executed_batch: Batch,
        snapshot: CarrySnapshot,
        step_output: Any,
    ) -> Tensor:
        """Return the predicted firing field from the step output."""
        _ = executed_batch, snapshot
        task = getattr(step_output, "task", None)
        if task is None:
            raise RuntimeError(
                "GoaltraceHRMV1ACTTaskBinding: step_output has no .task "
                "attribute."
            )
        field = getattr(task, "firing_field", None)
        if field is None:
            raise RuntimeError(
                "GoaltraceHRMV1ACTTaskBinding: step_output.task has no "
                ".firing_field attribute."
            )
        return field

    def extract_target_field(
        self,
        executed_batch: Batch,
        snapshot: CarrySnapshot,
        step_output: Any,
    ) -> Tensor:
        """Return the target firing field from the executed batch."""
        _ = snapshot, step_output
        if "target_field" not in executed_batch:
            raise RuntimeError(
                "GoaltraceHRMV1ACTTaskBinding: 'target_field' missing from "
                "executed_batch."
            )
        return executed_batch["target_field"].to(dtype=torch.float32)

    def extract_node_mask(
        self,
        executed_batch: Batch,
    ) -> Tensor:
        """Return the valid-node mask from the executed batch."""
        if "node_mask" not in executed_batch:
            raise RuntimeError(
                "GoaltraceHRMV1ACTTaskBinding: 'node_mask' missing from "
                "executed_batch."
            )
        return executed_batch["node_mask"].to(dtype=torch.bool)

    def compute_field_quality(
        self,
        pred_field: Tensor,
        target_field: Tensor,
        node_mask: Tensor,
        threshold: float,
    ) -> Tensor:
        """Return per-sample field quality flag for Q(done) supervision.

        A sample is considered 'done-well' if its masked field MSE is
        below *threshold*.
        """
        diff = pred_field - target_field
        squared = diff**2
        valid_count = node_mask.sum(dim=1).clamp(min=1)
        per_sample_mse = (squared * node_mask).sum(dim=1) / valid_count
        return per_sample_mse < threshold

    # ------------------------------------------------------------------
    # ACTObjectiveBinding legacy surface (unused by goaltrace objective)
    # ------------------------------------------------------------------

    def extract_logits(
        self,
        executed_batch: Batch,
        snapshot: CarrySnapshot,
        step_output: Any,
    ) -> Tensor:
        """Not used by GoaltraceFieldObjective.  Raises NotImplementedError."""
        _ = executed_batch, snapshot, step_output
        raise NotImplementedError(
            "GoaltraceHRMV1ACTTaskBinding.extract_logits is unused by "
            "GoaltraceFieldObjective.  Use extract_field instead."
        )

    def extract_targets(
        self,
        executed_batch: Batch,
        snapshot: CarrySnapshot,
        step_output: Any,
    ) -> Tensor:
        """Not used by GoaltraceFieldObjective.  Raises NotImplementedError."""
        _ = executed_batch, snapshot, step_output
        raise NotImplementedError(
            "GoaltraceHRMV1ACTTaskBinding.extract_targets is unused by "
            "GoaltraceFieldObjective.  Use extract_target_field instead."
        )

    def evaluate_sequences(
        self,
        logits: Tensor,
        targets: Any,
    ) -> AccuracyStats:
        """Not used by GoaltraceFieldObjective.  Raises NotImplementedError."""
        _ = logits, targets
        raise NotImplementedError(
            "GoaltraceHRMV1ACTTaskBinding.evaluate_sequences is unused by "
            "GoaltraceFieldObjective."
        )

    def extract_loss_labels(
        self,
        targets: Any,
    ) -> Tensor:
        """Not used by GoaltraceFieldObjective.  Raises NotImplementedError."""
        _ = targets
        raise NotImplementedError(
            "GoaltraceHRMV1ACTTaskBinding.extract_loss_labels is unused by "
            "GoaltraceFieldObjective."
        )


# =============================================================================
__all__ = [
    "GoaltraceHRMV1ACTTaskBinding",
    "MazeHardHRMV1ACTTaskBinding",
    "MazeHardHRMV2HybridTaskBinding",
    "SeqMazeHRMV1ACTTaskBinding",
    "SeqMazeHRMV2HybridTaskBinding",
]
