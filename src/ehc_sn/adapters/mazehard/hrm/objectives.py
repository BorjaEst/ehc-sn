"""MazeHard HRM-family task bindings for ACT and hybrid RL objectives.

The ACT binding reads from ``ACTStepOutput.backbone_output.task``.
The hybrid RL binding reads task-specific fields from
:class:`~ehc_sn.controllers.contracts.actor_critic.ActorCriticInteractionRecord` for
the actor-critic path.

Mirrors the pattern used by
:class:`~ehc_sn.adapters.arena.tem.objectives.ArenaTEMTaskBinding`
for the Arena+TEM family.
"""

from __future__ import annotations

from typing import Any, Protocol, cast

from torch import Tensor

from ehc_sn.controllers.contracts.actor_critic import ActorCriticInteractionRecord
from ehc_sn.objectives._token import AccuracyStats, TokenSupervisionBinding, compute_accuracy_stats
from ehc_sn.tasks.mazehard.contracts import MAZE_HARD_IGNORE_LABEL_ID, MazeHardTargets
from ehc_sn.training.actor_critic import HybridActorCriticTaskBinding
from ehc_sn.types import Batch


# =============================================================================
class _HasTaskLogits(Protocol):
    """Capability protocol for task payloads that expose supervised logits."""

    task_logits: Tensor


class _HasTaskPayload(Protocol):
    """Capability protocol for outputs that carry a task payload."""

    task: _HasTaskLogits


class _HasBackboneTaskPayload(Protocol):
    """Capability protocol for ACT step outputs with task logits."""

    backbone_output: _HasTaskPayload


def _extract_act_task_logits(step_output: _HasBackboneTaskPayload) -> Tensor:
    """Return task logits from any ACT-compatible step output."""
    return step_output.backbone_output.task.task_logits


def _extract_record_task_logits(record: ActorCriticInteractionRecord) -> Tensor:
    """Return task logits from any actor-critic record with task payload."""
    task_output = cast(_HasTaskLogits | None, record.task_output)
    if task_output is None:
        raise RuntimeError(
            "MazeHardHRMV2HybridTaskBinding: task_output is None. " "The controller must attach a task payload with task_logits."
        )
    return task_output.task_logits


# =============================================================================
class MazeHardHRMV1ACTTaskBinding(TokenSupervisionBinding[MazeHardTargets]):
    """ACT task binding for MazeHard token prediction via the HRM family.

    Extracts supervised logits from ``step_output.backbone_output.task.task_logits``
    and constructs targets from the canonical MazeHard batch ``"labels"`` key.
    """

    def extract_logits(self, batch: Batch, carry: Any, step_output: Any) -> Tensor:
        """Return token logits from the MazeHard task payload."""
        _ = batch, carry
        return _extract_act_task_logits(cast(_HasBackboneTaskPayload, step_output))

    def extract_targets(self, batch: Batch, carry: Any, step_output: Any) -> MazeHardTargets:
        """Return MazeHard token supervision targets from the carry buffer."""
        _ = batch, step_output
        return MazeHardTargets(labels=carry.data["labels"])

    def evaluate_sequences(self, logits: Tensor, targets: MazeHardTargets) -> AccuracyStats:
        """Return masked sequence-correctness statistics for MazeHard token prediction."""
        return compute_accuracy_stats(logits, targets.labels, ignore_label_id=MAZE_HARD_IGNORE_LABEL_ID)


# =============================================================================
class MazeHardHRMV2HybridTaskBinding:
    """MazeHard-specific extraction for the hybrid RL actor-critic batch path.

    Implements :class:`~ehc_sn.training.actor_critic.HybridActorCriticTaskBinding`
    for the HRM v2 + MazeHard pairing.  Extracts token logits from the
    task output on the interaction record and supervision labels from the
    observation dict used for the decision.

    Injected into :class:`~ehc_sn.training.actor_critic.TD0ActorCriticBatchBuilder`
    and :class:`~ehc_sn.training.actor_critic.ZeroBootstrapActorCriticValidationScorer`
    at wiring time in the Lightning module.
    """

    def extract_task_logits(self, record: ActorCriticInteractionRecord) -> Tensor:
        """Return token-prediction logits from ``record.task_output.task_logits``."""
        return _extract_record_task_logits(record)

    def extract_labels(self, record: ActorCriticInteractionRecord) -> Tensor:
        """Return supervision labels from ``record.observation_used_for_decision``."""
        if "labels" not in record.observation_used_for_decision:
            raise RuntimeError(
                "MazeHardHRMV2HybridTaskBinding: 'labels' key missing from "
                "observation_used_for_decision. The batch must include supervision labels."
            )
        return record.observation_used_for_decision["labels"]


# make the type-checker confirm the protocol is satisfied
_: HybridActorCriticTaskBinding = MazeHardHRMV2HybridTaskBinding()


# =============================================================================
__all__ = [
    "MazeHardHRMV1ACTTaskBinding",
    "MazeHardHRMV2HybridTaskBinding",
]
