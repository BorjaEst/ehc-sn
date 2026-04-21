"""MazeHard HRM-family task bindings for ACT and hybrid RL objectives.

These bindings live here — in the shared MazeHard+HRM bridge namespace — because
they read from HRM-family-specific output surfaces (``backbone_output.task``,
``backbone_output.policy``, ``backbone_output.critic``), making them HRM-specific
model-task bindings rather than generic MazeHard adapter logic.

Mirrors the pattern used by
:class:`~ehc_sn.adapters.arena.bridges.tem.objectives.ArenaTEMTaskBinding`
for the Arena+TEM family.
"""

from __future__ import annotations

from typing import Any

from torch import Tensor

from ehc_sn.objectives._token import AccuracyStats, TokenSupervisionBinding, compute_accuracy_stats
from ehc_sn.objectives.hybrid_rl import HybridRLObjectiveBinding
from ehc_sn.tasks.maze_hard.contracts import MAZE_HARD_IGNORE_LABEL_ID, MazeHardTargets
from ehc_sn.types import Batch


# =============================================================================
class MazeHardHRMACTTaskBinding(TokenSupervisionBinding[MazeHardTargets]):
    """ACT task binding for MazeHard token prediction via the HRM family.

    Extracts supervised logits from ``step_output.backbone_output.task.task_logits``
    and constructs targets from the canonical MazeHard batch ``"labels"`` key.
    """

    def extract_logits(self, batch: Batch, carry: Any, step_output: Any) -> Tensor:
        """Return token logits from the MazeHard task payload."""
        _ = batch, carry
        return step_output.backbone_output.task.task_logits

    def extract_targets(self, batch: Batch, carry: Any, step_output: Any) -> MazeHardTargets:
        """Return MazeHard token supervision targets from the carry buffer."""
        _ = batch, step_output
        return MazeHardTargets(labels=carry.data["labels"])

    def evaluate_sequences(self, logits: Tensor, targets: MazeHardTargets) -> AccuracyStats:
        """Return masked sequence-correctness statistics for MazeHard token prediction."""
        return compute_accuracy_stats(logits, targets.labels, ignore_label_id=MAZE_HARD_IGNORE_LABEL_ID)


# =============================================================================
class MazeHardHRMHybridRLTaskBinding(HybridRLObjectiveBinding[MazeHardTargets]):
    """Hybrid RL task binding for MazeHard: token prediction plus actor-critic readouts.

    Used by the ``maze_hard`` HRM v2 training path via
    :class:`~ehc_sn.objectives.hybrid_rl.HybridRLLossHead`.
    """

    def extract_logits(self, batch: Batch, carry: Any, step_output: Any) -> Tensor:
        """Return token logits from the MazeHard task payload."""
        _ = batch, carry
        return step_output.backbone_output.task.task_logits

    def extract_targets(self, batch: Batch, carry: Any, step_output: Any) -> MazeHardTargets:
        """Return MazeHard token supervision targets from the carry buffer."""
        _ = batch, step_output
        return MazeHardTargets(labels=carry.data["labels"])

    def evaluate_sequences(self, logits: Tensor, targets: MazeHardTargets) -> AccuracyStats:
        """Return masked sequence-correctness statistics for MazeHard token prediction."""
        return compute_accuracy_stats(logits, targets.labels, ignore_label_id=MAZE_HARD_IGNORE_LABEL_ID)

    def extract_policy_logits(self, step_output: Any) -> Tensor:
        """Return policy logits from the MazeHard task payload."""
        return step_output.backbone_output.policy.q_logits

    def extract_state_value(self, step_output: Any) -> Tensor:
        """Return state value from the MazeHard task payload."""
        critic = step_output.backbone_output.critic
        if critic is None:
            raise ValueError("Hybrid RL objective requires a critic output.")
        return critic.state_value


# =============================================================================
__all__ = [
    "MazeHardHRMACTTaskBinding",
    "MazeHardHRMHybridRLTaskBinding",
]
