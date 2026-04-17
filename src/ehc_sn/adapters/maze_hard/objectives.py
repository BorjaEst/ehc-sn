"""MazeHard ACT task binding for the ACT objective.

This module provides :class:`MazeHardACTTaskBinding`, which wires controller
loss head to the MazeHard task surface.  It is the sole place in the codebase
that knows both the objective API and the MazeHard output shape.
"""

from __future__ import annotations

from typing import Any

from torch import Tensor

from ehc_sn.heads._token import AccuracyStats, TokenSupervisionBinding, compute_accuracy_stats
from ehc_sn.tasks.maze_hard.contracts import MAZE_HARD_IGNORE_LABEL_ID, MazeHardTargets
from ehc_sn.types import Batch


# =============================================================================
class MazeHardACTTaskBinding(TokenSupervisionBinding[MazeHardTargets]):
    """ACT task binding for MazeHard token prediction.

    Extracts supervised logits from ``step_output.backbone_output.task.task_logits``
    and constructs targets from the canonical MazeHard batch ``"labels"`` key.
    """

    def extract_logits(  # ----------------------------------------------------
        self,
        batch: Batch,
        carry: Any,
        step_output: Any,
    ) -> Tensor:
        """Return token logits from the MazeHard task payload.

        Args:
            batch: Generic batch mapping; unused here.
            carry: Controller carry state; unused here.
            step_output: ACTStepOutput with ``backbone_output.task.task_logits``.

        Returns:
            Tensor of shape ``(B, S, vocab_size)``.
        """
        _ = batch, carry
        return step_output.backbone_output.task.task_logits

    def extract_targets(  # ---------------------------------------------------
        self,
        batch: Batch,
        carry: Any,
        step_output: Any,
    ) -> MazeHardTargets:
        """Return MazeHard token supervision targets from the carry buffer.

        Args:
            batch: Generic batch mapping; unused here.
            carry: Controller carry with ``carry.data["labels"]``.
            step_output: ACTStepOutput; unused here.

        Returns:
            :class:`~ehc_sn.tasks.maze_hard.contracts.MazeHardTargets` wrapping the labels tensor.
        """
        _ = batch, step_output
        return MazeHardTargets(labels=carry.data["labels"])

    def evaluate_sequences(  # ------------------------------------------------
        self,
        logits: Tensor,
        targets: MazeHardTargets,
    ) -> AccuracyStats:
        """Return masked sequence-correctness statistics for MazeHard token prediction.

        Args:
            logits: Token logits of shape ``(B, S, vocab_size)``.
            targets: :class:`~ehc_sn.tasks.maze_hard.contracts.MazeHardTargets`.

        Returns:
            :class:`~ehc_sn.heads._token.AccuracyStats`.
        """
        return compute_accuracy_stats(
            logits,
            targets.labels,
            ignore_label_id=MAZE_HARD_IGNORE_LABEL_ID,
        )


# =============================================================================
__all__ = ["MazeHardACTTaskBinding"]
