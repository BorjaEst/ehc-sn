"""MazeHard+EHC hybrid value-control task binding.

Owns the extraction logic from ValueControlInteractionRecord for the EHC
family's hybrid RL objective. Consumed by the family barrel and by the EHC
reason_pretrain Lightning controller.
"""

from __future__ import annotations

from typing import cast

from torch import Tensor

from ehc_sn.controllers.contracts.value_control import (
    ValueControlInteractionRecord,
)
from ehc_sn.objectives.hybrid_rl import HybridValueObjectiveBinding
from ehc_sn.tasks.mazehard.contracts import MazeHardTaskOutput


# =============================================================================
def _extract_record_task_logits(
    record: ValueControlInteractionRecord,
) -> Tensor:
    task_output = cast(MazeHardTaskOutput | None, record.task_output)
    if task_output is None:
        raise RuntimeError(
            "MazeHardEHCV1HybridTaskBinding: task_output is None. "
            "The controller must attach a task payload with task_logits."
        )
    return task_output.task_logits


class MazeHardEHCV1HybridTaskBinding:
    """MazeHard-specific extraction for the EHC v1 hybrid RL value-control path.

    Implements HybridValueObjectiveBinding structurally.
    """

    def extract_task_logits(
        self, record: ValueControlInteractionRecord
    ) -> Tensor:
        """Return token-prediction logits from ``record.task_output.task_logits``."""
        return _extract_record_task_logits(record)

    def extract_labels(self, record: ValueControlInteractionRecord) -> Tensor:
        """Return supervision labels from ``record.observation_used_for_decision``."""
        if "labels" not in record.observation_used_for_decision:
            raise RuntimeError(
                "MazeHardEHCV1HybridTaskBinding: 'labels' key missing from "
                "observation_used_for_decision. The batch must include supervision labels."
            )
        return record.observation_used_for_decision["labels"]


# Structural protocol check — fails at import if the binding is incomplete.
_: HybridValueObjectiveBinding = MazeHardEHCV1HybridTaskBinding()


# =============================================================================
__all__ = [
    "MazeHardEHCV1HybridTaskBinding",
]
