"""TODO"""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import Tensor

from ehc_sn.controllers.contracts.value_control import (
    ValueControlInteractionRecord,
)
from ehc_sn.metrics.keys import (
    TEM_ACC_OBS_PATH_ALL,
    TEM_ACC_OBS_PATH_REVISIT,
    TEM_ACC_OBS_POST_ALL,
    TEM_ACC_OBS_POST_REVISIT,
    TEM_ACC_OBS_RECALL_ALL,
    TEM_ACC_OBS_RECALL_REVISIT,
)
from ehc_sn.metrics.step_metrics import RatioStat
from ehc_sn.objectives.hybrid_rl import HybridValueObjectiveBinding
from ehc_sn.rollouts.runtime import CarrySnapshot
from ehc_sn.tasks.arena.contracts import ArenaTargets
from ehc_sn.tasks.arena.evaluation import (
    ArenaStepScore,
    build_arena_step_score,
    coerce_observation_ids,
    coerce_revisit_mask,
)
from ehc_sn.tasks.arena.runtime import coerce_arena_targets
from ehc_sn.tasks.mazehard.contracts import MazeHardTaskOutput
from ehc_sn.types import Batch


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
    """MazeHard-specific extraction for the EHP v1 hybrid RL value-control path.

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
class ArenaEHCTaskBinding:
    """EHP objective binding for the arena task.

    Implements :class:`~ehc_sn.objectives.ehp.EHCObjectiveBinding`
    ``[ArenaTargets]``.  Extracts supervised observation ids and protocol
    masks from the controller carry, and fans out single-pathway observation
    evaluation (from the task layer) across the three EHP pathways, mapping
    results to the established EHP metric keys.
    """

    def extract_targets(
        self,
        executed_batch: Batch,
        snapshot: CarrySnapshot,
        step_output: Any,
    ) -> ArenaTargets:
        """Build an :class:`ArenaTargets` from the executed step payload.

        Args:
            executed_batch: Executed step payload (observation_id, is_revisit, ...).
                This is the executed_frame alias set by the runner — carry is
                NOT consulted so snapshot is not the execution authority.
            snapshot: Frozen post-step snapshot; unused here.
            step_output: Controller step output; unused here.
        """
        _ = snapshot, step_output
        return coerce_arena_targets(executed_batch)

    def extract_observation_id(
        self,
        targets: ArenaTargets,
    ) -> Tensor:
        """Return the current-step observation id tensor from ``targets``."""
        return coerce_observation_ids(targets.observation_id)

    def extract_protocol_mask(
        self,
        targets: ArenaTargets,
    ) -> Tensor:
        """Return the revisit-eligibility mask for protocol supervision."""
        is_revisit = targets.is_revisit
        if is_revisit is None:
            raise KeyError(
                "Arena carry data must provide 'is_revisit' for protocol-gated "
                "EHP supervision."
            )
        result = coerce_revisit_mask(is_revisit, device=is_revisit.device)
        return result.to(dtype=torch.bool)  # type: ignore[union-attr]

    def evaluate_observation_metrics(
        self,
        step_output: EHCStepOutput,
        targets: ArenaTargets,
    ) -> dict[str, RatioStat]:
        """Return EHP-pathway count-bearing accuracy metrics for one step."""
        m_inf = build_arena_step_score(step_output.logits_post, targets)
        m_ret = build_arena_step_score(step_output.logits_recall, targets)
        m_anc = build_arena_step_score(step_output.logits_PATH, targets)

        batch_count = m_inf.is_correct.new_tensor(
            float(m_inf.is_correct.shape[0]), dtype=torch.float32
        )
        protocol_count = (
            m_inf.is_revisit.sum().float()
            if m_inf.is_revisit is not None
            else m_inf.is_correct.new_zeros(())
        )

        def _correct(m: ArenaStepScore) -> Tensor:
            return m.is_correct.sum().float()

        def _correct_revisit(m: ArenaStepScore) -> Tensor:
            if m.is_revisit is None:
                return m.is_correct.new_zeros(())
            return (m.is_correct & m.is_revisit).sum().float()

        return {
            EHP_ACC_OBS_POST_REVISIT: RatioStat(
                _correct_revisit(m_inf), protocol_count
            ),
            EHP_ACC_OBS_RECALL_REVISIT: RatioStat(
                _correct_revisit(m_ret), protocol_count
            ),
            EHP_ACC_OBS_PATH_REVISIT: RatioStat(
                _correct_revisit(m_anc), protocol_count
            ),
            EHP_ACC_OBS_POST_ALL: RatioStat(_correct(m_inf), batch_count),
            EHP_ACC_OBS_RECALL_ALL: RatioStat(_correct(m_ret), batch_count),
            EHP_ACC_OBS_PATH_ALL: RatioStat(_correct(m_anc), batch_count),
        }


# =============================================================================
__all__ = [
    "MazeHardEHCV1HybridTaskBinding",
    "ArenaEHCTaskBinding",
]
