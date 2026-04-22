"""Arena-TEM task binding for the TEM objective.

This module provides :class:`ArenaTEMTaskBinding`, which wires the TEM
objective to the arena task surface.  It lives here — in the shared
Arena+TEM bridge namespace — because it knows both the TEM objective API
and the arena carry shape, making it a TEM-specific model-task binding
rather than generic arena adapter logic.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from ehc_sn.metrics.keys import (
    TEM_ACC_OBS_ANCESTRAL_ALL,
    TEM_ACC_OBS_ANCESTRAL_REVISIT,
    TEM_ACC_OBS_INFERENCE_ALL,
    TEM_ACC_OBS_INFERENCE_REVISIT,
    TEM_ACC_OBS_RETRIEVED_ALL,
    TEM_ACC_OBS_RETRIEVED_REVISIT,
)
from ehc_sn.objectives.tem import TEMStepOutputs
from ehc_sn.tasks.arena.contracts import ArenaTargets
from ehc_sn.tasks.arena.evaluation import coerce_observation_ids, coerce_revisit_mask, evaluate_observation_logits
from ehc_sn.tasks.arena.runtime import coerce_arena_targets
from ehc_sn.training.types import RatioStat
from ehc_sn.types import Batch


# =============================================================================
class ArenaTEMTaskBinding:
    """TEM objective binding for the arena task.

    Implements :class:`~ehc_sn.objectives.tem.TEMObjectiveBinding`
    ``[ArenaTargets]``.  Extracts supervised observation ids and protocol
    masks from the controller carry, and fans out single-pathway observation
    evaluation (from the task layer) across the three TEM pathways, mapping
    results to the established TEM metric keys.
    """

    def extract_targets(
        self,
        batch: Batch,
        carry: Any,
        step_output: Any,
    ) -> ArenaTargets:
        """Build an :class:`ArenaTargets` from carry data.

        Args:
            batch: Generic batch mapping; unused here.
            carry: Controller carry state.
            step_output: Controller step output; unused here.
        """
        _ = batch, step_output
        return coerce_arena_targets(carry.data)

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
            raise KeyError("Arena carry data must provide 'is_revisit' for protocol-gated TEM supervision.")
        result = coerce_revisit_mask(is_revisit, device=is_revisit.device)
        return result.to(dtype=torch.bool)  # type: ignore[union-attr]

    def evaluate_observation_metrics(
        self,
        step_output: TEMStepOutputs,
        targets: ArenaTargets,
    ) -> dict[str, RatioStat]:
        """Return TEM-pathway count-bearing accuracy metrics for one step."""
        m_inf = evaluate_observation_logits(step_output.logits_inference, targets)
        m_ret = evaluate_observation_logits(step_output.logits_retrieved, targets)
        m_anc = evaluate_observation_logits(step_output.logits_ancestral, targets)
        protocol_count = m_inf.count_revisit
        batch_count = m_inf.count_all
        return {
            TEM_ACC_OBS_INFERENCE_REVISIT: RatioStat(m_inf.correct_revisit, protocol_count),
            TEM_ACC_OBS_RETRIEVED_REVISIT: RatioStat(m_ret.correct_revisit, protocol_count),
            TEM_ACC_OBS_ANCESTRAL_REVISIT: RatioStat(m_anc.correct_revisit, protocol_count),
            TEM_ACC_OBS_INFERENCE_ALL: RatioStat(m_inf.correct_all, batch_count),
            TEM_ACC_OBS_RETRIEVED_ALL: RatioStat(m_ret.correct_all, batch_count),
            TEM_ACC_OBS_ANCESTRAL_ALL: RatioStat(m_anc.correct_all, batch_count),
        }


# =============================================================================
__all__ = ["ArenaTEMTaskBinding"]
