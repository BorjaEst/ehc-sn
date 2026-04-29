"""Arena-EHC task binding for the EHC objective.

This module provides :class:`ArenaEHCTaskBinding`, which wires the EHC
objective to the arena task surface.  It lives here — in the shared
Arena+EHC bridge namespace — because it knows both the EHC objective API
and the arena carry shape, making it a EHC-specific model-task binding
rather than generic arena adapter logic.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from ehc_sn.metrics.keys import (
    EHC_ACC_OBS_ANCESTRAL_ALL,
    EHC_ACC_OBS_ANCESTRAL_REVISIT,
    EHC_ACC_OBS_INFERENCE_ALL,
    EHC_ACC_OBS_INFERENCE_REVISIT,
    EHC_ACC_OBS_RETRIEVED_ALL,
    EHC_ACC_OBS_RETRIEVED_REVISIT,
)
from ehc_sn.objectives.ehc import EHCStepOutput
from ehc_sn.tasks.arena.contracts import ArenaTargets
from ehc_sn.tasks.arena.evaluation import build_arena_step_score, coerce_observation_ids, coerce_revisit_mask
from ehc_sn.tasks.arena.runtime import coerce_arena_targets
from ehc_sn.training.types import RatioStat
from ehc_sn.types import Batch


# =============================================================================
class ArenaEHCTaskBinding:
    """EHC objective binding for the arena task.

    Implements :class:`~ehc_sn.objectives.ehc.EHCObjectiveBinding`
    ``[ArenaTargets]``.  Extracts supervised observation ids and protocol
    masks from the controller carry, and fans out single-pathway observation
    evaluation (from the task layer) across the three EHC pathways, mapping
    results to the established EHC metric keys.
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
            raise KeyError("Arena carry data must provide 'is_revisit' for protocol-gated EHC supervision.")
        result = coerce_revisit_mask(is_revisit, device=is_revisit.device)
        return result.to(dtype=torch.bool)  # type: ignore[union-attr]

    def evaluate_observation_metrics(
        self,
        step_output: EHCStepOutput,
        targets: ArenaTargets,
    ) -> dict[str, RatioStat]:
        """Return EHC-pathway count-bearing accuracy metrics for one step."""
        m_inf = build_arena_step_score(step_output.logits_inference, targets)
        m_ret = build_arena_step_score(step_output.logits_retrieved, targets)
        m_anc = build_arena_step_score(step_output.logits_ancestral, targets)

        batch_count = m_inf.is_correct.new_tensor(float(m_inf.is_correct.shape[0]), dtype=torch.float32)
        protocol_count = m_inf.is_revisit.sum().float() if m_inf.is_revisit is not None else m_inf.is_correct.new_zeros(())

        def _correct(m: "ArenaStepScore") -> Tensor:
            return m.is_correct.sum().float()

        def _correct_revisit(m: "ArenaStepScore") -> Tensor:
            if m.is_revisit is None:
                return m.is_correct.new_zeros(())
            return (m.is_correct & m.is_revisit).sum().float()

        return {
            EHC_ACC_OBS_INFERENCE_REVISIT: RatioStat(_correct_revisit(m_inf), protocol_count),
            EHC_ACC_OBS_RETRIEVED_REVISIT: RatioStat(_correct_revisit(m_ret), protocol_count),
            EHC_ACC_OBS_ANCESTRAL_REVISIT: RatioStat(_correct_revisit(m_anc), protocol_count),
            EHC_ACC_OBS_INFERENCE_ALL: RatioStat(_correct(m_inf), batch_count),
            EHC_ACC_OBS_RETRIEVED_ALL: RatioStat(_correct(m_ret), batch_count),
            EHC_ACC_OBS_ANCESTRAL_ALL: RatioStat(_correct(m_anc), batch_count),
        }


# =============================================================================
__all__ = ["ArenaEHCTaskBinding"]
