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
from ehc_sn.objectives.tem import TEMStepOutput
from ehc_sn.tasks.arena.contracts import ArenaTargets
from ehc_sn.tasks.arena.evaluation import (
    ArenaStepScore,
    build_arena_step_score,
    coerce_observation_ids,
    coerce_revisit_mask,
)
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

    def extract_targets(  # ---------------------------------------------------
        self,
        batch: Batch,
        carry: Any,
        step_output: Any,
    ) -> ArenaTargets:
        """Build an :class:`ArenaTargets` from the executed step payload.

        Args:
            batch: Executed step payload (observation_id, is_revisit, ...).
                This is the executed_frame alias set by the runner — carry is
                NOT consulted so snapshot is not the execution authority.
            carry: Controller carry state; unused here.
            step_output: Controller step output; unused here.
        """
        _ = carry, step_output
        return coerce_arena_targets(batch)

    def extract_observation_id(  # --------------------------------------------
        self,
        batch: Batch,
        carry: Any,
        step_output: Any,
    ) -> Tensor:
        """Return the current-step observation id tensor from the executed step payload."""
        _ = carry, step_output
        targets = coerce_arena_targets(batch)
        return coerce_observation_ids(targets.observation_id)

    def extract_protocol_mask(  # ----------------------------------------------
        self,
        batch: Batch,
        carry: Any,
        step_output: Any,
    ) -> Tensor:
        """Return the revisit-eligibility mask for protocol supervision."""
        _ = carry, step_output
        targets = coerce_arena_targets(batch)
        is_revisit = targets.is_revisit
        if is_revisit is None:
            raise KeyError(
                "Arena carry data must provide 'is_revisit' for protocol-gated TEM supervision.",
            )
        result = coerce_revisit_mask(is_revisit, device=is_revisit.device)
        return result.to(dtype=torch.bool)  # type: ignore[union-attr]

    def evaluate_observation_metrics(  # --------------------------------------
        self,
        step_output: TEMStepOutput,
        targets: ArenaTargets,
    ) -> dict[str, RatioStat]:
        """Return TEM-pathway count-bearing accuracy metrics for one step."""
        m_inf = build_arena_step_score(step_output.logits_inference, targets)
        m_ret = build_arena_step_score(step_output.logits_retrieved, targets)
        m_anc = build_arena_step_score(step_output.logits_ancestral, targets)

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
            TEM_ACC_OBS_INFERENCE_REVISIT: RatioStat(
                numerator_sum=_correct_revisit(m_inf),
                denominator_sum=protocol_count,
            ),
            TEM_ACC_OBS_RETRIEVED_REVISIT: RatioStat(
                numerator_sum=_correct_revisit(m_ret),
                denominator_sum=protocol_count,
            ),
            TEM_ACC_OBS_ANCESTRAL_REVISIT: RatioStat(
                numerator_sum=_correct_revisit(m_anc),
                denominator_sum=protocol_count,
            ),
            TEM_ACC_OBS_INFERENCE_ALL: RatioStat(
                numerator_sum=_correct(m_inf),
                denominator_sum=batch_count,
            ),
            TEM_ACC_OBS_RETRIEVED_ALL: RatioStat(
                numerator_sum=_correct(m_ret),
                denominator_sum=batch_count,
            ),
            TEM_ACC_OBS_ANCESTRAL_ALL: RatioStat(
                numerator_sum=_correct(m_anc),
                denominator_sum=batch_count,
            ),
        }


# =============================================================================
__all__ = ["ArenaTEMTaskBinding"]
