"""Navigation TEM task binding for the TEM objective.

This module provides :class:`NavigationTEMTaskBinding`, which wires the TEM
objective to the navigation task surface.  It is the sole place in the
codebase that knows both the TEM objective API and the navigation carry shape.
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
from ehc_sn.tasks.navigation.contracts import NavigationTargets
from ehc_sn.tasks.navigation.evaluation import coerce_observation_ids, coerce_revisit_mask, evaluate_observation_logits
from ehc_sn.tasks.navigation.runtime import coerce_navigation_targets
from ehc_sn.training.types import RatioStat
from ehc_sn.types import Batch


# =============================================================================
class NavigationTEMTaskBinding:
    """TEM objective binding for the navigation task.

    Implements :class:`~ehc_sn.objectives.tem.TEMObjectiveBinding`
    ``[NavigationTargets]``.  Extracts supervised observation ids and protocol
    masks from the controller carry, and fans out single-pathway observation
    evaluation (from the task layer) across the three TEM pathways, mapping
    results to the established TEM metric keys.

    Mirrors the pattern used by
    :class:`~ehc_sn.adapters.maze_hard.objectives.MazeHardACTTaskBinding`
    for the ACT objective path.
    """

    def extract_targets(  # --------------------------------------------------
        self,
        batch: Batch,
        carry: Any,
        step_output: Any,
    ) -> NavigationTargets:
        """Build a :class:`NavigationTargets` from carry data.

        Delegates to the task-owned :func:`~ehc_sn.tasks.navigation.runtime.coerce_navigation_targets`
        helper.  ``observation_id`` must be present in ``carry.data``;
        there is no ``labels`` fallback.

        Args:
            batch: Generic batch mapping; unused here.
            carry: Controller carry state.
            step_output: Controller step output; unused here.

        Raises:
            KeyError: If ``observation_id`` is absent from ``carry.data``.
        """
        _ = batch, step_output
        return coerce_navigation_targets(carry.data)

    def extract_observation_id(  # -------------------------------------------
        self,
        targets: NavigationTargets,
    ) -> Tensor:
        """Return the current-step observation id tensor from ``targets``.

        Delegates to the task-owned :func:`~ehc_sn.tasks.navigation.evaluation.coerce_observation_ids`
        to guarantee a 1-D integer tensor of shape ``(B,)``.

        Args:
            targets: Navigation supervision targets produced by
                :meth:`extract_targets`.

        Returns:
            Integer observation-id tensor of shape ``(B,)``.
        """
        return coerce_observation_ids(targets.observation_id)

    def extract_protocol_mask(  # --------------------------------------------
        self,
        targets: NavigationTargets,
    ) -> Tensor:
        """Return the revisit-eligibility mask for protocol supervision.

        Delegates to the task-owned :func:`~ehc_sn.tasks.navigation.evaluation.coerce_revisit_mask`.

        Args:
            targets: Navigation supervision targets produced by
                :meth:`extract_targets`.

        Returns:
            Boolean tensor of shape ``(B,)`` indicating revisit positions.

        Raises:
            KeyError: If ``targets`` does not carry an ``is_revisit`` field.
        """
        is_revisit = targets.is_revisit
        if is_revisit is None:
            raise KeyError("Navigation carry data must provide 'is_revisit' for protocol-gated TEM supervision.")
        result = coerce_revisit_mask(is_revisit, device=is_revisit.device)
        return result.to(dtype=torch.bool)  # type: ignore[union-attr]

    def evaluate_observation_metrics(  # -------------------------------------
        self,
        step_output: Any,
        targets: NavigationTargets,
    ) -> dict[str, RatioStat]:
        """Return TEM-pathway count-bearing accuracy metrics for one step.

        Fans out the task-generic single-pathway evaluator across the three TEM
        observation pathways (inference, retrieved, ancestral) and maps results
        to the established TEM metric keys.  The fan-out is TEM-specific and
        lives here, not in the task layer.

        Args:
            step_output: TEM controller step output exposing
                ``logits_inference``, ``logits_retrieved``, and
                ``logits_ancestral``.
            targets: Navigation supervision targets produced by
                :meth:`extract_targets`.

        Returns:
            Dict of :class:`~ehc_sn.training.types.RatioStat` pairs keyed to
            TEM metric-key constants.  Values are raw counts (not reduced
            ratios) so the objective can accumulate them correctly.
        """
        m_inf = evaluate_observation_logits(step_output.logits_inference, targets)
        m_ret = evaluate_observation_logits(step_output.logits_retrieved, targets)
        m_anc = evaluate_observation_logits(step_output.logits_ancestral, targets)
        protocol_count = m_inf.count_revisit  # same for all pathways
        batch_count = m_inf.count_all  # same for all pathways
        return {
            TEM_ACC_OBS_INFERENCE_REVISIT: RatioStat(m_inf.correct_revisit, protocol_count),
            TEM_ACC_OBS_RETRIEVED_REVISIT: RatioStat(m_ret.correct_revisit, protocol_count),
            TEM_ACC_OBS_ANCESTRAL_REVISIT: RatioStat(m_anc.correct_revisit, protocol_count),
            TEM_ACC_OBS_INFERENCE_ALL: RatioStat(m_inf.correct_all, batch_count),
            TEM_ACC_OBS_RETRIEVED_ALL: RatioStat(m_ret.correct_all, batch_count),
            TEM_ACC_OBS_ANCESTRAL_ALL: RatioStat(m_anc.correct_all, batch_count),
        }


# =============================================================================
__all__ = ["NavigationTEMTaskBinding"]
