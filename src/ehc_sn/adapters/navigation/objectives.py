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
from ehc_sn.tasks.navigation.evaluation import evaluate_navigation_observation_logits
from ehc_sn.training.types import RatioStat
from ehc_sn.types import Batch


# =============================================================================
class NavigationTEMTaskBinding:
    """TEM objective binding for the navigation task.

    Implements :class:`~ehc_sn.objectives.tem.TEMObjectiveBinding`
    ``[NavigationTargets]``.  Extracts supervised observation ids and protocol
    masks from the controller carry, and delegates observation-correctness
    evaluation to the task-owned
    :func:`~ehc_sn.tasks.navigation.evaluation.evaluate_navigation_observation_logits`.

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

        Uses ``observation_id`` when present; falls back to ``labels`` only if
        ``observation_id`` is absent.  ``is_revisit`` is forwarded unchanged.

        Args:
            batch: Generic batch mapping; unused here.
            carry: Controller carry state.
            step_output: Controller step output; unused here.
        """
        _ = batch, step_output
        raw: Tensor | None = carry.data.get("observation_id")
        if raw is None:
            raw = carry.data["labels"]
        is_revisit: Tensor | None = carry.data.get("is_revisit")
        return NavigationTargets(observation_id=raw, is_revisit=is_revisit)

    def extract_observation_id(  # -------------------------------------------
        self,
        targets: NavigationTargets,
    ) -> Tensor:
        """Return the current-step observation id tensor from ``targets``.

        Applies squeeze/argmax normalisation to guarantee a 1-D integer target
        tensor of shape ``(B,)``.

        Args:
            targets: Navigation supervision targets produced by
                :meth:`extract_targets`.

        Returns:
            Integer observation-id tensor of shape ``(B,)``.
        """
        raw = targets.observation_id
        if raw.ndim > 1:
            if raw.shape[-1] == 1:
                return raw.squeeze(-1)
            if raw.is_floating_point():
                return raw.argmax(dim=-1)
        return raw

    def extract_protocol_mask(  # --------------------------------------------
        self,
        targets: NavigationTargets,
    ) -> Tensor:
        """Return the revisit-eligibility mask for protocol supervision.

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
        return is_revisit.reshape(-1).to(dtype=torch.bool)

    def evaluate_observation_metrics(  # -------------------------------------
        self,
        step_output: Any,
        targets: NavigationTargets,
    ) -> dict[str, RatioStat]:
        """Return task-owned count-bearing accuracy metrics for one step.

        Delegates to
        :func:`~ehc_sn.tasks.navigation.evaluation.evaluate_navigation_observation_logits`
        so all argmax correctness counting remains in the task layer.

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
        obs_logits = (step_output.logits_inference, step_output.logits_retrieved, step_output.logits_ancestral)
        m = evaluate_navigation_observation_logits(obs_logits, targets)
        protocol_count = m.inference.count_revisit  # same for all pathways
        batch_count = m.inference.count_all  # same for all pathways
        return {
            TEM_ACC_OBS_INFERENCE_REVISIT: RatioStat(m.inference.correct_revisit, protocol_count),
            TEM_ACC_OBS_RETRIEVED_REVISIT: RatioStat(m.retrieved.correct_revisit, protocol_count),
            TEM_ACC_OBS_ANCESTRAL_REVISIT: RatioStat(m.ancestral.correct_revisit, protocol_count),
            TEM_ACC_OBS_INFERENCE_ALL: RatioStat(m.inference.correct_all, batch_count),
            TEM_ACC_OBS_RETRIEVED_ALL: RatioStat(m.retrieved.correct_all, batch_count),
            TEM_ACC_OBS_ANCESTRAL_ALL: RatioStat(m.ancestral.correct_all, batch_count),
        }


# =============================================================================
__all__ = ["NavigationTEMTaskBinding"]
