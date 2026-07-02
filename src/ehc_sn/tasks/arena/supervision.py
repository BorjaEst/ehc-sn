"""Arena task supervision — target coercion for TEM/EHP families.

This module owns the conversion of Arena batch tensors into
learning-ready supervision structs.  It is task-owned and reusable
across model families (TEM, EHP).

Delegates to existing coercion helpers in :mod:`ehp_sn.tasks.arena.runtime`
and :mod:`ehp_sn.tasks.arena.evaluation`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ehc_sn.metrics.keys import (
    TEM_ACC_OBS_PATH_ALL,
    TEM_ACC_OBS_PATH_REVISIT,
    TEM_ACC_OBS_POST_ALL,
    TEM_ACC_OBS_POST_REVISIT,
    TEM_ACC_OBS_RECALL_ALL,
    TEM_ACC_OBS_RECALL_REVISIT,
)
from ehc_sn.metrics.step_metrics import RatioStat
from ehc_sn.objectives.composites.tem import TEMSupervision
from ehc_sn.types import Batch

from .contracts import ArenaTargets
from .evaluation import (
    ArenaStepScore,
    build_arena_step_score,
    coerce_observation_ids,
    coerce_revisit_mask,
)
from .runtime import coerce_arena_targets


# =============================================================================
@dataclass(frozen=True)
class ArenaTEMSupervision:
    """Typed supervision struct for Arena TEM/EHP training.

    Attributes:
        observation_id: Ground-truth observation id for the current step,
            shape ``(B,)`` int64.
        protocol_mask: Revisit-eligibility mask for protocol-gated
            supervision, shape ``(B,)`` bool.
    """

    observation_id: Tensor
    protocol_mask: Tensor


# =============================================================================
def build_arena_supervision(executed_batch: Batch) -> ArenaTEMSupervision:
    """Build Arena TEM supervision from an executed batch frame.

    Delegates to :func:`~ehp_sn.tasks.arena.runtime.coerce_arena_targets`
    and :func:`~ehp_sn.tasks.arena.evaluation.coerce_observation_ids` /
    :func:`~ehp_sn.tasks.arena.evaluation.coerce_revisit_mask`.

    Args:
        executed_batch: Must contain Arena replay step keys
            (``observation_id``, ``is_revisit``, etc.).

    Returns:
        Typed supervision struct.

    Raises:
        KeyError: If required keys are missing.
    """
    targets = coerce_arena_targets(executed_batch)
    observation_id = coerce_observation_ids(targets.observation_id)

    if targets.is_revisit is None:
        raise KeyError(
            "build_arena_supervision: 'is_revisit' missing from "
            "executed_batch. Protocol-gated supervision requires revisit info."
        )
    protocol_mask = coerce_revisit_mask(
        targets.is_revisit, device=targets.is_revisit.device
    )

    return ArenaTEMSupervision(
        observation_id=observation_id,
        protocol_mask=protocol_mask.to(dtype=torch.bool),
    )


# =============================================================================
def build_arena_tem_observation_metrics(
    step_output: object,
    targets: ArenaTargets,
) -> dict[str, RatioStat]:
    """Return TEM-pathway count-bearing accuracy metrics for one arena step.

    Fans out single-pathway observation evaluation (from the task layer)
    across the three TEM pathways (post, recall, path), mapping results
    to the established TEM metric keys.

    Args:
        step_output: TEM step output with ``.logits_post``, ``.logits_recall``,
            ``.logits_path`` attributes.
        targets: Arena supervision targets (``observation_id``, ``is_revisit``).

    Returns:
        Dict mapping metric keys to :class:`~ehp_sn.metrics.step_metrics.RatioStat`
        values.
    """
    m_inf = build_arena_step_score(step_output.logits_post, targets)  # type: ignore[union-attr]
    m_ret = build_arena_step_score(step_output.logits_recall, targets)  # type: ignore[union-attr]
    m_anc = build_arena_step_score(step_output.logits_path, targets)  # type: ignore[union-attr]

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
        TEM_ACC_OBS_POST_REVISIT: RatioStat(
            numerator_sum=_correct_revisit(m_inf),
            denominator_sum=protocol_count,
        ),
        TEM_ACC_OBS_RECALL_REVISIT: RatioStat(
            numerator_sum=_correct_revisit(m_ret),
            denominator_sum=protocol_count,
        ),
        TEM_ACC_OBS_PATH_REVISIT: RatioStat(
            numerator_sum=_correct_revisit(m_anc),
            denominator_sum=protocol_count,
        ),
        TEM_ACC_OBS_POST_ALL: RatioStat(
            numerator_sum=_correct(m_inf),
            denominator_sum=batch_count,
        ),
        TEM_ACC_OBS_RECALL_ALL: RatioStat(
            numerator_sum=_correct(m_ret),
            denominator_sum=batch_count,
        ),
        TEM_ACC_OBS_PATH_ALL: RatioStat(
            numerator_sum=_correct(m_anc),
            denominator_sum=batch_count,
        ),
    }


# =============================================================================
def build_arena_tem_objective_supervision(
    supervision: ArenaTEMSupervision,
) -> TEMSupervision:
    """Translate Arena task supervision into TEM objective supervision.

    Validates shape and dtype at the translation boundary so that the
    objective receives a contract-conforming input.

    Args:
        supervision: Arena-owned supervision struct.

    Returns:
        Objective-facing TEMSupervision.

    Raises:
        ValueError: If observation labels are not 1-D.
        ValueError: If protocol mask shape does not match labels.
        TypeError: If protocol mask is not bool.
    """
    labels = supervision.observation_id
    protocol_mask = supervision.protocol_mask

    if labels.ndim != 1:
        raise ValueError(
            "TEM observation labels must have shape (B,), "
            f"received {tuple(labels.shape)}."
        )
    if protocol_mask.shape != labels.shape:
        raise ValueError(
            "TEM protocol mask must match observation labels: "
            f"{tuple(protocol_mask.shape)} != {tuple(labels.shape)}."
        )
    if protocol_mask.dtype is not torch.bool:
        raise TypeError(
            "TEM protocol mask must be bool, "
            f"received {protocol_mask.dtype}."
        )

    return TEMSupervision(
        observation_labels=labels,
        protocol_mask=protocol_mask,
    )


# =============================================================================
__all__ = [
    "ArenaTEMSupervision",
    "build_arena_supervision",
    "build_arena_tem_objective_supervision",
    "build_arena_tem_observation_metrics",
]
