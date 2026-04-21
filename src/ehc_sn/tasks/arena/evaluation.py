"""Arena task evaluation helpers.

Provides the canonical additive structural score, full-episode evaluation
primitives, and coercion utilities used by arena objective bindings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor

from .contracts import ArenaTargets, ArenaTaskInput


# =============================================================================
@dataclass(frozen=True)
class ArenaEpisodeSemantics:
    """Episode-semantic flags attached to one arena task step."""

    step_count: Tensor | None = None
    episode_start: Tensor | None = None
    is_revisit: Tensor | None = None


# =============================================================================
@dataclass(frozen=True)
class ArenaPathwayMetrics:
    """Raw accuracy counts for one arena observation-logit pathway.

    Raw counts rather than normalised rates are stored so callers can
    accumulate across batches before computing global accuracy.
    """

    correct_all: Tensor
    count_all: Tensor
    correct_revisit: Tensor
    count_revisit: Tensor

    @property
    def accuracy_all(self) -> Tensor:
        """Return all-step accuracy for this pathway."""
        return self.correct_all / self.count_all.clamp_min(1.0)

    @property
    def accuracy_revisit(self) -> Tensor:
        """Return revisit-only accuracy for this pathway."""
        return self.correct_revisit / self.count_revisit.clamp_min(1.0)


# =============================================================================
@dataclass(frozen=True)
class ArenaStructuralScore:
    """Canonical additive structural score for one arena evaluation batch.

    The structural score is the task-owned benchmark primitive.  It exposes
    both raw counts (for correct cross-batch accumulation) and derived
    accuracy scalars (for per-step logging).  Multi-pathway fan-out
    (TEM-specific) is adapter-side and must not add new fields here.
    """

    accuracy_all: Tensor
    """Mean per-step observation accuracy across all steps."""
    accuracy_revisit: Tensor
    """Mean per-step observation accuracy restricted to revisit steps."""
    correct_all: Tensor
    """Raw correct-prediction count (all steps)."""
    count_all: Tensor
    """Raw total-step count (all steps)."""
    correct_revisit: Tensor
    """Raw correct-prediction count (revisit steps only)."""
    count_revisit: Tensor
    """Raw total revisit-step count."""


# =============================================================================
def extract_arena_episode_semantics(
    payload: ArenaTaskInput | Mapping[str, Tensor],
) -> ArenaEpisodeSemantics:
    """Return the episode-semantic subset from an arena task payload."""
    if isinstance(payload, ArenaTaskInput):
        return ArenaEpisodeSemantics(
            step_count=payload.step_count,
            episode_start=payload.episode_start,
            is_revisit=payload.is_revisit,
        )
    return ArenaEpisodeSemantics(
        step_count=payload.get("step_count"),
        episode_start=payload.get("episode_start"),
        is_revisit=payload.get("is_revisit"),
    )


# =============================================================================
def evaluate_observation_logits(
    logits: Tensor,
    targets: ArenaTargets,
) -> ArenaPathwayMetrics:
    """Return all-step and revisit-split accuracy counts for one observation logit pathway.

    This is the task-generic primitive.  Multi-pathway (TEM-specific) fan-out
    is handled by the adapter/objective layer, not here.

    Args:
        logits: Predicted observation logits of shape ``(B, obs_dim)``.
        targets: Arena supervision targets with ``observation_id`` and
            optional ``is_revisit``.

    Returns:
        :class:`ArenaPathwayMetrics` with raw counts (not normalised).
    """
    labels = coerce_observation_ids(targets.observation_id)
    revisit_mask = coerce_revisit_mask(targets.is_revisit, device=labels.device)
    return _evaluate_pathway(logits, labels, revisit_mask)


# =============================================================================
def compute_arena_structural_score(
    metrics: ArenaPathwayMetrics,
) -> ArenaStructuralScore:
    """Return the canonical additive structural score from pathway accuracy counts.

    The structural score exposes both raw counts and derived accuracies so
    callers can aggregate correctly across batches by summing counts rather
    than averaging accuracy scalars.

    Args:
        metrics: Pathway accuracy counts from :func:`evaluate_observation_logits`.

    Returns:
        :class:`ArenaStructuralScore` with accuracy scalars and raw counts.
    """
    return ArenaStructuralScore(
        accuracy_all=metrics.accuracy_all,
        accuracy_revisit=metrics.accuracy_revisit,
        correct_all=metrics.correct_all,
        count_all=metrics.count_all,
        correct_revisit=metrics.correct_revisit,
        count_revisit=metrics.count_revisit,
    )


# =============================================================================
def _evaluate_pathway(
    logits: Tensor,
    labels: Tensor,
    revisit_mask: Tensor | None,
) -> ArenaPathwayMetrics:
    correct = logits.argmax(dim=-1).eq(labels)
    dtype = logits.dtype
    correct_all = correct.sum().to(dtype=dtype)
    count_all = logits.new_tensor(float(labels.shape[0]), dtype=dtype)
    if revisit_mask is None:
        revisit = logits.new_zeros(())
        revisit_count = logits.new_zeros(())
    else:
        revisit = (correct & revisit_mask).sum().to(dtype=dtype)
        revisit_count = revisit_mask.to(dtype=dtype).sum()
    return ArenaPathwayMetrics(
        correct_all=correct_all,
        count_all=count_all,
        correct_revisit=revisit,
        count_revisit=revisit_count,
    )


# =============================================================================
def coerce_observation_ids(
    observation_id: Tensor,
) -> Tensor:
    """Return categorical observation labels with shape ``(B,)``.

    Accepts ``(B,)``, ``(B, 1)``, or ``(B, obs_dim)`` (one-hot / soft-label)
    inputs and normalises to a clean 1-D integer tensor regardless of how the
    source emits observation identifiers.

    Args:
        observation_id: Raw observation-id tensor.

    Returns:
        Integer tensor of shape ``(B,)``.
    """
    if observation_id.ndim > 1:
        if observation_id.shape[-1] == 1:
            return observation_id.squeeze(-1)
        if observation_id.is_floating_point():
            return observation_id.argmax(dim=-1)
    return observation_id


# =============================================================================
def coerce_revisit_mask(
    is_revisit: Tensor | None,
    *,
    device: torch.device,
) -> Tensor | None:
    """Return a flattened boolean revisit mask, or ``None`` when absent.

    Args:
        is_revisit: Optional revisit flag of any shape.
        device: Target device for the output tensor.

    Returns:
        Boolean tensor of shape ``(B,)`` or ``None``.
    """
    if is_revisit is None:
        return None
    return is_revisit.reshape(-1).to(device=device, dtype=torch.bool)


# =============================================================================
__all__ = [
    "ArenaEpisodeSemantics",
    "ArenaPathwayMetrics",
    "ArenaStructuralScore",
    "coerce_observation_ids",
    "coerce_revisit_mask",
    "compute_arena_structural_score",
    "evaluate_observation_logits",
    "extract_arena_episode_semantics",
]
