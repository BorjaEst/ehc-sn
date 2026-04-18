"""Navigation task evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor

from .contracts import NavigationTargets, NavigationTaskInput


# =============================================================================
@dataclass(frozen=True)
class NavigationEpisodeSemantics:
    """Episode-semantic flags attached to one navigation task step."""

    step_count: Tensor | None = None
    episode_start: Tensor | None = None
    is_revisit: Tensor | None = None


# =============================================================================
@dataclass(frozen=True)
class NavigationPathwayMetrics:
    """Accuracy counts for one navigation observation-logit pathway."""

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
def extract_navigation_episode_semantics(  # ----------------------------------
    payload: NavigationTaskInput | Mapping[str, Tensor],
) -> NavigationEpisodeSemantics:
    """Return the episode-semantic subset from a navigation task payload."""
    if isinstance(payload, NavigationTaskInput):
        return NavigationEpisodeSemantics(
            step_count=payload.step_count,
            episode_start=payload.episode_start,
            is_revisit=payload.is_revisit,
        )
    return NavigationEpisodeSemantics(
        step_count=payload.get("step_count"),
        episode_start=payload.get("episode_start"),
        is_revisit=payload.get("is_revisit"),
    )


# =============================================================================
def evaluate_observation_logits(  # -------------------------------------------
    logits: Tensor,
    targets: NavigationTargets,
) -> NavigationPathwayMetrics:
    """Return all-step and revisit-split accuracy counts for one observation logit pathway.

    This is the task-generic primitive. Multi-pathway (TEM-specific) fan-out is
    handled by the adapter/objective layer, not here.

    Args:
        logits: Predicted observation logits of shape ``(B, obs_dim)``.
        targets: Navigation supervision targets with ``observation_id`` and
            optional ``is_revisit``.

    Returns:
        :class:`NavigationPathwayMetrics` with raw counts (not normalised).
    """
    labels = coerce_observation_ids(targets.observation_id)
    revisit_mask = coerce_revisit_mask(targets.is_revisit, device=labels.device)
    return _evaluate_pathway(logits, labels, revisit_mask)


# =============================================================================
def _evaluate_pathway(  # -----------------------------------------------------
    logits: Tensor,
    labels: Tensor,
    revisit_mask: Tensor | None,
) -> NavigationPathwayMetrics:
    """Return all-step and revisit-only accuracy counts for one pathway."""
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
    return NavigationPathwayMetrics(
        correct_all=correct_all,
        count_all=count_all,
        correct_revisit=revisit,
        count_revisit=revisit_count,
    )


# =============================================================================
def coerce_observation_ids(  # ------------------------------------------------
    observation_id: Tensor,
) -> Tensor:
    """Return categorical observation labels with shape ``(B,)``.

    Applies squeeze or argmax normalisation so downstream objectives receive
    a clean 1-D integer target tensor regardless of how the environment emits
    observation identifiers.

    Args:
        observation_id: Raw observation-id tensor of shape ``(B,)`` or ``(B, 1)``
            or ``(B, obs_dim)`` (one-hot / soft-label).

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
def coerce_revisit_mask(  # ---------------------------------------------------
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
    "NavigationEpisodeSemantics",
    "NavigationPathwayMetrics",
    "coerce_observation_ids",
    "coerce_revisit_mask",
    "evaluate_observation_logits",
    "extract_navigation_episode_semantics",
]
