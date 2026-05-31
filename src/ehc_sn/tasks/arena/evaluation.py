"""Arena task evaluation helpers.

Owns the canonical additive structural score, the benchmark-facing
:class:`ArenaScoreReport`, full-episode evaluation primitives, per-case
aggregation from rollout steps, and coercion utilities used by arena
objective bindings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor

from ehc_sn.metrics.step_metrics import RatioStat, StepMetrics
from ehc_sn.objectives.rollout import EvaluatedChunk

from .contracts import ArenaTargets, ArenaTaskInput


# =============================================================================
@dataclass(frozen=True)
class ArenaScoreReport:
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
@dataclass(frozen=True)
class ArenaEpisodeSemantics:
    """Episode-semantic flags attached to one arena task step."""

    step_count: Tensor | None = None
    episode_start: Tensor | None = None
    is_revisit: Tensor | None = None


# =============================================================================
@dataclass(frozen=True)
class ArenaStepScore:
    """Per-slot observation prediction result for one arena step.

    Local semantic state — stores per-slot booleans, not batch aggregates.
    Aggregate statistics (counts, accuracies) are computed by
    :func:`build_arena_score_report` when needed.
    """

    is_correct: Tensor
    """Whether the predicted observation id matched, shape ``(B,)`` bool."""
    is_revisit: Tensor | None
    """Whether this step is a revisit, shape ``(B,)`` bool, or ``None`` if unknown."""


# =============================================================================
def extract_arena_episode_semantics(
    payload: ArenaTaskInput | Mapping[str, Tensor],
) -> ArenaEpisodeSemantics:
    """Return the episode-semantic subset from an arena task payload."""
    if isinstance(payload, ArenaTaskInput):
        return ArenaEpisodeSemantics(
            step_count=payload.step_count,
            episode_start=payload.episode_start,
            is_revisit=None,
        )
    return ArenaEpisodeSemantics(
        step_count=payload.get("step_count"),
        episode_start=payload.get("episode_start"),
        is_revisit=payload.get("is_revisit"),
    )


# =============================================================================
def build_arena_step_score(
    logits: Tensor,
    targets: ArenaTargets,
) -> ArenaStepScore:
    """Return local per-slot prediction semantics for one arena step.

    Args:
        logits: Predicted observation logits of shape ``(B, obs_dim)``.
        targets: Arena supervision targets with ``observation_id`` and
            optional ``is_revisit``.

    Returns:
        :class:`ArenaStepScore` with per-slot boolean correctness and revisit flag.
    """
    labels = coerce_observation_ids(targets.observation_id)
    revisit_mask = coerce_revisit_mask(targets.is_revisit, device=labels.device)
    if logits.ndim != 2:
        raise ValueError(
            f"Arena observation logits must have shape (B, obs_dim), got {tuple(logits.shape)}."
        )
    if labels.ndim != 1:
        raise ValueError(
            f"Arena observation labels must have shape (B,), got {tuple(labels.shape)}."
        )
    if logits.shape[0] != labels.shape[0]:
        raise ValueError(
            "Arena logits/labels batch size mismatch: "
            f"logits batch={logits.shape[0]}, labels batch={labels.shape[0]}."
        )
    if revisit_mask is not None and revisit_mask.shape[0] != labels.shape[0]:
        raise ValueError(
            "Arena revisit mask/labels batch size mismatch: "
            f"revisit batch={revisit_mask.shape[0]}, labels batch={labels.shape[0]}."
        )
    is_correct = logits.argmax(dim=-1).eq(labels)
    return ArenaStepScore(is_correct=is_correct, is_revisit=revisit_mask)


# =============================================================================
def build_arena_score_report(
    step: ArenaStepScore,
) -> ArenaScoreReport:
    """Return the canonical additive structural score from local step semantics.

    Aggregates per-slot local state into count-bearing scalars suitable for
    cross-batch accumulation.  Callers accumulate correctly by summing counts
    rather than averaging accuracy scalars.

    Args:
        step: Per-slot prediction semantics from :func:`build_arena_step_score`.

    Returns:
        :class:`ArenaScoreReport` with accuracy scalars and raw counts.
    """
    dtype = torch.float32
    correct_all = step.is_correct.sum().to(dtype=dtype)
    count_all = step.is_correct.new_tensor(
        float(step.is_correct.shape[0]), dtype=dtype
    )
    if step.is_revisit is not None:
        correct_revisit = (
            (step.is_correct & step.is_revisit).sum().to(dtype=dtype)
        )
        count_revisit = step.is_revisit.sum().to(dtype=dtype)
    else:
        correct_revisit = step.is_correct.new_zeros(())
        count_revisit = step.is_correct.new_zeros(())
    return ArenaScoreReport(
        accuracy_all=correct_all / count_all.clamp_min(1.0),
        accuracy_revisit=correct_revisit / count_revisit.clamp_min(1.0),
        correct_all=correct_all,
        count_all=count_all,
        correct_revisit=correct_revisit,
        count_revisit=count_revisit,
    )


# =============================================================================
@dataclass(frozen=True)
class ArenaCaseMetrics:
    """Per-case scalar aggregates for one Arena-Struct evaluation case.

    Accumulated from per-step :class:`~ehc_sn.metrics.step_metrics.RatioStat`
    extras across all steps in an :class:`~ehc_sn.objectives.rollout.EvaluatedChunk`.
    """

    case_id: str
    accuracy_all: float
    accuracy_revisit: float
    correct_all: float
    count_all: float
    correct_revisit: float
    count_revisit: float
    n_steps: int
    is_failure: bool


# =============================================================================
def _accumulate_ratio_stat(
    key: str,
    steps: tuple,
) -> tuple[Tensor, Tensor]:
    """Walk a tuple of observed steps and sum numerator/denominator for one
    :class:`~ehc_sn.metrics.step_metrics.RatioStat` key from
    ``step.outputs.metrics.extras``."""
    num_sum = torch.zeros(())
    den_sum = torch.zeros(())
    for step in steps:
        extras: Mapping[str, RatioStat] = step.outputs.metrics.extras
        rs = extras.get(key)
        if rs is None:
            available = ", ".join(sorted(extras))
            raise KeyError(
                f"Metric key {key!r} not found in step {step.index} extras. "
                f"Available: {available}."
            )
        num_sum = num_sum + rs.numerator_sum.detach().cpu().float()
        den_sum = den_sum + rs.denominator_sum.detach().cpu().float()
    return num_sum, den_sum


# =============================================================================
def aggregate_arena_case_metrics(
    evaluated: EvaluatedChunk,
    case_id: str,
    *,
    acc_all_key: str = "accuracy_obs_inference_all",
    acc_revisit_key: str = "accuracy_obs_inference_revisit",
    failure_threshold_revisit: float = 0.5,
) -> ArenaCaseMetrics:
    """Return per-case scalar aggregates from an evaluated Arena rollout.

    Accumulates ``RatioStat`` values from every step in *evaluated* using
    the metric keys configured for the model family.

    Args:
        evaluated: Objective-scored rollout chunk containing per-step metrics.
        case_id: Provider-owned case identifier.
        acc_all_key: Extras key for all-step accuracy (family-dependent).
        acc_revisit_key: Extras key for revisit-step accuracy (family-dependent).
        failure_threshold_revisit: Revisit accuracy below which the case is
            flagged as a failure.

    Returns:
        :class:`ArenaCaseMetrics` with scalar aggregates.
    """
    if not evaluated.steps:
        raise ValueError(
            f"Cannot aggregate arena case {case_id!r}: evaluated chunk has no "
            f"observed steps."
        )

    correct_all, count_all = _accumulate_ratio_stat(
        acc_all_key, evaluated.steps
    )
    correct_revisit, count_revisit = _accumulate_ratio_stat(
        acc_revisit_key, evaluated.steps
    )

    count_all_safe = float(count_all.clamp_min(1.0).item())
    count_revisit_safe = float(count_revisit.clamp_min(1.0).item())

    metrics = ArenaCaseMetrics(
        case_id=case_id,
        accuracy_all=float(correct_all.item()) / count_all_safe,
        accuracy_revisit=float(correct_revisit.item()) / count_revisit_safe,
        correct_all=float(correct_all.item()),
        count_all=float(count_all.item()),
        correct_revisit=float(correct_revisit.item()),
        count_revisit=float(count_revisit.item()),
        n_steps=len(evaluated.steps),
        is_failure=float(correct_revisit.item()) / count_revisit_safe
        < failure_threshold_revisit,
    )
    return metrics


# =============================================================================
def coerce_observation_ids(
    observation_id: Tensor,
) -> Tensor:
    """Return categorical observation labels with shape ``(B,)``.

    Accepts ``(B,)``, ``(B, 1)``, or ``(B, obs_dim)`` (one-hot / soft-label)
    inputs and normalizes to a clean 1-D integer tensor regardless of how the
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
    "ArenaCaseMetrics",
    "ArenaEpisodeSemantics",
    "ArenaScoreReport",
    "ArenaStepScore",
    "aggregate_arena_case_metrics",
    "build_arena_score_report",
    "build_arena_step_score",
    "coerce_observation_ids",
    "coerce_revisit_mask",
    "extract_arena_episode_semantics",
]
