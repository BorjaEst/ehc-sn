"""Canonical public surface for objective modules.

Three contract levels are defined by this package:

1. **Atomic objectives** (``supervised/``, ``control/``): stateless
   ``nn.Module`` components that compute one differentiable loss from
   typed tensors and return an ``ObjectiveResult``.  Batch-oriented; do
   not consume ``StepRecord`` or rollout chunks.

2. **Composite regime step scorers** (``composites/``): learning-regime
   step scorers that compose atomic objectives into family-specific
   losses (ACT, TEM, hybrid RL).  A composite may consume one
   ``StepRecord`` with typed supervision, but it must not initiate
   rollout traversal or import task semantics.

3. **Rollout traversal** is owned by ``rollouts/scoring.py`` and
   ``training/rollout.py``, not by any module under ``objectives/``.

Prefer ``from ehc_sn.objectives import ...`` over any sub-module import.
"""

from ehc_sn.metrics.token import (
    AccuracyStats,
    build_token_step_metrics,
    compute_accuracy_stats,
)
from ehc_sn.objectives.composites.act import (
    ACTStepLosses,
    ACTSupervisedScorer,
    ACTSupervisedScorerConfig,
    ACTSupervisedStep,
)
from ehc_sn.objectives.composites.hybrid_rl import (
    HybridRLLossConfig,
    HybridRLLosses,
    HybridRLObjective,
    HybridRLObjectiveStep,
)
from ehc_sn.objectives.composites.tem import (
    TEMLosses,
    TEMObjective,
    TEMObjectiveConfig,
    TEMObjectiveStep,
    TEMStepOutput,
    VariationalLosses,
    build_variational_step_metrics,
)
from ehc_sn.objectives.supervised.token import (
    IGNORE_LABEL_ID,
)
from ehc_sn.objectives.types import ObjectiveResult

# =============================================================================
__all__ = [
    # types
    "ObjectiveResult",
    # token family — implementation / compat
    "IGNORE_LABEL_ID",
    "AccuracyStats",
    "build_token_step_metrics",
    "compute_accuracy_stats",
    # variational family — canonical
    "VariationalLosses",
    # variational family — implementation / compat
    "build_variational_step_metrics",
    # act — canonical
    "ACTStepLosses",
    "ACTSupervisedScorer",
    "ACTSupervisedScorerConfig",
    "ACTSupervisedStep",
    # hybrid rl — batch-loss path (no *Objective* aliases; not a rollout scorer)
    "HybridRLLossConfig",
    "HybridRLObjective",
    "HybridRLLosses",
    "HybridRLObjectiveStep",
    # continuous field — canonical
    "ACTSupervisedScorerConfig",
    "ACTSupervisedScorer",
    "ACTSupervisedStep",
    # tem — canonical
    "TEMObjectiveConfig",
    "TEMObjective",
    "TEMObjectiveStep",
    "TEMLosses",
    "TEMStepOutput",
]
