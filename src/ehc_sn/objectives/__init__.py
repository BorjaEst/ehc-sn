"""Canonical public surface for rollout-scoring objective modules.

All objective-scoring implementations reside in the submodules of this package.
For the ACT and TEM families the canonical names follow the ``*Objective*``
vocabulary; the legacy ``*LossHead*`` names are preserved as backward-compatible
aliases. The hybrid RL family uses ``*LossHead*`` names directly — it is a
learner-owned batch-loss path, not a rollout-scoring objective.

Prefer ``from ehc_sn.objectives import ...`` over any sub-module import.
"""

from ehc_sn.objectives._base import BaseObjective
from ehc_sn.objectives._token import (
    IGNORE_LABEL_ID,
    AccuracyStats,
    TokenLosses,
    TokenLossHeadBase,
    TokenSupervisionBinding,
    build_token_step_metrics,
    compute_accuracy_stats,
    compute_lm_loss_sum,
)
from ehc_sn.objectives._variational import (
    VariationalLosses,
    VariationalLossHeadBase,
    VariationalLossStep,
    build_variational_step_metrics,
    get_reg_term,
    require_latent_relation,
)
from ehc_sn.objectives.act import ACTLossConfig, ACTLossHead, ACTLossStep, ACTTaskBinding
from ehc_sn.objectives.ehc import EHCLosses, EHCObjective, EHCObjectiveBinding, EHCObjectiveConfig, EHCObjectiveStep
from ehc_sn.objectives.hybrid_rl import HybridRLLossConfig, HybridRLLosses, HybridRLLossHead, HybridRLLossStep
from ehc_sn.objectives.tem import TEMLosses, TEMObjective, TEMObjectiveBinding, TEMObjectiveConfig, TEMObjectiveStep

# ── Canonical objective aliases (preferred) ──────────────────────────────────
# ACT
ACTObjectiveConfig = ACTLossConfig
ACTObjective = ACTLossHead
ACTObjectiveStep = ACTLossStep
ACTObjectiveBinding = ACTTaskBinding
# Base families
TokenObjectiveBase = TokenLossHeadBase
VariationalObjectiveBase = VariationalLossHeadBase

__all__ = [
    # base
    "BaseObjective",
    # token family — canonical
    "TokenObjectiveBase",
    # token family — implementation / compat
    "IGNORE_LABEL_ID",
    "AccuracyStats",
    "TokenLossHeadBase",
    "TokenLosses",
    "TokenSupervisionBinding",
    "build_token_step_metrics",
    "compute_accuracy_stats",
    "compute_lm_loss_sum",
    # variational family — canonical
    "VariationalObjectiveBase",
    # variational family — implementation / compat
    "VariationalLossHeadBase",
    "VariationalLosses",
    "VariationalLossStep",
    "build_variational_step_metrics",
    "get_reg_term",
    "require_latent_relation",
    # act — canonical
    "ACTObjectiveConfig",
    "ACTObjective",
    "ACTObjectiveStep",
    "ACTObjectiveBinding",
    # act — compat
    "ACTLossConfig",
    "ACTLossHead",
    "ACTLossStep",
    "ACTTaskBinding",
    # ehc — canonical
    "EHCObjectiveBinding",
    "EHCObjectiveConfig",
    "EHCObjective",
    "EHCObjectiveStep",
    "EHCLosses",
    # hybrid rl — batch-loss path (no *Objective* aliases; not a rollout scorer)
    "HybridRLLossConfig",
    "HybridRLLossHead",
    "HybridRLLosses",
    "HybridRLLossStep",
    # tem — canonical
    "TEMObjectiveBinding",
    "TEMObjectiveConfig",
    "TEMObjective",
    "TEMObjectiveStep",
    "TEMLosses",
]
