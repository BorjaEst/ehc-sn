"""Canonical public surface for rollout-scoring objective modules.

All objective-scoring implementations reside in the submodules of this package.
The canonical names follow the ``*Objective*`` vocabulary; the legacy ``*LossHead*``
names are preserved as backward-compatible aliases.

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
from ehc_sn.objectives.hybrid_rl import HybridRLLossConfig, HybridRLLosses, HybridRLLossHead, HybridRLLossStep, HybridRLObjectiveBinding
from ehc_sn.objectives.rl import RewardBearingStep, RLLossConfig, RLLosses, RLLossHead, RLLossStep, RLObjectiveBinding
from ehc_sn.objectives.tem import (
    TEMLossConfig,
    TEMLosses,
    TEMLossHead,
    TEMLossStep,
    TEMObjective,
    TEMObjectiveBinding,
    TEMObjectiveConfig,
    TEMObjectiveStep,
    TEMSupervisionBinding,
)
from ehc_sn.objectives.var import (
    LatentCode,
    VARLossConfig,
    VARLosses,
    VARLossHead,
    VARLossStep,
    VARObjective,
    VARObjectiveBinding,
    VARObjectiveConfig,
    VARObjectiveStep,
)

# ── Canonical objective aliases (preferred) ──────────────────────────────────
# ACT
ACTObjectiveConfig = ACTLossConfig
ACTObjective = ACTLossHead
ACTObjectiveStep = ACTLossStep
ACTObjectiveBinding = ACTTaskBinding
# RL — pure reward-first family (canonical names)
RLObjectiveConfig = RLLossConfig
RLObjective = RLLossHead
RLObjectiveStep = RLLossStep
# RL — hybrid token-supervised + actor-critic family (canonical names)
HybridRLObjectiveConfig = HybridRLLossConfig
HybridRLObjective = HybridRLLossHead
HybridRLObjectiveStep = HybridRLLossStep
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
    # rl — canonical (pure reward-first)
    "RLObjectiveConfig",
    "RLObjective",
    "RLObjectiveStep",
    # rl — compat / binding
    "RewardBearingStep",
    "RLLossConfig",
    "RLLossHead",
    "RLLosses",
    "RLLossStep",
    "RLObjectiveBinding",
    # hybrid rl — canonical
    "HybridRLObjectiveConfig",
    "HybridRLObjective",
    "HybridRLObjectiveStep",
    # hybrid rl — compat
    "HybridRLLossConfig",
    "HybridRLLossHead",
    "HybridRLLosses",
    "HybridRLLossStep",
    "HybridRLObjectiveBinding",
    # tem — canonical
    "TEMObjectiveBinding",
    "TEMObjectiveConfig",
    "TEMObjective",
    "TEMObjectiveStep",
    # tem — compat
    "TEMSupervisionBinding",
    "TEMLossConfig",
    "TEMLossHead",
    "TEMLosses",
    "TEMLossStep",
    # var — canonical
    "VARObjectiveBinding",
    "VARObjectiveConfig",
    "VARObjective",
    "VARObjectiveStep",
    # var — compat
    "LatentCode",
    "VARLossConfig",
    "VARLossHead",
    "VARLosses",
    "VARLossStep",
]
