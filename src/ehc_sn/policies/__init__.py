"""Reusable action-selection policies."""

from typing import TypeAlias

from ehc_sn.policies._base import ActionPolicy, PolicyDecision, PolicyInput
from ehc_sn.policies.categorical import CategoricalPolicy, CategoricalPolicyConfig
from ehc_sn.policies.random_walk import RandomWalkPolicy, RandomWalkPolicyConfig
from ehc_sn.policies.stay import StayPolicy, StayPolicyConfig

ScriptedPolicyConfig: TypeAlias = StayPolicyConfig | RandomWalkPolicyConfig

# =================================================================================================
__all__ = [
    "ActionPolicy", "CategoricalPolicy", "CategoricalPolicyConfig", "PolicyDecision", "PolicyInput",
    "RandomWalkPolicy", "RandomWalkPolicyConfig", "ScriptedPolicyConfig", "StayPolicy",
    "StayPolicyConfig",
]  # fmt: skip
