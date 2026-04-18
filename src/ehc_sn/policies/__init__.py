"""Reusable action-selection policy public export surface."""

from typing import Annotated

from pydantic import Field
from typing_extensions import TypeAlias

from ._base import ActionPolicy, PolicyDecision, PolicyInput
from .random_walk import RandomWalkPolicy, RandomWalkPolicyConfig
from .stay import StayPolicy, StayPolicyConfig

# Discriminated union of all scripted (non-learned) policy configs.
# Pydantic selects the concrete type by matching the ``kind`` field.
ScriptedPolicyConfig: TypeAlias = Annotated[
    StayPolicyConfig | RandomWalkPolicyConfig,
    Field(discriminator="kind"),
]

__all__ = [
    "ActionPolicy",
    "PolicyDecision",
    "PolicyInput",
    "RandomWalkPolicy",
    "RandomWalkPolicyConfig",
    "ScriptedPolicyConfig",
    "StayPolicy",
    "StayPolicyConfig",
]
