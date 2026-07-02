"""Online actor-critic (RL) controller family.

Leaf module: :mod:`ehp_sn.controllers.online.actor_critic`.
"""

from ehc_sn.controllers.online.actor_critic import (
    RLController,
    RLControllerConfig,
)

__all__ = [
    "RLController",
    "RLControllerConfig",
]
