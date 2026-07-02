"""Controller carry types used during execution.

All major carry definitions remain in their original defining modules
for backward compatibility.  This module re-exports them under a single
canonical import path.

Stable import::

    from ehc_sn.controllers.state import RolloutState

Controller-family carry types::

    from ehc_sn.controllers.state import (
        ACTRolloutState,
        DeliberationQHaltingRolloutState,
        RLRolloutState,
        ReplayRolloutState,
    )

Snapshot types::

    from ehc_sn.controllers.state import CarrySnapshot
"""

from ehc_sn.controllers._base import RolloutState
from ehc_sn.controllers.deliberation.act import ACTRolloutState
from ehc_sn.controllers.deliberation.q_halting import (
    DeliberationQHaltingRolloutState,
)
from ehc_sn.controllers.online.actor_critic import RLRolloutState
from ehc_sn.controllers.replay.trajectory import ReplayRolloutState
from ehc_sn.rollouts.runtime import CarrySnapshot

__all__ = [
    "ACTRolloutState",
    "CarrySnapshot",
    "DeliberationQHaltingRolloutState",
    "RLRolloutState",
    "ReplayRolloutState",
    "RolloutState",
]
