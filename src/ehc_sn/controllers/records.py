"""Controller-produced interaction records and step outputs.

All major record definitions remain in their original defining modules
for backward compatibility.  This module re-exports them under a single
canonical import path.

Stable import::

    from ehc_sn.controllers.records import (
        ACTControllerStepOutput,
        QHaltingInteractionRecord,
        ReplayStepOutput,
    )
"""

from ehc_sn.controllers.contracts.actor_critic import (
    QHaltingInteractionRecord,
)
from ehc_sn.controllers.deliberation.act import ACTControllerStepOutput
from ehc_sn.controllers.replay.trajectory import ReplayStepOutput

__all__ = [
    "ACTControllerStepOutput",
    "QHaltingInteractionRecord",
    "ReplayStepOutput",
]
