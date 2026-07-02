"""Controller families for EHP-SN rollout execution.

One-step, task-agnostic control transitions over recurrent model execution.

A controller owns:

    slot admission/reset → backbone transition → control decision → next carry

The outer runner (``ehp_sn.rollouts.RecurrentRunner``) owns iteration.
See :doc:`/design/controllers` for the full architectural rationale.

Public API — stable names that survive internal refactors::

    from ehc_sn.controllers import (
        ACTController,
        ACTControllerConfig,
        DeliberationQHaltingController,
        DeliberationQHaltingControllerConfig,
        ReplayTrajectoryController,
        ReplayTrajectoryControllerConfig,
        RLController,
        RLControllerConfig,
        StepController,
    )

Advanced callers import infrastructure from defining modules::

    from ehc_sn.controllers.state import RolloutState, ReplayRolloutState
    from ehc_sn.controllers.records import QHaltingInteractionRecord
    from ehc_sn.controllers._base import BaseController
    from ehc_sn.controllers.factory import build_controller
"""

from ehc_sn.controllers.base import StepController
from ehc_sn.controllers.deliberation import (
    ACTController,
    ACTControllerConfig,
    DeliberationQHaltingController,
    DeliberationQHaltingControllerConfig,
)
from ehc_sn.controllers.online import RLController, RLControllerConfig
from ehc_sn.controllers.replay import (
    ReplayTrajectoryController,
    ReplayTrajectoryControllerConfig,
)

__all__ = [
    "ACTController",
    "ACTControllerConfig",
    "DeliberationQHaltingController",
    "DeliberationQHaltingControllerConfig",
    "ReplayTrajectoryController",
    "ReplayTrajectoryControllerConfig",
    "RLController",
    "RLControllerConfig",
    "StepController",
]
