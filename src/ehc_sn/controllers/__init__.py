"""Controller families for EHC-SN rollout execution.

Three canonical controller families are exported here:

- **Deliberation** — :class:`ACTController`: adaptive computation-time
  deliberation with halt/continue logic.
- **Replay** — :class:`ReplayTrajectoryController`: stepwise recurrent replay
  over source-provided batch-major trajectory tensors.  Generic and
  task-agnostic; use for arena-family replay.
- **Online environment rollout** — :class:`RLController`: policy-driven
  interaction with a live :class:`~torchrl.envs.EnvBase` environment.

Internal helpers live in submodules prefixed with ``_``.
"""

from ehc_sn.controllers.act import ACTController, ACTControllerConfig, ACTRolloutState, ACTStepOutput
from ehc_sn.controllers.replay import (
    ReplayRolloutState,
    ReplayStepOutput,
    ReplayTrajectoryController,
    ReplayTrajectoryControllerConfig,
    ReplayTrajectoryRuntime,
)
from ehc_sn.controllers.rl import (
    InteractionRecord,
    RLBackboneOutput,
    RLController,
    RLControllerConfig,
    RLCriticOutput,
    RLPolicyOutput,
    RLRolloutBackbone,
    RLRolloutState,
    RLTaskRuntime,
)

__all__ = [
    # Deliberation family
    "ACTController",
    "ACTControllerConfig",
    "ACTRolloutState",
    "ACTStepOutput",
    # Replay family
    "ReplayRolloutState",
    "ReplayStepOutput",
    "ReplayTrajectoryController",
    "ReplayTrajectoryControllerConfig",
    "ReplayTrajectoryRuntime",
    # Online rollout — RL
    "InteractionRecord",
    "RLBackboneOutput",
    "RLController",
    "RLControllerConfig",
    "RLCriticOutput",
    "RLPolicyOutput",
    "RLRolloutBackbone",
    "RLRolloutState",
    "RLTaskRuntime",
]
