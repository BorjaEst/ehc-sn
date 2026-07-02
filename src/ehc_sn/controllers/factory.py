"""Controller construction from resolved specifications and dependencies.

**Experimental.**  Explicit constructors are the preferred pattern.
Import from this module only when configuration-driven construction
is genuinely required::

    from ehc_sn.controllers.factory import build_controller, ControllerDependencies
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Union

from ehc_sn.contracts.task_environment import TaskEnvironmentAdapter
from ehc_sn.contracts.task_runtime import TaskRuntime
from ehc_sn.controllers.base import StepController
from ehc_sn.controllers.deliberation.act import (
    ACTController,
    ACTControllerConfig,
)
from ehc_sn.controllers.deliberation.q_halting import (
    DeliberationQHaltingController,
    DeliberationQHaltingControllerConfig,
)
from ehc_sn.controllers.online.actor_critic import (
    RLController,
    RLControllerConfig,
)
from ehc_sn.controllers.replay.trajectory import (
    ReplayTrajectoryController,
    ReplayTrajectoryControllerConfig,
    ReplayTrajectoryRuntime,
)

# Re-export spec types for ``build_controller`` callers.
ControllerSpec = Union[
    ACTControllerConfig,
    DeliberationQHaltingControllerConfig,
    RLControllerConfig,
    ReplayTrajectoryControllerConfig,
]


# =============================================================================
@dataclass(frozen=True)
class ControllerDependencies:
    """Explicit dependency bundle for controller construction.

    Concrete controllers require specific subsets of these.  The factory
    validates availability before construction.
    """

    task_runtime: TaskRuntime[Any] | None = field(default=None)
    """Required by :class:`DeliberationQHaltingController`."""
    environment_adapter: TaskEnvironmentAdapter | None = field(default=None)
    """Required by :class:`RLController`."""
    replay_runtime: ReplayTrajectoryRuntime[Any] | None = field(default=None)
    """Required by :class:`ReplayTrajectoryController`."""
    environment: Any | None = field(default=None)
    """TorchRL ``EnvBase``; required by :class:`RLController`."""
    policy: Any | None = field(default=None)
    """Action policy; defaults to :class:`~ehp_sn.policies.categorical.CategoricalPolicy`
    when not provided."""


# =============================================================================
def build_controller(
    spec: ControllerSpec,
    *,
    backbone: Any,
    dependencies: ControllerDependencies = ControllerDependencies(),
) -> StepController:
    """Construct a controller from a resolved spec and dependencies.

    The factory constructs controllers only.  It does not resolve datasets,
    recipes, checkpoints, trainers, or full experiment graphs.

    Args:
        spec: Controller specification dataclass.
        backbone: Recurrent backbone satisfying the relevant protocol.
        dependencies: Explicit dependency bundle.

    Returns:
        A concrete controller instance.

    Raises:
        AssertionError: If a required dependency is missing for the
            requested controller family.
    """
    match spec:
        case ACTControllerConfig():
            return ACTController(backbone=backbone, config=spec)

        case DeliberationQHaltingControllerConfig():
            assert (
                dependencies.task_runtime is not None
            ), "DeliberationQHaltingController requires task_runtime dependency."
            return DeliberationQHaltingController(
                backbone=backbone,
                config=spec,
                runtime=dependencies.task_runtime,
            )

        case RLControllerConfig():
            assert (
                dependencies.environment is not None
            ), "RLController requires environment dependency."
            assert (
                dependencies.environment_adapter is not None
            ), "RLController requires environment_adapter dependency."
            return RLController(
                backbone=backbone,
                env=dependencies.environment,
                config=spec,
                runtime=dependencies.environment_adapter,
            )

        case ReplayTrajectoryControllerConfig():
            assert (
                dependencies.replay_runtime is not None
            ), "ReplayTrajectoryController requires replay_runtime dependency."
            return ReplayTrajectoryController(
                backbone=backbone,
                config=spec,
                runtime=dependencies.replay_runtime,
            )

        case _:
            raise TypeError(
                f"Unsupported controller spec type: {type(spec).__name__}"
            )


__all__ = [
    "ControllerDependencies",
    "ControllerSpec",
    "build_controller",
]
