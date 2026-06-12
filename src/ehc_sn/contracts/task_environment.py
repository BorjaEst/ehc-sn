# SPDX-License-Identifier: MIT
"""Task-owned environment adapter contract for EnvBase-backed online RL.

Implement :class:`TaskEnvironmentAdapter` together with an ``EnvBase`` subclass
when a task exposes an interactive environment whose observations, rewards,
and termination arise from stepping external state.

Fixed-instance reasoning tasks should usually implement
:class:`~ehc_sn.contracts.task_step.TaskStepEvaluator` instead.
"""

from __future__ import annotations

from typing import Protocol

from tensordict import TensorDictBase
from torch import Tensor

from ehc_sn.controllers.contracts.actor_critic import OnlineBootstrapCarry
from ehc_sn.types import Batch


# =============================================================================
class TaskEnvironmentAdapter(Protocol):
    """Task-owned adapter for EnvBase-backed online RL rollouts.

    Converts task batches, model outputs, and controller state into the
    TensorDict format expected by a TorchRL ``EnvBase``.

    Implement this when a task exposes an interactive environment.  Fixed-
    instance reasoning tasks should usually implement
    :class:`~ehc_sn.contracts.task_step.TaskStepEvaluator` instead.
    """

    def build_reset_td(self, batch: Batch) -> TensorDictBase:
        """Build an initial environment TensorDict from a batch sample."""
        ...

    def build_env_step_td(
        self,
        env_td: TensorDictBase,
        *,
        reset_mask: Tensor,
        action: Tensor,
        task_output: object,
        data: Batch,
    ) -> TensorDictBase:
        """Prepare the concrete env input for the current step."""
        ...

    def finalize_env_transition(
        self,
        previous_env_td: TensorDictBase,
        next_env_td: TensorDictBase,
        *,
        reset_mask: Tensor,
        action: Tensor,
        task_output: object,
        data: Batch,
    ) -> TensorDictBase:
        """Attach task-owned reward or diagnostic fields after env.step()."""
        ...

    def extract_next_step_obs(self, carry: OnlineBootstrapCarry) -> Batch:
        """Return the task-shaped bootstrap observation batch from carry."""
        ...


# =============================================================================
__all__ = [
    "TaskEnvironmentAdapter",
]
