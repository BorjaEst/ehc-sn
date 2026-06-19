"""TorchRL EnvBase for goaltrace field-prediction (stub).

Goaltrace is a single-step field prediction task with no interactive
environment for online rollouts.  The production training path uses
supervised single-step forward passes (``ACTSupervisedModule`` with
``single_step=True``), and the ACT deliberation path uses
``ACTController`` with ``ACTSupervisedScorer`` (field modality) — neither path
requires an interactive environment.

This module exists for contract symmetry with other task families that
expose an interactive environment.  Use this environment only for
experiments that explicitly need ``EnvBase`` / ``TensorDict`` rollout
semantics.
"""

from __future__ import annotations

import torch
from pydantic import BaseModel, Field
from torch import device as Device
from torchrl.data import Categorical, Composite, Unbounded
from torchrl.envs import EnvBase


# =============================================================================
class EnvConfig(BaseModel, extra="forbid"):
    """Configuration for :class:`GoaltraceEnv`.

    Attributes:
        max_episode_steps: Maximum steps per episode before truncation.
        schema_length: Total schema-token sequence length (N = num_observations).
        halt_action: Action index interpreted as 'halt'.
    """

    max_episode_steps: int = Field(
        default=8,
        ge=1,
        description="Maximum steps per episode before truncation.",
    )
    schema_length: int = Field(
        ...,
        ge=1,
        description="Schema-token sequence length (N = num_observations).",
    )
    halt_action: int = Field(
        default=0,
        ge=0,
        description="Action index that terminates the episode.",
    )


# =============================================================================
class GoaltraceEnv(EnvBase):
    """Minimal TorchRL environment for goaltrace field-prediction.

    A single-step environment where the action space is ``{halt, continue}``
    and the observation is the schema-token sequence length.  The environment
    does not implement state transitions — the observed task (observation IDs,
    weights, flags) is static across deliberation steps.
    """

    def __init__(  # ----------------------------------------------------------
        self,
        config: EnvConfig,
        *,
        device: Device | None = None,
    ) -> None:
        super().__init__(device=device)
        self._config = config

        # Observation: schema length (static per episode)
        self.observation_spec = Composite(
            schema_length=Categorical(
                n=config.schema_length,
                dtype=torch.int32,
                device=device,
            ),
            shape=(),
            device=device,
        )

        # Action: halt index (0 = halt, 1 = continue)
        self.action_spec = Composite(
            action=Categorical(
                n=2,
                dtype=torch.int64,
                device=device,
            ),
            shape=(),
            device=device,
        )

        # Reward: zero placeholder; overwritten by the capability layer
        self.reward_spec = Composite(
            reward=Unbounded(
                shape=(1,),
                dtype=torch.float32,
                device=device,
            ),
            shape=(),
            device=device,
        )

    def _reset(self, tensordict: torch.Tensor) -> torch.Tensor:
        """Reset the environment (no-op for single-step field prediction)."""
        return tensordict

    def _step(self, tensordict: torch.Tensor) -> torch.Tensor:
        """Step the environment (no-op; reward/done handled by capability)."""
        return tensordict


# =============================================================================
__all__ = [
    "EnvConfig",
    "GoaltraceEnv",
]
