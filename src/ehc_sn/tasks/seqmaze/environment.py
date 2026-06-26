"""TorchRL EnvBase for experimental SeqMaze online rollouts.

This module exposes SeqMaze as an interactive environment interface for the
deferred EnvBase-backed RL path.  The production HRM v2 reasoning path uses
    :class:`~ehc_sn.controllers.deliberation.q_halting.DeliberationQHaltingController`
with :class:`~ehc_sn.tasks.seqmaze.runtime.SeqMazeRuntime` instead, because
SeqMaze behaves as a fixed-instance deliberation task rather than a mutable
closed-loop world.

Use this environment only for experiments that explicitly need EnvBase/TensorDict
rollout semantics.

SeqMaze supervision labels, accuracy tracking, and reward shaping are
task-owned; they belong in the evaluation layer, not here.

TensorDict contract:
    state_spec / observation_spec:
        "schema_length" : (1,) int32  — S = N_max + T_max, static
        "step_count"    : (1,) int32  — steps taken so far
    action_spec:
        "action"        : (1,) int64  — halt index (0 = halt, 1 = continue)
    reward_spec:
        "reward"        : (1,) float32 — zero placeholder; overwritten by the capability/finalizer layer
    done_spec (auto):
        "done"          : (1,) bool
        "terminated"    : (1,) bool
        "truncated"     : (1,) bool
"""

from __future__ import annotations

import torch
from pydantic import BaseModel, Field
from tensordict import TensorDict, TensorDictBase
from torch import device as Device
from torchrl.data import Categorical, Composite, Unbounded
from torchrl.envs import EnvBase


# =============================================================================
class EnvConfig(BaseModel, extra="forbid"):
    """Configuration for :class:`SeqMazeEnv`."""

    max_episode_steps: int = Field(
        default=16,
        ge=1,
        description="Maximum steps per episode before truncation.",
    )
    schema_length: int = Field(
        ...,
        ge=1,
        description="Total schema-token sequence length (S = N_max + T_max).",
    )
    halt_action: int = Field(
        default=0,
        ge=0,
        description=(
            "Action index the environment interprets as 'halt' (terminates "
            "the episode). Must match the halt action configured in the "
            "RL controller."
        ),
    )


# =============================================================================
class SeqMazeEnv(EnvBase):
    """Mechanical env kernel for SeqMaze (TorchRL, batch-locked).

    Wraps a batched graph-instance as a TorchRL environment. Owns only the
    mechanical kernel: ``schema_length`` carry, ``step_count`` increment, and
    halt/truncation transitions. Reward is a zero placeholder; supervision
    labels and accuracy tracking are task-owned.

    This class is **not** the active HRM v2 deliberation surface. The active
    training path uses ``DeliberationQHaltingController`` with ``SeqMazeRuntime``
    and does not step through this env.

    The environment is batch-locked: all B slots step simultaneously.
    Per-slot auto-reset is handled by the controller, not this class.
    """

    batch_locked = True
    SPATIAL_GEOMETRY = "dag"

    def __init__(  # ----------------------------------------------------------
        self,
        config: EnvConfig,
        batch_size: int,
        device: Device | str | None = None,
    ) -> None:
        super().__init__(batch_size=[batch_size], device=device)
        self._config = config
        self._make_specs()

    @property
    def config(self) -> EnvConfig:
        """Return the environment configuration."""
        return self._config

    @property
    def spatial_geometry(self) -> str:
        """Return the declared spatial geometry for this environment."""
        return self.SPATIAL_GEOMETRY

    def _make_specs(  # -------------------------------------------------------
        self,
    ) -> None:
        bs = self.batch_size  # torch.Size([B])

        self.observation_spec = Composite(
            schema_length=Unbounded(shape=(*bs, 1), dtype=torch.int32),
            step_count=Unbounded(shape=(*bs, 1), dtype=torch.int32),
            shape=bs,
        )
        self.state_spec = self.observation_spec.clone()
        self.action_spec = Composite(
            action=Categorical(n=2, shape=(*bs, 1), dtype=torch.int64),
            shape=bs,
        )
        self.reward_spec = Unbounded(shape=(*bs, 1), dtype=torch.float32)

    def _reset(  # ------------------------------------------------------------
        self,
        tensordict: TensorDictBase | None,
    ) -> TensorDictBase:
        """Initialise episode state from external data.

        The controller injects ``schema_length`` from the dataloader.
        """
        if tensordict is None or tensordict.is_empty():
            raise ValueError(
                "SeqMazeEnv._reset requires tensordict with 'schema_length'."
            )
        if "schema_length" not in tensordict:
            raise ValueError(
                "SeqMazeEnv._reset requires tensordict with 'schema_length'."
            )

        B = self.batch_size[0]
        kw = {"device": self.device}
        return TensorDict(
            {
                "schema_length": tensordict["schema_length"],
                "step_count": torch.zeros(B, 1, dtype=torch.int32, **kw),
            },
            batch_size=self.batch_size,
            device=self.device,
        )

    @torch.no_grad()
    def _step(  # -------------------------------------------------------------
        self,
        tensordict: TensorDictBase,
    ) -> TensorDictBase:
        """Advance the mechanical env state from the current action.

        Owns only halt/truncation transitions. Emits a zero reward placeholder;
        task-owned reward semantics live in the capability/finalizer layer.
        """
        action = tensordict["action"]  # (B, 1)
        step_count = tensordict["step_count"]  # (B, 1)

        reward = torch.zeros_like(step_count, dtype=torch.float32)
        terminated = action == self._config.halt_action  # (B, 1)
        truncated = (step_count + 1) >= self._config.max_episode_steps  # (B, 1)
        done = terminated | truncated  # (B, 1)

        return TensorDict(
            {
                "schema_length": tensordict[
                    "schema_length"
                ],  # static — carry unchanged
                "step_count": step_count + 1,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "done": done,
            },
            batch_size=self.batch_size,
            device=self.device,
        )

    def _set_seed(  # ---------------------------------------------------------
        self,
        seed: int | None,
    ) -> None:
        """No-op: all randomness lives in the dataloader."""
        pass


# =============================================================================
__all__ = ["EnvConfig", "SeqMazeEnv"]
