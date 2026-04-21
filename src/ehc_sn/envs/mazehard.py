"""Mazehard deliberation environment (TorchRL).

The environment owns only the mechanical rollout kernel: static token tape,
step count, and halt/truncation transitions. MazeHard supervision labels,
accuracy tracking, and reward shaping are task-owned and are attached by the
task runtime after env stepping.

TensorDict contract:
    state_spec / observation_spec:
        "input_ids"   : (S,) int64   — static token sequence for the episode
        "step_count"  : ()   int32   — steps taken so far
    action_spec:
        "action"      : ()   int64   — halt index (0 = halt, 1 = continue)
    reward_spec:
        "reward"      : (1,) float32 — placeholder overwritten by task runtime
    done_spec (auto):
        "done"       : (1,) bool
        "terminated" : (1,) bool
        "truncated"  : (1,) bool
"""

import torch
from pydantic import BaseModel, Field
from tensordict import TensorDict, TensorDictBase
from torch import device as Device
from torchrl.data import Categorical, Composite, Unbounded
from torchrl.envs import EnvBase


# =================================================================================================
class EnvConfig(BaseModel, extra="forbid"):
    """Configuration for :class:`MazeHardEnv`."""

    max_episode_steps: int = Field(
        default=10,
        ge=1,
        description="Maximum steps per episode before truncation.",
    )
    seq_length: int = Field(
        ...,
        ge=1,
        description="Length of input and prediction sequences.",
    )
    vocab_size: int = Field(
        ...,
        ge=1,
        description="Token vocabulary size retained for MazeHard config compatibility.",
    )
    halt_action: int = Field(
        default=0,
        ge=0,
        description=(
            "Action index the environment interprets as 'halt' (terminates the episode). "
            "Must match the halt action configured in the RL controller."
        ),
    )


# =================================================================================================
class MazeHardEnv(EnvBase):
    """Deliberation environment for maze-solving (TorchRL, batch-locked).

    Wraps a batched token-prediction task as a TorchRL environment.
    No subprocess, no numpy — all ops are batched torch kernels.

    Usage::

        env = MazeHardEnv(config, batch_size=32, device="cuda")
        td = env.reset(TensorDict({"input_ids": x}, batch_size=[32]))
        td["action"] = policy(td)
        td = env.step(td)

    The environment is batch-locked: all B slots step simultaneously.
    Per-slot auto-reset is handled by the controller, not this class.
    """

    batch_locked = True
    SPATIAL_GEOMETRY = "maze"

    def __init__(  # ------------------------------------------------------------------------------
        self, config: EnvConfig, batch_size: int, device: Device | str | None = None,
    ) -> None:  # fmt: skip
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

    def _make_specs(  # ---------------------------------------------------------------------------
        self,
    ) -> None:  # fmt: skip
        S = self._config.seq_length
        bs = self.batch_size  # torch.Size([B])

        self.observation_spec = Composite(
            input_ids=Unbounded(shape=(*bs, S), dtype=torch.int64),
            step_count=Unbounded(shape=(*bs, 1), dtype=torch.int32),
            shape=bs,
        )
        self.state_spec = self.observation_spec.clone()
        self.action_spec = Composite(
            action=Categorical(n=2, shape=(*bs, 1), dtype=torch.int64),
            shape=bs,
        )
        self.reward_spec = Unbounded(shape=(*bs, 1), dtype=torch.float32)

    def _reset(  # -------------------------------------------------------------------------------
        self, tensordict: TensorDictBase | None,
    ) -> TensorDictBase:  # fmt: skip
        """Initialise episode state from external data.

        The controller injects ``input_ids`` from the dataloader.
        """
        if tensordict is None or tensordict.is_empty():
            raise ValueError("MazeHardEnv._reset requires tensordict with 'input_ids'.")

        B = self.batch_size[0]
        kw = {"device": self.device}
        return TensorDict(
            {
                "input_ids": tensordict["input_ids"],
                "step_count": torch.zeros(B, 1, dtype=torch.int32, **kw),
            },
            batch_size=self.batch_size,
            device=self.device,
        )

    @torch.no_grad()
    def _step(  # --------------------------------------------------------------------------------
        self, tensordict: TensorDictBase,
    ) -> TensorDictBase:  # fmt: skip
        """Advance the mechanical env state from the current action.

        The env owns only halt/truncation transitions. It emits a zero reward
        placeholder so the task runtime can attach task-owned reward semantics
        after stepping.
        """
        action = tensordict["action"]  # (B, 1)
        step_count = tensordict["step_count"]  # (B, 1)

        reward = torch.zeros_like(step_count, dtype=torch.float32)
        terminated = action == self._config.halt_action  # (B, 1)
        truncated = (step_count + 1) >= self._config.max_episode_steps  # (B, 1)
        done = terminated | truncated  # (B, 1)

        return TensorDict(
            {
                "input_ids": tensordict["input_ids"],  # static — carry unchanged
                "step_count": step_count + 1,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "done": done,
            },
            batch_size=self.batch_size,
            device=self.device,
        )

    def _set_seed(self, seed: int | None) -> None:  # --------------------------------------------
        """No-op: all randomness lives in the dataloader."""
        pass


# =================================================================================================
__all__ = ["EnvConfig", "MazeHardEnv"]
