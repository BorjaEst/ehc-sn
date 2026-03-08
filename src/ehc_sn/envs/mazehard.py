"""Mazehard deliberation environment (TorchRL).

The agent submits predictions as actions. The environment evaluates
prediction quality and returns improvement-based reward.

TensorDict contract:
    state_spec / observation_spec:
        "inputs"        : (S,)   int64   — static token sequence
        "labels"        : (S,)   int64   — ground truth labels
        "prev_accuracy" : ()     float32 — accuracy at previous step
        "step_count"    : ()     int32   — steps taken so far
    action_spec:
        "action"  : ()     int64   — halt index (0 = halt, 1 = continue)
        "logits"  : (S, V) float32 — prediction logits from the policy
    reward_spec:
        "reward"    : (1,) float32
    done_spec (auto):
        "done"       : (1,) bool
        "terminated" : (1,) bool
        "truncated"  : (1,) bool
"""

import torch
from pydantic import BaseModel, Field
from tensordict import TensorDict, TensorDictBase
from torchrl.data import Categorical, Composite, Unbounded
from torchrl.envs import EnvBase

IGNORE_LABEL_ID = -100


# =================================================================================================
class EnvConfig(BaseModel, extra="forbid"):
    """Configuration for :class:`MazeHardEnv`."""

    max_steps: int = Field(default=10, ge=1, description="Maximum steps per episode before truncation.")
    seq_length: int = Field(..., ge=1, description="Length of input and prediction sequences.")
    vocab_size: int = Field(..., ge=1, description="Size of the token vocabulary.")
    halt_action: int = Field(
        default=0,
        ge=0,
        description=(
            "Action index the environment interprets as 'halt' (terminates the episode). "
            "Must match RLLossConfig.halt_action."
        ),
    )


# =================================================================================================
class MazeHardEnv(EnvBase):
    """Deliberation environment for maze-solving (TorchRL, batch-locked).

    Wraps a batched token-prediction task as a TorchRL environment.
    No subprocess, no numpy — all ops are batched torch kernels.

    Usage::

        env = MazeHardEnv(config, batch_size=32, device="cuda")
        td = env.reset(TensorDict({"inputs": x, "labels": y}, batch_size=[32]))
        td["action"] = policy(td)
        td = env.step(td)

    The environment is batch-locked: all B slots step simultaneously.
    Per-slot auto-reset is handled by the controller, not this class.
    """

    batch_locked = True

    def __init__(  # ------------------------------------------------------------------------------
        self, config: EnvConfig, batch_size: int, device: torch.device | str | None = None,
    ) -> None:  # fmt: skip
        super().__init__(batch_size=[batch_size], device=device)
        self._config = config
        self._make_specs()

    @property
    def config(self) -> EnvConfig:
        """Return the environment configuration."""
        return self._config

    def _make_specs(  # ---------------------------------------------------------------------------
        self,
    ) -> None:  # fmt: skip
        S = self._config.seq_length
        V = self._config.vocab_size
        bs = self.batch_size  # torch.Size([B])

        self.observation_spec = Composite(
            inputs=Unbounded(shape=(*bs, S), dtype=torch.int64),
            labels=Unbounded(shape=(*bs, S), dtype=torch.int64),
            prev_accuracy=Unbounded(shape=(*bs, 1), dtype=torch.float32),
            step_count=Unbounded(shape=(*bs, 1), dtype=torch.int32),
            shape=bs,
        )
        self.state_spec = self.observation_spec.clone()
        self.action_spec = Composite(
            action=Categorical(n=2, shape=(*bs, 1), dtype=torch.int64),
            logits=Unbounded(shape=(*bs, S, V), dtype=torch.float32),
            shape=bs,
        )
        self.reward_spec = Unbounded(shape=(*bs, 1), dtype=torch.float32)

    def _reset(  # -------------------------------------------------------------------------------
        self, tensordict: TensorDictBase | None,
    ) -> TensorDictBase:  # fmt: skip
        """Initialise episode state from external data.

        The controller injects ``inputs`` and ``labels`` from the dataloader.
        """
        if tensordict is None or tensordict.is_empty():
            raise ValueError("MazeHardEnv._reset requires tensordict with 'inputs' and 'labels'.")

        B = self.batch_size[0]
        kw = {"device": self.device}
        return TensorDict(
            {
                "inputs": tensordict["inputs"],
                "labels": tensordict["labels"],
                "prev_accuracy": torch.zeros(B, 1, **kw),
                "step_count": torch.zeros(B, 1, dtype=torch.int32, **kw),
            },
            batch_size=self.batch_size,
            device=self.device,
        )

    @torch.no_grad()
    def _step(  # --------------------------------------------------------------------------------
        self, tensordict: TensorDictBase,
    ) -> TensorDictBase:  # fmt: skip
        """Compute reward, done-flags, and next state from action + current state.

        Reward = exp(acc) - exp(prev_acc): smooth, bounded, rewards improvement.
        Terminated when agent selects halt_action. Truncated at max_steps.
        """
        logits = tensordict["logits"]  # (B, S, V)
        action = tensordict["action"]  # (B, 1)
        labels = tensordict["labels"]  # (B, S)
        prev_accuracy = tensordict["prev_accuracy"]  # (B, 1)
        step_count = tensordict["step_count"]  # (B, 1)

        mask = labels != IGNORE_LABEL_ID  # (B, S) bool
        counts = mask.sum(-1, keepdim=True).clamp_min(1).float()  # (B, 1)
        acc = ((logits.argmax(-1) == labels) & mask).sum(-1, keepdim=True).float() / counts

        reward = torch.exp(acc) - torch.exp(prev_accuracy)  # (B, 1)
        terminated = action == self._config.halt_action  # (B, 1)
        truncated = (step_count + 1) >= self._config.max_steps  # (B, 1)
        done = terminated | truncated  # (B, 1)

        return TensorDict(
            {
                "inputs": tensordict["inputs"],  # static — carry unchanged
                "labels": labels,
                "prev_accuracy": acc,
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
