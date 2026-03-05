""" """

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pydantic import BaseModel, Field

IGNORE_LABEL_ID = -100


# =================================================================================================
class EnvConfig(BaseModel, extra="forbid"):
    """ """

    max_steps: int = Field(
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
        description="Size of the token vocabulary.",
    )
    halt_action: int = Field(
        default=0,
        ge=0,
        description=(
            "Action index the environment interprets as \u2018halt\u2019 (terminates the episode). "
            "Must match RLLossConfig.halt_action so the training loop and env agree on semantics."
        ),
    )


# =================================================================================================
class Env(gym.Env):
    """Deliberation environment for maze-solving.

    The agent submits predictions as actions. The environment evaluates
    prediction quality and returns improvement-based reward.

    Observation: raw token sequence (static across steps).
    Action: {"halt": Discrete(2), "prediction": Box(vocab_size, seq_length)}.
    Reward: accuracy_t - accuracy_{t-1} (incremental improvement).
    Terminated: agent chose halt.
    Truncated: step_count >= max_steps.
    """

    metadata = {"render_modes": []}

    def __init__(  # ------------------------------------------------------------------------------
        self, config: EnvConfig,
    ) -> None:  # fmt: skip
        super().__init__()
        """ """
        self._config = config
        self.observation_space = spaces.Dict(
            {
                "inputs": spaces.MultiDiscrete(np.full(*self.shape)),
            }
        )
        self.action_space = spaces.Dict(
            {
                "halt": spaces.Discrete(2),
                "prediction": spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32),
            }
        )
        # Episode state (set on reset)
        self._inputs: np.ndarray | None = None
        self._labels: np.ndarray | None = None
        self._prev_accuracy: float = 0.0
        self._step_count: int = 0

    @property
    def config(self):
        """ """
        return self._config

    @property
    def shape(self):
        """ """
        return (self.config.seq_length, self.config.vocab_size)

    def reset(  # ---------------------------------------------------------------------------------
        self, *, seed: int | None = None, options: dict | None = None,
    ) -> tuple[dict[str, np.ndarray], dict]:  # fmt: skip
        """ """
        super().reset(seed=seed)
        # options must carry the sample for this episode
        sample = options["sample"]  # {"inputs": np.ndarray, "labels": np.ndarray}
        self._inputs = sample["inputs"]
        self._labels = sample["labels"]
        self._prev_accuracy = 0.0
        self._step_count = 0
        obs = {"inputs": self._inputs.copy()}
        return obs, {}

    def step(  # ----------------------------------------------------------------------------------
        self, action: dict[str, np.ndarray | int],
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:  # fmt: skip
        """ """
        self._step_count += 1
        prediction = action["prediction"]  # (S, V) logits
        halt = bool(action["halt"] == self._config.halt_action)  # action index → halt semantics

        # Reward: improvement in prediction accuracy
        accuracy = self._compute_accuracy(prediction, self._labels)
        reward = accuracy - self._prev_accuracy
        self._prev_accuracy = accuracy

        terminated = halt
        truncated = self._step_count >= self._config.max_steps

        obs = {"inputs": self._inputs.copy()}
        info = {"accuracy": accuracy, "step_count": self._step_count}
        return obs, reward, terminated, truncated, info

    @staticmethod
    def _compute_accuracy(  # ---------------------------------------------------------------------
        logits: np.ndarray, labels: np.ndarray,
    ) -> float:  # fmt: skip
        """ """
        mask = labels != IGNORE_LABEL_ID
        if mask.sum() == 0:
            return 0.0
        preds = logits.argmax(axis=-1)
        correct = (preds == labels) & mask
        return float(correct.sum()) / float(mask.sum())


# =================================================================================================
__all__ = ["EnvConfig", "Env"]
