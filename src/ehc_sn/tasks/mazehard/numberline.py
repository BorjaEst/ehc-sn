"""NumberLine mechanical environment kernel.

Owns only the hidden-state stepping kernel for a bounded integer line under
PREV, NEXT, and STAY actions.  Does not own Countwalk scoring, query logic,
anchor semantics, or rewards.

This env is required as a clean mechanical kernel for the NumberLine state
space.  V1 Countwalk replay training uses the replay trajectory controller and
does not step through this env at training time, but the env is the canonical
reference for the transition function.

Actions:
    STAY = 0  — remain at current position
    PREV = 1  — move to position - 1 (clamped at 0; boundary = illegal)
    NEXT = 2  — move to position + 1 (clamped at n_states-1; boundary = illegal)
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# =============================================================================
# Action constants
# =============================================================================

STAY: int = 0
"""Action: remain at the current state."""

PREV: int = 1
"""Action: move to state - 1.  Illegal at state 0."""

NEXT: int = 2
"""Action: move to state + 1.  Illegal at state n_states - 1."""

N_ACTIONS: int = 3
"""Total number of actions in the NumberLine action space."""


# =============================================================================
class NumberLineEnvConfig(BaseModel, extra="forbid"):
    """Configuration for :class:`NumberLineEnv`."""

    n_states: int = Field(
        ...,
        ge=2,
        description="Number of states on the number line.  Must match the parent shared substrate.",
    )
    max_episode_steps: int = Field(
        default=32,
        ge=1,
        description="Maximum steps per episode before mechanical truncation.  None = no truncation.",
    )


# =============================================================================
class NumberLineEnv:
    """Mechanical kernel for the NumberLine state space.

    Owns:
    - hidden integer state ``current`` in ``[0, n_states - 1]``
    - per-step step counter
    - deterministic PREV / NEXT / STAY stepping
    - boundary legality checks
    - mechanical truncation when ``max_episode_steps`` is reached

    Does NOT own:
    - Countwalk cue surfaces or anchor regimes
    - query logic or supervision targets
    - scoring or rewards
    """

    def __init__(self, config: NumberLineEnvConfig) -> None:
        self._n = config.n_states
        self._max_steps = config.max_episode_steps
        self._state: int = 0
        self._step_count: int = 0
        self._done: bool = False

    # ------------------------------------------------------------------
    @property
    def n_states(self) -> int:
        """Number of states on the number line."""
        return self._n

    @property
    def current(self) -> int:
        """Current hidden state (integer in ``[0, n_states - 1]``)."""
        return self._state

    @property
    def step_count(self) -> int:
        """Number of steps taken since last reset."""
        return self._step_count

    @property
    def done(self) -> bool:
        """Whether the episode has ended (truncated)."""
        return self._done

    # ------------------------------------------------------------------
    def reset(self, *, start: int | None = None) -> int:
        """Reset the environment to an initial state.

        Args:
            start: Initial hidden state.  Defaults to 0.

        Returns:
            The initial hidden state.
        """
        if start is None:
            start = 0
        if not (0 <= start < self._n):
            raise ValueError(f"start={start} out of range [0, {self._n - 1}]")
        self._state = start
        self._step_count = 0
        self._done = False
        return self._state

    # ------------------------------------------------------------------
    def is_legal(self, action: int) -> bool:
        """Return whether *action* is legal at the current state.

        STAY is always legal.  PREV is illegal at state 0.
        NEXT is illegal at state ``n_states - 1``.
        """
        if action == STAY:
            return True
        if action == PREV:
            return self._state > 0
        if action == NEXT:
            return self._state < self._n - 1
        raise ValueError(f"Unknown action {action}; must be STAY={STAY}, PREV={PREV}, or NEXT={NEXT}")

    # ------------------------------------------------------------------
    def step(self, action: int) -> tuple[int, bool]:
        """Apply *action* and return ``(new_state, truncated)``.

        Illegal boundary actions are clamped (PREV at 0 stays at 0; NEXT at
        n_states-1 stays at n_states-1).  The env itself does not raise on
        illegal actions; legality enforcement is the caller's responsibility
        for primary-corpus generation.

        Args:
            action: One of STAY, PREV, NEXT.

        Returns:
            ``(new_state, truncated)`` — new hidden state and whether the
            episode was mechanically truncated on this step.

        Raises:
            RuntimeError: When called after the episode has already ended.
            ValueError: When *action* is not a valid action integer.
        """
        if self._done:
            raise RuntimeError("Cannot step after the episode has ended.  Call reset() first.")

        if action == STAY:
            pass
        elif action == PREV:
            self._state = max(0, self._state - 1)
        elif action == NEXT:
            self._state = min(self._n - 1, self._state + 1)
        else:
            raise ValueError(f"Unknown action {action}; must be STAY={STAY}, PREV={PREV}, or NEXT={NEXT}")

        self._step_count += 1
        truncated = self._step_count >= self._max_steps
        if truncated:
            self._done = True
        return self._state, truncated


# =============================================================================
__all__ = [
    "STAY",
    "PREV",
    "NEXT",
    "N_ACTIONS",
    "NumberLineEnvConfig",
    "NumberLineEnv",
]
