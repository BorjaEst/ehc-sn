"""Observation placement policy for layout substrates.

Controls how observation identities are assigned to traversable cells.

Policies
--------
``"dense_uniform"``:
    Every traversable cell gets a random observation ID from ``[0, vocab_size)``.
    This is the legacy behavior: all traversable cells are ``CELL_OBSERVATION``.

``"exactly_one"``:
    Each of the ``vocab_size`` observation IDs appears at exactly one traversable
    cell.  Remaining traversable cells become ``CELL_FREE`` (no observation).
    Raises ``ValueError`` when ``vocab_size > n_states``.

``"bounded"``:
    Each observation ID appears at most ``max_occurrences`` times.  Cells beyond
    the per-observation cap become ``CELL_FREE``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ehc_sn.tasks.routebind.contracts import CELL_FREE, CELL_OBSERVATION


@dataclass(frozen=True)
class ObservationPlacementConfig:
    """Configuration for observation placement on a layout substrate.

    Attributes:
        policy: Placement policy name (``"dense_uniform"``, ``"exactly_one"``,
            or ``"bounded"``).
        max_occurrences: Maximum occurrences per observation ID (used by
            ``"bounded"`` policy).  Ignored for other policies.
        seed: RNG seed for observation placement (deterministic across calls
            with matching topology states).
    """

    policy: str = "dense_uniform"
    max_occurrences: int = 1
    seed: int = 42

    def __post_init__(self) -> None:
        valid = ("dense_uniform", "exactly_one", "bounded")
        if self.policy not in valid:
            raise ValueError(
                f"ObservationPlacementConfig.policy={self.policy!r} "
                f"must be one of {valid}."
            )
        if self.max_occurrences < 1:
            raise ValueError(
                f"ObservationPlacementConfig.max_occurrences must be >= 1, "
                f"got {self.max_occurrences}."
            )


def assign_observations(
    n_states: int,
    vocab_size: int,
    placement: ObservationPlacementConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign observation IDs to traversable cells according to *placement*.

    Args:
        n_states: Number of traversable cells in the layout.
        vocab_size: Number of distinct observation IDs (``0..vocab_size-1``).
        placement: Placement policy configuration.
        rng: Random number generator for stochastic policies.

    Returns:
        Tuple ``(observation_id, cell_type)``:
        - ``observation_id``: ``(n_states,)`` int32.  Values in
          ``[0, vocab_size)`` for observation-bearing cells; ``-1`` for
          ``CELL_FREE`` cells.
        - ``cell_type``: ``(n_states,)`` int32.  ``CELL_OBSERVATION`` (2) or
          ``CELL_FREE`` (1).

    Raises:
        ValueError: When ``policy == "exactly_one"`` and
            ``vocab_size > n_states``.
    """
    if placement.policy == "dense_uniform":
        obs_ids = rng.integers(0, vocab_size, size=n_states).astype(np.int32)
        cell_types = np.full(n_states, CELL_OBSERVATION, dtype=np.int32)
        return obs_ids, cell_types

    if placement.policy == "exactly_one":
        if vocab_size > n_states:
            raise ValueError(
                f"exactly_one policy requires vocab_size ({vocab_size}) <= "
                f"n_states ({n_states})."
            )
        obs_ids = np.full(n_states, -1, dtype=np.int32)
        cell_types = np.full(n_states, CELL_FREE, dtype=np.int32)
        landmarks = rng.choice(n_states, size=vocab_size, replace=False)
        obs_ids[landmarks] = np.arange(vocab_size, dtype=np.int32)
        cell_types[landmarks] = CELL_OBSERVATION
        return obs_ids, cell_types

    if placement.policy == "bounded":
        k = placement.max_occurrences
        obs_ids = np.full(n_states, -1, dtype=np.int32)
        cell_types = np.full(n_states, CELL_FREE, dtype=np.int32)
        max_cells = k * vocab_size
        n_assign = min(max_cells, n_states)
        cells = rng.choice(n_states, size=n_assign, replace=False)
        for i, cell_idx in enumerate(cells):
            oid = i % vocab_size
            obs_ids[cell_idx] = np.int32(oid)
            cell_types[cell_idx] = CELL_OBSERVATION
        return obs_ids, cell_types

    # Unreachable — __post_init__ validates policy.
    raise ValueError(f"Unknown policy: {placement.policy}")
