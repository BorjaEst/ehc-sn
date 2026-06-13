"""Pull-based episode source with coverage-guaranteed shuffled access."""

from __future__ import annotations

import warnings
from typing import Any, Callable, ClassVar

import numpy as np
import torch
from torch import Tensor

from ehc_sn.data.datasets import ProcessedDataset
from ehc_sn.types import Batch


# =============================================================================
def _default_collate(samples: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Stack a list of episode dicts into a single collated batch.

    Each key is stacked along a new leading dimension.
    """
    if not samples:
        return {}
    keys = list(samples[0].keys())
    result: dict[str, Tensor] = {}
    for k in keys:
        tensors = [s[k] for s in samples]
        result[k] = torch.stack(tensors, dim=0)
    return result


# =============================================================================
class ShuffledEpisodeSource:
    """Rank-local shuffled episode source without replacement.

    Each rank owns a deterministic shard of the global permutation.
    Cycles are rank-local — ranks advance independently.

    Permutations are deterministic: given ``(base_seed, cycle, rank,
    world_size)``, the same permutation is always reconstructed.
    No mutable RNG state is stored in checkpoints — ``state_dict``
    stores only the deterministic reconstruction parameters.

    Contract
    --------
    - ``take(k)`` returns exactly ``k`` episodes, advancing the cursor only by ``k``.
    - Every index in the local shard is returned exactly once per cycle
      before reshuffling.
    - Crossing a cycle boundary deterministically reshuffles and continues serving.
    - No selected-but-unreturned index is discarded.
    """

    SHUFFLE_VERSION: ClassVar[int] = 1
    """Increment when the shuffle algorithm or sharding policy changes,
    so that saved checkpoints can be rejected as incompatible."""

    STATE_VERSION: ClassVar[int] = 1
    """Increment when the ``state_dict`` format changes."""

    def __init__(  # ----------------------------------------------------------
        self,
        dataset: ProcessedDataset,
        *,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
        collate_fn: (
            Callable[[list[dict[str, Tensor]]], dict[str, Tensor]] | None
        ) = None,
    ) -> None:
        if world_size < 1:
            raise ValueError("world_size must be >= 1")
        if rank < 0 or rank >= world_size:
            raise ValueError(
                f"rank={rank} is out of bounds for world_size={world_size}"
            )
        if not isinstance(seed, int):
            raise TypeError(f"seed must be an int, got {type(seed).__name__}")

        self._dataset = dataset
        self._rank = rank
        self._world_size = world_size
        self._base_seed = seed
        self._collate_fn = collate_fn or _default_collate
        self._cycle = 0
        self._cursor = 0
        self._total_admitted = 0
        self._permutation = self._build_permutation()

    # ── Public API ──────────────────────────────────────────────────────────

    def take(self, n: int) -> Batch:
        """Return exactly ``n`` collated episodes.

        Guarantees:
        - Each local dataset index is returned once per coverage cycle.
        - Cursor advances only by ``n``.
        - Crossing a cycle boundary reshuffles and continues.
        - No selected-but-unreturned index is discarded.
        - ``n=0`` returns an empty batch.

        Raises:
            ValueError: if ``n < 0`` or ``n > len(self._dataset)``.
        """
        if n < 0:
            raise ValueError(f"take(n) requires n >= 0, got {n}.")
        if n > len(self._dataset):
            raise ValueError(
                f"take({n}) exceeds dataset size {len(self._dataset)}."
            )
        if n == 0:
            return {}

        episodes: list[dict[str, Tensor]] = []
        remaining = n
        while remaining > 0:
            available = len(self._permutation) - self._cursor
            take_now = min(available, remaining)
            for i in range(take_now):
                idx = int(self._permutation[self._cursor + i])
                sample = self._dataset[idx]
                episodes.append(sample)
            self._cursor += take_now
            remaining -= take_now
            if remaining > 0:
                self._advance_cycle()
        self._total_admitted += n
        return self._collate_fn(episodes)

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def local_cycle(self) -> int:
        """Current rank-local coverage cycle (0-based)."""
        return self._cycle

    @property
    def coverage_cycle(self) -> int:
        """Number of completed cycles.  Equals the index of the active cycle."""
        return self._cycle

    @property
    def local_cursor(self) -> int:
        """Current position within the local shard."""
        return self._cursor

    @property
    def cycle_cursor(self) -> int:
        """Current position within the local shard (alias for ``local_cursor``)."""
        return self._cursor

    @property
    def shard_size(self) -> int:
        """Number of episodes in the current rank's local shard."""
        return len(self._permutation)

    @property
    def cycle_progress(self) -> float:
        """Fraction of the current shard consumed (0.0 – 1.0)."""
        shard = self.shard_size
        return self._cursor / shard if shard > 0 else 0.0

    @property
    def total_admitted(self) -> int:
        """Total number of episodes ever returned by ``take()``."""
        return self._total_admitted

    # ── State persistence ───────────────────────────────────────────────────

    def state_dict(self) -> dict[str, Any]:
        """Return a fingerprint-verified resume state.

        No mutable RNG state — the permutation is deterministically
        reconstructable from ``(base_seed, cycle, rank, world_size)``.
        """
        return {
            "state_version": self.STATE_VERSION,
            "fingerprint": self._make_fingerprint(),
            "cycle": self._cycle,
            "cursor": self._cursor,
            "total_admitted": self._total_admitted,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore exact source position.  Validates fingerprint first.

        Raises:
            ValueError: if the saved fingerprint is incompatible with the
                current source configuration.
        """
        self._validate_fingerprint(state.get("fingerprint", {}))
        cycle = state["cycle"]
        cursor = state["cursor"]
        total = state["total_admitted"]

        # Rebuild permutation for the restored cycle before validating cursor.
        self._cycle = cycle
        self._permutation = self._build_permutation()

        if cursor < 0 or cursor > self.shard_size:
            raise ValueError(
                f"Restored cursor {cursor} out of range [0, {self.shard_size}]."
            )
        if total < cursor:
            raise ValueError(f"total_admitted ({total}) < cursor ({cursor}).")
        shard = self.shard_size
        expected = cycle * shard + cursor
        if total != expected:
            raise ValueError(
                f"total_admitted ({total}) != cycle*shard+cursor "
                f"= {cycle}*{shard}+{cursor} = {expected}."
            )

        self._cursor = cursor
        self._total_admitted = total

    def restart_cycle(self, cycle: int) -> None:
        """Restart the given cycle from its beginning (cursor=0).

        Does NOT validate fingerprint — the caller is explicitly choosing
        to restart rather than continue.  The cycle must be >= 0.
        """
        if cycle < 0:
            raise ValueError(f"restart_cycle requires cycle >= 0, got {cycle}.")
        self._cycle = cycle
        self._cursor = 0
        self._total_admitted = cycle * self.shard_size
        self._permutation = self._build_permutation()

    # ── Fingerprint ─────────────────────────────────────────────────────────

    def _make_fingerprint(self) -> dict[str, Any]:
        """Return an immutable description of source identity.

        Two sources with the same fingerprint produce the same sequence
        of permutations given the same cycle number.
        """
        return {
            "state_version": self.STATE_VERSION,
            "shuffle_version": self.SHUFFLE_VERSION,
            "dataset_digest": self._dataset.dataset_digest,
            "dataset_size": self._dataset.dataset_size,
            "base_seed": self._base_seed,
            "rank": self._rank,
            "world_size": self._world_size,
            "sharding_policy": "strided",
        }

    def _validate_fingerprint(self, saved: dict[str, Any]) -> None:
        """Raise ``ValueError`` if the saved fingerprint is incompatible."""
        current = self._make_fingerprint()
        hard_keys = (
            "state_version",
            "shuffle_version",
            "dataset_digest",
            "dataset_size",
            "base_seed",
            "rank",
            "world_size",
            "sharding_policy",
        )
        mismatches = []
        for k in hard_keys:
            expected = saved.get(k)
            actual = current.get(k)
            if expected != actual:
                mismatches.append(
                    f"  {k}: saved={expected!r}, current={actual!r}"
                )
        if mismatches:
            raise ValueError(
                "Episode source compatibility fingerprint mismatch:\n"
                + "\n".join(mismatches)
                + "\nSource cursor cannot be safely restored. "
                "Use restart_cycle() to restart the current coverage cycle."
            )

    # ── Internal helpers ────────────────────────────────────────────────────

    def _build_permutation(self) -> np.ndarray:
        """Deterministic per-cycle, per-rank shard of the full dataset.

        No instance-level RNG is used — a local generator is created
        from ``(base_seed + cycle)`` each time, guaranteeing
        reproducibility without mutable state.
        """
        rng = np.random.default_rng(self._base_seed + self._cycle)
        total = len(self._dataset)
        indices = np.arange(total)
        rng.shuffle(indices)
        return indices[self._rank :: self._world_size]

    def _advance_cycle(self) -> None:
        self._cycle += 1
        self._cursor = 0
        self._total_admitted = self._cycle * self.shard_size
        self._permutation = self._build_permutation()


# =============================================================================
__all__ = ["ShuffledEpisodeSource"]
