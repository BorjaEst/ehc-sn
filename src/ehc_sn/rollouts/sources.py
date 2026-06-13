"""Passive and demand-driven rollout sources used by rollout runners."""

from __future__ import annotations

import torch
from torch import Tensor

from ehc_sn.rollouts.runtime import EpisodeSource, HaltedCarry
from ehc_sn.types import Batch


# =============================================================================
class RepeatSource:
    """Yield the same batch repeatedly, optionally with a fixed horizon."""

    def __init__(  # ----------------------------------------------------------
        self,
        batch: Batch,
        *,
        max_rollout_steps: int | None = None,
        stop_on_halt: bool = False,
        freeze_halted: bool = False,
    ) -> None:
        """Initialize the source with a batch to repeat and an optional rollout horizon."""
        self._batch = batch
        self._max_rollout_steps = max_rollout_steps
        self._steps = 0
        self._halted: Tensor | None = None
        self._stop_on_halt = stop_on_halt
        self._freeze_halted = freeze_halted

    def __iter__(self) -> RepeatSource:
        """Return self as an iterator."""
        return self

    def __next__(self) -> Batch:
        """Yield the batch, stopping if the maximum rollout steps have been reached."""
        if self._stop_on_halt and self._halted is not None:
            if bool(self._halted.all()):
                raise StopIteration
        if (
            self._max_rollout_steps is not None
            and self._steps >= self._max_rollout_steps
        ):
            raise StopIteration
        self._steps += 1
        return self._batch

    def update(  # ------------------------------------------------------------
        self,
        *,
        carry: HaltedCarry,
    ) -> None:
        """Update halt state and optionally freeze halted rows to their last data."""
        self._halted = carry.halted
        if not self._freeze_halted:
            return
        carry_data = getattr(carry, "data", None)
        if not isinstance(carry_data, dict) or not carry_data:
            return
        halted = carry.halted
        updated: dict[str, Tensor] = {}
        for key, value in self._batch.items():
            carry_value = carry_data.get(key)
            if (
                isinstance(value, Tensor)
                and isinstance(carry_value, Tensor)
                and value.shape == carry_value.shape
            ):
                updated[key] = torch.where(
                    halted.view((-1,) + (1,) * (value.ndim - 1)),
                    carry_value,
                    value,
                )
            else:
                updated[key] = value
        self._batch = updated


# =============================================================================
class DemandDrivenReplaySource:
    """Demand-driven replacement source for partial-reset training.

    Satisfies the ``Source`` protocol.  On each ``__next__()``:
    1. Reads the ``halted`` mask from the carry (stored by the latest
       ``update()`` call).
    2. Requests exactly ``n_halted`` replacement episodes from an
       ``EpisodeSource`` (e.g. ``ShuffledEpisodeSource``).
    3. Builds a step batch where halted rows receive new episode data
       and active rows receive a template placeholder (the controller
       reads ``resident_payload`` for active slots, so the placeholder
       is never consumed).

    Contract
    --------
    - Never discards an unconsumed episode.
    - Advances the episode source cursor only by the number of halted
      slots.
    - ``__next__()`` with all slots halted returns a full batch of
      ``B`` fresh episodes — does not raise ``StopIteration``.
    """

    def __init__(  # ----------------------------------------------------------
        self,
        *,
        episode_source: EpisodeSource,
        carry0: HaltedCarry,
    ) -> None:
        """Initialize with an episode source and initial carry.

        Args:
            episode_source: Pull-based source that provides episodes on demand.
            carry0: Initial carry whose ``halted`` mask is ``True`` for all
                slots, triggering a full initial admission batch.
        """
        self._episode_source = episode_source
        self._carry = carry0
        self._template: Batch | None = None
        self._batch_size: int = int(carry0.halted.shape[0])
        self._device: torch.device = carry0.halted.device

    def __iter__(self) -> "DemandDrivenReplaySource":
        """Return self as an iterator."""
        return self

    def __next__(self) -> Batch:
        """Yield the next step batch, with replacement rows for halted slots."""
        halted = self._carry.halted  # (B,) bool
        n_halted = int(halted.sum())

        if n_halted == 0:
            # No replacements needed: return the template.
            # The controller will read active slots from resident_payload.
            if self._template is None:
                raise RuntimeError(
                    "DemandDrivenReplaySource: no template available and "
                    "no halted slots to build one from."
                )
            return self._template

        replacements = self._episode_source.take(n_halted)
        # Move replacements from CPU (episode source) to the carry's device.
        replacements = {
            k: v.to(self._device, non_blocking=True) if isinstance(v, Tensor) else v
            for k, v in replacements.items()
        }

        if self._template is None:
            # First call: build the template from first replacement keys,
            # sized to match the full batch.
            B = self._batch_size
            self._template = {
                k: torch.zeros(
                    (B,) + tuple(v.shape[1:]),
                    dtype=v.dtype,
                    device=self._device,
                )
                for k, v in replacements.items()
            }

        # Scatter replacement rows into halted positions.
        B = self._batch_size
        result = dict(self._template)
        halted_idx = halted.nonzero(as_tuple=False).flatten()

        for k, replacement_tensor in replacements.items():
            if k not in result:
                continue
            if isinstance(replacement_tensor, Tensor):
                x = result[k]
                if (
                    isinstance(x, Tensor)
                    and x.shape[1:] == replacement_tensor.shape[1:]
                ):
                    x = x.clone()
                    x.index_copy_(0, halted_idx, replacement_tensor)
                    result[k] = x
        return result

    def update(  # ------------------------------------------------------------
        self,
        *,
        carry: HaltedCarry,
    ) -> None:
        """Store the latest controller carry for the next ``__next__()`` call.

        The source must have completed at least one successful ``__next__``
        (which builds its template) before ``update`` is called.
        """
        self._carry = carry


# =============================================================================
__all__ = ["DemandDrivenReplaySource", "RepeatSource"]
