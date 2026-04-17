"""Navigation TEM task binding for the TEM objective.

This module provides :class:`NavigationTEMTaskBinding`, which wires the TEM
loss head to the navigation task surface.  It is the sole place in the
codebase that knows both the TEM objective API and the navigation carry shape.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from ehc_sn.tasks.navigation.contracts import NavigationTargets
from ehc_sn.types import Batch


# =============================================================================
class NavigationTEMTaskBinding:
    """TEM task binding for the navigation task.

    Extracts supervised observation ids and protocol masks from the controller
    carry, using the canonical navigation carry-data keys populated by
    :class:`~ehc_sn.tasks.navigation.runtime.NavigationControllerRuntime`.
    Mirrors the pattern used by
    :class:`~ehc_sn.adapters.maze_hard.objectives.MazeHardACTTaskBinding`
    for the ACT objective path.
    """

    def _extract_targets(  # --------------------------------------------------
        self,
        carry: Any,
    ) -> NavigationTargets:
        """Build a :class:`NavigationTargets` from carry data.

        Uses ``observation_id`` when present; falls back to ``labels`` only if
        ``observation_id`` is absent.  ``is_revisit`` is forwarded unchanged.
        """
        raw: Tensor | None = carry.data.get("observation_id")
        if raw is None:
            raw = carry.data["labels"]
        is_revisit: Tensor | None = carry.data.get("is_revisit")
        return NavigationTargets(observation_id=raw, is_revisit=is_revisit)

    def extract_observation_id(  # -------------------------------------------
        self,
        batch: Batch,
        carry: Any,
        step_output: Any,
    ) -> Tensor:
        """Return the current-step observation id tensor from the carry buffer.

        Reads ``observation_id`` from carry data (falling back to ``labels``
        when absent), applying squeeze/argmax normalisation to guarantee a
        1-D integer target tensor.

        Args:
            batch: Generic batch mapping; unused here.
            carry: Controller carry state.
            step_output: Controller step output; unused here.

        Returns:
            Integer observation-id tensor of shape ``(B,)``.
        """
        _ = batch, step_output
        raw = self._extract_targets(carry).observation_id
        if raw.ndim > 1:
            if raw.shape[-1] == 1:
                return raw.squeeze(-1)
            if raw.is_floating_point():
                return raw.argmax(dim=-1)
        return raw

    def extract_protocol_mask(  # --------------------------------------------
        self,
        batch: Batch,
        carry: Any,
        step_output: Any,
    ) -> Tensor:
        """Return the revisit-eligibility mask for protocol supervision.

        Reads ``is_revisit`` from :meth:`_extract_targets` and reshapes to
        ``(B,)``.

        Args:
            batch: Generic batch mapping; unused here.
            carry: Controller carry state.
            step_output: Controller step output; unused here.

        Returns:
            Boolean tensor of shape ``(B,)`` indicating revisit positions.

        Raises:
            KeyError: If the carry does not contain ``"is_revisit"``.
        """
        _ = batch, step_output
        is_revisit = self._extract_targets(carry).is_revisit
        if is_revisit is None:
            raise KeyError("Navigation carry data must provide 'is_revisit' for protocol-gated TEM supervision.")
        return is_revisit.reshape(-1).to(dtype=torch.bool)


# =============================================================================
__all__ = ["NavigationTEMTaskBinding"]
