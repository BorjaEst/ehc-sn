"""Shared MEC and OVC layout derivation.

This module centralizes how MEC grid modules and optional OVC modules are laid
out so callers do not re-derive shape and slice policy independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence

from ehc_sn import utils

OVCMode = Literal["off", "merged", "separate"]


# =================================================================================================
@dataclass(frozen=True)
class MECLayout:
    """Resolved MEC layout facts derived from grid and OVC config.

    Attributes:
        grid_shape: Base grid-module shape before any appended OVC modules.
        full_shape: Full MEC shape after applying the configured OVC mode.
        appended_ovc_shape: Explicit OVC modules appended in `separate` mode.
        ovc_correction_start: Index of the first module corrected by OVC.
        ovc_correction_count: Number of modules corrected by OVC.
    """

    grid_shape: list[int]
    full_shape: list[int]
    appended_ovc_shape: list[int]
    ovc_correction_start: int
    ovc_correction_count: int

    @property
    def appended_ovc_count(self) -> int:
        """Return the number of explicit OVC modules appended to MEC."""
        return len(self.appended_ovc_shape)

    @property
    def ovc_correction_shape(self) -> list[int]:
        """Return the per-frequency shape corrected by OVC."""
        stop = self.ovc_correction_start + self.ovc_correction_count
        return self.full_shape[self.ovc_correction_start : stop]

    @property
    def n_total_freq(self) -> int:
        """Return the total number of MEC frequencies after layout expansion."""
        return len(self.full_shape)


# =================================================================================================
def validate_ovc_shape_policy(mode: OVCMode, shape: Optional[Sequence[int]]) -> None:
    """Validate whether an explicit OVC shape is allowed for the given mode."""
    if mode == "separate" and not shape:
        raise ValueError("mec.ovc.shape is required when mec.ovc.mode='separate'.")
    if mode != "separate" and shape is not None:
        raise ValueError("mec.ovc.shape is only allowed when mec.ovc.mode='separate'.")


# =================================================================================================
def resolve_mec_layout(
    grid_shape: Sequence[int], *, ovc_mode: OVCMode, ovc_shape: Optional[Sequence[int]]
) -> MECLayout:
    """Resolve MEC shape expansion and OVC correction layout from config values."""
    validate_ovc_shape_policy(ovc_mode, ovc_shape)

    grid_shape_resolved = list(grid_shape)
    appended_ovc_shape = list(ovc_shape or []) if ovc_mode == "separate" else []
    full_shape = grid_shape_resolved + appended_ovc_shape

    if ovc_mode == "off":
        n_freq_ovc = 0
    elif ovc_mode == "merged":
        n_freq_ovc = None
    else:
        n_freq_ovc = len(appended_ovc_shape)

    ovc_correction_start, ovc_correction_count = utils.resolve_ovc_slice(len(full_shape), n_freq_ovc)
    return MECLayout(
        grid_shape=grid_shape_resolved,
        full_shape=full_shape,
        appended_ovc_shape=appended_ovc_shape,
        ovc_correction_start=ovc_correction_start,
        ovc_correction_count=ovc_correction_count,
    )


# =================================================================================================
__all__ = ["MECLayout", "resolve_mec_layout", "validate_ovc_shape_policy"]
