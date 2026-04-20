"""Shared settings and constants for EHC backbone versions.

This module intentionally stays thin. It holds only:
  - Shared projection edge settings grouped for EHC v2 (and reusable by later versions).
  - Slot-name constants for the fixed body workspace slots.
  - No version-specific cue-timing logic (that lives in ehc_v2.py / ehc_v3.py).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ehc_sn.modules.projection import ProjectionSettings

# ---------------------------------------------------------------------------
# Fixed body-workspace slot names shared across EHC versions.
# ---------------------------------------------------------------------------
SLOT_STATE: str = "state"
SLOT_REPLAY: str = "replay"
SLOT_CUE: str = "cue"
FAMILY_CONTENT: str = "content"


# ===========================================================================
class EHCProjectionSettingsV2(BaseModel, extra="forbid", strict=False):
    """Inter-region projection settings for EHC v2.

    ``lec_to_hpc`` and ``mec_to_hpc`` are multiscale-to-multiscale edges.
    ``pfc_to_hpc`` is a flat-to-multiscale broadcast edge that maps the
    previous-step public PFC summary into the hippocampal contextual cue
    family ``c``.
    ``hpc_to_pfc`` is an aligned workspace-to-workspace edge that projects the
    three fixed HPC interface slots (state, replay, cue) from the flattened HPC
    dimension into PFC hidden size.  Each fixed role gets its own independent
    parameter block; weights are not shared across roles by default.
    """

    lec_to_hpc: ProjectionSettings = Field(
        default_factory=lambda: ProjectionSettings(mode="tiling", learnable=False),
        description="Projection settings mapping LEC features into hippocampal query space.",
    )
    mec_to_hpc: ProjectionSettings = Field(
        default_factory=lambda: ProjectionSettings(mode="low_rank", learnable=False, rank=[10, 10, 8, 6, 6]),
        description="Projection settings mapping MEC codes into hippocampal query space.",
    )
    pfc_to_hpc: ProjectionSettings = Field(
        default_factory=lambda: ProjectionSettings(mode="linear", bridge="broadcast", init="random", learnable=True),
        description="Projection settings mapping the previous PFC summary into the hippocampal c-cue family.",
    )
    hpc_to_pfc: ProjectionSettings = Field(
        default_factory=lambda: ProjectionSettings(mode="linear", init="random", learnable=True),
        description=(
            "Projection settings for the workspace-aligned HPC->PFC reverse edge. "
            "Maps flattened HPC codes into PFC hidden size with one parameter block per fixed role."
        ),
    )


# ===========================================================================
__all__ = [
    "EHCProjectionSettingsV2",
    "SLOT_CUE",
    "SLOT_REPLAY",
    "SLOT_STATE",
    "FAMILY_CONTENT",
]
