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
    """Inter-region multiscale projection settings for EHC v2.

    Only the LEC→HPC and MEC→HPC edges are expressed here because both are
    multiscale-to-multiscale and fit the shared ``ProjectionBundle`` contract.

    The PFC↔HPC flat projectors (pfc_to_hpc_c and hpc_to_pfc_*) are
    constructed directly in ``EHCModelV2.__init__`` as plain ``nn.Linear``
    modules; they do not belong here because the projection framework only
    understands aligned multiscale endpoints.
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
        ...,  # TODO: complete
    )


# ===========================================================================
__all__ = [
    "EHCProjectionSettingsV2",
    "SLOT_CUE",
    "SLOT_REPLAY",
    "SLOT_STATE",
    "FAMILY_CONTENT",
]
