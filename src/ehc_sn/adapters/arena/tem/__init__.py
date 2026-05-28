"""Arena+TEM bridge family public surface.

Import stable adapter settings, family-stable TEM diagnostics, versioned bridge
adapters, task bindings, and trace helpers from this barrel.
"""

from .core import (
    ArenaTEMAdapterSettings,
    ArenaTEMBridgeOutput,
    ArenaTEMDiagnostics,
)
from .objectives import ArenaTEMTaskBinding
from .tem_v1 import ArenaTEMV1BridgeAdapter
from .tem_v2 import ArenaTEMV2BridgeAdapter
from .traces import (
    ARENA_TEM_TRACE_FIELDS,
    ARENA_TEM_TRACE_IS_REVISIT,
    ARENA_TEM_TRACE_PRED_ANCESTRAL,
    ARENA_TEM_TRACE_PRED_INFERENCE,
    ARENA_TEM_TRACE_PRED_RETRIEVED,
    ARENA_TEM_TRACE_WORLD_OBS_ID,
    TARGET_OBSERVATION_ID_META_KEY,
    build_arena_tem_trace_meta,
    select_arena_tem_trace_fields,
)

__all__ = [
    "ArenaTEMAdapterSettings",
    "ArenaTEMBridgeOutput",
    "ArenaTEMDiagnostics",
    "ArenaTEMTaskBinding",
    "ArenaTEMV1BridgeAdapter",
    "ArenaTEMV2BridgeAdapter",
    "ARENA_TEM_TRACE_FIELDS",
    "ARENA_TEM_TRACE_IS_REVISIT",
    "ARENA_TEM_TRACE_PRED_ANCESTRAL",
    "ARENA_TEM_TRACE_PRED_INFERENCE",
    "ARENA_TEM_TRACE_PRED_RETRIEVED",
    "ARENA_TEM_TRACE_WORLD_OBS_ID",
    "build_arena_tem_trace_meta",
    "select_arena_tem_trace_fields",
    "TARGET_OBSERVATION_ID_META_KEY",
]
