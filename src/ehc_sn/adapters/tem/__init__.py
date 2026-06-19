"""Arena+TEM bridge family public surface.

Import stable adapter settings, family-stable TEM diagnostics, versioned bridge
adapters, task bindings, and trace helpers from this barrel.
"""

from ._base import (
    ArenaTEMAdapterSettings,
    ArenaTEMBridgeOutput,
    TEMLearningState,
)
from .arena import ArenaTEMV1BridgeAdapter, ArenaTEMV2BridgeAdapter
from .traces import (
    ARENA_TEM_TRACE_FIELDS,
    ARENA_TEM_TRACE_IS_REVISIT,
    ARENA_TEM_TRACE_PRED_PATH,
    ARENA_TEM_TRACE_PRED_POST,
    ARENA_TEM_TRACE_PRED_RECALL,
    ARENA_TEM_TRACE_WORLD_OBS_ID,
    build_arena_tem_trace_meta,
    select_arena_tem_trace_fields,
)

__all__ = [
    "ArenaTEMAdapterSettings",
    "ArenaTEMBridgeOutput",
    "ArenaTEMV1BridgeAdapter",
    "ArenaTEMV2BridgeAdapter",
    "ARENA_TEM_TRACE_FIELDS",
    "ARENA_TEM_TRACE_IS_REVISIT",
    "ARENA_TEM_TRACE_PRED_PATH",
    "ARENA_TEM_TRACE_PRED_POST",
    "ARENA_TEM_TRACE_PRED_RECALL",
    "ARENA_TEM_TRACE_WORLD_OBS_ID",
    "TEMLearningState",
    "build_arena_tem_trace_meta",
    "select_arena_tem_trace_fields",
]
