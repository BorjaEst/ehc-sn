"""Shared Arena+TEM bridge namespace.

Re-exports the public symbols shared across both TEM v1 and TEM v2 Lightning
runtimes.  Version-specific bridge classes live in :mod:`tem_v1` and
:mod:`tem_v2` respectively and are not re-exported here.
"""

from ehc_sn.adapters.arena.bridges.tem.objectives import ArenaTEMTaskBinding
from ehc_sn.adapters.arena.bridges.tem.traces import (
    ARENA_TEM_TRACE_FIELDS,
    ARENA_TEM_TRACE_IS_REVISIT,
    ARENA_TEM_TRACE_PRED_ANCESTRAL,
    ARENA_TEM_TRACE_PRED_INFERENCE,
    ARENA_TEM_TRACE_PRED_RETRIEVED,
    ARENA_TEM_TRACE_WORLD_OBS_ID,
    select_arena_tem_trace_fields,
)

__all__ = [
    "ArenaTEMTaskBinding",
    "ARENA_TEM_TRACE_FIELDS",
    "ARENA_TEM_TRACE_IS_REVISIT",
    "ARENA_TEM_TRACE_PRED_ANCESTRAL",
    "ARENA_TEM_TRACE_PRED_INFERENCE",
    "ARENA_TEM_TRACE_PRED_RETRIEVED",
    "ARENA_TEM_TRACE_WORLD_OBS_ID",
    "select_arena_tem_trace_fields",
]
