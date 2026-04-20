"""Shared Navigation+TEM bridge namespace.

Re-exports the public symbols shared across both TEM v1 and TEM v2 Lightning
runtimes.  Version-specific bridge classes live in :mod:`tem_v1` and
:mod:`tem_v2` respectively and are not re-exported here.
"""

from ehc_sn.adapters.navigation.bridges.tem.objectives import NavigationTEMTaskBinding
from ehc_sn.adapters.navigation.bridges.tem.traces import (
    NAVIGATION_TEM_TRACE_FIELDS,
    NAVIGATION_TEM_TRACE_IS_REVISIT,
    NAVIGATION_TEM_TRACE_PRED_ANCESTRAL,
    NAVIGATION_TEM_TRACE_PRED_INFERENCE,
    NAVIGATION_TEM_TRACE_PRED_RETRIEVED,
    NAVIGATION_TEM_TRACE_WORLD_OBS_ID,
    select_navigation_tem_trace_fields,
)

__all__ = [
    "NavigationTEMTaskBinding",
    "NAVIGATION_TEM_TRACE_FIELDS",
    "NAVIGATION_TEM_TRACE_IS_REVISIT",
    "NAVIGATION_TEM_TRACE_PRED_ANCESTRAL",
    "NAVIGATION_TEM_TRACE_PRED_INFERENCE",
    "NAVIGATION_TEM_TRACE_PRED_RETRIEVED",
    "NAVIGATION_TEM_TRACE_WORLD_OBS_ID",
    "select_navigation_tem_trace_fields",
]
