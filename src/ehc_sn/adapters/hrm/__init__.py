"""HRM bridge adapters for MazeHard and SeqMaze task families."""

from ._base import (
    MazeHardHRMAdapterSettings,
    SeqMazeAdapterSettings,
    SeqMazeProbeAdapterSettings,
)
from .mazehard import MazeHardHRMV1BridgeAdapter, MazeHardHRMV2BridgeAdapter
from .objectives import (
    MazeHardHRMV1ACTTaskBinding,
    MazeHardHRMV2HybridTaskBinding,
    SeqMazeHRMV1ACTTaskBinding,
)
from .seqmaze import (
    SeqMazeBridgeOutput,
    SeqMazeHRMV1BridgeAdapter,
    SeqMazeHRMV1BridgeOutput,
    SeqMazeHRMV1ControlOutput,
    SeqMazeHRMV2BridgeAdapter,
    SeqMazeProbeHRMV2BridgeAdapter,
    SeqMazeProbeOutput,
)
from .traces import (
    MAZE_HARD_HRM_ACT_TRACE_FIELDS,
    MAZE_HARD_HRM_ACTOR_CRITIC_TRACE_FIELDS,
    MAZE_HARD_HRM_TRACE_SOLUTION_OVERLAY,
    build_mazehard_hrm_trace_meta,
    build_seqmaze_hrm_trace_meta,
)

__all__ = [
    "MazeHardHRMAdapterSettings",
    "MazeHardHRMV1BridgeAdapter",
    "MazeHardHRMV1ACTTaskBinding",
    "MazeHardHRMV2BridgeAdapter",
    "MazeHardHRMV2HybridTaskBinding",
    "MAZE_HARD_HRM_ACTOR_CRITIC_TRACE_FIELDS",
    "MAZE_HARD_HRM_ACT_TRACE_FIELDS",
    "MAZE_HARD_HRM_TRACE_SOLUTION_OVERLAY",
    "build_mazehard_hrm_trace_meta",
    "build_seqmaze_hrm_trace_meta",
    "SeqMazeAdapterSettings",
    "SeqMazeBridgeOutput",
    "SeqMazeHRMV1ACTTaskBinding",
    "SeqMazeHRMV1BridgeAdapter",
    "SeqMazeHRMV1BridgeOutput",
    "SeqMazeHRMV1ControlOutput",
    "SeqMazeHRMV2BridgeAdapter",
    "SeqMazeProbeAdapterSettings",
    "SeqMazeProbeHRMV2BridgeAdapter",
    "SeqMazeProbeOutput",
]
