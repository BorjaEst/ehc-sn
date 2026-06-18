"""HRM bridge adapters for MazeHard, SeqMaze, and Goaltrace task families."""

from ._base import (
    MazeHardHRMAdapterSettings,
    SeqMazeAdapterSettings,
    SeqMazeProbeAdapterSettings,
)
from .goaltrace import (
    GoaltraceAdapterSettings,
    GoaltraceHRMV1BridgeAdapter,
    GoaltraceHRMV1BridgeOutput,
    GoaltraceHRMV1ControlOutput,
    build_goaltrace_hrm_trace_meta,
)
from .mazehard import MazeHardHRMV1BridgeAdapter, MazeHardHRMV2BridgeAdapter
from .objectives import (
    GoaltraceHRMV1ACTTaskBinding,
    MazeHardHRMV1ACTTaskBinding,
    MazeHardHRMV2HybridTaskBinding,
    SeqMazeHRMV1ACTTaskBinding,
    SeqMazeHRMV2HybridTaskBinding,
)
from .seqmaze import (
    SeqMazeBridgeOutput,
    SeqMazeHRMV1BridgeAdapter,
    SeqMazeHRMV1BridgeOutput,
    SeqMazeHRMV1ControlOutput,
    SeqMazeHRMV2BridgeAdapter,
    SeqMazeHRMV2BridgeOutput,
    SeqMazeHRMV2CriticOutput,
    SeqMazeHRMV2PolicyOutput,
    SeqMazeProbeHRMV2BridgeAdapter,
    SeqMazeProbeOutput,
)
from .traces import (
    MAZE_HARD_HRM_ACT_TRACE_FIELDS,
    MAZE_HARD_HRM_ACTOR_CRITIC_TRACE_FIELDS,
    MAZE_HARD_HRM_TRACE_SOLUTION_OVERLAY,
    SEQMAZE_HRM_ACTOR_CRITIC_TRACE_FIELDS,
    build_mazehard_hrm_trace_meta,
    build_seqmaze_hrm_actor_critic_trace_meta,
    build_seqmaze_hrm_trace_meta,
)

__all__ = [
    "GoaltraceAdapterSettings",
    "GoaltraceHRMV1ACTTaskBinding",
    "GoaltraceHRMV1BridgeAdapter",
    "GoaltraceHRMV1BridgeOutput",
    "GoaltraceHRMV1ControlOutput",
    "build_goaltrace_hrm_trace_meta",
    "MazeHardHRMAdapterSettings",
    "MazeHardHRMV1BridgeAdapter",
    "MazeHardHRMV1ACTTaskBinding",
    "MazeHardHRMV2BridgeAdapter",
    "MazeHardHRMV2HybridTaskBinding",
    "MAZE_HARD_HRM_ACTOR_CRITIC_TRACE_FIELDS",
    "MAZE_HARD_HRM_ACT_TRACE_FIELDS",
    "MAZE_HARD_HRM_TRACE_SOLUTION_OVERLAY",
    "SEQMAZE_HRM_ACTOR_CRITIC_TRACE_FIELDS",
    "build_mazehard_hrm_trace_meta",
    "build_seqmaze_hrm_actor_critic_trace_meta",
    "build_seqmaze_hrm_trace_meta",
    "SeqMazeAdapterSettings",
    "SeqMazeBridgeOutput",
    "SeqMazeHRMV1ACTTaskBinding",
    "SeqMazeHRMV1BridgeAdapter",
    "SeqMazeHRMV1BridgeOutput",
    "SeqMazeHRMV1ControlOutput",
    "SeqMazeHRMV2BridgeAdapter",
    "SeqMazeHRMV2BridgeOutput",
    "SeqMazeHRMV2CriticOutput",
    "SeqMazeHRMV2HybridTaskBinding",
    "SeqMazeHRMV2PolicyOutput",
    "SeqMazeProbeAdapterSettings",
    "SeqMazeProbeHRMV2BridgeAdapter",
    "SeqMazeProbeOutput",
]
