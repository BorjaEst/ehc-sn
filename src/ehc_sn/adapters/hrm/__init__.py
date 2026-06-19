"""HRM bridge adapters for MazeHard, SeqMaze, and Goaltrace task families."""

from ._base import (
    GoaltraceHRMAdapterSettings,
    MazeHardHRMAdapterSettings,
    SeqMazeAdapterSettings,
    SeqMazeProbeAdapterSettings,
)
from .goaltrace import (
    GoaltraceHRMV1BridgeAdapter,
    GoaltraceHRMV1BridgeOutput,
    GoaltraceHRMV1ControlOutput,
)
from .mazehard import MazeHardHRMV1BridgeAdapter, MazeHardHRMV2BridgeAdapter
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
    GOALTRACE_HRM_ACT_FIRING_FIELD,
    GOALTRACE_HRM_ACT_TRACE_FIELDS,
    MAZE_HARD_HRM_ACT_TRACE_FIELDS,
    MAZE_HARD_HRM_ACTOR_CRITIC_TRACE_FIELDS,
    MAZE_HARD_HRM_TRACE_SOLUTION_OVERLAY,
    SEQMAZE_HRM_ACTOR_CRITIC_TRACE_FIELDS,
    build_goaltrace_hrm_trace_meta,
    build_mazehard_hrm_trace_meta,
    build_seqmaze_hrm_actor_critic_trace_meta,
    build_seqmaze_hrm_trace_meta,
)

__all__ = [
    "MazeHardHRMAdapterSettings",
    "GoaltraceHRMAdapterSettings",
    "GoaltraceHRMV1BridgeAdapter",
    "GoaltraceHRMV1BridgeOutput",
    "GoaltraceHRMV1ControlOutput",
    "GOALTRACE_HRM_ACT_FIRING_FIELD",
    "GOALTRACE_HRM_ACT_TRACE_FIELDS",
    "build_goaltrace_hrm_trace_meta",
    "MazeHardHRMAdapterSettings",
    "MazeHardHRMV1BridgeAdapter",
    "MazeHardHRMV2BridgeAdapter",
    "MAZE_HARD_HRM_ACTOR_CRITIC_TRACE_FIELDS",
    "MAZE_HARD_HRM_ACT_TRACE_FIELDS",
    "MAZE_HARD_HRM_TRACE_SOLUTION_OVERLAY",
    "SEQMAZE_HRM_ACTOR_CRITIC_TRACE_FIELDS",
    "build_mazehard_hrm_trace_meta",
    "build_seqmaze_hrm_actor_critic_trace_meta",
    "build_seqmaze_hrm_trace_meta",
    "SeqMazeAdapterSettings",
    "SeqMazeBridgeOutput",
    "SeqMazeHRMV1BridgeAdapter",
    "SeqMazeHRMV1BridgeOutput",
    "SeqMazeHRMV1ControlOutput",
    "SeqMazeHRMV2BridgeAdapter",
    "SeqMazeHRMV2BridgeOutput",
    "SeqMazeHRMV2CriticOutput",
    "SeqMazeHRMV2PolicyOutput",
    "SeqMazeProbeAdapterSettings",
    "SeqMazeProbeHRMV2BridgeAdapter",
    "SeqMazeProbeOutput",
]
