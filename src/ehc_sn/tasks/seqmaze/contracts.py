"""SeqMaze probe and v1 task-owned contracts.

Phase 0 edge-lookup probe contracts: input graph structure, supervised edge
targets, and model output shape.

Phase 1 (v1) path-prediction contracts: graph-task input (shared fields with
probe), path supervision targets, and path-logit output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from torch import Tensor

# Canonical ignore label for supervised positions not contributing to loss.
SEQMAZE_IGNORE_LABEL_ID: Final[int] = -100


# =============================================================================
@dataclass(frozen=True)
class SeqMazeProbeInput:
    """Task input for the edge-lookup probe.

    Attributes:
        node_obs_id: (B, N) int64 — observation ids for candidate nodes.
        node_candidate_index: (B, N) int64 — candidate index (0..N-1).
        node_start_flag: (B, N) bool — is this the start node?
        node_goal_flag: (B, N) bool — is this the goal node?
        successor_indices: (B, N, K) int64 — successor candidate indices per node (padded).
        successor_mask: (B, N, K) bool — valid successor slots.
        node_mask: (B, N) bool — valid nodes (padding).
    """

    node_obs_id: Tensor
    node_candidate_index: Tensor
    node_start_flag: Tensor
    node_goal_flag: Tensor
    successor_indices: Tensor
    successor_mask: Tensor
    node_mask: Tensor


# =============================================================================
@dataclass(frozen=True)
class SeqMazeProbeTargets:
    """Edge lookup targets.

    Attributes:
        edge_label: (B, N, N) int64 — 1 if node j is a valid successor of node i, 0 otherwise.
        edge_mask: (B, N, N) bool — valid pairs (both i and j are real nodes).
    """

    edge_label: Tensor
    edge_mask: Tensor


# =============================================================================
@dataclass(frozen=True)
class SeqMazeProbeOutput:
    """Edge prediction output from the probe decoder.

    Attributes:
        edge_logits: (B, N, N, 2) — binary edge logits for each (i, j) pair.
    """

    edge_logits: Tensor


# =============================================================================
# Phase 1 -- Path prediction (v1)
# =============================================================================


@dataclass(frozen=True)
class SeqMazeTaskInput:
    """Task input for seqmaze v1 path prediction.

    Shares the same graph-structure fields as the probe input.  The adapter
    packs these into both the graph region and the path region of the schema.

    Attributes:
        node_obs_id: (B, N) int64 -- observation ids for candidate nodes.
        node_candidate_index: (B, N) int64 -- candidate index (0..N-1).
        node_start_flag: (B, N) bool -- is this the start node?
        node_goal_flag: (B, N) bool -- is this the goal node?
        successor_indices: (B, N, K) int64 -- successor candidate indices (padded).
        successor_mask: (B, N, K) bool -- valid successor slots.
        node_mask: (B, N) bool -- valid nodes (padding).
    """

    node_obs_id: Tensor
    node_candidate_index: Tensor
    node_start_flag: Tensor
    node_goal_flag: Tensor
    successor_indices: Tensor
    successor_mask: Tensor
    node_mask: Tensor


@dataclass(frozen=True)
class SeqMazeTargets:
    """Path supervision targets for seqmaze v1.

    Attributes:
        path_index: (B, T) int64 -- target path token indices in [0, N_max+1].
            N_max = EOS token, N_max+1 = PAD token.
        path_mask: (B, T) bool -- True for supervised positions (includes EOS,
            excludes PAD).
        path_length: (B,) int64 -- length of the path including EOS.
    """

    path_index: Tensor
    path_mask: Tensor
    path_length: Tensor


@dataclass(frozen=True)
class SeqMazeTaskOutput:
    """Path prediction output for seqmaze v1.

    Attributes:
        path_logits: (B, T, N_max+2) float32 -- logits over the path vocabulary
            for each output position.
        oracle_path_logits: (B, T, N_max+2) float32, optional -- logits decoded
            from graph-region slots via the oracle decoder (diagnostic ablation
            only, None when disabled).
        edge_logits: (B, N, N, 2) float32, optional -- auxiliary edge-prediction
            logits from graph-region PFC output slots (multi-task training).
    """

    path_logits: Tensor
    oracle_path_logits: Tensor | None = None
    edge_logits: Tensor | None = None


# =============================================================================
__all__ = [
    "SeqMazeProbeInput",
    "SeqMazeProbeTargets",
    "SeqMazeProbeOutput",
    "SeqMazeTaskInput",
    "SeqMazeTargets",
    "SeqMazeTaskOutput",
]
