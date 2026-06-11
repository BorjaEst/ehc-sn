"""SeqMaze probe task-owned contracts.

Phase 0 edge-lookup probe contracts: input graph structure, supervised edge
targets, and model output shape.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


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
