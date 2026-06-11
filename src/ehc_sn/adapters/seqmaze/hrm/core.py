"""SeqMaze probe adapter settings and encoder/decoder logic.

Implements the successor_index_embedding edge encoding for the Phase 0 probe:
mean-pool E_slot(k) + E_candidate_index(succ[i,k]) over valid successor slots.
"""

from __future__ import annotations

from typing import Literal

import torch
from pydantic import BaseModel, Field
from torch import Tensor, nn


# =============================================================================
class SeqMazeProbeAdapterSettings(BaseModel, extra="forbid"):
    """Adapter configuration for the seqmaze edge-lookup probe.

    Attributes:
        n_max: Maximum candidate nodes per batch (N).
        k_max: Maximum out-degree per node (K).
        vocab_size_obs: Vocabulary size for observation-id embeddings.
        vocab_size_candidate: Vocabulary size for candidate-index embeddings.
        edge_encoding: Edge encoding mode (v1 default: successor_index_embedding).
        hidden_size: Embedding dimension (must match PFC hidden size).
    """

    n_max: int = Field(default=8, ge=1)
    k_max: int = Field(default=3, ge=1)
    vocab_size_obs: int = Field(default=64, ge=1)
    vocab_size_candidate: int = Field(default=16, ge=1)
    edge_encoding: Literal["successor_index_embedding"] = "successor_index_embedding"
    hidden_size: int = Field(default=64, ge=1)


# =============================================================================
class SeqMazeProbeEncoder(nn.Module):
    """Encoder that packs graph nodes into schema tokens for the HRM workspace.

    Implements the successor-index edge embedding:

        edge_embedding_i =
            Pool_k [ E_successor_slot(k) + E_candidate_index(successor_indices[i, k]) ]
            masked by successor_mask[i, k]

    Then:

        node_embedding_i =
            E_obs(obs_id_i) + E_candidate(candidate_index_i)
            + E_start(start_flag_i) + E_goal(goal_flag_i)
            + edge_embedding_i
            + E_region("graph")
    """

    def __init__(
        self,
        config: SeqMazeProbeAdapterSettings,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.config = config

        # Content embeddings
        self.E_obs = nn.Embedding(
            config.vocab_size_obs, config.hidden_size, device=device, dtype=dtype
        )
        self.E_candidate = nn.Embedding(
            config.vocab_size_candidate, config.hidden_size, device=device, dtype=dtype
        )

        # Flag embeddings (2 values: False=0, True=1)
        self.E_start = nn.Embedding(
            2, config.hidden_size, device=device, dtype=dtype
        )
        self.E_goal = nn.Embedding(
            2, config.hidden_size, device=device, dtype=dtype
        )

        # Edge encoding embeddings
        self.E_successor_slot = nn.Embedding(
            config.k_max, config.hidden_size, device=device, dtype=dtype
        )
        # Candidate index for successor (vocab includes padding at 0, values 0..N_max)
        self.E_candidate_index = nn.Embedding(
            config.n_max + 1, config.hidden_size, device=device, dtype=dtype
        )

        # Region embedding (graph region only for probe with no path region)
        self.E_region_graph = nn.Embedding(
            1, config.hidden_size, device=device, dtype=dtype
        )

        # Scaling
        self.embedding_scale = config.hidden_size**0.5

    def forward(
        self,
        node_obs_id: Tensor,                # (B, N)
        node_candidate_index: Tensor,        # (B, N)
        node_start_flag: Tensor,             # (B, N)
        node_goal_flag: Tensor,              # (B, N)
        successor_indices: Tensor,           # (B, N, K)
        successor_mask: Tensor,              # (B, N, K)
        node_mask: Tensor,                   # (B, N)
    ) -> Tensor:
        """Encode graph nodes into schema tokens (B, N, D).

        Returns:
            schema_tokens: (B, N, D) — node embeddings for the graph region.
            schema_mask: (B, N) — False for padded nodes.
        """
        B, N, K = successor_indices.shape
        D = self.config.hidden_size

        # Content embedding
        obs_emb = self.E_obs(node_obs_id)                     # (B, N, D)
        cand_emb = self.E_candidate(node_candidate_index)     # (B, N, D)
        start_emb = self.E_start(node_start_flag.to(torch.int64))  # (B, N, D)
        goal_emb = self.E_goal(node_goal_flag.to(torch.int64))     # (B, N, D)

        content = obs_emb + cand_emb + start_emb + goal_emb

        # Edge embedding: successor_index_embedding
        # For each node i, pool over successor slots
        # Clamp successor_indices to valid range for embedding lookup
        succ_idx_clamped = successor_indices.clamp(min=0, max=self.config.n_max)
        succ_idx_mask = successor_mask  # (B, N, K)

        # E_successor_slot(k) — broadcast over (B, N)
        slot_emb = self.E_successor_slot.weight.unsqueeze(0).unsqueeze(0)  # (1, 1, K, D)
        slot_emb = slot_emb.expand(B, N, -1, -1)                          # (B, N, K, D)

        # E_candidate_index(succ[i,k])
        index_emb = self.E_candidate_index(succ_idx_clamped)               # (B, N, K, D)

        # Sum slot + index, then mean-pool over K
        edge_per_slot = slot_emb + index_emb                               # (B, N, K, D)
        edge_emb = edge_per_slot * succ_idx_mask.unsqueeze(-1)             # zero out invalid
        denom = succ_idx_mask.sum(dim=-1, keepdim=True).clamp(min=1)       # (B, N, 1)
        edge_emb = edge_emb.sum(dim=-2) / denom                            # (B, N, D)

        # Region embedding (graph)
        region_emb = self.E_region_graph.weight.unsqueeze(0)  # (1, 1, D)
        region_emb = region_emb.expand(B, N, -1)

        # Final node embedding
        node_emb = self.embedding_scale * (content + edge_emb) + region_emb
        schema_mask = node_mask

        return node_emb, schema_mask


# =============================================================================
class SeqMazeProbeDecoder(nn.Module):
    """Decoder that reads schema slots and produces pairwise edge logits.

    For each (i,j) pair: concat(slot_i, slot_j) -> Linear(2*D, 2) -> 2 logits.
    """

    def __init__(
        self,
        config: SeqMazeProbeAdapterSettings,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.pairwise = nn.Linear(
            2 * config.hidden_size, 2, device=device, dtype=dtype
        )

    def forward(
        self,
        schema_slots: Tensor,  # (B, N, D)
    ) -> Tensor:
        """Decode schema slots into edge logits.

        Args:
            schema_slots: (B, N, D) — output slots from HRM schema workspace.

        Returns:
            edge_logits: (B, N, N, 2) — binary edge logits for each (i,j) pair.
        """
        B, N, D = schema_slots.shape

        # Create all pairs: slot_i (B, N, 1, D) and slot_j (B, 1, N, D)
        slot_i = schema_slots.unsqueeze(2)   # (B, N, 1, D)
        slot_j = schema_slots.unsqueeze(1)   # (B, 1, N, D)

        # Broadcast to (B, N, N, 2*D) and apply linear
        pair_emb = torch.cat(
            [slot_i.expand(-1, -1, N, -1), slot_j.expand(-1, N, -1, -1)],
            dim=-1,
        )  # (B, N, N, 2*D)
        edge_logits = self.pairwise(pair_emb)  # (B, N, N, 2)
        return edge_logits
