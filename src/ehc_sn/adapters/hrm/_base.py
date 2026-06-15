"""Shared MazeHard+HRM bridge family core.

Holds the task-side settings and token encoder/decoder glue shared by the
MazeHard+HRM v1 and v2 bridge adapters.  Versioned bridge modules keep the
model-native input and controller-output types local.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, Literal, Protocol, TypeVar

import torch
from pydantic import BaseModel, Field
from torch import Tensor
from torch import device as Device
from torch import dtype as Dtype
from torch import nn

from ehc_sn.tasks.mazehard.contracts import (
    MazeHardTaskInput,
    MazeHardTaskOutput,
)
from ehc_sn.tasks.mazehard.runtime import PATH_ID as O_ID
from ehc_sn.tasks.mazehard.runtime import SEM_VOCAB_SIZE as MAZE_SEM_VOCAB_SIZE

TInput = TypeVar("TInput")


DEFAULT_MAZE_HARD_HRM_VOCAB_SIZE: int = max(MAZE_SEM_VOCAB_SIZE, O_ID + 1)
"""Default vocabulary size for MazeHard HRM bridges plus the solution-overlay token."""


# =============================================================================
class MazeHardHRMAdapterSettings(BaseModel, extra="forbid"):
    """Task-side MazeHard settings shared by the HRM bridge family."""

    encoder_kind: Literal["learned", "rope"] = Field(
        default="rope",
        description="Positional front-end used by the MazeHard token encoder.",
    )
    vocab_size: int = Field(
        default=DEFAULT_MAZE_HARD_HRM_VOCAB_SIZE,
        ge=1,
        description=(
            "MazeHard token vocabulary size used by encoder and decoder heads. "
            "Defaults to the canonical SEM vocabulary plus the solution-overlay token."
        ),
    )


# =============================================================================
class MazeHardLearnedEncoder(nn.Module, Generic[TInput]):
    """Encoder for MazeHard token inputs using learned positional embeddings."""

    def __init__(  # ----------------------------------------------------------
        self,
        seq_length: int,
        vocab_size: int,
        hidden_size: int,
        *,
        input_factory: Callable[[Tensor, Tensor | None], TInput],
        device: Device | None = None,
        dtype: Dtype | None = None,
    ) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(
            vocab_size, hidden_size, device=device, dtype=dtype
        )
        self.embed_pos = nn.Embedding(
            seq_length, hidden_size, device=device, dtype=dtype
        )
        self.embedding_scale = 0.707106781 * (hidden_size**0.5)
        self._input_factory = input_factory

    def forward(  # -----------------------------------------------------------
        self,
        batch: MazeHardTaskInput,
    ) -> TInput:
        """Encode MazeHard tokens with learned positional embeddings."""
        token_embeddings = self.embed_tokens(
            batch.input_ids.to(dtype=torch.int32)
        )
        positions = torch.arange(
            self.embed_pos.num_embeddings, device=batch.input_ids.device
        )
        pos_embeddings = self.embed_pos(positions).unsqueeze(0)
        return self._input_factory(
            self.embedding_scale * (token_embeddings + pos_embeddings),
            None,
        )


# =============================================================================
class MazeHardRoPEEncoder(nn.Module, Generic[TInput]):
    """Encoder for MazeHard token inputs using a RoPE-compatible front-end."""

    def __init__(  # ----------------------------------------------------------
        self,
        seq_length: int,
        vocab_size: int,
        hidden_size: int,
        *,
        input_factory: Callable[[Tensor, Tensor | None], TInput],
        device: Device | None = None,
        dtype: Dtype | None = None,
    ) -> None:
        super().__init__()
        _ = seq_length
        self.embed_tokens = nn.Embedding(
            vocab_size, hidden_size, device=device, dtype=dtype
        )
        self.embedding_scale = hidden_size**0.5
        self._input_factory = input_factory

    def forward(  # -----------------------------------------------------------
        self,
        batch: MazeHardTaskInput,
    ) -> TInput:
        """Encode MazeHard tokens without a learned positional table."""
        token_embeddings = self.embed_tokens(
            batch.input_ids.to(dtype=torch.int32)
        )
        return self._input_factory(
            self.embedding_scale * token_embeddings,
            None,
        )


# =============================================================================
def build_token_encoder(
    *,
    seq_length: int,
    vocab_size: int,
    hidden_size: int,
    encoder_kind: Literal["learned", "rope"],
    input_factory: Callable[[Tensor, Tensor | None], TInput],
    device: Device | None = None,
    dtype: Dtype | None = None,
) -> MazeHardLearnedEncoder[TInput] | MazeHardRoPEEncoder[TInput]:
    """Construct the MazeHard token encoder front-end for one HRM bridge."""
    match encoder_kind:
        case "learned":
            encoder_cls = MazeHardLearnedEncoder[TInput]
        case "rope":
            encoder_cls = MazeHardRoPEEncoder[TInput]
        case _:
            raise ValueError(f"Unsupported encoder kind: {encoder_kind}")

    return encoder_cls(
        seq_length=seq_length,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        input_factory=input_factory,
        device=device,
        dtype=dtype,
    )


# =============================================================================
class HasSchemaSlots(Protocol):
    """Minimal model-output surface required by the shared MazeHard decoder."""

    schema_slots: Tensor


# =============================================================================
class MazeHardMLPDecoder(nn.Module):
    """Decoder mapping HRM schema-slot features to MazeHard task logits."""

    def __init__(  # ----------------------------------------------------------
        self,
        hidden_size: int,
        vocab_size: int,
        *,
        device: Device | None = None,
        dtype: Dtype | None = None,
    ) -> None:
        super().__init__()
        self.lm_head = nn.Linear(
            hidden_size, vocab_size, bias=False, device=device, dtype=dtype
        )

    def forward(  # -----------------------------------------------------------
        self,
        outputs: HasSchemaSlots,
    ) -> MazeHardTaskOutput:
        """Decode schema-slot activations into MazeHard token logits."""
        return MazeHardTaskOutput(
            task_logits=self.lm_head(outputs.schema_slots)
        )


# =============================================================================
def build_token_decoder(
    *,
    hidden_size: int,
    vocab_size: int,
    device: Device | None = None,
    dtype: Dtype | None = None,
) -> MazeHardMLPDecoder:
    """Construct the MazeHard token decoder head for one HRM bridge."""
    return MazeHardMLPDecoder(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        device=device,
        dtype=dtype,
    )


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
    edge_encoding: Literal["successor_index_embedding"] = (
        "successor_index_embedding"
    )
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
            config.vocab_size_obs,
            config.hidden_size,
            device=device,
            dtype=dtype,
        )
        self.E_candidate = nn.Embedding(
            config.vocab_size_candidate,
            config.hidden_size,
            device=device,
            dtype=dtype,
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
        node_obs_id: Tensor,  # (B, N)
        node_candidate_index: Tensor,  # (B, N)
        node_start_flag: Tensor,  # (B, N)
        node_goal_flag: Tensor,  # (B, N)
        successor_indices: Tensor,  # (B, N, K)
        successor_mask: Tensor,  # (B, N, K)
        node_mask: Tensor,  # (B, N)
    ) -> Tensor:
        """Encode graph nodes into schema tokens (B, N, D).

        Returns:
            schema_tokens: (B, N, D) — node embeddings for the graph region.
            schema_mask: (B, N) — False for padded nodes.
        """
        B, N, K = successor_indices.shape
        D = self.config.hidden_size

        # Content embedding
        obs_emb = self.E_obs(node_obs_id)  # (B, N, D)
        cand_emb = self.E_candidate(node_candidate_index)  # (B, N, D)
        start_emb = self.E_start(node_start_flag.to(torch.int64))  # (B, N, D)
        goal_emb = self.E_goal(node_goal_flag.to(torch.int64))  # (B, N, D)

        content = obs_emb + cand_emb + start_emb + goal_emb

        # Edge embedding: successor_index_embedding
        # For each node i, pool over successor slots
        # Clamp successor_indices to valid range for embedding lookup
        succ_idx_clamped = successor_indices.clamp(min=0, max=self.config.n_max)
        succ_idx_mask = successor_mask  # (B, N, K)

        # E_successor_slot(k) — broadcast over (B, N)
        slot_emb = self.E_successor_slot.weight.unsqueeze(0).unsqueeze(
            0
        )  # (1, 1, K, D)
        slot_emb = slot_emb.expand(B, N, -1, -1)  # (B, N, K, D)

        # E_candidate_index(succ[i,k])
        index_emb = self.E_candidate_index(succ_idx_clamped)  # (B, N, K, D)

        # Sum slot + index, then mean-pool over K
        edge_per_slot = slot_emb + index_emb  # (B, N, K, D)
        edge_emb = edge_per_slot * succ_idx_mask.unsqueeze(
            -1
        )  # zero out invalid
        denom = succ_idx_mask.sum(dim=-1, keepdim=True).clamp(
            min=1
        )  # (B, N, 1)
        edge_emb = edge_emb.sum(dim=-2) / denom  # (B, N, D)

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
        slot_i = schema_slots.unsqueeze(2)  # (B, N, 1, D)
        slot_j = schema_slots.unsqueeze(1)  # (B, 1, N, D)

        # Broadcast to (B, N, N, 2*D) and apply linear
        pair_emb = torch.cat(
            [slot_i.expand(-1, -1, N, -1), slot_j.expand(-1, N, -1, -1)],
            dim=-1,
        )  # (B, N, N, 2*D)
        edge_logits = self.pairwise(pair_emb)  # (B, N, N, 2)
        return edge_logits


# =============================================================================
class SeqMazeAdapterSettings(BaseModel, extra="forbid"):
    """Adapter configuration for seqmaze v1 path prediction.

    Attributes:
        n_max: Maximum candidate nodes per batch (N).
        t_max: Maximum generated path length (T).
        k_max: Maximum out-degree per node (K).
        vocab_size_obs: Vocabulary size for observation-id embeddings.
        vocab_size_candidate: Vocabulary size for candidate-index embeddings.
        edge_encoding: Edge encoding mode.
            - "successor_index_embedding": v1 default. Embeds adjacency list
              as E_slot(k) + E_index(succ[i,k]), pooled per node.
            - "successor_node_pool": Pools successor node base embeddings
              for one-hop structural summary.
            - "none": Negative control. Removes all transition information.
        hidden_size: Embedding dimension (must match PFC hidden size).
        share_path_position_embeddings: Whether E_decode_position shares
            weights with E_path_position.
    """

    n_max: int = Field(default=32, ge=1)
    t_max: int = Field(default=32, ge=1)
    k_max: int = Field(default=4, ge=1)
    vocab_size_obs: int = Field(default=64, ge=1)
    vocab_size_candidate: int = Field(default=64, ge=1)
    edge_encoding: Literal[
        "successor_index_embedding",
        "successor_node_pool",
        "none",
    ] = Field(default="successor_index_embedding")
    hidden_size: int = Field(default=128, ge=1)
    share_path_position_embeddings: bool = Field(default=True)


# =============================================================================
class SeqMazeEncoder(nn.Module):
    """Encoder that packs graph nodes + path queries into HRM schema tokens.

    Schema layout (S = N + T):

        positions [0 : N):
            graph region — node embeddings for candidate graph nodes.

        positions [N : N + T):
            path region — learned query embeddings for output positions.

    The graph region uses the same successor-index embedding as the probe
    encoder.  The path region uses learned position-specific query embeddings.
    A region tag embedding is added to every slot.
    """

    def __init__(
        self,
        config: SeqMazeAdapterSettings,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.config = config

        D = config.hidden_size

        # --- Graph region: content embeddings ---
        self.E_obs = nn.Embedding(
            config.vocab_size_obs, D, device=device, dtype=dtype
        )
        self.E_candidate = nn.Embedding(
            max(config.n_max + 1, config.vocab_size_candidate),
            D,
            device=device,
            dtype=dtype,
        )
        # Flag embeddings (2 values: False=0, True=1)
        self.E_start = nn.Embedding(2, D, device=device, dtype=dtype)
        self.E_goal = nn.Embedding(2, D, device=device, dtype=dtype)

        # --- Graph region: edge encoding ---
        self.E_successor_slot = nn.Embedding(
            config.k_max, D, device=device, dtype=dtype
        )
        self.E_candidate_index = nn.Embedding(
            config.n_max + 1, D, device=device, dtype=dtype
        )

        # --- Graph region: successor_node_pool encoding ---
        # Shared with candidate-index embedding above (same index space)
        # Pooled by mean over valid successors

        # --- Path region: learned query embeddings ---
        self.E_path_query = nn.Embedding(1, D, device=device, dtype=dtype)
        self.E_path_position = nn.Embedding(
            config.t_max, D, device=device, dtype=dtype
        )

        # --- Region embeddings ---
        self.E_region_graph = nn.Embedding(1, D, device=device, dtype=dtype)
        self.E_region_path = nn.Embedding(1, D, device=device, dtype=dtype)

        # Scaling
        self.embedding_scale = D**0.5

    def forward(
        self,
        # Graph region inputs
        node_obs_id: Tensor,  # (B, N)
        node_candidate_index: Tensor,  # (B, N)
        node_start_flag: Tensor,  # (B, N)
        node_goal_flag: Tensor,  # (B, N)
        successor_indices: Tensor,  # (B, N, K)
        successor_mask: Tensor,  # (B, N, K)
        node_mask: Tensor,  # (B, N)
    ) -> tuple[Tensor, Tensor]:
        """Encode graph nodes and path queries into schema tokens.

        Returns:
            schema_tokens: (B, S, D) — full schema token sequence.
            schema_mask: (B, S) — True for valid tokens.
        """
        config = self.config
        B, N, K = successor_indices.shape
        T = config.t_max
        S = N + T
        D = config.hidden_size
        device = successor_indices.device

        # ---- Graph region ----
        # Content embedding
        obs_emb = self.E_obs(node_obs_id)  # (B, N, D)
        cand_emb = self.E_candidate(node_candidate_index)  # (B, N, D)
        start_emb = self.E_start(node_start_flag.to(torch.int64))  # (B, N, D)
        goal_emb = self.E_goal(node_goal_flag.to(torch.int64))  # (B, N, D)
        content = obs_emb + cand_emb + start_emb + goal_emb

        # Edge embedding
        edge_emb = self._encode_edges(
            successor_indices, successor_mask, node_candidate_index, node_obs_id
        )  # (B, N, D)

        if self.training and self.config.n_max <= 8:
            n_actual = node_mask.sum(dim=-1).float().mean().item()
            succ_min = successor_indices.min().item()
            succ_max = successor_indices.max().item()
            mask_ratio = successor_mask.float().mean().item()
            print(
                f"[seqmaze-encoder] n_actual={n_actual:.1f} "
                f"succ_range=[{succ_min},{succ_max}] "
                f"mask_ratio={mask_ratio:.3f} "
                f"edge_emb_norm={edge_emb.norm(dim=-1).mean().item():.4f}"
            )

        # Region tag
        region_graph = self.E_region_graph.weight.unsqueeze(0)  # (1, 1, D)

        # Final node embedding
        graph_tokens = (
            self.embedding_scale * (content + edge_emb) + region_graph
        )  # (B, N, D)

        # Graph mask
        graph_mask = node_mask  # (B, N)

        # ---- Path region ----
        # Learned query embedding (same for all path slots)
        query = self.E_path_query.weight.unsqueeze(
            0
        )  # (1, 1, D) — weight is (1, D)
        query = query.expand(B, T, -1)  # (B, T, D)

        # Position embedding
        pos_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        pos_emb = self.E_path_position(pos_ids)  # (B, T, D)

        region_path = self.E_region_path.weight.unsqueeze(0)  # (1, 1, D)

        path_tokens = (
            self.embedding_scale * (query + pos_emb) + region_path
        )  # (B, T, D)

        # Path mask: path positions are always valid
        path_mask = torch.ones(B, T, dtype=torch.bool, device=device)

        # ---- Concatenate ----
        schema_tokens = torch.cat(
            [graph_tokens, path_tokens], dim=1
        )  # (B, S, D)
        schema_mask = torch.cat([graph_mask, path_mask], dim=1)  # (B, S)

        return schema_tokens, schema_mask

    def forward_padded(
        self,
        # Graph region inputs
        node_obs_id: Tensor,  # (B, N)
        node_candidate_index: Tensor,  # (B, N)
        node_start_flag: Tensor,  # (B, N)
        node_goal_flag: Tensor,  # (B, N)
        successor_indices: Tensor,  # (B, N, K)
        successor_mask: Tensor,  # (B, N, K)
        node_mask: Tensor,  # (B, N)
        model_seq_length: int,  # total PFC slot capacity
    ) -> tuple[Tensor, Tensor]:
        """Run ``forward()`` then pad output to *model_seq_length*.

        When the task schema length (N + T) is less than the model's
        PFC slot capacity, trailing positions are filled with zero
        vectors and masked as invalid.

        Returns:
            schema_tokens: ``(B, model_seq_length, D)``.
            schema_mask: ``(B, model_seq_length)``.

        Raises:
            ValueError: If ``N + T > model_seq_length``.
        """
        schema_tokens, schema_mask = self(
            node_obs_id=node_obs_id,
            node_candidate_index=node_candidate_index,
            node_start_flag=node_start_flag,
            node_goal_flag=node_goal_flag,
            successor_indices=successor_indices,
            successor_mask=successor_mask,
            node_mask=node_mask,
        )
        config = self.config
        B, N, _ = successor_indices.shape
        T = config.t_max
        S_task = N + T
        D = config.hidden_size
        device = successor_indices.device

        if S_task > model_seq_length:
            raise ValueError(
                f"Task schema length {S_task} (N={N} + T={T}) exceeds "
                f"PFC capacity {model_seq_length}."
            )

        if S_task < model_seq_length:
            pad_len = model_seq_length - S_task
            pad_tokens = torch.zeros(
                B, pad_len, D, device=device, dtype=schema_tokens.dtype
            )
            pad_mask = torch.zeros(B, pad_len, dtype=torch.bool, device=device)
            schema_tokens = torch.cat([schema_tokens, pad_tokens], dim=1)
            schema_mask = torch.cat([schema_mask, pad_mask], dim=1)

        return schema_tokens, schema_mask

    def _encode_edges(
        self,
        successor_indices: Tensor,  # (B, N, K)
        successor_mask: Tensor,  # (B, N, K)
        node_candidate_index: Tensor,  # (B, N)
        node_obs_id: Tensor,  # (B, N)
    ) -> Tensor:
        """Encode transition structure into per-node edge embeddings.

        Returns:
            edge_emb: (B, N, D)
        """
        config = self.config
        B, N, K = successor_indices.shape
        D = config.hidden_size

        if config.edge_encoding == "none":
            return torch.zeros(B, N, D, device=successor_indices.device)

        if config.edge_encoding == "successor_index_embedding":
            return self._encode_successor_index_embedding(
                successor_indices, successor_mask
            )

        if config.edge_encoding == "successor_node_pool":
            return self._encode_successor_node_pool(
                successor_indices, successor_mask, node_obs_id
            )

        raise ValueError(f"Unknown edge_encoding: {config.edge_encoding!r}")

    def _encode_successor_index_embedding(
        self,
        successor_indices: Tensor,  # (B, N, K)
        successor_mask: Tensor,  # (B, N, K)
    ) -> Tensor:
        """Encode edges as pooled slot+index embeddings.

        edge_embedding_i =
            Pool_k [ E_successor_slot(k) + E_candidate_index(succ[i,k]) ]
            masked by successor_mask[i, k]
        """
        config = self.config
        B, N, K = successor_indices.shape
        D = config.hidden_size

        succ_idx_clamped = successor_indices.clamp(min=0, max=config.n_max)
        succ_idx_mask = successor_mask

        # E_successor_slot(k) — broadcast over (B, N)
        slot_emb = self.E_successor_slot.weight.unsqueeze(0).unsqueeze(0)
        slot_emb = slot_emb.expand(B, N, -1, -1)  # (B, N, K, D)

        # E_candidate_index(succ[i,k])
        index_emb = self.E_candidate_index(succ_idx_clamped)  # (B, N, K, D)

        # Sum slot + index, then mean-pool over K
        edge_per_slot = slot_emb + index_emb  # (B, N, K, D)
        edge_emb = edge_per_slot * succ_idx_mask.unsqueeze(-1)
        denom = succ_idx_mask.sum(dim=-1, keepdim=True).clamp(min=1)
        edge_emb = edge_emb.sum(dim=-2) / denom  # (B, N, D)

        return edge_emb

    def _encode_successor_node_pool(
        self,
        successor_indices: Tensor,  # (B, N, K)
        successor_mask: Tensor,  # (B, N, K)
        node_obs_id: Tensor,  # (B, N)
    ) -> Tensor:
        """Encode edges by pooling successor node base embeddings.

        edge_embedding_i =
            Pool_k [ E_obs(obs_id[succ[i,k]]) ]
            masked by successor_mask[i, k]
        """
        config = self.config
        B, N, K = successor_indices.shape
        D = config.hidden_size

        # Clamp indices to valid range
        succ_idx_clamped = successor_indices.clamp(min=0, max=N - 1)
        succ_idx_mask = successor_mask  # (B, N, K)

        # Gather obs_id of successors: node_obs_id[b, succ[i,k]]
        # Expand node_obs_id to (B, N, 1) -> gather along dim=1 with succ indices
        batch_idx = (
            torch.arange(B, device=node_obs_id.device)
            .unsqueeze(1)
            .unsqueeze(2)
            .expand(-1, N, K)
        )
        node_idx = succ_idx_clamped  # (B, N, K)
        succ_obs_id = node_obs_id[batch_idx, node_idx]  # (B, N, K)

        # Embed and mean-pool
        obs_emb = self.E_obs(succ_obs_id)  # (B, N, K, D)
        obs_emb = obs_emb * succ_idx_mask.unsqueeze(-1)
        denom = succ_idx_mask.sum(dim=-1, keepdim=True).clamp(min=1)
        edge_emb = obs_emb.sum(dim=-2) / denom  # (B, N, D)

        return edge_emb


# =============================================================================
class SeqMazeDecoder(nn.Module):
    """Decoder that reads path-region slots and produces path logits.

    Applies an explicit output-position embedding before the linear head:

        decoder_input_t = path_states[:, t, :] + E_decode_position(t)
        path_logits = Linear(D, N_max + 2)(decoder_input)

    E_decode_position optionally shares weights with E_path_position
    (controlled by share_path_position_embeddings in the config).
    """

    def __init__(
        self,
        config: SeqMazeAdapterSettings,
        path_position_emb: nn.Embedding | None = None,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        D = config.hidden_size
        V = config.n_max + 2  # path vocabulary size

        if (
            config.share_path_position_embeddings
            and path_position_emb is not None
        ):
            self.E_decode_position = path_position_emb
        else:
            self.E_decode_position = nn.Embedding(
                config.t_max, D, device=device, dtype=dtype
            )

        self.lm_head = nn.Linear(D, V, bias=False, device=device, dtype=dtype)

    def forward(
        self,
        schema_slots: Tensor,  # (B, S, D) — full schema output from HRM
        n_max: int,
        t_max: int,
    ) -> Tensor:
        """Decode path-region schema slots into path logits.

        Args:
            schema_slots: (B, S, D) — all schema slots from HRM output.
            n_max: Number of graph region slots (N).
            t_max: Number of path region slots (T).

        Returns:
            path_logits: (B, T, N+2) float32.
        """
        # Extract path region
        path_states = schema_slots[:, n_max : n_max + t_max, :]  # (B, T, D)
        B, T, D = path_states.shape

        # Position embedding
        device = schema_slots.device
        pos_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        pos_emb = self.E_decode_position(pos_ids)  # (B, T, D)

        decoder_input = path_states + pos_emb  # (B, T, D)
        path_logits = self.lm_head(decoder_input)  # (B, T, N+2)
        return path_logits


# =============================================================================
__all__ = [
    "O_ID",
    "DEFAULT_MAZE_HARD_HRM_VOCAB_SIZE",
    "MazeHardHRMAdapterSettings",
    "MazeHardLearnedEncoder",
    "MazeHardRoPEEncoder",
    "MazeHardMLPDecoder",
    "HasSchemaSlots",
    "build_token_encoder",
    "build_token_decoder",
    "SeqMazeProbeAdapterSettings",
    "SeqMazeProbeEncoder",
    "SeqMazeProbeDecoder",
    "SeqMazeAdapterSettings",
    "SeqMazeEncoder",
    "SeqMazeDecoder",
]
