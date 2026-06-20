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

from ehc_sn.tasks.goaltrace.contracts import (
    GoaltraceTaskInput,
    GoaltraceTaskOutput,
)
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
class GoaltraceHRMAdapterSettings(BaseModel, extra="forbid"):
    """Task-side Goaltrace settings shared by the HRM bridge family.

    Attributes:
        encoder_kind: Positional front-end (``"learned"`` or ``"rope"``).
        num_observations: Padded node count N_max.  How many node slots
            are encoded in one task instance.  The observation-ID
            embedding ``E_obs`` is sized to ``num_observations`` — under
            the current corpus contract observation IDs are dense
            permutations of ``[0, N_max)``, so V = N_max.
    """

    encoder_kind: Literal["learned", "rope"] = Field(
        default="rope",
        description="Positional front-end used by the Goaltrace token encoder.",
    )
    num_observations: int = Field(
        default=45,
        ge=1,
        description="Padded node count N_max.  Determines the number of "
        "schema slots reserved for goaltrace nodes and the observation-ID "
        "embedding table size.  Must match the goaltrace corpus "
        "``n_observations``.",
    )
    padding_obs_id: int | None = Field(
        default=None,
        description="Observation ID sentinel for padding slots.  When set, "
        "the encoder uses ``vocab_size = padding_obs_id + 1`` and applies "
        "``padding_idx = padding_obs_id`` so padding slots embed to zero. "
        "When ``None`` (default), the encoder uses ``vocab_size = num_observations`` "
        "with no padding_idx (backward compatible with v1 corpora).",
    )


# =============================================================================
# =============================================================================
class RoutebindHRMAdapterSettings(BaseModel, extra="forbid"):
    """Task-side Routebind settings shared by the HRM bridge family.

    Attributes:
        encoder_kind: Positional front-end (``"learned"`` or ``"rope"``).
        num_cell_types: Number of cell type categories (WALL=0, FREE=1,
            OBSERVATION=2).  Default 3.
        num_observations: Number of observation identities in the hidden
            DAG.  Determines the observation-ID embedding table size.
        grid_height: Grid height in cells (default 30 for v1).
        grid_width: Grid width in cells (default 30 for v1).
        padding_obs_id: Observation ID sentinel for non-observation cells.
            Defaults to ``num_observations`` (the sentinel value used during
            data generation; the embedding at this index is frozen to zero).
    """

    encoder_kind: Literal["learned", "rope"] = Field(
        default="rope",
        description="Positional front-end used by the Routebind token encoder.",
    )
    num_cell_types: int = Field(
        default=3,
        ge=1,
        description="Number of cell type categories (WALL, FREE, OBSERVATION).",
    )
    num_observations: int = Field(
        default=16,
        ge=1,
        description="Number of observation identities in the hidden DAG.",
    )
    grid_height: int = Field(
        default=30,
        ge=1,
        description="Grid height in cells.",
    )
    grid_width: int = Field(
        default=30,
        ge=1,
        description="Grid width in cells.",
    )


class GoaltraceRoPEEncoder(nn.Module, Generic[TInput]):
    """Encoder for Goaltrace task inputs using a RoPE-compatible front-end.

    Encodes observation IDs, relational weights, current/goal flags into
    schema tokens without adapter-level positional embeddings.  Sequence
    position is left to the HRM's internal RoPE mechanism.
    """

    def __init__(  # ----------------------------------------------------------
        self,
        seq_length: int,
        vocab_size: int,
        hidden_size: int,
        *,
        input_factory: Callable[[Tensor, Tensor | None], TInput],
        padding_idx: int | None = None,
        device: Device | None = None,
        dtype: Dtype | None = None,
    ) -> None:
        super().__init__()
        self._seq_length = seq_length
        self._hidden_size = hidden_size
        self.E_obs = nn.Embedding(
            vocab_size,
            hidden_size,
            padding_idx=padding_idx,
            device=device,
            dtype=dtype,
        )
        self.f_weight = nn.Linear(1, hidden_size, device=device, dtype=dtype)
        self.E_current = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.E_goal = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.embedding_scale = hidden_size**0.5
        self._input_factory = input_factory
        self.reset_parameters()

    def reset_parameters(  # ------------------------------------------------------
        self,
    ) -> None:
        """Initialize weights with small normal to avoid HRM input overdrive."""
        nn.init.normal_(self.E_obs.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.f_weight.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.f_weight.bias)
        nn.init.normal_(self.E_current, mean=0.0, std=0.02)
        nn.init.normal_(self.E_goal, mean=0.0, std=0.02)

    def forward(  # -----------------------------------------------------------
        self,
        batch: GoaltraceTaskInput,
    ) -> TInput:
        """Encode Goaltrace task inputs without learned positional embeddings."""
        obs_id = batch.observation_id.to(dtype=torch.int32)
        B, N = obs_id.shape
        D = self._hidden_size
        S = self._seq_length

        obs_emb = self.E_obs(obs_id)  # (B, N, D)
        w_emb = self.f_weight(batch.weight.unsqueeze(-1))  # (B, N, D)
        current_emb = self.E_current * batch.current_flag.unsqueeze(-1).float()
        goal_emb = self.E_goal * batch.goal_flag.unsqueeze(-1).float()

        content = obs_emb + w_emb + current_emb + goal_emb
        encoded = self.embedding_scale * content  # (B, N, D)

        # Pad to model slot capacity
        if N < S:
            pad = torch.zeros(
                B, S - N, D, device=encoded.device, dtype=encoded.dtype
            )
            encoded = torch.cat([encoded, pad], dim=1)

        return self._input_factory(encoded, None)


# =============================================================================
class GoaltraceLearnedEncoder(nn.Module, Generic[TInput]):
    """Encoder for Goaltrace task inputs using learned positional embeddings."""

    def __init__(  # ----------------------------------------------------------
        self,
        seq_length: int,
        vocab_size: int,
        hidden_size: int,
        *,
        input_factory: Callable[[Tensor, Tensor | None], TInput],
        padding_idx: int | None = None,
        device: Device | None = None,
        dtype: Dtype | None = None,
    ) -> None:
        super().__init__()
        self._seq_length = seq_length
        self._hidden_size = hidden_size
        self.E_obs = nn.Embedding(
            vocab_size,
            hidden_size,
            padding_idx=padding_idx,
            device=device,
            dtype=dtype,
        )
        self.f_weight = nn.Linear(1, hidden_size, device=device, dtype=dtype)
        self.E_current = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.E_goal = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.E_pos = nn.Embedding(
            seq_length, hidden_size, device=device, dtype=dtype
        )
        self.embedding_scale = 0.707106781 * (hidden_size**0.5)
        self._input_factory = input_factory
        self.reset_parameters()

    def reset_parameters(  # ------------------------------------------------------
        self,
    ) -> None:
        """Initialize weights with small normal to avoid HRM input overdrive."""
        nn.init.normal_(self.E_obs.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.f_weight.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.f_weight.bias)
        nn.init.normal_(self.E_current, mean=0.0, std=0.02)
        nn.init.normal_(self.E_goal, mean=0.0, std=0.02)
        nn.init.normal_(self.E_pos.weight, mean=0.0, std=0.02)

    def forward(  # -----------------------------------------------------------
        self,
        batch: GoaltraceTaskInput,
    ) -> TInput:
        """Encode Goaltrace task inputs with learned positional embeddings."""
        obs_id = batch.observation_id.to(dtype=torch.int32)
        B, N = obs_id.shape
        D = self._hidden_size
        S = self._seq_length

        obs_emb = self.E_obs(obs_id)  # (B, N, D)
        w_emb = self.f_weight(batch.weight.unsqueeze(-1))  # (B, N, D)
        current_emb = self.E_current * batch.current_flag.unsqueeze(-1).float()
        goal_emb = self.E_goal * batch.goal_flag.unsqueeze(-1).float()
        content = obs_emb + w_emb + current_emb + goal_emb

        positions = torch.arange(N, device=obs_id.device)
        pos_emb = self.E_pos(positions).unsqueeze(0)  # (1, N, D)

        encoded = self.embedding_scale * (content + pos_emb)  # (B, N, D)

        # Pad to model slot capacity
        if N < S:
            pad = torch.zeros(
                B, S - N, D, device=encoded.device, dtype=encoded.dtype
            )
            encoded = torch.cat([encoded, pad], dim=1)

        return self._input_factory(encoded, None)


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
    task_family: Literal["mazehard", "goaltrace"] = "mazehard",
    padding_idx: int | None = None,
    device: Device | None = None,
    dtype: Dtype | None = None,
) -> (
    MazeHardLearnedEncoder[TInput]
    | MazeHardRoPEEncoder[TInput]
    | GoaltraceLearnedEncoder[TInput]
    | GoaltraceRoPEEncoder[TInput]
):
    """Construct a token encoder front-end for one HRM bridge.

    Args:
        seq_length: Model PFC slot capacity (S).
        vocab_size: Token or observation vocabulary size.
        hidden_size: Embedding dimension (must match PFC hidden size).
        encoder_kind: ``"learned"`` adds adapter-level learned positional
            embeddings; ``"rope"`` leaves position handling to the HRM's
            internal RoPE mechanism.
        input_factory: Callable that wraps encoded tensors into the
            model-native input type (e.g. ``HRMInputV1``).
        task_family: Which task family the encoder belongs to.
        padding_idx: Optional padding index for the observation-ID embedding.
            ``None`` (default) means no padding index (v1 backward compat).
            When set, the embedding at that index is frozen to zero.
    """
    if task_family == "mazehard":
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
    else:
        match encoder_kind:
            case "learned":
                encoder_cls = GoaltraceLearnedEncoder[TInput]
            case "rope":
                encoder_cls = GoaltraceRoPEEncoder[TInput]
            case _:
                raise ValueError(f"Unsupported encoder kind: {encoder_kind}")
        return encoder_cls(
            seq_length=seq_length,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            input_factory=input_factory,
            padding_idx=padding_idx,
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
class GoaltraceMLPDecoder(nn.Module):
    """Decoder mapping HRM schema-slot features to Goaltrace firing field."""

    def __init__(  # ----------------------------------------------------------
        self,
        hidden_size: int,
        num_observations: int,
        *,
        device: Device | None = None,
        dtype: Dtype | None = None,
    ) -> None:
        super().__init__()
        self._num_observations = num_observations
        self.field_head = nn.Linear(hidden_size, 1, device=device, dtype=dtype)
        self.reset_parameters()

    def reset_parameters(  # ------------------------------------------------------
        self,
    ) -> None:
        """Initialize decoder head with small normal."""
        nn.init.normal_(self.field_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.field_head.bias)

    def forward(  # -----------------------------------------------------------
        self,
        outputs: HasSchemaSlots,
    ) -> GoaltraceTaskOutput:
        """Decode the first ``num_observations`` schema slots into a
        scalar firing field via linear head + sigmoid."""
        N = self._num_observations
        slots = outputs.schema_slots[:, :N, :]  # (B, N, D)
        field_logits = self.field_head(slots).squeeze(-1)  # (B, N)
        return GoaltraceTaskOutput(firing_field=torch.sigmoid(field_logits))


# =============================================================================
class RoutebindRoPEEncoder(nn.Module, Generic[TInput]):
    """Encoder for Routebind spatial-grid task inputs using a RoPE-compatible
    front-end.

    For each spatial position slot ``p = (r, c)``, constructs:

        e_p = E_cell(t_p) + E_obs(o_p) + s_p * E_start + g_p * E_goal
              + E_row(r) + E_col(c)

    Sequence position is left to the HRM's internal RoPE mechanism.
    """

    def __init__(
        self,
        seq_length: int,
        num_cell_types: int,
        num_observations: int,
        hidden_size: int,
        grid_height: int,
        grid_width: int,
        *,
        input_factory: Callable[[Tensor, Tensor | None], TInput],
        padding_idx: int | None = None,
        device: Device | None = None,
        dtype: Dtype | None = None,
    ) -> None:
        super().__init__()
        self._seq_length = seq_length
        self._hidden_size = hidden_size
        self._grid_height = grid_height
        self._grid_width = grid_width
        self.E_cell = nn.Embedding(
            num_cell_types, hidden_size, device=device, dtype=dtype
        )
        self.E_obs = nn.Embedding(
            num_observations + 1,
            hidden_size,
            padding_idx=padding_idx,
            device=device,
            dtype=dtype,
        )
        self.E_start = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.E_goal = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.E_row = nn.Embedding(
            grid_height, hidden_size, device=device, dtype=dtype
        )
        self.E_col = nn.Embedding(
            grid_width, hidden_size, device=device, dtype=dtype
        )
        self.embedding_scale = hidden_size**0.5
        self._input_factory = input_factory
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.E_cell.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.E_obs.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.E_start, mean=0.0, std=0.02)
        nn.init.normal_(self.E_goal, mean=0.0, std=0.02)
        nn.init.normal_(self.E_row.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.E_col.weight, mean=0.0, std=0.02)
        # Explicitly zero the padding index so sentinel cells embed to zero
        if self.E_obs.padding_idx is not None:
            with torch.no_grad():
                self.E_obs.weight[self.E_obs.padding_idx] = 0.0

    def forward(
        self,
        cell_type: Tensor,
        observation_id: Tensor,
        start_flag: Tensor,
        goal_flag: Tensor,
    ) -> TInput:
        """Encode routebind grid inputs into schema tokens.

        Args:
            cell_type: ``(B, S)`` int32 — cell type per position.
            observation_id: ``(B, S)`` int32 — observation IDs, with
                sentinel = ``num_observations`` for non-observation cells.
            start_flag: ``(B, S)`` bool.
            goal_flag: ``(B, S)`` bool.

        Returns:
            Model-native input with schema tokens ``(B, S, D)``.
        """
        B, S = cell_type.shape
        D = self._hidden_size
        device = cell_type.device

        # Compute row/col per slot
        rows = (
            torch.arange(self._grid_height, device=device)
            .view(-1, 1)
            .expand(self._grid_height, self._grid_width)
            .reshape(-1)[:S]
            .unsqueeze(0)
            .expand(B, -1)
        )  # (B, S)
        cols = (
            torch.arange(self._grid_width, device=device)
            .view(1, -1)
            .expand(self._grid_height, self._grid_width)
            .reshape(-1)[:S]
            .unsqueeze(0)
            .expand(B, -1)
        )  # (B, S)

        cell_emb = self.E_cell(cell_type.to(dtype=torch.int32))  # (B, S, D)
        obs_emb = self.E_obs(observation_id.to(dtype=torch.int32))  # (B, S, D)
        start_emb = self.E_start * start_flag.unsqueeze(-1).float()  # (B, S, D)
        goal_emb = self.E_goal * goal_flag.unsqueeze(-1).float()  # (B, S, D)
        row_emb = self.E_row(rows.to(dtype=torch.int32))  # (B, S, D)
        col_emb = self.E_col(cols.to(dtype=torch.int32))  # (B, S, D)

        content = cell_emb + obs_emb + start_emb + goal_emb + row_emb + col_emb
        encoded = self.embedding_scale * content  # (B, S, D)

        # Pad to model slot capacity
        S_model = self._seq_length
        if S < S_model:
            pad = torch.zeros(
                B, S_model - S, D, device=device, dtype=encoded.dtype
            )
            encoded = torch.cat([encoded, pad], dim=1)
        elif S > S_model:
            raise ValueError(
                f"Routebind grid ({S} slots) exceeds model capacity "
                f"({S_model} slots)."
            )

        return self._input_factory(encoded, None)


# =============================================================================
class RoutebindDecoder(nn.Module):
    """Multi-head decoder for Routebind field prediction.

    Decodes schema-slot features into trajectory field, waypoint field,
    next-direction logits (from the start slot), and next-observation
    logits (from the start slot).
    """

    def __init__(
        self,
        hidden_size: int,
        num_slots: int,
        num_observations: int,
        *,
        device: Device | None = None,
        dtype: Dtype | None = None,
    ) -> None:
        super().__init__()
        self._num_slots = num_slots
        self._num_observations = num_observations
        self.trajectory_head = nn.Linear(
            hidden_size, 1, device=device, dtype=dtype
        )
        self.waypoint_head = nn.Linear(
            hidden_size, 1, device=device, dtype=dtype
        )
        self.direction_head = nn.Linear(
            hidden_size, 4, device=device, dtype=dtype
        )
        self.observation_head = nn.Linear(
            hidden_size, num_observations, device=device, dtype=dtype
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for head in (
            self.trajectory_head,
            self.waypoint_head,
            self.direction_head,
            self.observation_head,
        ):
            nn.init.normal_(head.weight, mean=0.0, std=0.02)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        outputs: HasSchemaSlots,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Decode schema-slot features into routebind outputs.

        Args:
            outputs: Model output carrying ``schema_slots`` shaped
                ``(B, S, D)``.

        Returns:
            Tuple of (trajectory_field, waypoint_field,
            next_direction_logits, next_observation_logits) where:
            - trajectory_field: ``(B, S)`` float32 via sigmoid
            - waypoint_field: ``(B, S)`` float32 via sigmoid
            - next_direction_logits: ``(B, 4)`` float32
            - next_observation_logits: ``(B, N_obs)`` float32
        """
        slots = outputs.schema_slots
        S = self._num_slots
        B = slots.shape[0]

        # Per-slot field heads (operate on all S slots)
        traj_logits = self.trajectory_head(slots[:, :S, :]).squeeze(-1)
        wp_logits = self.waypoint_head(slots[:, :S, :]).squeeze(-1)
        trajectory_field = torch.sigmoid(traj_logits)
        waypoint_field = torch.sigmoid(wp_logits)

        # Start-slot heads use the full pooled representation across all slots.
        # H_start = slots[:, 0, :] is a heuristic; the actual start position
        # must be located by the caller.  Here we use the *pooled* representation
        # from the full schema-slot bank so that HRM self-attention can integrate
        # the start position information into any slot.
        pooled = slots.mean(dim=1)  # (B, D)
        next_direction_logits = self.direction_head(pooled)
        next_observation_logits = self.observation_head(pooled)

        return (
            trajectory_field,
            waypoint_field,
            next_direction_logits,
            next_observation_logits,
        )


# =============================================================================
def build_token_decoder(
    *,
    hidden_size: int,
    vocab_size: int,
    task_family: Literal["mazehard", "goaltrace"] = "mazehard",
    num_observations: int | None = None,
    device: Device | None = None,
    dtype: Dtype | None = None,
) -> MazeHardMLPDecoder | GoaltraceMLPDecoder:
    """Construct a token decoder head for one HRM bridge.

    Args:
        hidden_size: Embedding dimension (must match PFC hidden size).
        vocab_size: Token vocabulary size (mazehard) or observation
            vocabulary size (goaltrace).  Only used by mazehard.
        task_family: Which task family the decoder belongs to.
        num_observations: Required for ``task_family="goaltrace"``.
            Number of observation nodes (N) to decode into field values.
    """
    if task_family == "mazehard":
        return MazeHardMLPDecoder(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            device=device,
            dtype=dtype,
        )
    if num_observations is None:
        raise ValueError(
            "num_observations is required for goaltrace token decoder"
        )
    return GoaltraceMLPDecoder(
        hidden_size=hidden_size,
        num_observations=num_observations,
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

        # E_candidate(succ[i,k]) — shared identity table with node slots
        index_emb = self.E_candidate(succ_idx_clamped)  # (B, N, K, D)

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
        oracle_decoder_enabled: When True, also decode from graph-region
            slots as a diagnostic.  The oracle decoder is an ablation-only
            shortcut — it must never be active during production training.
    """

    n_max: int = Field(default=45, ge=1)
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
    oracle_decoder_enabled: bool = Field(
        default=False,
        description="Diagnostic ablation: decode path logits from graph-region "
        "slots directly.  Disabled by default — must not be enabled in "
        "production training.",
    )


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
        # Edge-to-node identity grounding: successor references use the same
        # E_candidate table as node slots (shared identity code space).

        # --- Path region: learned query embeddings ---
        self.E_path_query = nn.Embedding(1, D, device=device, dtype=dtype)
        self.E_path_position = nn.Embedding(
            config.t_max, D, device=device, dtype=dtype
        )

        # --- Region embeddings ---
        self.E_region_graph = nn.Embedding(1, D, device=device, dtype=dtype)
        self.E_region_path = nn.Embedding(1, D, device=device, dtype=dtype)

        # Scaling: D**0.5 matches transformer input convention used by PFC
        # and MazeHard encoder.  The PFC was designed for this scale.
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
                successor_indices,
                successor_mask,
                node_obs_id,
                node_candidate_index,
            )

        raise ValueError(f"Unknown edge_encoding: {config.edge_encoding!r}")

    def _encode_successor_index_embedding(
        self,
        successor_indices: Tensor,  # (B, N, K)
        successor_mask: Tensor,  # (B, N, K)
    ) -> Tensor:
        """Encode edges as pooled slot+index embeddings.

        edge_embedding_i =
            Pool_k [ E_successor_slot(k) + E_candidate(succ[i,k]) ]
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

        # E_candidate(succ[i,k]) — shared identity table with node slots
        index_emb = self.E_candidate(succ_idx_clamped)  # (B, N, K, D)

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
        node_candidate_index: Tensor,  # (B, N)
    ) -> Tensor:
        """Encode edges by pooling successor full node content embeddings.

        edge_embedding_i =
            Pool_k [ E_obs(obs_id[succ]) + E_candidate(candidate_index[succ]) ]
            masked by successor_mask[i, k]

        Pools both E_obs and E_candidate of each successor so the edge
        representation shares the same identity code space as graph-region
        node slots.
        """
        config = self.config
        B, N, K = successor_indices.shape
        D = config.hidden_size

        # Clamp indices to valid range
        succ_idx_clamped = successor_indices.clamp(min=0, max=N - 1)
        succ_idx_mask = successor_mask  # (B, N, K)

        # Gather inputs of successors via advanced indexing.
        batch_idx = (
            torch.arange(B, device=node_obs_id.device)
            .unsqueeze(1)
            .unsqueeze(2)
            .expand(-1, N, K)
        )
        node_idx = succ_idx_clamped  # (B, N, K)

        # E_obs of each successor
        succ_obs_id = node_obs_id[batch_idx, node_idx]  # (B, N, K)
        obs_emb = self.E_obs(succ_obs_id)  # (B, N, K, D)

        # E_candidate of each successor (shared identity code)
        succ_cand_idx = node_candidate_index[batch_idx, node_idx]  # (B, N, K)
        cand_emb = self.E_candidate(succ_cand_idx)  # (B, N, K, D)

        # Mean-pool over successors
        content_emb = obs_emb + cand_emb  # (B, N, K, D)
        content_emb = content_emb * succ_idx_mask.unsqueeze(-1)
        denom = succ_idx_mask.sum(dim=-1, keepdim=True).clamp(min=1)
        edge_emb = content_emb.sum(dim=-2) / denom  # (B, N, D)

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
class SeqMazeOracleDecoder(nn.Module):
    """Diagnostic decoder that reads graph-region slots to produce path logits.

    This is an ablation-only shortcut decoder.  It proves whether the full
    data pipeline (encoder, targets, loss, output vocabulary) is correct when
    graph information is explicitly exposed to the decoder.

    It is NOT the production architecture — the production :class:`SeqMazeDecoder`
    reads only path-region slots.
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
        V = config.n_max + 2  # path vocabulary size

        # Position embedding for the T output positions
        self.E_decode_position = nn.Embedding(
            config.t_max, D, device=device, dtype=dtype
        )
        self.ln = nn.LayerNorm(D, device=device, dtype=dtype)
        self.lm_head = nn.Linear(D, V, bias=False, device=device, dtype=dtype)

    def forward(
        self,
        schema_slots: Tensor,  # (B, S, D) — full schema output from HRM
        n_max: int,
        t_max: int,
    ) -> Tensor:
        """Decode graph-region schema slots into path logits.

        Each graph node state is decoded independently into one path position,
        so position ``t`` reads from graph slot ``t`` (clamped to N-1 when
        ``t >= N``).  This exposes graph information to the decoder while
        keeping the output shape identical to :class:`SeqMazeDecoder`.

        Args:
            schema_slots: (B, S, D) — all schema slots from HRM output.
            n_max: Number of graph region slots (N).
            t_max: Number of path region slots (T).

        Returns:
            path_logits: (B, T, N+2) float32.
        """
        # Extract graph region
        graph_states = schema_slots[:, :n_max, :]  # (B, N, D)
        B, N, D = graph_states.shape
        device = schema_slots.device

        # For each output position t, read from graph slot min(t, N-1).
        # This keeps all logits inside the graph region, avoiding path-region
        # slot access.
        src_idx = torch.arange(t_max, device=device).clamp(max=N - 1)  # (T,)
        decoder_input = graph_states[:, src_idx, :]  # (B, T, D)

        # Add position embedding
        pos_ids = torch.arange(t_max, device=device).unsqueeze(0).expand(B, -1)
        pos_emb = self.E_decode_position(pos_ids)  # (B, T, D)
        decoder_input = decoder_input + pos_emb
        decoder_input = self.ln(decoder_input)

        path_logits = self.lm_head(decoder_input)  # (B, T, N+2)
        return path_logits


# =============================================================================
__all__ = [
    "O_ID",
    "DEFAULT_MAZE_HARD_HRM_VOCAB_SIZE",
    "GoaltraceHRMAdapterSettings",
    "GoaltraceLearnedEncoder",
    "GoaltraceMLPDecoder",
    "GoaltraceRoPEEncoder",
    "MazeHardHRMAdapterSettings",
    "MazeHardLearnedEncoder",
    "MazeHardMLPDecoder",
    "MazeHardRoPEEncoder",
    "HasSchemaSlots",
    "RoutebindHRMAdapterSettings",
    "RoutebindRoPEEncoder",
    "RoutebindDecoder",
    "build_token_encoder",
    "build_token_decoder",
    "SeqMazeProbeAdapterSettings",
    "SeqMazeProbeEncoder",
    "SeqMazeProbeDecoder",
    "SeqMazeAdapterSettings",
    "SeqMazeEncoder",
    "SeqMazeDecoder",
    "SeqMazeOracleDecoder",
]
