"""MazeHard adapter pieces for task-side query construction and projection."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ehc_sn.models.hrm.hrm_v1 import HRMBatchV1, HRModelV1, HRMStateV1
from ehc_sn.models.hrm.hrm_v2 import HRMBatchV2, HRModelV2, HRMStateV2
from ehc_sn.types import Batch


# =============================================================================
class MazeHardTokenAdapter(nn.Module):
    """Build MazeHard token-position queries and project decoded states."""

    def __init__(  # ----------------------------------------------------------
        self,
        seq_length: int,
        vocab_size: int,
        hidden_size: int,
        *,
        num_heads: int = 1,
    ):
        """Initialize the MazeHard token adapter with query embeddings and projection head."""
        super().__init__()
        self.query_embed = nn.Embedding(seq_length, hidden_size)
        self.decoder = nn.MultiheadAttention(hidden_size, num_heads=num_heads, batch_first=True)
        self.out_proj = nn.Linear(hidden_size, vocab_size)

    def build_queries(  # -----------------------------------------------------
        self,
        batch: Batch,
        *,
        device: torch.device,
    ) -> Tensor:
        """Return task-owned position queries for MazeHard token decoding."""
        input_ids = batch["input_ids"]
        if int(input_ids.shape[1]) != self.query_embed.num_embeddings:
            raise ValueError(
                "MazeHard input_ids length must match adapter seq_length "
                f"{self.query_embed.num_embeddings}, got {int(input_ids.shape[1])}."
            )
        batch_size = input_ids.shape[0]
        positions = torch.arange(self.query_embed.num_embeddings, device=device)
        return self.query_embed(positions).unsqueeze(0).expand(batch_size, -1, -1)

    def decode_content_bank(  # -----------------------------------------------
        self,
        queries: Tensor,
        content_bank: Workspace,
    ) -> Tensor:
        """Decode token-position queries against the decoder-ready content bank."""
        bank_tokens = content_bank.tokens
        decoded_queries, _ = self.decoder(queries, bank_tokens, bank_tokens, need_weights=False)
        return decoded_queries

    def project_task_content_logits(  # ---------------------------------------
        self,
        decoded_queries: Tensor,
    ) -> Tensor:
        """Project decoded task states into MazeHard token logits."""
        return self.out_proj(decoded_queries)  # (B, S, V)

    def forward(  # -----------------------------------------------------------
        self,
        batch: Batch,
        content_bank: Workspace,
        *,
        device: torch.device | None = None,
    ) -> Tensor:
        """Build token-position queries over the content bank and return MazeHard token logits."""
        if device is None:
            device = content_bank.tokens.device
        queries = self.build_queries(batch, device=device)
        decoded_queries = self.decode_content_bank(queries, content_bank)
        return self.project_task_content_logits(decoded_queries)


# =============================================================================
class MazeHardHRMV1BridgeAdapter(nn.Module):
    """MazeHard bridge that preserves the ACT controller contract over canonical HRM v1."""

    def __init__(  # ----------------------------------------------------------
        self,
        model: HRModelV1,
    ) -> None:
        super().__init__()
        self._model = model

    @property
    def model(self) -> HRModelV1:
        """Return the wrapped canonical HRM v1 model."""
        return self._model

    def init_state(  # --------------------------------------------------------
        self,
        batch_size: int,
    ) -> HRMStateV1:
        """Delegate recurrent-state initialization to the wrapped HRM model."""
        return self.model.init_state(batch_size)

    def reset_state(  # -------------------------------------------------------
        self,
        reset_flag: Tensor,
        state: HRMStateV1,
    ) -> HRMStateV1:
        """Delegate selective recurrent resets to the wrapped HRM model."""
        return self.model.reset_state(reset_flag, state)

    def forward(  # -------------------------------------------------------------------------------
        self,
        batch: HRMBatchV1,
        state: HRMStateV1 | None = None,
    ):  # TODO: the output is defined by the
        """Encode MazeHard tokens, run canonical HRM v1, and decode task logits."""
        next_state, output = self.model.step(
            self.model.build_input_workspace(batch["input_ids"]),
            state=state,
        )
        logits = self.model.decode_content_bank(
            output.content.content_bank.tokens,
        )
        return


# =============================================================================
class MazeHardHRMV2BridgeAdapter(nn.Module):
    """MazeHard bridge that preserves the RL controller contract over canonical HRM v2."""

    def __init__(  # ----------------------------------------------------------
        self,
        model: HRModelV2,
    ) -> None:
        """Initialize the MazeHard HRM v2 bridge adapter with the wrapped canonical model."""
        super().__init__()
        self._model = model

    @property
    def model(self) -> HRModelV2:
        """Return the wrapped canonical HRM v2 model."""
        return self._model

    def init_state(  # --------------------------------------------------------
        self,
        batch_size: int,
    ) -> HRMStateV2:
        """Delegate recurrent-state initialization to the wrapped HRM model."""
        return self.model.init_state(batch_size)

    def reset_state(  # -------------------------------------------------------
        self,
        reset_flag: Tensor,
        state: HRMStateV2,
    ) -> HRMStateV2:
        """Delegate selective recurrent resets to the wrapped HRM model."""
        return self.model.reset_state(reset_flag, state)

    def forward(  # -----------------------------------------------------------
        self,
        batch: HRMBatchV2,
        state: HRMStateV2 | None = None,
    ):  # TODO: the output is defined by the
        """Encode MazeHard tokens, run canonical HRM v2, and decode task logits."""
        next_state, output = self.model.step(
            self.model.build_input_workspace(batch["input_ids"]),
            state=state,
        )
        logits = self.model.decode_content_bank(output.content.content_bank.tokens)
        reward_logits = output.control.reward_logits
        if reward_logits is None:
            raise RuntimeError("HRM v2 bridge expected reward logits from the canonical HRM core.")

        return


# =============================================================================
__all__ = [
    "MazeHardHRMV1BridgeAdapter",
    "MazeHardHRMV2BridgeAdapter",
    "MazeHardTokenAdapter",
]
