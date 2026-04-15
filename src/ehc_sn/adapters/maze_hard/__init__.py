"""MazeHard adapter pieces for task-side query construction and projection."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ehc_sn.models.hrm.core import HRMOutput
from ehc_sn.models.hrm.hrm_v1 import Batch as HRMBatchV1
from ehc_sn.models.hrm.hrm_v1 import HRModelV1
from ehc_sn.models.hrm.hrm_v1 import HRMState as HRMStateV1
from ehc_sn.models.hrm.hrm_v2 import Batch as HRMBatchV2
from ehc_sn.models.hrm.hrm_v2 import HRModelV2
from ehc_sn.models.hrm.hrm_v2 import HRMState as HRMStateV2
from ehc_sn.modules.pfc import NamedWorkspace
from ehc_sn.tasks.maze_hard import MazeHardTaskOutput
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
    ) -> None:
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
        content_bank: NamedWorkspace,
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
        content_bank: NamedWorkspace,
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
@dataclass(frozen=True)
class MazeHardHRMV1ControlOutput:
    """Control payload emitted by the MazeHard+HRM v1 bridge."""

    theta_summary: Tensor
    q_logits: Tensor


# =============================================================================
@dataclass(frozen=True)
class MazeHardHRMV1BridgeOutput:
    """Concrete task+model output emitted by the MazeHard+HRM v1 bridge."""

    task: MazeHardTaskOutput
    control: MazeHardHRMV1ControlOutput
    raw_model_output: HRMOutput


# =============================================================================
@dataclass(frozen=True)
class MazeHardHRMV2ControlOutput:
    """Control payload emitted by the MazeHard+HRM v2 bridge."""

    theta_summary: Tensor
    q_logits: Tensor
    value_logits: Tensor


# =============================================================================
@dataclass(frozen=True)
class MazeHardHRMV2BridgeOutput:
    """Concrete task+model output emitted by the MazeHard+HRM v2 bridge."""

    task: MazeHardTaskOutput
    control: MazeHardHRMV2ControlOutput
    raw_model_output: HRMOutput


# =============================================================================
class MazeHardHRMV1BridgeAdapter(nn.Module):
    """MazeHard+HRM v1 model-task binding over the canonical HRM core."""

    def __init__(  # ----------------------------------------------------------
        self,
        model: HRModelV1,
        task_adapter: MazeHardTokenAdapter | None = None,
    ) -> None:
        super().__init__()
        self._model = model
        self._task_adapter = task_adapter

    @property
    def model(self) -> HRModelV1:
        """Return the wrapped canonical HRM v1 model."""
        return self._model

    @property
    def task_adapter(self) -> MazeHardTokenAdapter | None:
        """Return the optional task-side decoder owned by this bridge."""
        return self._task_adapter

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
    ) -> tuple[HRMStateV1, MazeHardHRMV1BridgeOutput]:
        """Encode MazeHard tokens, run canonical HRM v1, and return task+control outputs."""
        next_state, output = self.model.step(
            self.model.build_input_workspace(batch["input_ids"]),
            state=state,
        )
        if self.task_adapter is None:
            logits = self.model.decode_content_bank(output.content.content_bank.tokens)
        else:
            logits = self.task_adapter(batch, output.content.content_bank)
        return next_state.detach(), MazeHardHRMV1BridgeOutput(
            task=MazeHardTaskOutput(task_logits=logits),
            control=MazeHardHRMV1ControlOutput(
                theta_summary=output.control.theta_summary,
                q_logits=output.control.q_logits,
            ),
            raw_model_output=output,
        )


# =============================================================================
class MazeHardHRMV2BridgeAdapter(nn.Module):
    """MazeHard+HRM v2 model-task binding over the canonical HRM core."""

    def __init__(  # ----------------------------------------------------------
        self,
        model: HRModelV2,
        task_adapter: MazeHardTokenAdapter | None = None,
    ) -> None:
        """Initialize the MazeHard HRM v2 bridge adapter with the wrapped canonical model."""
        super().__init__()
        self._model = model
        self._task_adapter = task_adapter

    @property
    def model(self) -> HRModelV2:
        """Return the wrapped canonical HRM v2 model."""
        return self._model

    @property
    def task_adapter(self) -> MazeHardTokenAdapter | None:
        """Return the optional task-side decoder owned by this bridge."""
        return self._task_adapter

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
    ) -> tuple[HRMStateV2, MazeHardHRMV2BridgeOutput]:
        """Encode MazeHard tokens, run canonical HRM v2, and return task+control outputs."""
        next_state, output = self.model.step(
            self.model.build_input_workspace(batch["input_ids"]),
            state=state,
        )
        if self.task_adapter is None:
            logits = self.model.decode_content_bank(output.content.content_bank.tokens)
        else:
            logits = self.task_adapter(batch, output.content.content_bank)
        reward_logits = output.control.reward_logits
        if reward_logits is None:
            raise RuntimeError("HRM v2 bridge expected reward logits from the canonical HRM core.")

        return next_state.detach(), MazeHardHRMV2BridgeOutput(
            task=MazeHardTaskOutput(task_logits=logits),
            control=MazeHardHRMV2ControlOutput(
                theta_summary=output.control.theta_summary,
                q_logits=output.control.q_logits,
                value_logits=reward_logits,
            ),
            raw_model_output=output,
        )


# =============================================================================
__all__ = [
    "MazeHardHRMV1BridgeAdapter",
    "MazeHardHRMV1BridgeOutput",
    "MazeHardHRMV1ControlOutput",
    "MazeHardHRMV2BridgeAdapter",
    "MazeHardHRMV2BridgeOutput",
    "MazeHardHRMV2ControlOutput",
    "MazeHardTaskOutput",
    "MazeHardTokenAdapter",
]
