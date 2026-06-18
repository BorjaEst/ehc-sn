"""Goaltrace HRM v1 bridge adapter.

Encodes goaltrace task inputs (observation_id, weight, current/goal flags)
into schema tokens via element-wise embedding composition, runs HRM v1
deliberation, and decodes schema slots into a goal-conditioned prospective
firing field.

The adapter follows the SeqMazeHRMV1BridgeAdapter public surface
(init_state, reset_state, prepare_inputs, postprocess, forward) but is
simpler — no path region, no edge head, no oracle decoder, no output
vocabulary masking.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from pydantic import BaseModel, Field
from torch import Tensor, nn

from ehc_sn.models.hrm.hrm_v1 import (
    HRMInputV1,
    HRModelV1,
    HRMOutputV1,
    HRMStateV1,
)
from ehc_sn.tasks.goaltrace.contracts import (
    GoaltraceTaskInput,
    GoaltraceTaskOutput,
)
from ehc_sn.tasks.goaltrace.runtime import extract_goaltrace_task_input
from ehc_sn.types import Batch


# =============================================================================
class GoaltraceAdapterSettings(BaseModel, extra="forbid"):
    """Adapter configuration for goaltrace field prediction.

    Attributes:
        num_observations: Padded node count N.  Must match the goaltrace
            corpus ``n_observations``.
        hidden_size: Embedding dimension — must match PFC hidden size.
        vocab_size_obs: Vocabulary size for observation-id embeddings.
    """

    num_observations: int = Field(default=45, ge=1)
    hidden_size: int = Field(default=128, ge=1)
    vocab_size_obs: int = Field(default=64, ge=1)


# =============================================================================
@dataclass(frozen=True)
class GoaltraceHRMV1ControlOutput:
    """ACT control readouts emitted by the Goaltrace HRM v1 bridge."""

    q_logits: Tensor


# =============================================================================
@dataclass(frozen=True)
class GoaltraceHRMV1BridgeOutput:
    """Controller-consumable Goaltrace HRM v1 bridge output bundle."""

    task: GoaltraceTaskOutput
    control: GoaltraceHRMV1ControlOutput


# =============================================================================
class GoaltraceHRMV1BridgeAdapter(nn.Module):
    """Goaltrace field-prediction bridge over HRM v1 (ACT-supervised).

    Wraps HRModelV1, encodes goaltrace task inputs (observation IDs,
    relational weights, current/goal flags) into schema tokens via
    element-wise embedding composition, runs HRM deliberation, and decodes
    schema slots into a continuous firing field via a learned linear head
    with sigmoid activation.

    The adapter has no path region, no edge head, and no output vocabulary
    masking — every node position is a direct field prediction.
    """

    def __init__(
        self,
        model: HRModelV1,
        config: GoaltraceAdapterSettings | None = None,
    ) -> None:
        super().__init__()
        self._config = config or GoaltraceAdapterSettings()
        self.model = model

        N = self._config.num_observations
        D = self._config.hidden_size

        # Validate profile coupling
        if N > model.config.num_schema_slots:
            raise ValueError(
                f"Task requires {N} schema slots (num_observations={N}), "
                f"but PFC capacity is only {model.config.num_schema_slots}."
            )
        if model.config.pfc.hidden_size != D:
            raise ValueError(
                f"Model hidden size ({model.config.pfc.hidden_size}) must match "
                f"adapter hidden_size ({D})."
            )

        # Embedding tables
        self.E_obs = nn.Embedding(self._config.vocab_size_obs, D)
        self.f_weight = nn.Linear(1, D)
        self.E_current = nn.Parameter(torch.zeros(1, 1, D))
        self.E_goal = nn.Parameter(torch.zeros(1, 1, D))

        self.embedding_scale = D**0.5

        # Field decoder: project each schema slot to a scalar field value
        self.field_head = nn.Linear(D, 1)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights for all learnable parameters."""
        nn.init.normal_(self.E_obs.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.f_weight.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.f_weight.bias)
        nn.init.normal_(self.E_current, mean=0.0, std=0.02)
        nn.init.normal_(self.E_goal, mean=0.0, std=0.02)
        nn.init.normal_(self.field_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.field_head.bias)

    @property
    def config(self) -> GoaltraceAdapterSettings:
        """Return the adapter settings used to build this module."""
        return self._config

    def init_state(self, batch_size: int) -> HRMStateV1:
        """Delegate to the wrapped HRM model."""
        return self.model.init_state(batch_size)

    def reset_state(
        self,
        reset_flag: Tensor,
        state: HRMStateV1,
    ) -> HRMStateV1:
        """Delegate to the wrapped HRM model."""
        return self.model.reset_state(reset_flag, state)

    def prepare_inputs(self, batch: Batch) -> HRMInputV1:
        """Extract goaltrace task input from a raw batch and encode into
        schema tokens.

        Constructs the per-node embedding:
            e_j = E_obs(obs_id_j) + f_weight(w_j) + E_current * [j = g_t]
                  + E_goal * [j = x_goal]

        Pads the token sequence from ``N`` to the model's PFC slot capacity
        (``S``) so the workspace layout validation passes.

        Returns:
            HRMInputV1 with schema tokens shaped ``(B, S, D)``.
        """
        N = self._config.num_observations
        D = self._config.hidden_size
        S = self.model.config.num_schema_slots
        device = next(self.parameters()).device

        # Extract typed task input from batch (validates keys at boundary)
        task_input = extract_goaltrace_task_input(batch)
        observation_id = task_input.observation_id.to(device=device)
        weight = task_input.weight.to(device=device)
        current_flag = task_input.current_flag.to(device=device)
        goal_flag = task_input.goal_flag.to(device=device)

        B = observation_id.shape[0]

        # Content embedding
        obs_emb = self.E_obs(observation_id.to(torch.int32))  # (B, N, D)

        # Weight projection
        w_emb = self.f_weight(weight.unsqueeze(-1))  # (B, N, D)

        # Flag embeddings: broadcast learned parameter over flagged positions
        current_emb = self.E_current.expand(B, N, -1) * (
            current_flag.unsqueeze(-1).float()
        )
        goal_emb = self.E_goal.expand(B, N, -1) * (
            goal_flag.unsqueeze(-1).float()
        )

        schema_tokens = self.embedding_scale * (
            obs_emb + w_emb + current_emb + goal_emb
        )  # (B, N, D)

        # Pad to model slot capacity
        if N < S:
            pad = torch.zeros(
                B, S - N, D, device=device, dtype=schema_tokens.dtype
            )
            schema_tokens = torch.cat([schema_tokens, pad], dim=1)  # (B, S, D)

        return HRMInputV1(schema_tokens=schema_tokens, prefix_bias=None)

    def postprocess(self, outputs: HRMOutputV1) -> GoaltraceHRMV1BridgeOutput:
        """Decode PFC schema slots into a firing field via linear head + sigmoid.

        Args:
            outputs: Architecture-native HRM v1 output with ``schema_slots``
                shaped ``(B, S, D)``.

        Returns:
            Bridge output bundle with:
                - ``task.firing_field``: ``(B, N)`` float32 in ``[0, 1]``
                - ``control.q_logits``: ``(B, 2)`` from the HRM core
        """
        # Slice only the first N slots (observation positions)
        N = self._config.num_observations
        schema_slots = outputs.schema_slots[:, :N, :]  # (B, N, D)

        # Decode each slot to a scalar field value
        field_logits = self.field_head(schema_slots).squeeze(-1)  # (B, N)
        firing_field = torch.sigmoid(field_logits)

        return GoaltraceHRMV1BridgeOutput(
            task=GoaltraceTaskOutput(firing_field=firing_field),
            control=GoaltraceHRMV1ControlOutput(q_logits=outputs.q_logits),
        )

    def forward(
        self,
        batch: Batch,
        state: HRMStateV1 | None = None,
    ) -> tuple[GoaltraceHRMV1BridgeOutput, HRMStateV1]:
        """Run a forward pass on one batch of goaltrace samples."""
        inputs = self.prepare_inputs(batch)
        outputs, next_state = self.model.step(inputs, state=state)
        bridge_out = self.postprocess(outputs)
        return bridge_out, next_state


# =============================================================================
def build_goaltrace_hrm_trace_meta(
    config: GoaltraceAdapterSettings,
) -> dict:
    """Build trace metadata for goaltrace evaluation traces.

    Returns an empty dict initially — trace keys will be added when
    evaluation regimes are implemented.
    """
    _ = config
    return {}


# =============================================================================
__all__ = [
    "GoaltraceAdapterSettings",
    "GoaltraceHRMV1BridgeAdapter",
    "GoaltraceHRMV1BridgeOutput",
    "GoaltraceHRMV1ControlOutput",
    "build_goaltrace_hrm_trace_meta",
]
