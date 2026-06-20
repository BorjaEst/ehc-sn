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
from typing import TypeAlias

import torch
from torch import Tensor, nn

from ehc_sn.adapters.hrm._base import (
    GoaltraceHRMAdapterSettings,
    GoaltraceLearnedEncoder,
    GoaltraceMLPDecoder,
    GoaltraceRoPEEncoder,
    build_token_decoder,
    build_token_encoder,
)
from ehc_sn.models.hrm.hrm_v1 import (
    HRMInputV1,
    HRModelV1,
    HRMOutputV1,
    HRMStateV1,
)
from ehc_sn.models.hrm.hrm_v2 import (
    HRMInputV2,
    HRModelV2,
    HRMOutputV2,
    HRMStateV2,
)
from ehc_sn.tasks.goaltrace.contracts import GoaltraceTaskOutput
from ehc_sn.tasks.goaltrace.runtime import extract_goaltrace_task_input
from ehc_sn.types import Batch


# =============================================================================
@dataclass(frozen=True)
class GoaltraceHRMV1ControlOutput:
    """ACT control readouts emitted by the Goaltrace HRM v1 bridge."""

    action_logits: Tensor


# =============================================================================
@dataclass(frozen=True)
class GoaltraceHRMV1BridgeOutput:
    """Controller-consumable Goaltrace HRM v1 bridge output bundle."""

    task: GoaltraceTaskOutput
    control: GoaltraceHRMV1ControlOutput


GoaltraceTokenEncoder: TypeAlias = (
    GoaltraceLearnedEncoder[HRMInputV1] | GoaltraceRoPEEncoder[HRMInputV1]
)


def _make_input_v1(
    schema_tokens: Tensor, prefix_bias: Tensor | None
) -> HRMInputV1:
    """Construct the model-native HRM v1 input payload."""
    return HRMInputV1(schema_tokens=schema_tokens, prefix_bias=prefix_bias)


def _build_encoder(  # ----------------------------------------------------
    model: HRModelV1,
    config: GoaltraceHRMAdapterSettings,
) -> GoaltraceTokenEncoder:
    """Construct the token encoder front-end for the bridge adapter based on config."""
    params = next(model.parameters())
    # vocab_size: when a padding sentinel is configured use that + 1,
    # otherwise fall back to num_observations (backward compat).
    vocab_size = (
        config.padding_obs_id + 1
        if config.padding_obs_id is not None
        else config.num_observations
    )
    return build_token_encoder(
        seq_length=model.config.num_schema_slots,
        vocab_size=vocab_size,
        hidden_size=model.config.pfc.hidden_size,
        encoder_kind=config.encoder_kind,
        task_family="goaltrace",
        input_factory=_make_input_v1,
        device=params.device,
        dtype=params.dtype,
    )


GoaltraceTokenDecoder: TypeAlias = GoaltraceMLPDecoder


def _build_decoder(  # --------------------------------------------------------
    model: HRModelV1,
    config: GoaltraceHRMAdapterSettings,
) -> GoaltraceTokenDecoder:
    """Construct the token decoder head for the bridge adapter."""
    params = next(model.parameters())
    return build_token_decoder(
        hidden_size=model.config.pfc.hidden_size,
        # V = N_max per corpus contract (see _build_encoder).
        vocab_size=config.num_observations,
        task_family="goaltrace",
        num_observations=config.num_observations,
        device=params.device,
        dtype=params.dtype,
    )


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

    def __init__(  # ----------------------------------------------------------
        self,
        model: HRModelV1,
        config: GoaltraceHRMAdapterSettings,
    ) -> None:
        super().__init__()
        self._config = config
        self.model = model
        self._encoder = _build_encoder(model, config)
        self._decoder = _build_decoder(model, config)
        self.reset_parameters()

    def reset_parameters(  # --------------------------------------------------
        self,
    ) -> None:
        """Initialize encoder and decoder with small normal weights."""
        self._encoder.reset_parameters()
        self._decoder.reset_parameters()

    @property
    def config(self) -> GoaltraceHRMAdapterSettings:
        """Return the adapter settings used to build this module."""
        return self._config

    def init_state(  # --------------------------------------------------------
        self,
        batch_size: int,
    ) -> HRMStateV1:
        """Delegate to the wrapped HRM model."""
        return self.model.init_state(batch_size)

    def reset_state(  # -------------------------------------------------------
        self,
        reset_flag: Tensor,
        state: HRMStateV1,
    ) -> HRMStateV1:
        """Delegate to the wrapped HRM model."""
        return self.model.reset_state(reset_flag, state)

    def prepare_inputs(  # ----------------------------------------------------
        self,
        batch: Batch,
    ) -> HRMInputV1:
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
        task_input = extract_goaltrace_task_input(batch)
        return self._encoder(task_input)

    def postprocess(  # -------------------------------------------------------
        self,
        outputs: HRMOutputV1,
    ) -> GoaltraceHRMV1BridgeOutput:
        """Decode PFC schema slots into a firing field via linear head + sigmoid.

        Args:
            outputs: Architecture-native HRM v1 output with ``schema_slots``
                shaped ``(B, S, D)``.

        Returns:
            Bridge output bundle with:
                - ``task.firing_field``: ``(B, N)`` float32 in ``[0, 1]``
                - ``control.action_logits``: ``(B, 2)`` from the HRM core
        """
        return GoaltraceHRMV1BridgeOutput(
            task=self._decoder(outputs),  # Identity decoder in this case
            control=GoaltraceHRMV1ControlOutput(
                action_logits=outputs.action_logits
            ),
        )

    def forward(  # -----------------------------------------------------------
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
__all__ = [
    "GoaltraceHRMV1BridgeAdapter",
    "GoaltraceHRMV1BridgeOutput",
    "GoaltraceHRMV1ControlOutput",
]
