"""SeqMaze probe and v1 bridge adapters for HRM v2.

Phase 0: edge-lookup probe (SeqMazeProbeHRMV2BridgeAdapter).
Phase 1: path-prediction adapter (SeqMazeHRMV2BridgeAdapter).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ehc_sn.adapters.hrm._base import (
    SeqMazeAdapterSettings,
    SeqMazeDecoder,
    SeqMazeEncoder,
    SeqMazeProbeAdapterSettings,
    SeqMazeProbeDecoder,
    SeqMazeProbeEncoder,
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
from ehc_sn.tasks.seqmaze.contracts import (
    SeqMazeProbeInput,
    SeqMazeProbeOutput,
    SeqMazeTaskOutput,
)
from ehc_sn.tasks.seqmaze.runtime import (
    extract_seqmaze_probe_input,
    extract_seqmaze_task_input,
)
from ehc_sn.types import Batch


# =============================================================================
@dataclass(frozen=True)
class SeqMazeProbeBridgeOutput:
    """Bundle of probe output and unused model readouts."""

    probe: SeqMazeProbeOutput


# =============================================================================
# HRM v1 bridge adapter
# =============================================================================


@dataclass(frozen=True)
class SeqMazeHRMV1ControlOutput:
    """ACT control readouts emitted by the SeqMaze HRM v1 bridge."""

    q_logits: Tensor


@dataclass(frozen=True)
class SeqMazeHRMV1BridgeOutput:
    """Controller-consumable SeqMaze HRM v1 bridge output bundle."""

    task: SeqMazeTaskOutput
    control: SeqMazeHRMV1ControlOutput


# =============================================================================
class SeqMazeHRMV1BridgeAdapter(nn.Module):
    """SeqMaze v1 path-prediction bridge over HRM v1 (ACT-supervised).

    Wraps HRModelV1, encodes graph nodes + path queries into schema tokens,
    runs HRM deliberation, decodes path-region slots into path logits, and
    applies sample-specific output vocabulary masking.

    The adapter caches ``node_valid_mask`` from ``prepare_inputs`` for use
    in ``postprocess`` within the same forward pass.
    """

    def __init__(
        self,
        model: HRModelV1,
        config: SeqMazeAdapterSettings | None = None,
    ) -> None:
        super().__init__()
        self._config = config or SeqMazeAdapterSettings()
        self.model = model

        # Validate profile coupling: N + T <= S
        expected_slots = self._config.n_max + self._config.t_max
        if expected_slots > model.config.num_schema_slots:
            raise ValueError(
                f"Task requires {expected_slots} schema slots "
                f"(n_max={self._config.n_max} + t_max={self._config.t_max}), "
                f"but PFC capacity is only {model.config.num_schema_slots}."
            )
        if model.config.pfc.hidden_size != self._config.hidden_size:
            raise ValueError(
                f"Model hidden size ({model.config.pfc.hidden_size}) must match "
                f"adapter hidden_size ({self._config.hidden_size})."
            )

        self._encoder = SeqMazeEncoder(self._config)

        # Build decoder with optional weight-sharing
        if self._config.share_path_position_embeddings:
            path_position_emb = self._encoder.E_path_position
        else:
            path_position_emb = None
        self._decoder = SeqMazeDecoder(
            self._config,
            path_position_emb=path_position_emb,
        )

        # Cached per-forward node_valid_mask for output masking
        self._node_valid_mask: Tensor | None = None

    @property
    def config(self) -> SeqMazeAdapterSettings:
        return self._config

    def init_state(self, batch_size: int) -> HRMStateV1:
        return self.model.init_state(batch_size)

    def reset_state(
        self,
        reset_flag: Tensor,
        state: HRMStateV1,
    ) -> HRMStateV1:
        return self.model.reset_state(reset_flag, state)

    def prepare_inputs(self, batch: Batch) -> HRMInputV1:
        """Extract task input from batch and encode into schema tokens."""
        task_input = extract_seqmaze_task_input(batch)
        schema_tokens, _ = self._encoder.forward_padded(
            node_obs_id=task_input.node_obs_id,
            node_candidate_index=task_input.node_candidate_index,
            node_start_flag=task_input.node_start_flag,
            node_goal_flag=task_input.node_goal_flag,
            successor_indices=task_input.successor_indices,
            successor_mask=task_input.successor_mask,
            node_mask=task_input.node_mask,
            model_seq_length=self.model.config.num_schema_slots,
        )
        # Cache node_valid_mask for output masking in postprocess
        self._node_valid_mask = task_input.node_mask
        return HRMInputV1(schema_tokens=schema_tokens, prefix_bias=None)

    def _apply_output_mask(self, logits: Tensor) -> Tensor:
        """Apply sample-specific output vocabulary masking.

        Delegates to the module-level :func:`_apply_seqmaze_output_mask`
        using the cached ``node_valid_mask``.
        """
        node_valid = self._node_valid_mask
        if node_valid is None:
            raise RuntimeError(
                "SeqMazeHRMV1BridgeAdapter: node_valid_mask not cached. "
                "prepare_inputs must be called before postprocess."
            )
        return _apply_seqmaze_output_mask(
            logits, node_valid, n_max=self._config.n_max
        )

    def postprocess(self, outputs: HRMOutputV1) -> SeqMazeHRMV1BridgeOutput:
        """Decode path-region schema slots into constrained path logits."""
        path_logits = self._decoder(
            outputs.schema_slots,
            n_max=self._config.n_max,
            t_max=self._config.t_max,
        )
        constrained_logits = self._apply_output_mask(path_logits)
        return SeqMazeHRMV1BridgeOutput(
            task=SeqMazeTaskOutput(path_logits=constrained_logits),
            control=SeqMazeHRMV1ControlOutput(q_logits=outputs.q_logits),
        )

    def forward(
        self,
        batch: Batch,
        state: HRMStateV1 | None = None,
    ) -> tuple[SeqMazeHRMV1BridgeOutput, HRMStateV1]:
        """Run a forward pass on one batch of seqmaze v1 samples."""
        inputs = self.prepare_inputs(batch)
        outputs, next_state = self.model(inputs, state=state)
        bridge_out = self.postprocess(outputs)
        return bridge_out, next_state


# =============================================================================
class SeqMazeProbeHRMV2BridgeAdapter(nn.Module):
    """SeqMaze edge-lookup probe bridge over HRM v2.

    Wraps HRModelV2, encodes graph nodes via successor-index embedding into
    schema tokens, runs HRM deliberation, and decodes schema slots into
    pairwise edge logits.
    """

    def __init__(
        self,
        model: HRModelV2,
        config: SeqMazeProbeAdapterSettings | None = None,
    ) -> None:
        super().__init__()
        self._config = config or SeqMazeProbeAdapterSettings()
        self.model = model
        self._encoder = SeqMazeProbeEncoder(self._config)
        self._decoder = SeqMazeProbeDecoder(self._config)

        # Validate seq_length matches n_max (no path region in probe mode)
        if model.config.num_schema_slots != self._config.n_max:
            raise ValueError(
                f"Model seq_length ({model.config.num_schema_slots}) must equal "
                f"n_max ({self._config.n_max}) for probe mode (no path region)."
            )

    @property
    def config(self) -> SeqMazeProbeAdapterSettings:
        return self._config

    def init_state(self, batch_size: int) -> HRMStateV2:
        return self.model.init_state(batch_size)

    def prepare_inputs(self, batch: Batch) -> HRMInputV2:
        """Extract probe input from batch and encode into schema tokens."""
        task_input = extract_seqmaze_probe_input(batch)
        schema_tokens, schema_mask = self._encoder(
            node_obs_id=task_input.node_obs_id,
            node_candidate_index=task_input.node_candidate_index,
            node_start_flag=task_input.node_start_flag,
            node_goal_flag=task_input.node_goal_flag,
            successor_indices=task_input.successor_indices,
            successor_mask=task_input.successor_mask,
            node_mask=task_input.node_mask,
        )
        return HRMInputV2(schema_tokens=schema_tokens)

    def postprocess(self, outputs: HRMOutputV2) -> SeqMazeProbeBridgeOutput:
        """Decode schema slots into edge logits."""
        edge_logits = self._decoder(outputs.schema_slots)
        return SeqMazeProbeBridgeOutput(
            probe=SeqMazeProbeOutput(edge_logits=edge_logits),
        )

    def forward(
        self,
        batch: Batch,
        state: HRMStateV2 | None = None,
    ) -> tuple[SeqMazeProbeBridgeOutput, HRMStateV2]:
        """Run a forward pass on one batch of probe samples."""
        inputs = self.prepare_inputs(batch)
        outputs, next_state = self.model(inputs, state=state)
        bridge_out = self.postprocess(outputs)
        return bridge_out, next_state


# =============================================================================
@dataclass(frozen=True)
class SeqMazeBridgeOutput:
    """V1 path-prediction bridge output bundle (no policy/critic)."""

    task: SeqMazeTaskOutput


# =============================================================================
@dataclass(frozen=True)
class SeqMazeHRMV2PolicyOutput:
    """Value-control readouts for the halt/continue policy."""

    q_values: Tensor  # (B, A) Q-values over halt/continue actions
    valid_action_mask: Tensor | None = None


# =============================================================================
@dataclass(frozen=True)
class SeqMazeHRMV2CriticOutput:
    """Critic readouts for the state-value estimate."""

    state_value: Tensor  # (B, 1) STR state-value


# =============================================================================
@dataclass(frozen=True)
class SeqMazeHRMV2BridgeOutput:
    """Controller-consumable SeqMaze HRM v2 bridge output bundle."""

    task: SeqMazeTaskOutput
    policy: SeqMazeHRMV2PolicyOutput
    critic: SeqMazeHRMV2CriticOutput


# =============================================================================
# Module-level vocabulary masking helper (shared by v1 and v2 bridges)
# =============================================================================


def _apply_seqmaze_output_mask(
    logits: Tensor,
    node_valid_mask: Tensor,
    *,
    n_max: int,
) -> Tensor:
    """Apply sample-specific output vocabulary masking.

    Invalid node indices (those not present in the sample) and PAD
    are masked to -inf.  EOS and valid node indices pass unmasked.

    Args:
        logits: (B, T, V) global path logits, V = n_max + 2.
        node_valid_mask: (B, N) bool mask of valid node indices in each sample.
        n_max: Maximum candidate nodes.

    Returns:
        (B, T, V) constrained path logits.
    """
    B, T, V = logits.shape
    N = n_max
    device = logits.device

    vocab_mask = torch.zeros(B, V, dtype=torch.bool, device=device)
    vocab_mask[:, :N] = node_valid_mask  # (B, N)
    vocab_mask[:, N] = True  # EOS is always valid
    vocab_mask = vocab_mask.unsqueeze(1)  # (B, 1, V)

    return torch.where(
        vocab_mask, logits, torch.tensor(float("-inf"), device=device)
    )


# =============================================================================
class SeqMazeHRMV2BridgeAdapter(nn.Module):
    """SeqMaze v2 path-prediction bridge over HRM v2 (actor-critic).

    Wraps HRModelV2, encodes graph nodes + path queries into schema tokens,
    runs HRM deliberation, decodes path-region slots into path logits, applies
    sample-specific output vocabulary masking, and exposes policy/critic
    readouts for RL halt/continue control.

    Validates at construction that:

        n_max + t_max == model.config.num_schema_slots
    """

    def __init__(
        self,
        model: HRModelV2,
        config: SeqMazeAdapterSettings | None = None,
    ) -> None:
        super().__init__()
        self._config = config or SeqMazeAdapterSettings()
        self.model = model
        self._node_valid_mask: Tensor | None = None

        # Validate profile coupling: N + T <= S
        expected_slots = self._config.n_max + self._config.t_max
        if expected_slots > model.config.num_schema_slots:
            raise ValueError(
                f"Task requires {expected_slots} schema slots "
                f"(n_max={self._config.n_max} + t_max={self._config.t_max}), "
                f"but PFC capacity is only {model.config.num_schema_slots}."
            )

        self._encoder = SeqMazeEncoder(self._config)

        # Build decoder with optional weight-sharing
        if self._config.share_path_position_embeddings:
            path_position_emb = self._encoder.E_path_position
        else:
            path_position_emb = None
        self._decoder = SeqMazeDecoder(
            self._config,
            path_position_emb=path_position_emb,
        )

    @property
    def config(self) -> SeqMazeAdapterSettings:
        return self._config

    def init_state(self, batch_size: int) -> HRMStateV2:
        return self.model.init_state(batch_size)

    def reset_state(
        self,
        reset_flag: Tensor,
        state: HRMStateV2,
    ) -> HRMStateV2:
        return self.model.reset_state(reset_flag, state)

    def prepare_inputs(self, batch: Batch) -> HRMInputV2:
        """Extract task input from batch and encode into schema tokens.

        Caches ``node_mask`` for output vocabulary masking in
        :meth:`postprocess`.
        """
        task_input = extract_seqmaze_task_input(batch)
        schema_tokens, schema_mask = self._encoder.forward_padded(
            node_obs_id=task_input.node_obs_id,
            node_candidate_index=task_input.node_candidate_index,
            node_start_flag=task_input.node_start_flag,
            node_goal_flag=task_input.node_goal_flag,
            successor_indices=task_input.successor_indices,
            successor_mask=task_input.successor_mask,
            node_mask=task_input.node_mask,
            model_seq_length=self.model.config.num_schema_slots,
        )
        self._node_valid_mask = task_input.node_mask
        return HRMInputV2(schema_tokens=schema_tokens)

    def postprocess(self, outputs: HRMOutputV2) -> SeqMazeHRMV2BridgeOutput:
        """Decode path-region slots into path logits, apply output vocabulary
        masking, and expose policy/critic readouts."""
        path_logits = self._decoder(
            outputs.schema_slots,
            n_max=self._config.n_max,
            t_max=self._config.t_max,
        )
        if self._node_valid_mask is not None:
            path_logits = _apply_seqmaze_output_mask(
                path_logits,
                self._node_valid_mask,
                n_max=self._config.n_max,
            )
        return SeqMazeHRMV2BridgeOutput(
            task=SeqMazeTaskOutput(path_logits=path_logits),
            policy=SeqMazeHRMV2PolicyOutput(q_values=outputs.q_values),
            critic=SeqMazeHRMV2CriticOutput(state_value=outputs.state_value),
        )

    def forward(
        self,
        batch: Batch,
        state: HRMStateV2 | None = None,
    ) -> tuple[SeqMazeHRMV2BridgeOutput, HRMStateV2]:
        """Run a forward pass on one batch of v2 path-prediction samples."""
        inputs = self.prepare_inputs(batch)
        outputs, next_state = self.model(inputs, state=state)
        bridge_out = self.postprocess(outputs)
        return bridge_out, next_state


# =============================================================================
__all__ = [
    "SeqMazeProbeBridgeOutput",
    "SeqMazeBridgeOutput",
    "SeqMazeHRMV1ControlOutput",
    "SeqMazeHRMV1BridgeOutput",
    "SeqMazeHRMV1BridgeAdapter",
    "SeqMazeProbeHRMV2BridgeAdapter",
    "SeqMazeHRMV2BridgeAdapter",
    "SeqMazeHRMV2PolicyOutput",
    "SeqMazeHRMV2CriticOutput",
    "SeqMazeHRMV2BridgeOutput",
    "_apply_seqmaze_output_mask",
]
