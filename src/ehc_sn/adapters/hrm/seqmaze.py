"""SeqMaze probe bridge adapter for HRM v2.

Binds the seqmaze phase-0 edge-lookup probe to HRModelV2.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ehc_sn.adapters.hrm._base import (
    SeqMazeProbeAdapterSettings,
    SeqMazeProbeDecoder,
    SeqMazeProbeEncoder,
)
from ehc_sn.models.hrm.hrm_v2 import (
    HRMInputV2,
    HRModelV2,
    HRMOutputV2,
    HRMStateV2,
)
from ehc_sn.tasks.seqmaze.contracts import SeqMazeProbeInput, SeqMazeProbeOutput
from ehc_sn.tasks.seqmaze.runtime import extract_seqmaze_probe_input
from ehc_sn.types import Batch


# =============================================================================
@dataclass(frozen=True)
class SeqMazeProbeBridgeOutput:
    """Bundle of probe output and unused model readouts."""

    probe: SeqMazeProbeOutput


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
