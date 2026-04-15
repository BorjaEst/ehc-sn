"""MazeHard plus HRM v2 bridge implementation."""

from __future__ import annotations

from torch import nn

from ehc_sn.adapters.maze_hard.decoders import MazeHardTokenDecoder
from ehc_sn.adapters.maze_hard.encoders import MazeHardTokenEncoder
from ehc_sn.models.hrm.hrm_v2 import HRModelV2, HRMStateV2
from ehc_sn.tasks.maze_hard import MazeHardTaskBatch, MazeHardTaskOutput


# =============================================================================
class MazeHardHRMV2BridgeAdapter(nn.Module):
    """MazeHard plus HRM v2 model-task binding over the canonical HRM core."""

    def __init__(  # ----------------------------------------------------------
        self,
        model: HRModelV2,
        encoder: MazeHardTokenEncoder,
        decoder: MazeHardTokenDecoder,
    ) -> None:
        """Initialize the HRM v2 bridge adapter with its component modules."""
        super().__init__()
        self.model = model
        self.encoder = encoder
        self.decoder = decoder

    def forward(  # -----------------------------------------------------------
        self,
        batch: MazeHardTaskBatch,
        state: HRMStateV2 | None = None,
    ) -> tuple[HRMStateV2, MazeHardTaskOutput]:
        """Run a forward pass of the HRM v2 bridge adapter on a MazeHard task batch."""

        input = self.encoder(batch)
        if state is None:
            state = self.model.init_state(batch_size=input.batch_size)

        logits, next_state = self.model(input, state=state)
        output = self.decoder(logits)

        return next_state, output


# =============================================================================
__all__ = ["MazeHardHRMV2BridgeAdapter"]
