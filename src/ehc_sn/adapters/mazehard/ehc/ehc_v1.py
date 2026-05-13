"""MazeHard plus EHC v1 bridge implementation.

Binds model-native EHC v1 types to the MazeHard task family.

Optimizer ownership in phase 2 (mazehard_reason_pretrain):
    - Trained:  model.pfc (excl. pfc.estimator), model.pfc.estimator,
                model.str, _encoder, _decoder.
    - Frozen:   model.lec, model.mec, model.hpc, model.lec_to_hpc,
                model.mec_to_hpc, model.pfc_to_hpc, model.hpc_to_pfc.

Critic surface:
    EHCControlV1.reward_prediction is the STR V(s) scalar.  STR is the
    canonical V(s) owner in this repo (see ehc_sn.modules.str.__init__).
    The bridge re-exposes it as critic.state_value with shape (B, 1) to
    satisfy the neutral ActorCriticCriticOutput protocol.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ehc_sn.adapters.mazehard.ehc.core import MazeHardEHCAdapterSettings
from ehc_sn.adapters.mazehard.ehc.objectives import MazeHardEHCV1HybridTaskBinding
from ehc_sn.models.ehc.ehc_v1 import EHCInputV1, EHCModelV1, EHCOutputV1, EHCStateV1
from ehc_sn.tasks.mazehard.contracts import MazeHardTaskOutput
from ehc_sn.types import Batch


# =============================================================================
@dataclass(frozen=True)
class MazeHardEHCV1PolicyOutput:
    """Policy readouts emitted by the MazeHard EHC v1 bridge."""

    policy_logits: Tensor
    valid_action_mask: Tensor | None = None


# =============================================================================
@dataclass(frozen=True)
class MazeHardEHCV1CriticOutput:
    """Critic readouts emitted by the MazeHard EHC v1 bridge.

    re-exposes EHCControlV1.reward_prediction as the canonical actor-critic
    critic surface.  STR is the V(s) owner in this codebase; the field name
    ``reward_prediction`` on EHCControlV1 reflects naming drift, not a
    semantic difference from V(s).  See ehc_sn.modules.str.__init__ for the
    authoritative STR contract.
    """

    state_value: Tensor  # (B, 1)


# =============================================================================
@dataclass(frozen=True)
class MazeHardEHCV1BridgeOutput:
    """Controller-consumable MazeHard EHC v1 bridge output bundle."""

    task: MazeHardTaskOutput
    policy: MazeHardEHCV1PolicyOutput
    critic: MazeHardEHCV1CriticOutput


# =============================================================================
class MazeHardEHCV1Encoder(nn.Module):
    """Encodes a MazeHard token grid into an EHCInputV1 sensory payload.

    MazeHard provides a flattened grid of token IDs (B, S).  This encoder
    embeds the tokens and pools over the sequence dimension to produce the
    multi-scale sensory-code list that EHCInputV1 expects.

    The LEC module is frozen in phase 2, so the sensory codes serve as a
    compressed static context for the frozen spatial pathway rather than a
    dynamic per-step signal.
    """

    def __init__(
        self,
        n_freq: int,
        feature_dim: int,
        vocab_size: int,
        *,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self._n_freq = n_freq
        self._feature_dim = feature_dim
        self.embed = nn.Embedding(vocab_size, n_freq * feature_dim, device=device, dtype=dtype)

    def forward(self, batch: Batch) -> EHCInputV1:
        """Encode a MazeHard batch into an EHCInputV1 payload."""
        input_ids = batch["input_ids"].long()  # (B, S)
        B = input_ids.shape[0]
        device = input_ids.device

        # Embed tokens → (B, S, n_freq * feature_dim); pool over S → (B, n_freq * feature_dim)
        x = self.embed(input_ids).mean(dim=1)

        # Split into n_freq multi-scale sensory codes
        observation_embedding = list(x.split(self._feature_dim, dim=-1))  # n_freq × (B, feature_dim)

        # Horizon-1 deliberation: no previous navigation action; treat each step as
        # a fresh episode so MEC resets its grid-prior state.
        prev_action = torch.zeros(B, dtype=torch.long, device=device)
        episode_start = torch.ones(B, dtype=torch.bool, device=device)

        return EHCInputV1(
            observation_embedding=observation_embedding,
            previous_action=prev_action,
            episode_start=episode_start,
        )


# =============================================================================
class MazeHardEHCV1TaskDecoder(nn.Module):
    """Decodes EHC v1 PFC body workspace tokens into MazeHard task logits.

    Stacks the three fixed body slots (state, replay, cue) with the content
    family to recover the full (B, pfc.seq_length, D) body tensor, then
    applies a linear head to produce per-position token logits.

    With pfc.seq_length = 900 (30x30 MazeHard) the output shape is
    (B, 900, vocab_size), mirroring the MazeHardMLPDecoder used by HRM v2.
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        *,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False, device=device, dtype=dtype)

    def forward(self, output: EHCOutputV1) -> MazeHardTaskOutput:
        """Decode PFC body workspace to MazeHard token-prediction logits."""
        c = output.content
        # Reconstruct full body: 3 fixed slots + content family = pfc.seq_length
        body = torch.cat(
            [
                c.state_slot.unsqueeze(1),  # (B, 1, D)
                c.replay_slot.unsqueeze(1),  # (B, 1, D)
                c.cue_slot.unsqueeze(1),  # (B, 1, D)
                c.content_slots,  # (B, content_size, D)
            ],
            dim=1,
        )  # (B, pfc.seq_length, D)
        return MazeHardTaskOutput(task_logits=self.lm_head(body))


# =============================================================================
class MazeHardEHCV1BridgeAdapter(nn.Module):
    """MazeHard plus EHC v1 model-task binding over the EHC v1 backbone.

    Bridges the horizon-1 MazeHard deliberation controller to EHC v1.
    The encoder and decoder are trained in phase 2; the spatial pathway
    (LEC, MEC, HPC, and all four inter-region projections) is frozen by
    the phase-2 Lightning surface.

    Public surface (stable):
        config, model, init_state(), reset_state(), prepare_inputs(),
        postprocess(), forward().
    """

    def __init__(
        self,
        model: EHCModelV1,
        config: MazeHardEHCAdapterSettings | None = None,
    ) -> None:
        super().__init__()
        self._config = config or MazeHardEHCAdapterSettings()
        self.model = model

        n_freq = len(model.config.f_initial)
        feature_dim = model.config.lec.feature_dim
        hidden_size = model.config.hidden_size
        params = next(model.parameters())

        self._encoder = MazeHardEHCV1Encoder(
            n_freq=n_freq,
            feature_dim=feature_dim,
            vocab_size=self._config.vocab_size,
            device=params.device,
            dtype=params.dtype,
        )
        self._decoder = MazeHardEHCV1TaskDecoder(
            hidden_size=hidden_size,
            vocab_size=self._config.vocab_size,
            device=params.device,
            dtype=params.dtype,
        )

    @property
    def config(self) -> MazeHardEHCAdapterSettings:
        """Return the immutable adapter settings used to configure the bridge."""
        return self._config

    def init_state(self, batch_size: int) -> EHCStateV1:
        """Create a fresh EHC recurrent state for one rollout batch."""
        return self.model.init_state(batch_size)

    def reset_state(self, reset_flag: Tensor, state: EHCStateV1) -> EHCStateV1:
        """Reset halted rows of the EHC recurrent state."""
        return self.model.reset_state(reset_flag, state)

    def prepare_inputs(self, batch: Batch) -> EHCInputV1:
        """Prepare the EHC-native input payload from one generic rollout batch."""
        return self._encoder(batch)

    def postprocess(self, output: EHCOutputV1) -> MazeHardEHCV1BridgeOutput:
        """Split one EHC step output into task, policy, and critic surfaces."""
        return MazeHardEHCV1BridgeOutput(
            task=self._decoder(output),
            policy=MazeHardEHCV1PolicyOutput(policy_logits=output.control.control_logits),
            # STR scalar V(s) re-exposed as the canonical critic surface.
            critic=MazeHardEHCV1CriticOutput(state_value=output.control.reward_prediction.unsqueeze(-1)),
        )

    def forward(
        self,
        batch: Batch,
        state: EHCStateV1 | None = None,
    ) -> tuple[MazeHardEHCV1BridgeOutput, EHCStateV1]:
        """Run a forward pass of the EHC v1 bridge adapter on a MazeHard batch."""
        inputs = self.prepare_inputs(batch)
        output, next_state = self.model(inputs, state=state)
        return self.postprocess(output), next_state


# =============================================================================
__all__ = [
    "MazeHardEHCAdapterSettings",
    "MazeHardEHCV1BridgeAdapter",
    "MazeHardEHCV1BridgeOutput",
    "MazeHardEHCV1CriticOutput",
    "MazeHardEHCV1Encoder",
    "MazeHardEHCV1HybridTaskBinding",
    "MazeHardEHCV1PolicyOutput",
    "MazeHardEHCV1TaskDecoder",
]
