"""STR (Striatum) actor-critic gating module.

Biologically, the striatum receives dense cortical projections from dlPFC and
dopaminergic input from the midbrain.  It acts as the reward-integration and
prediction-error computation site in the cortico-striatal loop:

    - **Actor** (policy): two-logit categorical head selecting actions.
    - **Critic** (value): scalar V(s) estimator for TD advantage computation.
    - **RPE** (reward prediction error): δ = r + γ·V(s') − V(s), computed via
      ``compute_rpe()``.  γ is owned by STR because temporal credit assignment
      is a basal-ganglia computation.

Inputs: detached dlPFC theta-level CLS features concatenated with the previous
step's reward (``STRState.prev_reward``), modelling dopaminergic tone.

Note on vmPFC (``QValueEstimator`` inside PFC):
    The PFC module contains an auxiliary Q-value predictor (vmPFC) trained with
    a separate TD target.  It is **not** the advantage critic; advantage is always
    computed from STR's V(s) here.  vmPFC updates via a dedicated optimizer.

"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Tuple

import torch
from pydantic import BaseModel, Field
from torch import Tensor
from torch import device as Device
from torch import dtype as Dtype
from torch import nn

from ehc_sn.utils.detach import DetachMixin


# =================================================================================================
class STRSettings(BaseModel, extra="forbid"):
    """Configuration for STR (striatum) modules."""

    n_features: int = Field(
        ...,
        ge=1,
        description="Dimensionality of STR's internal feature representation (cortical features).",
    )
    n_actions: int = Field(
        ...,
        ge=1,
        description="Number of possible actions in the environment (for prediction logits).",
    )
    hidden_size: int = Field(
        default=64,
        ge=1,
        description="Hidden layer size for STR's internal MLPs.",
    )


# =================================================================================================
@dataclass
class STRState(DetachMixin):
    """Recurrent state for STR.

    Note:
        The current linear STR implementation is stateless; this is a placeholder
        for future recurrent variants.
    """

    dummy_placeholder: int = 0  # TODO: STRState currently has no internal state


# =================================================================================================
class STRModelLinear(nn.Module):
    """STR actor-critic with simple linear layers."""

    def __init__(  # ------------------------------------------------------------------------------
        self, config: STRSettings, *,
        device: Optional[Device] = None, dtype: Optional[Dtype] = None,
    ) -> None:  # fmt: skip
        """Create a linear STR reward/value estimator.

        Args:
            config: STR configuration.
            device: Optional torch device.
            dtype: Optional torch dtype.
        """
        super().__init__()
        self._config = config
        in_dim = config.n_features + config.n_actions  # cortical features + q_values
        self.reward_head = nn.Sequential(
            nn.Linear(in_dim, config.hidden_size, bias=False, device=device, dtype=dtype),
            nn.SiLU(),
            nn.Linear(config.hidden_size, 1, bias=True, device=device, dtype=dtype),
        )

    def init_state(  # ----------------------------------------------------------------------------
        self, batch_size: int, *, device=None,
    ) -> STRState:  # fmt: skip
        """Create an initial STR state."""
        return STRState()

    def reset_state(  # ---------------------------------------------------------------------------
        self, state: STRState, reset_flag: Tensor,
    ) -> STRState:  # fmt: skip
        """Reset the STR state for flagged rows.

        The linear STR implementation is stateless; this returns a shallow copy.
        """
        return replace(state)

    def forward(  # -------------------------------------------------------------------------------
        self, features: Tensor, q_values: Tensor, state: STRState,
    ) -> tuple[STRState, Tensor]:  # fmt: skip
        """Predict reward/value from features and Q-values.

        Args:
            features: CLS feature tensor of shape ``(B, D)``.
            q_values: Q logits/tensor of shape ``(B, A)``.
            state: STR state.

        Returns:
            ``(new_state, state_value)`` where ``state_value`` has shape ``(B,)``.
        """
        x = torch.cat([features.to(torch.float32), q_values.to(torch.float32)], dim=-1)
        reward_hat = self.reward_head(x).squeeze(-1)  # (B,)
        new_state = STRState()  # Placeholder for Linear STR since it has no internal state
        return new_state, reward_hat


class STRModelGRU(nn.Module):
    """STR actor-critic with GRU recurrence."""

    def __init__(  # ------------------------------------------------------------------------------
        self, config: STRSettings, *,
        device: Optional[Device] = None, dtype: Optional[Dtype] = None,
    ) -> None:  # fmt: skip
        raise NotImplementedError("STRModelGRU is not implemented yet.")

    def forward(  # -------------------------------------------------------------------------------
        self, features: Tensor, q_values: Tensor, state: STRState,
    ) -> tuple[STRState, Tensor]:  # fmt: skip
        """Forward pass (not implemented)."""
        raise NotImplementedError("STRModelGRU.forward() is not implemented yet.")


class STRModelLSTM(nn.Module):
    """STR actor-critic with LSTM recurrence."""

    def __init__(  # ------------------------------------------------------------------------------
        self, config: STRSettings, *,
        device: Optional[Device] = None, dtype: Optional[Dtype] = None,
    ) -> None:  # fmt: skip
        raise NotImplementedError("STRModelLSTM is not implemented yet.")

    def forward(  # -------------------------------------------------------------------------------
        self, features: Tensor, q_values: Tensor, state: STRState,
    ) -> tuple[STRState, Tensor]:  # fmt: skip
        """Forward pass (not implemented)."""
        raise NotImplementedError("STRModelLSTM.forward() is not implemented yet.")


class STRModelGoNoGo(nn.Module):
    """STR actor-critic with separate Go/NoGo pathways."""

    def __init__(  # ------------------------------------------------------------------------------
        self, config: STRSettings, *,
        device: Optional[Device] = None, dtype: Optional[Dtype] = None,
    ) -> None:  # fmt: skip
        raise NotImplementedError("STRModelGoNoGo is not implemented yet.")

    def forward(  # -------------------------------------------------------------------------------
        self, features: Tensor, q_values: Tensor, state: STRState,
    ) -> tuple[STRState, Tensor]:  # fmt: skip
        """Forward pass (not implemented)."""
        raise NotImplementedError("STRModelGoNoGo.forward() is not implemented yet.")


__all__ = ["STRSettings", "STRState", "STRModelLinear"]
