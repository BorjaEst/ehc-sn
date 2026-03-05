"""STR (Striatum) actor-critic gating module.

Biologically, the striatum receives projections from dlPFC and decides when to
act (halt deliberation) vs. continue computing via Go/NoGo (D1/D2-like) pathways.

This module implements the computational analogue:
    - **Actor** (policy): two-logit categorical head over {halt, continue}.
    - **Critic** (value): scalar V(s) estimator — the **sole** advantage critic
      for TD-error ("dopamine") computation and policy gradient updates.

Note on vmPFC (``QValueEstimator`` inside PFC):
    The PFC module contains an auxiliary Q-value predictor (vmPFC) that is trained
    with a separate TD target for credit assignment.  It is **not** the advantage
    critic; advantage is always computed from STR's ``V(s)`` here.  vmPFC updates
    only its own parameters via a third dedicated optimizer.

Both heads consume *detached* PFC theta-level features (``z_H[:, 0]``), enforcing
the PFC/STR gradient isolation required by the HRM v2 design (STR-only RL updates).

Action indices (canonical; must match caller usage):
    0 = HALT
    1 = CONTINUE
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Tuple

import torch
from pydantic import BaseModel, Field
from torch import Tensor, nn

from ehc_sn.types import Device, Dtype
from ehc_sn.utils.detach import DetachMixin

# Canonical action indices shared with hrm_v2.
HALT_ACTION: int = 0
CONTINUE_ACTION: int = 1


# =================================================================================================
class STRSettings(BaseModel, extra="forbid"):
    """Configuration for :class:`STRModel`.

    Attributes:
        hidden_size: Input feature dimensionality (must match the PFC hidden size fed to STR).
        policy_hidden_layers: Sizes of hidden layers in the actor MLP. Empty → single linear.
        value_hidden_layers: Sizes of hidden layers in the critic MLP. Empty → single linear.
        init_policy_bias: Bias initialisation for the HALT logit. Negative values start the
            policy biased toward *continue* (σ(-2) ≈ 0.12), preventing trivial halt collapse
            during the λ=0 warmup phase.
        init_value_bias: Initial scalar bias for the value head.
    """

    hidden_size: int = Field(..., ge=1, description="Input feature dimensionality.")
    policy_hidden_layers: list[int] = Field(
        default_factory=list,
        description="Hidden layer sizes for the actor MLP. Empty = single linear projection.",
    )
    value_hidden_layers: list[int] = Field(
        default_factory=list,
        description="Hidden layer sizes for the critic MLP. Empty = single linear projection.",
    )
    init_policy_bias: float = Field(
        default=-2.0,
        description=(
            "Initial bias for the HALT action logit. Negative → continue-favoring init. "
            "σ(-2) ≈ 0.12; the model almost never halts at initialisation."
        ),
    )
    init_value_bias: float = Field(
        default=0.0,
        description="Initial bias for the value head output.",
    )


# =================================================================================================
@dataclass
class STRState(DetachMixin):
    """ """

    prev_reward: Tensor = torch.tensor(0.0)  # (B,) float32 — reward from previous env.step()


# =================================================================================================
class STRModel(nn.Module):
    """STR actor-critic: maps detached PFC theta-cell features to policy + value.

    Uses a **two-logit categorical** policy (not a Bernoulli/single-logit BCE formulation)
    so the design generalises to N-action selection without modification.

    Args:
        config: :class:`STRSettings` describing architecture and init.
        device: Target device for parameter allocation.
        dtype: Parameter dtype.
    """

    def __init__(  # ------------------------------------------------------------------------------
        self, config: STRSettings, *,
        device: Optional[Device] = None, dtype: Optional[Dtype] = None,
    ) -> None:  # fmt: skip
        super().__init__()
        self._config = config

        # Actor: features (B, D) → policy_logits (B, 2)
        policy_layers: list[nn.Module] = []
        in_dim = config.hidden_size
        for h_dim in config.policy_hidden_layers:
            policy_layers.append(nn.Linear(in_dim, h_dim, bias=False, device=device, dtype=dtype))
            policy_layers.append(nn.SiLU())
            in_dim = h_dim
        policy_layers.append(nn.Linear(in_dim, 2, bias=True, device=device, dtype=dtype))
        self.policy_head = nn.Sequential(*policy_layers)

        # Critic: features (B, D) → value (B, 1) → squeezed to (B,)
        value_layers: list[nn.Module] = []
        in_dim = config.hidden_size
        for h_dim in config.value_hidden_layers:
            value_layers.append(nn.Linear(in_dim, h_dim, bias=False, device=device, dtype=dtype))
            value_layers.append(nn.SiLU())
            in_dim = h_dim
        value_layers.append(nn.Linear(in_dim, 1, bias=True, device=device, dtype=dtype))
        self.value_head = nn.Sequential(*value_layers)

        self.reset_parameters()

    @property
    def config(self) -> STRSettings:
        """Settings used to construct this module."""
        return self._config

    def reset_parameters(  # ----------------------------------------------------------------------
        self,
    ) -> None:  # fmt: skip
        """Continue-favouring initialization.

        - Policy weights zeroed; HALT bias = ``init_policy_bias`` (negative = suppressed).
        - CONTINUE bias = 0.
        - Value weights and bias zeroed.

        This prevents the trivial "halt immediately" collapse at the start of training
        when λ ≈ 0 and the agent has no evidence to prefer halting.
        """
        with torch.no_grad():
            policy_linear: nn.Linear = self.policy_head[-1]  # type: ignore[assignment]
            policy_linear.weight.zero_()
            policy_linear.bias[HALT_ACTION] = self._config.init_policy_bias
            policy_linear.bias[CONTINUE_ACTION] = 0.0

            value_linear: nn.Linear = self.value_head[-1]  # type: ignore[assignment]
            value_linear.weight.zero_()
            value_linear.bias.zero_()

    def init_state(  # ----------------------------------------------------------------------------
        self, batch_size: int, *, 
        device: Optional[Device] = None,
    ) -> STRState:  # fmt: skip
        """Create initial STR state (placeholder; no tensors yet)."""
        return STRState(prev_reward=torch.zeros(batch_size, device=device))

    def reset_state(  # ---------------------------------------------------------------------------
        self, state: STRState, reset_flag: Tensor,
    ) -> STRState:  # fmt: skip
        """Reset STR state for slots where reset_flag is True (no-op for now)."""
        prev_reward = torch.where(reset_flag, torch.zeros_like(state.prev_reward), state.prev_reward)
        return replace(state, prev_reward=prev_reward)

    def forward(  # -------------------------------------------------------------------------------
        self, features: Tensor, state: STRState,
    ) -> Tuple[Tensor, Tensor, STRState]:  # fmt: skip
        """Compute policy logits and state value from PFC theta-cell features.

        Args:
            features: Detached dlPFC theta-level CLS features, shape ``(B, D)``.
                      **Must be detached by the caller** to enforce STR-only learning.
            state: Current STRState (placeholder for now; no tensors or recurrence yet).

        Returns:
            policy_logits: Two-logit categorical distribution ``(B, 2)``;
                           index 0 = halt, index 1 = continue.
            value: Scalar state-value estimate ``(B,)``.
        """
        reward_input = state.prev_reward.unsqueeze(-1)  # (B, 1)
        x = torch.cat([features.to(torch.float32), reward_input], dim=-1)  # (B, D+1)
        policy_logits = self.policy_head(x)  # (B, A)
        value = self.value_head(x).squeeze(-1)  # (B,)
        return policy_logits, value, state


class STRModelGRU(nn.Module):
    """STR actor-critic with GRU recurrence. Not currently used.

    This is an alternative STR implementation with a GRU layer for temporal integration.
    It is not currently used in the HRM v2 design, but it may be useful for future experiments
    with more complex STR dynamics.

    The forward method and state management would need to be adapted to handle the GRU's
    recurrent state and the temporal dependencies it introduces.
    """

    def __init__(  # ------------------------------------------------------------------------------
        self, config: STRSettings, *,
        device: Optional[Device] = None, dtype: Optional[Dtype] = None,
    ) -> None:  # fmt: skip
        raise NotImplementedError("STRModelGRU is not implemented yet. Use STRModel instead.")


class STRModelLSTM(nn.Module):
    """STR actor-critic with LSTM recurrence. Not currently used.

    This is an alternative STR implementation with an LSTM layer for temporal integration.
    It is not currently used in the HRM v2 design, but it may be useful for future experiments
    with more complex STR dynamics.

    The forward method and state management would need to be adapted to handle the LSTM's
    recurrent state and the temporal dependencies it introduces.
    """

    def __init__(  # ------------------------------------------------------------------------------
        self, config: STRSettings, *,
        device: Optional[Device] = None, dtype: Optional[Dtype] = None,
    ) -> None:  # fmt: skip
        raise NotImplementedError("STRModelLSTM is not implemented yet. Use STRModel instead.")


class STRModelGoNoGo(nn.Module):
    """STR actor-critic with separate Go/NoGo pathways. Not currently used.

    This is an alternative STR implementation with separate pathways for Go (D1-like) and NoGo (D2-like)
    action selection. It is not currently used in the HRM v2 design, but it may be useful for future experiments
    exploring more biologically detailed STR architectures.

    The forward method and state management would need to be adapted to compute separate Go/NoGo activations
    and combine them into a final policy distribution.
    """

    def __init__(  # ------------------------------------------------------------------------------
        self, config: STRSettings, *,
        device: Optional[Device] = None, dtype: Optional[Dtype] = None,
    ) -> None:  # fmt: skip
        raise NotImplementedError("STRModelGoNoGo is not implemented yet. Use STRModel instead.")


__all__ = ["STRSettings", "STRState", "STRModel", "HALT_ACTION", "CONTINUE_ACTION"]
