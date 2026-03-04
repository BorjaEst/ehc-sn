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

from dataclasses import dataclass
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
    """STR recurrent state placeholder.

    Kept in the STR module so we can later add short-timescale recurrence
    (e.g., GRU hidden, eligibility traces, filtered RPE) without refactoring
    HRM wiring/signatures.
    """

    # Empty for now.
    pass


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
        _ = (batch_size, device)
        return STRState()

    def reset_state(  # ---------------------------------------------------------------------------
        self, state: STRState, reset_flag: Tensor,
    ) -> STRState:  # fmt: skip
        """Reset STR state for slots where reset_flag is True (no-op for now)."""
        _ = (state, reset_flag)
        return state

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
        x = features.to(torch.float32)
        policy_logits = self.policy_head(x)  # (B, 2)
        value = self.value_head(x).squeeze(-1)  # (B,)
        return policy_logits, value, state


__all__ = ["STRSettings", "STRState", "STRModel", "HALT_ACTION", "CONTINUE_ACTION"]
