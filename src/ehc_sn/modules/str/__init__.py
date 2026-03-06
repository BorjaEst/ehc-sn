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
from torch import Tensor, nn

from ehc_sn.types import Device, Dtype
from ehc_sn.utils.detach import DetachMixin


# =================================================================================================
class STRSettings(BaseModel, extra="forbid"):
    """ """


# =================================================================================================
@dataclass
class STRState(DetachMixin):
    """STR recurrent state.

    Carries the reward signal from the previous environment step into the next
    STR forward pass, modelling dopaminergic tone from the preceding trial epoch.

    Attributes:
        prev_reward: Reward received at the *previous* step, shape ``(B,)``.
            Read by ``STRModel.forward()`` (concatenated with cortical features).
            Written externally by the training loop after ``env.step()``.
            Zeroed on episode start (``init_state``) and slot reset (``reset_state``).
    """

    prev_reward: Tensor  # (B,) float32 — reward from the previous env step


# =================================================================================================
class STRModelLinear(nn.Module):
    """STR actor-critic with simple linear layers. This is the default STR implementation in HRM v2.

    The policy head outputs logits for the HALT action and the prediction (action) distribution.
    The value head outputs a scalar V(s) estimate for TD advantage computation.

    The forward method takes cortical features and the previous reward, concatenates them,
    and passes them through the actor and critic MLPs to produce the policy distribution and value estimate.
    """

    def __init__(  # ------------------------------------------------------------------------------
        self, config: STRSettings, *,
        device: Optional[Device] = None, dtype: Optional[Dtype] = None,
    ) -> None:  # fmt: skip
        raise NotImplementedError("STRModelLinear is not implemented yet.")

    def forward(  # -------------------------------------------------------------------------------
        self, features: Tensor, q_values: Tensor, state: STRState,
    ) -> Tuple[STRState, Tensor]:  # fmt: skip
        """ """
        raise NotImplementedError("STRModelLinear.forward() is not implemented yet.")


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
        raise NotImplementedError("STRModelGRU is not implemented yet.")

    def forward(  # -------------------------------------------------------------------------------
        self, features: Tensor, q_values: Tensor, state: STRState,
    ) -> Tuple[STRState, Tensor]:  # fmt: skip
        """ """
        raise NotImplementedError("STRModelGRU.forward() is not implemented yet.")


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
        raise NotImplementedError("STRModelLSTM is not implemented yet.")

    def forward(  # -------------------------------------------------------------------------------
        self, features: Tensor, q_values: Tensor, state: STRState,
    ) -> Tuple[STRState, Tensor]:  # fmt: skip
        """ """
        raise NotImplementedError("STRModelLSTM.forward() is not implemented yet.")


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
        raise NotImplementedError("STRModelGoNoGo is not implemented yet.")

    def forward(  # -------------------------------------------------------------------------------
        self, features: Tensor, q_values: Tensor, state: STRState,
    ) -> Tuple[STRState, Tensor]:  # fmt: skip
        """ """
        raise NotImplementedError("STRModelGoNoGo.forward() is not implemented yet.")


__all__ = ["STRSettings", "STRState", "STRModelLinear"]
