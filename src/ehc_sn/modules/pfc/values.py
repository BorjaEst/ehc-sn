"""vmPFC Q-value estimation module.

Biologically, the ventromedial prefrontal cortex (vmPFC) estimates the subjective
value of available actions given the current working memory state from dlPFC.

The :class:`QValueEstimator` receives the full spatial representation from the
dlPFC reasoning modules (theta cells ``z_H`` and/or gamma cells ``z_L``), pools
it into a summary vector, and projects to per-action Q-values.

The pooling strategy is configurable:
    - ``"first"``: use a single position (legacy parity with [CLS]-like tokens).
    - ``"mean"``: global average (population-code readout).
    - ``"max"``: winner-take-all readout.
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
from pydantic import BaseModel, Field
from torch import Tensor
from torch import device as Device
from torch import dtype as Dtype
from torch import nn


# =============================================================================
class QEstimatorSettings(BaseModel, extra="forbid"):
    """Configuration for :class:`QValueEstimator` (vmPFC analogue).

    Controls the number of actions, how spatial features are pooled, and
    optional MLP depth between the pooled features and the Q-value head.
    """

    hidden_size: int = Field(
        ...,
        ge=1,
        description=(
            "Dimensionality of the input feature vector "
            "(must match backbone output)."
        ),
    )
    n_actions: int = Field(
        default=2,
        ge=1,
        description="Number of discrete actions to estimate Q-values for.",
    )
    pool_mode: Literal["first", "first_detached", "mean"] = Field(
        default="first",
        description=(
            "How to reduce spatial positions (B, S, D) into a summary vector (B, D). "
            "'first': select a single position index (legacy parity). "
            "'first_detached': same but no gradient through features. "
            "'mean': average over all positions (population-code readout). "
        ),
    )
    pool_index: int = Field(
        default=0,
        ge=0,
        description=(
            "Position index to select when pool_mode='first'. "
            "Legacy default: 0."
        ),
    )
    hidden_layers: list[int] = Field(
        default_factory=list,
        description=(
            "Sizes of optional hidden layers between pooled features and Q-head. "
            "Empty list (default) = single linear projection (legacy parity)."
        ),
    )
    init_bias: float = Field(
        default=-5.0,
        description=(
            "Initial bias for all Q-value outputs. "
            "Negative values → conservative init: σ(-5) ≈ 0.007 at init."
        ),
    )


# =============================================================================
class QValueEstimator(nn.Module):
    """Q-value head mapping dlPFC features to per-action value estimates.

    With ``hidden_layers=[]``, ``n_actions=2``, and ``pool_mode="first"``,
    this is functionally equivalent to the legacy ``LinearHaltingHead``.

    Accepts both pre-pooled ``(B, D)`` and spatial ``(B, S, D)`` inputs.
    When given 3-D input, applies the configured pooling strategy first.
    """

    def __init__(  # ----------------------------------------------------------
        self,
        config: QEstimatorSettings,
        *,
        device: Optional[Device] = None,
        dtype: Optional[Dtype] = None,
    ) -> None:
        super().__init__()
        self._config = config

        # Optional MLP trunk: [] → Identity, [h1, h2] → Linear+SiLU+Linear+SiLU
        layers: list[nn.Module] = []
        in_dim = config.hidden_size
        for h_dim in config.hidden_layers:
            layers.append(
                nn.Linear(in_dim, h_dim, bias=False, device=device, dtype=dtype)
            )
            layers.append(nn.SiLU())
            in_dim = h_dim
        self.trunk = nn.Sequential(*layers) if layers else nn.Identity()

        # Final projection to Q-values
        self.head = nn.Linear(
            in_dim, config.n_actions, bias=True, device=device, dtype=dtype
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Zero-weight, negative-bias init (legacy parity).

        At init all actions have equal Q ≈ init_bias regardless of input,
        so σ(Q) is near zero → the model starts by almost never halting.
        """
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.bias.fill_(self._config.init_bias)

    @property
    def config(self) -> QEstimatorSettings:
        return self._config

    def _pool(  # -------------------------------------------------------------
        self,
        z_H: Tensor,
        z_L,
    ) -> Tensor:
        """Reduce (B, S, D) → (B, D) using the configured strategy."""
        mode = self._config.pool_mode
        if mode == "first":
            return z_H[:, self._config.pool_index]
        if mode == "first_detached":
            return z_H[:, self._config.pool_index].detach()
        if mode == "mean":
            return z_H.mean(dim=1)
        raise ValueError(f"Unknown pool_mode: {mode!r}")

    def forward(  # -----------------------------------------------------------
        self,
        z_H: Tensor,
        z_L: Tensor,
    ) -> Tensor:
        """Estimate Q-values from dlPFC features.

        Args:
            z_H: High-level (theta) features from dlPFC, shape (B, S, D) or (B, D).
            z_L: Low-level (gamma) features from dlPFC, shape (B, S, D) or (B, D).

        Returns:
            Q-value logits ``(B, n_actions)``.
        """
        features = self._pool(z_H, z_L)
        x = self.trunk(features.to(torch.float32))
        return self.head(x).to(torch.float32)


# =============================================================================
__all__ = ["QEstimatorSettings", "QValueEstimator"]
