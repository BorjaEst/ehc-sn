"""Striatum action-selection modules.

The ``LinearHaltingHead`` is retained for backward compatibility with code that
expects ``HaltingHead`` protocol (returns ``Tuple[Tensor, Tensor]``). New code
should use :class:`~ehc_sn.modules.pfc.values.QValueEstimator` directly.
"""

# FIXME: This module is currently a placeholder for striatum-related components,
# but it is not yet implemented.
# The striatum is a subcortical structure that interacts with the PFC to provide
# action selection and reinforcement learning signals.
# For now, we are reusing the QValueEstimator from the PFC module as a simple action-selection head,
# but in the future we may want to implement more biologically realistic striatum modules.


from __future__ import annotations

from torch import Tensor, nn

from ehc_sn.modules.pfc.values import QEstimatorSettings, QValueEstimator


class LinearHaltingHead(nn.Module):
    """Legacy compat shim: wraps QValueEstimator to return (q_halt, q_continue).

    .. deprecated::
        Use :class:`~ehc_sn.modules.pfc.values.QValueEstimator` directly.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self._estimator = QValueEstimator(
            QEstimatorSettings(hidden_size=hidden_size, n_actions=2),
        )

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        q = self._estimator(features)
        return q[..., 0], q[..., 1]


__all__ = ["LinearHaltingHead"]
