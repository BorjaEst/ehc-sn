"""Lightning modules organized by training regime.

Each module defines a generic LightningModule for one training regime
(supervised, variational replay, actor-critic, hybrid), parameterized
by component classes injected via a components bundle.

Sub-modules:

- :mod:`~ehc_sn.lightning.modules.variational_replay` — variational/replay training
  (TEM-like: chunked TBPTT, replay trajectory controller, variational objective).
- :mod:`~ehc_sn.lightning.modules.act_supervised` — ACT-supervised training
  (HRM v1-like: halting, partial reset, target network).
- :mod:`~ehc_sn.lightning.modules.q_halting` — supervised Q-halting training
  (HRM v2-like: deliberation, three-optimizer, warmup gating).
- (future) ``hybrid`` — combined memory-reasoning training (EHP-like).
"""

from ehc_sn.lightning.modules.act_supervised import (
    ACTSupervisedConfig,
    ACTSupervisedModule,
)
from ehc_sn.lightning.modules.q_halting import (
    QHaltingConfig,
    QHaltingModule,
)
from ehc_sn.lightning.modules.variational_replay import (
    VariationalReplayConfig,
    VariationalReplayModule,
)

__all__ = [
    "ACTSupervisedConfig",
    "ACTSupervisedModule",
    "QHaltingConfig",
    "QHaltingModule",
    "VariationalReplayConfig",
    "VariationalReplayModule",
]
