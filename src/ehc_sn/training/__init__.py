"""Training surfaces for optimization, rollout orchestration, and
family-specific training helpers.

Sub-modules:

- :mod:`~ehc_sn.training.hrm` — HRM runtime config and weight-loading.
- :mod:`~ehc_sn.training.tem` — TEM runtime config, schedules, and weight-loading.
- :mod:`~ehc_sn.training.ehc` — EHC weight-loading helpers.
- :mod:`~ehc_sn.training.gradients` — gradient diagnostic helpers.
"""

from ehc_sn.training.rollout import (
    CapturedRolloutResult,
    StreamingRolloutResult,
    run_captured_rollout,
    score_captured_rollout,
    score_rollout_streaming,
)

__all__ = [
    "CapturedRolloutResult",
    "StreamingRolloutResult",
    "run_captured_rollout",
    "score_captured_rollout",
    "score_rollout_streaming",
]
