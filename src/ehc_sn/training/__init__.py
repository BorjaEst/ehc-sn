"""Training surfaces for optimization, rollout orchestration, and
family-specific training helpers.

Sub-modules:

- :mod:`~ehc_sn.training.hrm` — HRM runtime config and weight-loading.
- :mod:`~ehc_sn.training.tem` — TEM runtime config, schedules, and weight-loading.
- :mod:`~ehc_sn.training.ehp` — EHP weight-loading helpers.
- :mod:`~ehc_sn.training.gradients` — gradient diagnostic helpers.
"""

from ehc_sn.training.rollout import (
    CapturedRolloutResult,
    StreamingRolloutResult,
    StreamingRolloutResultWithTrace,
    run_captured_rollout,
    score_captured_rollout,
    score_rollout_streaming,
    score_rollout_streaming_with_trace,
)

__all__ = [
    "CapturedRolloutResult",
    "StreamingRolloutResult",
    "StreamingRolloutResultWithTrace",
    "run_captured_rollout",
    "score_captured_rollout",
    "score_rollout_streaming",
    "score_rollout_streaming_with_trace",
]
