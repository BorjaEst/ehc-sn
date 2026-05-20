("""Training surfaces for optimization and rollout orchestration.""")

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
