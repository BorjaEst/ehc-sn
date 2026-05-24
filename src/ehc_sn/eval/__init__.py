"""Top-level reusable evaluation seam for named replay regimes.

This package defines shared contracts consumed by task-owned providers and
offers reusable execution helpers that Lightning families can delegate to.
"""

from ehc_sn.eval.contracts import (
    EvaluationCaseResult,
    EvaluationCaseBatch,
    EvaluationExecutor,
    EvaluationRegimeResult,
    EvaluationSourceProvider,
    EvaluationTraceRequest,
)
from ehc_sn.eval.executor import (
    execute_replay_evaluation_batch,
    iter_evaluation_regime,
)

__all__ = [
    "EvaluationCaseResult",
    "EvaluationCaseBatch",
    "EvaluationExecutor",
    "EvaluationRegimeResult",
    "EvaluationSourceProvider",
    "EvaluationTraceRequest",
    "execute_replay_evaluation_batch",
    "iter_evaluation_regime",
]
