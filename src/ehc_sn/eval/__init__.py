"""Top-level reusable evaluation seam for named replay regimes.

This package defines shared contracts consumed by task-owned providers and
offers reusable execution helpers that Lightning families can delegate to.
"""

from ehc_sn.eval.artifacts import (
    EvalArtifactExecutorRef,
    LoadedArtifactCase,
    collect_regime_artifact_bundle,
    collect_regime_artifact_bundle_from_ref,
    load_artifact_run_cases,
    load_executor_from_artifact,
    persist_regime_artifact_bundle,
    resolve_provider,
)
from ehc_sn.eval.contracts import (
    EvaluationCaseBatch,
    EvaluationCaseResult,
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
    "EvalArtifactExecutorRef",
    "LoadedArtifactCase",
    "collect_regime_artifact_bundle",
    "collect_regime_artifact_bundle_from_ref",
    "execute_replay_evaluation_batch",
    "iter_evaluation_regime",
    "load_artifact_run_cases",
    "load_executor_from_artifact",
    "persist_regime_artifact_bundle",
    "resolve_provider",
]
