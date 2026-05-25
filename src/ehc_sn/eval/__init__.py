"""Top-level reusable evaluation seam for named replay regimes.

This package defines shared contracts consumed by task-owned providers and
offers reusable execution helpers that Lightning families can delegate to.
"""

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
from ehc_sn.eval.figure_bundle import (
    FigureBundleExecutorArtifact,
    PersistedTraceCase,
    collect_regime_figure_bundle,
    collect_regime_figure_bundle_from_artifact,
    load_executor_from_artifact,
    load_persisted_regime_run_cases,
    persist_regime_figure_bundle,
    resolve_provider,
)

__all__ = [
    "EvaluationCaseResult",
    "EvaluationCaseBatch",
    "EvaluationExecutor",
    "EvaluationRegimeResult",
    "EvaluationSourceProvider",
    "EvaluationTraceRequest",
    "FigureBundleExecutorArtifact",
    "PersistedTraceCase",
    "collect_regime_figure_bundle",
    "collect_regime_figure_bundle_from_artifact",
    "execute_replay_evaluation_batch",
    "iter_evaluation_regime",
    "load_persisted_regime_run_cases",
    "load_executor_from_artifact",
    "persist_regime_figure_bundle",
    "resolve_provider",
]
