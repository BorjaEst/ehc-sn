"""Shared evaluation regime layer for Lightning training surfaces.

Public API:

- :mod:`ehc_sn.lightning.eval.contracts` — shared contract types and protocols.
- :mod:`ehc_sn.lightning.eval.runner` — stateless regime runner.
"""

from ehc_sn.lightning.eval.contracts import (
    EvaluationBatchArtifacts,
    EvaluationCaseBatch,
    EvaluationRegimeSettings,
    EvaluationScheduleSettings,
    EvaluationSourceProvider,
    EvaluationTraceRequest,
    SupportsEvaluationRegimes,
)
from ehc_sn.lightning.eval.runner import RegimeRunResult, load_provider, run_evaluation_regime

__all__ = [
    "EvaluationBatchArtifacts",
    "EvaluationCaseBatch",
    "EvaluationRegimeSettings",
    "EvaluationScheduleSettings",
    "EvaluationSourceProvider",
    "EvaluationTraceRequest",
    "RegimeRunResult",
    "SupportsEvaluationRegimes",
    "load_provider",
    "run_evaluation_regime",
]
