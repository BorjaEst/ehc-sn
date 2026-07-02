"""Top-level reusable evaluation seam for named replay regimes.

This package defines shared contracts consumed by task-owned providers and
offers reusable execution helpers that Lightning families can delegate to.
"""

from __future__ import annotations

from ehc_sn.evaluation.accumulators import SpatialPopulationAccumulator
from ehc_sn.evaluation.consumers import TraceConsumer
from ehc_sn.evaluation.contracts import (
    ArtifactKey,
    ArtifactKind,
    ArtifactRequirement,
    EvaluationCaseBatch,
    EvaluationCaseResult,
    EvaluationConsumer,
    EvaluationExecutor,
    EvaluationExperiment,
    EvaluationIdentity,
    EvaluationRegimeResult,
    EvaluationSourceProvider,
    EvaluationTraceRequest,
    ProducedArtifact,
    ProviderSpec,
)
from ehc_sn.evaluation.executor import (
    execute_replay_evaluation_batch,
    iter_evaluation_regime,
)

# =============================================================================
__all__ = [
    "ArtifactKey",
    "ArtifactKind",
    "ArtifactRequirement",
    "EvaluationCaseBatch",
    "EvaluationCaseResult",
    "EvaluationConsumer",
    "EvaluationExecutor",
    "EvaluationExperiment",
    "EvaluationIdentity",
    "EvaluationRegimeResult",
    "EvaluationSourceProvider",
    "EvaluationTraceRequest",
    "ProducedArtifact",
    "ProviderSpec",
    "SpatialPopulationAccumulator",
    "TraceConsumer",
    "execute_replay_evaluation_batch",
    "iter_evaluation_regime",
]
