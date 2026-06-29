"""MLflow evaluation recorder — always-active tracking with local fallback.

Every successful ``ehp-sn eval run`` creates exactly one MLflow run.
Tracking is not optional; there is no null/no-op recorder.

Tracking URI resolution precedence::

    CLI ``--tracking-uri``
        > invocation ``[tracking].tracking_uri``
        > ``MLFLOW_TRACKING_URI`` env var
        > ``sqlite:///outputs/mlflow/mlflow.db``  (local fallback)
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

_EHP_LOCAL_TRACKING_URI = "sqlite:///outputs/mlflow/mlflow.db"
"""Default local SQLite tracking URI when no other source is configured."""


# =============================================================================
@runtime_checkable
class EvaluationRecorder(Protocol):
    """Protocol for recording one evaluation run's metadata."""

    def start(self) -> None:
        """Begin recording."""

    def log_params(self, params: Mapping[str, object]) -> None:
        """Log a set of parameters."""

    def log_metrics(self, metrics: Mapping[str, float]) -> None:
        """Log a set of scalar metrics."""

    def log_artifact(self, path: str | Path) -> None:
        """Log a single artifact file."""

    def finish(self) -> None:
        """Finalise and close the run."""

    def __enter__(self) -> EvaluationRecorder:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.finish()


# =============================================================================
_EHP_LOCAL_ARTIFACTS_DIR = "outputs/mlflow/artifacts"


def _resolve_tracking_uri(explicit_uri: str | None) -> str:
    """Resolve the effective MLflow tracking URI.

    Precedence: explicit → ``MLFLOW_TRACKING_URI`` env var → local fallback.
    """
    if explicit_uri is not None:
        return explicit_uri
    env_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if env_uri:
        return env_uri
    return _EHP_LOCAL_TRACKING_URI


# =============================================================================
@dataclass
class MLflowEvaluationRecorder:
    """MLflow-backed evaluation recorder — always active.

    Parameters
    ----------
    tracking_uri:
        Explicit tracking URI.  Falls back to ``MLFLOW_TRACKING_URI`` env var,
        then to ``sqlite:///outputs/mlflow/mlflow.db``.
    experiment_name:
        MLflow experiment name.
    run_name:
        Optional human-readable run name.
    tags:
        Additional tags applied to the MLflow run.
    """

    tracking_uri: str | None = None
    experiment_name: str = "ehp-evaluation"
    run_name: str | None = None
    tags: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._run: object = None

    def start(self) -> None:
        import mlflow

        uri = _resolve_tracking_uri(self.tracking_uri)
        if uri != _EHP_LOCAL_TRACKING_URI:
            mlflow.set_tracking_uri(uri)
        else:
            # Local fallback: ensure the artifact dir exists, then configure.
            artifacts_dir = Path(_EHP_LOCAL_ARTIFACTS_DIR)
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            mlflow.set_tracking_uri(uri)

        mlflow.set_experiment(self.experiment_name)

        tags = dict(self.tags)
        self._run = mlflow.start_run(run_name=self.run_name, tags=tags)

    def log_params(self, params: Mapping[str, object]) -> None:
        import mlflow

        mlflow.log_params(params)

    def log_metrics(self, metrics: Mapping[str, float]) -> None:
        import mlflow

        mlflow.log_metrics(metrics)

    def log_artifact(self, path: str | Path) -> None:
        import mlflow

        mlflow.log_artifact(str(path))

    def finish(self) -> None:
        import mlflow

        mlflow.end_run()
        self._run = None

    @property
    def run_id(self) -> str | None:
        if self._run is not None:
            return getattr(self._run.info, "run_id", None)
        return None

    def __enter__(self) -> EvaluationRecorder:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.finish()


__all__ = [
    "EvaluationRecorder",
    "MLflowEvaluationRecorder",
]
