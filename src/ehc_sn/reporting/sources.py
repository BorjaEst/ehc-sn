"""Source resolution for evaluation artifacts — local and MLflow URIs.

Usage::

    from ehc_sn.reporting.sources import materialize_evaluation_source

    source = materialize_evaluation_source("artifacts/evaluation/tem-v1-arena")
    # source.backend == "local"
    # source.local_root == Path("artifacts/evaluation/tem-v1-arena").resolve()
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


# =============================================================================
@dataclass(frozen=True)
class MaterializedEvaluationSource:
    """Resolved evaluation source — local or downloaded.

    Attributes:
        input_uri: The original URI string supplied by the user.
        local_root: Absolute path to the local directory containing the
            evaluation artifact.
        backend: ``"local"`` for direct filesystem paths, ``"mlflow"``
            for artifacts downloaded from an MLflow tracking server.
        run_id: MLflow run ID, if resolved from an MLflow URI.
        artifact_uri: Original MLflow artifact URI, if applicable.
    """

    input_uri: str
    local_root: Path
    backend: Literal["local", "mlflow"]
    run_id: str | None = None
    artifact_uri: str | None = None


# =============================================================================
def materialize_evaluation_source(
    source_uri: str,
    *,
    destination: Path | None = None,
) -> MaterializedEvaluationSource:
    """Resolve *source_uri* to a local artifact directory.

    Supports two URI schemes:

    - **Local path**: returned directly after validation.
    - **``runs:/<run_id>/<path>``**: downloaded from MLflow and cached
      at *destination* (or a temporary directory if not provided).

    Args:
        source_uri: A local filesystem path or an MLflow artifact URI.
        destination: Optional local download cache directory for MLflow
            artifacts.  Ignored for local paths.

    Returns:
        A :class:`MaterializedEvaluationSource` with the resolved local
        directory.

    Raises:
        FileNotFoundError: If the local path does not exist.
    """
    uri = source_uri.strip()

    # --- Local path ----------------------------------------------------------
    local_path = Path(uri)
    if local_path.exists():
        return MaterializedEvaluationSource(
            input_uri=uri,
            local_root=local_path.resolve(),
            backend="local",
        )

    # --- runs:/ URI ----------------------------------------------------------
    if uri.startswith("runs:/"):
        return _materialize_mlflow_source(uri, destination=destination)

    # --- Other MLflow artifact URI -------------------------------------------
    if uri.startswith("models:/") or "mlflow" in uri.lower():
        return _materialize_mlflow_source(uri, destination=destination)

    # --- Not found locally and not a recognised remote scheme ----------------
    raise FileNotFoundError(
        f"Evaluation source not found: {source_uri!r}. "
        "Provide a local path to an evaluation artifact directory or a "
        "supported MLflow URI (runs:/<run_id>/<path>)."
    )


# =============================================================================
def _materialize_mlflow_source(
    uri: str,
    destination: Path | None,
) -> MaterializedEvaluationSource:
    """Download an MLflow artifact and return its local path.

    Requires ``mlflow`` to be installed.  Raises ``ImportError`` at call
    time if it is not available.
    """
    try:
        import mlflow
    except ImportError as exc:
        raise ImportError(
            "mlflow is required to resolve MLflow artifact URIs. "
            "Install with: pip install 'ehp-sn[mlflow]'"
        ) from exc

    # Determine the run ID and artifact path from the URI.
    if uri.startswith("runs:/"):
        # runs:/<run_id>/<path>
        parts = uri[len("runs:/") :].split("/", 1)
        run_id = parts[0]
        artifact_path = parts[1] if len(parts) > 1 else ""
    else:
        # Unsupported MLflow URI format — attempt to parse generically.
        raise ValueError(
            f"Unsupported MLflow URI format: {uri!r}. "
            "Use runs:/<run_id>/<path>."
        )

    if destination is None:
        import tempfile

        destination = Path(tempfile.mkdtemp(prefix="ehp-mlflow-source-"))

    destination.mkdir(parents=True, exist_ok=True)

    mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path=artifact_path or None,
        dst_path=str(destination),
    )

    return MaterializedEvaluationSource(
        input_uri=uri,
        local_root=destination.resolve(),
        backend="mlflow",
        run_id=run_id,
        artifact_uri=uri,
    )


# =============================================================================
__all__ = [
    "MaterializedEvaluationSource",
    "materialize_evaluation_source",
]
