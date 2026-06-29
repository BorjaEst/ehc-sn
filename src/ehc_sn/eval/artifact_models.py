"""Typed manifest models for evaluation and regime artifact bundles.

This module owns the typed in-memory representation of both evaluation-level
and regime-level artifact manifests.  It must not import from ``pandas``,
``lightning``, ``models``, ``tasks``, ``adapters``, ``experiments``,
``reporting``, or ``figures``.

Usage::

    from ehc_sn.eval.artifact_models import (
        EvaluationArtifactSet,
        EvaluationManifest,
        RegimeArtifactManifest,
        RegimeArtifactSet,
    )

    evaluation = EvaluationArtifactSet.load(Path("artifacts/eval/arena-tem-v1"))
    regime = evaluation.open_regime("diagnostic")
    regime.metrics  # -> Mapping[str, float]
    regime.read_cases()  # -> tuple[LoadedArtifactCase, ...]
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ehc_sn.eval.artifacts import (
    LoadedArtifactCase,
    UnsupportedEvaluationArtifactSchema,
    load_artifact_run_cases,
)
from ehc_sn.eval.artifacts import (
    load_evaluation_artifact_manifest as _load_regime_manifest_raw,
)

# =============================================================================
# Schemas
# =============================================================================

_EVALUATION_SCHEMA = "ehp_sn.eval.evaluation.v1"
"""Schema identifier for the evaluation-level manifest (``evaluation-manifest.json``)."""

_REGIME_SCHEMA_V1 = "ehc_sn.eval.artifact.v1"
"""Schema identifier for legacy v1 regime manifests."""

_MANIFEST_FILENAME = "manifest.json"
_EVALUATION_MANIFEST_FILENAME = "evaluation-manifest.json"
_METRICS_FILENAME = "metrics.json"
_SUCCESS_FILENAME = "_SUCCESS"

# =============================================================================
# Types
# =============================================================================

ValidationResult: type = object
"""Placeholder for ``ValidationResult`` — imported lazily where needed."""

# =============================================================================
# Manifest models
# =============================================================================


@dataclass(frozen=True)
class EvaluationIdentity:
    """Immutable identity fields for one evaluation invocation."""

    evaluation_id: str
    alias: str
    task: str
    model_family: str


@dataclass(frozen=True)
class ModelProvenance:
    """Provenance for the model used in an evaluation."""

    uri: str
    resolved_uri: str | None = None
    digest: str | None = None


@dataclass(frozen=True)
class DatasetProvenance:
    """Provenance for the dataset used in an evaluation."""

    uri: str
    split: str
    digest: str | None = None


@dataclass(frozen=True)
class EvaluationManifest:
    """Typed evaluation-level manifest — one per evaluation invocation.

    The ``regimes`` mapping keys are regime IDs (e.g. ``"diagnostic"``,
    ``"test"``) and values are relative paths from the evaluation root
    to the regime directory's ``manifest.json``.
    """

    schema: str
    identity: EvaluationIdentity
    model: ModelProvenance
    dataset: DatasetProvenance
    code_revision: str | None = None
    regimes: Mapping[str, str] = field(default_factory=dict)
    extensions: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RegimeArtifactManifest:
    """Typed regime-level manifest — one per evaluation regime.

    Semantically matches the existing regime manifest schema
    (``ehc_sn.eval.artifact.v1``).
    """

    schema: str
    status: str
    regime_id: str
    regime_kind: str
    phase_kind: str
    trigger_kind: str
    epoch: int
    step: int
    task: str | None = None
    capture: Mapping[str, object] | None = None
    summary: Mapping[str, object] = field(default_factory=dict)
    artifacts: Mapping[str, object] = field(default_factory=dict)
    extensions: Mapping[str, object] = field(default_factory=dict)


# =============================================================================
# Artifact set containers
# =============================================================================


@dataclass(frozen=True)
class EvaluationArtifactSet:
    """Typed handle to a completed evaluation invocation.

    Load via :meth:`EvaluationArtifactSet.load` or the ``load_evaluation``
    convenience function in ``ehc_sn.reporting``.

    Use :meth:`open_regime` to access a specific regime artifact bundle.
    """

    root: Path
    manifest: EvaluationManifest

    def open_regime(self, regime_id: str) -> RegimeArtifactSet:
        """Open a regime artifact bundle by its regime ID.

        Args:
            regime_id: Regime identifier (e.g. ``"diagnostic"``).

        Returns:
            A :class:`RegimeArtifactSet` for the requested regime.

        Raises:
            KeyError: If the regime is not listed in the evaluation manifest.
        """
        rel_path = self.manifest.regimes.get(regime_id)
        if rel_path is None:
            available = sorted(self.manifest.regimes.keys())
            raise KeyError(
                f"Regime {regime_id!r} not found in evaluation at "
                f"{self.root}. Available regimes: {available}"
            )
        regime_root = (self.root / rel_path).resolve()
        return RegimeArtifactSet.load(regime_root)

    @classmethod
    def load(cls, root: Path) -> EvaluationArtifactSet:
        """Load an evaluation-level artifact set from a directory.

        Requires ``evaluation-manifest.json`` in *root*.

        Args:
            root: Path to an evaluation artifact directory containing
                ``evaluation-manifest.json``.

        Returns:
            A fully typed :class:`EvaluationArtifactSet`.

        Raises:
            FileNotFoundError: If ``evaluation-manifest.json`` is missing.
            UnsupportedEvaluationArtifactSchema: If the schema field is
                unrecognised.
        """
        root = root.resolve()
        manifest_path = root / _EVALUATION_MANIFEST_FILENAME
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Missing {_EVALUATION_MANIFEST_FILENAME} under {root}. "
                "This directory is not a valid evaluation-level artifact root."
            )

        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        schema = raw.get("schema", "")
        if schema != _EVALUATION_SCHEMA:
            raise UnsupportedEvaluationArtifactSchema(
                expected=_EVALUATION_SCHEMA,
                actual=schema,
                path=manifest_path,
            )

        identity_raw = raw.get("identity", {})
        model_raw = raw.get("model", {})
        dataset_raw = raw.get("dataset", {})

        manifest = EvaluationManifest(
            schema=schema,
            identity=EvaluationIdentity(
                evaluation_id=identity_raw.get("evaluation_id", ""),
                alias=identity_raw.get("alias", ""),
                task=identity_raw.get("task", ""),
                model_family=identity_raw.get("model_family", ""),
            ),
            model=ModelProvenance(
                uri=model_raw.get("uri", ""),
                resolved_uri=model_raw.get("resolved_uri"),
                digest=model_raw.get("digest"),
            ),
            dataset=DatasetProvenance(
                uri=dataset_raw.get("uri", ""),
                split=dataset_raw.get("split", ""),
                digest=dataset_raw.get("digest"),
            ),
            code_revision=raw.get("code", {}).get("revision"),
            regimes=dict(raw.get("regimes", {})),
        )
        return cls(root=root, manifest=manifest)


# =============================================================================
_COUNT_KEYS = frozenset(
    {
        "n_cases",
        "n_samples",
        "n_sequence_completed",
        "n_sequence_eligible",
        "n_token_correct",
        "n_token_total",
        "n_traced_cases",
    }
)
"""Keys in ``manifest.summary`` that are operational counts, not scientific metrics."""


@dataclass(frozen=True)
class RegimeArtifactSet:
    """Typed handle to one completed evaluation regime.

    Load via :meth:`RegimeArtifactSet.load` or by calling
    ``EvaluationArtifactSet.open_regime()``.

    This container lazily defers case/trace loading to explicit reader
    methods — it does not eagerly load traces.
    """

    root: Path
    manifest: RegimeArtifactManifest
    metrics: Mapping[str, float] = field(default_factory=dict)

    def read_cases(self) -> tuple[LoadedArtifactCase, ...]:
        """Load all trace-bearing cases from this regime artifact.

        Returns:
            Tuple of :class:`LoadedArtifactCase` objects (same format as
            ``load_artifact_run_cases``).
        """
        return tuple(load_artifact_run_cases(self.root))

    @classmethod
    def load(cls, root: Path) -> RegimeArtifactSet:
        """Load a regime artifact set from a regime artifact directory.

        Works with both v1 (no ``metrics.json``) and v2 (with
        ``metrics.json``) regime artifacts.

        Args:
            root: Path to a regime artifact directory containing
                ``manifest.json`` and ``_SUCCESS``.

        Returns:
            A fully typed :class:`RegimeArtifactSet`.

        Raises:
            FileNotFoundError: If ``manifest.json`` is missing.
            RuntimeError: If ``_SUCCESS`` is missing.
            UnsupportedEvaluationArtifactSchema: If schema is unrecognised.
        """
        root = root.resolve()
        raw_manifest = _load_regime_manifest_raw(root)
        regime_manifest = _normalize_v1_regime_manifest(raw_manifest)

        # Load metrics — prefer metrics.json, fall back to summary.
        metrics_path = root / _METRICS_FILENAME
        if metrics_path.exists():
            raw_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics: dict[str, float] = {}
            for k, v in raw_metrics.items():
                if isinstance(v, (int, float)):
                    metrics[k] = float(v)
        else:
            # v1 fallback: extract scalar measurements from manifest.summary.
            metrics = _extract_metrics_from_summary(regime_manifest.summary)

        return cls(root=root, manifest=regime_manifest, metrics=metrics)


# =============================================================================
# Normalisation helpers
# =============================================================================


def _normalize_v1_regime_manifest(
    raw: dict[str, Any],
) -> RegimeArtifactManifest:
    """Normalize a raw regime manifest dict (v1 or v2) into a typed model.

    Args:
        raw: The raw manifest dict returned by
            ``load_evaluation_artifact_manifest``.

    Returns:
        A :class:`RegimeArtifactManifest` with all known fields mapped.
    """
    return RegimeArtifactManifest(
        schema=raw.get("schema", _REGIME_SCHEMA_V1),
        status=raw.get("status", "complete"),
        regime_id=raw.get("regime_id", "unknown"),
        regime_kind=raw.get("regime_kind", "diagnostic"),
        phase_kind=raw.get("phase_kind", "diag"),
        trigger_kind=raw.get("trigger_kind", "manual"),
        epoch=raw.get("epoch", 0),
        step=raw.get("step", 0),
        task=raw.get("task"),
        capture=raw.get("capture"),
        summary=raw.get("summary", {}),
        artifacts=raw.get("artifacts", {}),
    )


def _extract_metrics_from_summary(
    summary: Mapping[str, object],
) -> dict[str, float]:
    """Extract scalar scientific metrics from a regime summary dict.

    Filters out operational counts (``n_cases``, ``n_samples``, etc.)
    and returns only float-compatible values.
    """
    metrics: dict[str, float] = {}
    for k, v in summary.items():
        if k in _COUNT_KEYS:
            continue
        if isinstance(v, (int, float)):
            metrics[k] = float(v)
    return metrics


# =============================================================================
# Evaluation-manifest writer
# =============================================================================


def write_evaluation_manifest(
    root: Path,
    *,
    identity: EvaluationIdentity,
    model: ModelProvenance,
    dataset: DatasetProvenance,
    regimes: Mapping[str, str],
    code_revision: str | None = None,
) -> None:
    """Write an evaluation-level manifest (``evaluation-manifest.json``) to *root*.

    This is called once per evaluation invocation, after all regimes have
    been collected.  It is invocation provenance, not a scientific artifact.

    Args:
        root: Evaluation artifact root directory.
        identity: Evaluation identity fields.
        model: Model provenance.
        dataset: Dataset provenance.
        regimes: Mapping from regime ID to relative manifest path
            (e.g. ``{"diagnostic": "regimes/diagnostic/manifest.json"}``).
        code_revision: Optional VCS revision string.
    """
    manifest: dict[str, object] = {
        "schema": _EVALUATION_SCHEMA,
        "identity": {
            "evaluation_id": identity.evaluation_id,
            "alias": identity.alias,
            "task": identity.task,
            "model_family": identity.model_family,
        },
        "model": {
            "uri": model.uri,
            "resolved_uri": model.resolved_uri,
            "digest": model.digest,
        },
        "dataset": {
            "uri": dataset.uri,
            "split": dataset.split,
            "digest": dataset.digest,
        },
        "code": {
            "revision": code_revision,
        },
        "regimes": dict(regimes),
    }
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / _EVALUATION_MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


# =============================================================================
__all__ = [
    "DatasetProvenance",
    "EvaluationArtifactSet",
    "EvaluationIdentity",
    "EvaluationManifest",
    "LoadedArtifactCase",
    "ModelProvenance",
    "RegimeArtifactManifest",
    "RegimeArtifactSet",
    "UnsupportedEvaluationArtifactSchema",
    "write_evaluation_manifest",
]
