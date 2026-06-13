"""Shared benchmark-owned runner for ready benchmark tracks only."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping

import torch

from ehc_sn.tasks.arena.evaluation import ArenaScoreReport
from ehc_sn.tasks.mazehard.evaluation import MazeHardScoreReport
from ehc_sn.tasks.scoring import scoring_spec_for_task

# =============================================================================
READY_TRACKS: tuple[str, ...] = (
    "arena-struct",
    "mazehard-delib",
)

_TRACK_ALIASES: dict[str, str] = {
    "b0": "mazehard-delib",
    "b0-mazehard": "mazehard-delib",
}

_TRACK_TO_TASK_FAMILY: dict[str, str] = {
    "arena-struct": "arena",
    "mazehard-delib": "mazehard",
}

_TRACK_TO_CLAIM_FAMILY: dict[str, str] = {
    "arena-struct": "structural_representation",
    "mazehard-delib": "deliberative_reasoning",
}

_TRACK_TO_SECONDARY_METRICS: dict[str, tuple[str, ...]] = {
    "arena-struct": (
        "accuracy_all",
        "correct_all",
        "count_all",
        "correct_revisit",
        "count_revisit",
    ),
    "mazehard-delib": (
        "sequences_accuracy",
        "tokens_accuracy",
    ),
}

_TRACK_MODEL_SUPPORT: dict[str, set[str]] = {
    "arena-struct": {"tem-v1", "tem-v2", "ehp-v1"},
    "mazehard-delib": {"hrm-v1", "hrm-v2", "ehp-v1"},
}


# =============================================================================
@dataclass(frozen=True)
class TrackDefinition:
    """Canonical metadata for one benchmark track."""

    track_id: str
    task_family: str
    claim_family: str
    compared_artifact_type: str
    readiness_state: str
    secondary_metrics: tuple[str, ...]
    supported_models: tuple[str, ...]


# =============================================================================
def build_track_report(  # ----------------------------------------------------
    track_id: str,
    model_family: str,
    score_report: ArenaScoreReport | MazeHardScoreReport,
    *,
    primary_metric: str,
    fixed_recipe: str,
    seed_count: int = 1,
    ood_slice: str | None = None,
) -> dict[str, object]:
    """Build the canonical benchmark report payload for one ready track."""
    if seed_count < 1:
        raise ValueError(f"seed_count must be >= 1, got {seed_count}.")

    track = _resolve_track_definition(track_id)
    canonical_model = _validate_model_support(track, model_family)
    metrics = _metrics_from_score(track, score_report)

    report: dict[str, object] = {
        "track_id": track.track_id,
        "claim_family": track.claim_family,
        "compared_artifact_type": track.compared_artifact_type,
        "readiness_state": track.readiness_state,
        "task_family": track.task_family,
        "model_family": canonical_model,
        "fixed_recipe": fixed_recipe,
        "primary_metric": {
            "name": primary_metric,
            "value": metrics[primary_metric],
        },
        "secondary_metrics": {
            metric_name: metrics[metric_name]
            for metric_name in track.secondary_metrics
        },
        "seed_count": seed_count,
        "ood_slice": ood_slice,
        "scores": metrics,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    return report


# =============================================================================
def build_track_report_from_seed_scores(
    track_id: str,
    model_family: str,
    seed_scores: tuple[ArenaScoreReport | MazeHardScoreReport, ...],
    *,
    primary_metric: str,
    fixed_recipe: str,
    ood_slice: str | None = None,
) -> dict[str, object]:
    """Build one benchmark report from concrete per-seed score reports."""
    if not seed_scores:
        raise ValueError("seed_scores must contain at least one score report.")

    track = _resolve_track_definition(track_id)
    # Validate that the recipe's primary_metric is a known benchmark-eligible
    # metric for this track's task family.
    scoring_spec_for_task(track.task_family).require_benchmark_metric(
        primary_metric
    )
    canonical_model = _validate_model_support(track, model_family)
    per_seed_metrics = [
        _metrics_from_score(track, seed_score) for seed_score in seed_scores
    ]
    metric_names = (primary_metric, *track.secondary_metrics)
    metric_summaries = {
        metric_name: _summarize_seed_values(
            tuple(
                float(seed_metrics[metric_name])
                for seed_metrics in per_seed_metrics
            )
        )
        for metric_name in metric_names
    }

    seed_count = len(seed_scores)
    canonical_ready = seed_count == 5

    report: dict[str, object] = {
        "track_id": track.track_id,
        "claim_family": track.claim_family,
        "compared_artifact_type": track.compared_artifact_type,
        "readiness_state": track.readiness_state,
        "task_family": track.task_family,
        "model_family": canonical_model,
        "fixed_recipe": fixed_recipe,
        "primary_metric": {
            "name": primary_metric,
            "value": metric_summaries[primary_metric]["mean"],
            **metric_summaries[primary_metric],
        },
        "secondary_metrics": {
            metric_name: {
                "value": metric_summaries[metric_name]["mean"],
                **metric_summaries[metric_name],
            }
            for metric_name in track.secondary_metrics
        },
        "seed_count": seed_count,
        "ood_slice": ood_slice,
        "scores": {
            metric_name: metric_summaries[metric_name]["mean"]
            for metric_name in metric_names
        },
        "seed_scatter": {
            metric_name: metric_summaries[metric_name]["per_seed"]
            for metric_name in metric_names
        },
        "canonical_ready_track_evidence": canonical_ready,
        "canonical_ready_track_requirement": "requires exactly 5 seeds",
        "canonical_ready_track_note": (
            None
            if canonical_ready
            else "Non-canonical ready-track evidence: report uses fewer than 5 seeds."
        ),
        "generated_at": datetime.now(UTC).isoformat(),
    }
    return report


# =============================================================================
def resolve_track_id(  # ------------------------------------------------------
    track_id: str,
) -> str:
    """Resolve a canonical ready-track id or raise a fast, explicit error."""
    normalized = _normalize_name(track_id)
    resolved = _TRACK_ALIASES.get(normalized, normalized)
    if resolved not in READY_TRACKS:
        supported = ", ".join(READY_TRACKS)
        aliases = ", ".join(sorted(_TRACK_ALIASES))
        raise ValueError(
            "Unsupported benchmark track "
            f"{track_id!r}. Ready tracks: {supported}. Aliases: {aliases}."
        )
    return resolved


# =============================================================================
def build_score_report(  # ----------------------------------------------------
    track_id: str,
    payload: Mapping[str, float | int],
) -> ArenaScoreReport | MazeHardScoreReport:
    """Build a task-owned score report object from scalar payload values."""
    canonical_track = resolve_track_id(track_id)
    if canonical_track == "arena-struct":
        values = _coerce_score_payload(
            canonical_track,
            payload,
            required_fields=(
                "accuracy_all",
                "accuracy_revisit",
                "correct_all",
                "count_all",
                "correct_revisit",
                "count_revisit",
            ),
        )
        return ArenaScoreReport(
            accuracy_all=torch.tensor(values["accuracy_all"]),
            accuracy_revisit=torch.tensor(values["accuracy_revisit"]),
            correct_all=torch.tensor(values["correct_all"]),
            count_all=torch.tensor(values["count_all"]),
            correct_revisit=torch.tensor(values["correct_revisit"]),
            count_revisit=torch.tensor(values["count_revisit"]),
        )
    values = _coerce_score_payload(
        canonical_track,
        payload,
        required_fields=(
            "tokens_accuracy",
            "sequences_accuracy",
            "sequences_exact",
        ),
    )
    return MazeHardScoreReport(
        tokens_accuracy=torch.tensor(values["tokens_accuracy"]),
        sequences_accuracy=torch.tensor(values["sequences_accuracy"]),
        sequences_exact=torch.tensor(values["sequences_exact"]),
    )


# =============================================================================
def write_track_report(  # ----------------------------------------------------
    report: Mapping[str, object],
    output_path: Path,
) -> Path:
    """Write one canonical benchmark report JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(dict(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


# =============================================================================
def _normalize_name(  # -------------------------------------------------------
    raw_name: str,
) -> str:
    """Normalize a track or model family name to a canonical form."""
    return raw_name.strip().lower().replace("_", "-")


# =============================================================================
def _resolve_track_definition(  # ---------------------------------------------
    track_id: str,
) -> TrackDefinition:
    """Resolve a canonical TrackDefinition for one ready track or raise a
    fast, explicit error.
    """
    canonical_track = resolve_track_id(track_id)
    return TrackDefinition(
        track_id=canonical_track,
        task_family=_TRACK_TO_TASK_FAMILY[canonical_track],
        claim_family=_TRACK_TO_CLAIM_FAMILY[canonical_track],
        compared_artifact_type="within-task architecture comparison",
        readiness_state="ready",
        secondary_metrics=_TRACK_TO_SECONDARY_METRICS[canonical_track],
        supported_models=tuple(sorted(_TRACK_MODEL_SUPPORT[canonical_track])),
    )


# =============================================================================
def _normalize_model_family(  # -----------------------------------------------
    model_family: str,
) -> str:
    """Normalize a model family name to a canonical form."""
    return _normalize_name(model_family)


# =============================================================================
def _validate_model_support(  # -----------------------------------------------
    track: TrackDefinition,
    model_family: str,
) -> str:
    """Validate that the given model family is supported for the track and
    return the canonical model family name, or raise a fast, explicit error.
    """
    canonical_model = _normalize_model_family(model_family)
    if canonical_model not in _TRACK_MODEL_SUPPORT[track.track_id]:
        supported = ", ".join(track.supported_models)
        raise ValueError(
            "Unsupported model-task benchmark pairing: "
            f"track={track.track_id!r}, model_family={model_family!r}. "
            f"Supported models for this track: {supported}."
        )
    return canonical_model


# =============================================================================
def _to_float(  # -------------------------------------------------------------
    value: torch.Tensor,
) -> float:
    """Coerce a single-value torch.Tensor to a float."""
    return float(value.detach().cpu().item())


# =============================================================================
def _coerce_score_payload(  # -------------------------------------------------
    track_id: str,
    payload: Mapping[str, float | int],
    *,
    required_fields: tuple[str, ...],
) -> dict[str, float]:
    """Coerce and validate a raw score payload mapping field names to numeric
    values, for one track's score report construction. Raise a fast, explicit
    error if the payload is invalid.
    """
    if not isinstance(payload, MappingABC):
        raise ValueError(
            "Invalid score payload for track "
            f"{track_id!r}: payload must be an object mapping field names "
            "to numeric values."
        )

    missing_fields = [
        field for field in required_fields if field not in payload
    ]
    coerced: dict[str, float] = {}
    invalid_fields: list[str] = []
    for field in required_fields:
        if field not in payload:
            continue
        value = payload[field]
        try:
            coerced[field] = float(value)
        except (TypeError, ValueError):
            invalid_fields.append(f"{field}={value!r}")

    if missing_fields or invalid_fields:
        details: list[str] = []
        if missing_fields:
            details.append("missing fields: " + ", ".join(missing_fields))
        if invalid_fields:
            details.append(
                "invalid numeric fields: " + ", ".join(invalid_fields)
            )
        raise ValueError(
            "Invalid score payload for track "
            f"{track_id!r}: " + "; ".join(details) + "."
        )

    return coerced


# =============================================================================
def _summarize_seed_values(values: tuple[float, ...]) -> dict[str, object]:
    """Return mean, 95% CI half-width, and per-seed values."""
    if not values:
        raise ValueError("Cannot summarize empty seed values.")
    mean = sum(values) / len(values)
    if len(values) == 1:
        ci95 = 0.0
    else:
        variance = sum((value - mean) ** 2 for value in values) / (
            len(values) - 1
        )
        ci95 = 1.96 * math.sqrt(variance / len(values))
    return {
        "mean": float(mean),
        "ci95": float(ci95),
        "per_seed": [float(value) for value in values],
    }


# =============================================================================
def _metrics_from_score(  # ---------------------------------------------------
    track: TrackDefinition,
    score_report: ArenaScoreReport | MazeHardScoreReport,
) -> dict[str, float]:
    """Extract a mapping of metric names to float values from a task-owned
    score report object, for one track's report construction. Raise a fast,
    explicit error if the score report is invalid for the track.
    """
    if track.track_id == "arena-struct":
        if not isinstance(score_report, ArenaScoreReport):
            raise TypeError(
                "Arena-Struct benchmark requires "
                "ehc_sn.tasks.arena.evaluation.ArenaScoreReport."
            )
        return {
            "accuracy_revisit": _to_float(score_report.accuracy_revisit),
            "accuracy_all": _to_float(score_report.accuracy_all),
            "correct_all": _to_float(score_report.correct_all),
            "count_all": _to_float(score_report.count_all),
            "correct_revisit": _to_float(score_report.correct_revisit),
            "count_revisit": _to_float(score_report.count_revisit),
        }

    if not isinstance(score_report, MazeHardScoreReport):
        raise TypeError(
            "MazeHard-Delib benchmark requires "
            "ehc_sn.tasks.mazehard.evaluation.MazeHardScoreReport."
        )
    return {
        "sequences_exact": _to_float(score_report.sequences_exact),
        "sequences_accuracy": _to_float(score_report.sequences_accuracy),
        "tokens_accuracy": _to_float(score_report.tokens_accuracy),
    }


# =============================================================================
# =============================================================================
def task_family_for_track(track_id: str) -> str:
    """Return the task-family name for a ready-track id, or raise."""
    canonical = resolve_track_id(track_id)
    return _TRACK_TO_TASK_FAMILY[canonical]


# =============================================================================
__all__ = [
    "READY_TRACKS",
    "build_score_report",
    "build_track_report",
    "build_track_report_from_seed_scores",
    "resolve_track_id",
    "task_family_for_track",
    "write_track_report",
]
