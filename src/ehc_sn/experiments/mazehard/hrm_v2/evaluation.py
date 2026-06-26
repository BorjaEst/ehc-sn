"""Evaluation executor assembly for MazeHard × HRM-v2."""

from __future__ import annotations

from ehc_sn.experiments._infra import (
    EvaluationExperiment,
    EvaluationIdentity,
    ProviderSpec,
)
from ehc_sn.lightning.modules.q_halting import QHaltingModule
from ehc_sn.tasks.mazehard.runtime import MazeHardRuntimeConfig
from ehc_sn.traces import resolve_capture_profile

from .config import MazeHardHRMV2EvaluationExperimentConfig
from .model import build_mazehard_hrm_v2_model


def build_mazehard_hrm_v2_evaluation_experiment(
    config: MazeHardHRMV2EvaluationExperimentConfig,
) -> EvaluationExperiment:
    """Build a resolved evaluation experiment for MazeHard × HRM-v2."""
    executor = build_mazehard_hrm_v2_model(
        config.model,
        execution=config.execution.to_runtime_config(),
    )
    trace_paradigm = getattr(executor, "_trace_paradigm", "rl")

    trace_spec = (
        resolve_capture_profile(
            paradigm=trace_paradigm,
            profile=config.capture.profile,
            profile_version=config.capture.profile_version,
            include=config.capture.include,
            exclude=config.capture.exclude,
            extra_fields=executor._bindings.trace_fields,
        )
        if config.capture.profile != "metrics_only"
        else None
    )

    return EvaluationExperiment(
        executor=executor,
        provider_spec=ProviderSpec(
            ref=config.provider.ref,
            settings=config.provider.settings,
            batch_size=config.provider.batch_size,
        ),
        regime_id=config.regime.id,
        regime_kind=config.regime.kind,
        trace_spec=trace_spec,
        capture_profile=config.capture.profile,
        capture_include=config.capture.include,
        capture_exclude=config.capture.exclude,
        capture_max_cases=config.capture.max_cases,
        identity=EvaluationIdentity(
            task="mazehard",
            model_family="hrm-v2",
            trace_paradigm=trace_paradigm,
        ),
    )


__all__ = ["build_mazehard_hrm_v2_evaluation_experiment"]
