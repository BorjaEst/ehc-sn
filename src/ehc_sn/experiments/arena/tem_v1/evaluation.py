"""Evaluation executor assembly for Arena × TEM-v1."""

from __future__ import annotations

from ehc_sn.experiments._infra import (
    EvaluationExperiment,
    EvaluationIdentity,
    ProviderSpec,
)
from ehc_sn.lightning.modules.variational_replay import (
    VariationalReplayModule,
)
from ehc_sn.traces import resolve_capture_profile

from .config import ArenaTEMV1EvaluationExperimentConfig
from .model import build_arena_tem_v1_model


def build_arena_tem_v1_evaluation_experiment(
    config: ArenaTEMV1EvaluationExperimentConfig,
) -> EvaluationExperiment:
    """Build a resolved evaluation experiment for Arena × TEM-v1."""
    executor = build_arena_tem_v1_model(
        config.model,
        execution=config.execution,
    )
    trace_paradigm = getattr(executor, "_trace_paradigm", "tem")

    trace_spec = (
        resolve_capture_profile(
            paradigm=trace_paradigm,
            profile=config.capture.profile,
            profile_version=config.capture.profile_version,
            include=config.capture.include,
            exclude=config.capture.exclude,
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
            task="arena",
            model_family="tem-v1",
            trace_paradigm=trace_paradigm,
        ),
    )


__all__ = ["build_arena_tem_v1_evaluation_experiment"]
