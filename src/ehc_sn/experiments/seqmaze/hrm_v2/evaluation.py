"""Evaluation executor assembly for SeqMaze × HRM-v2."""

from __future__ import annotations

from ehc_sn.experiments._infra import (
    EvaluationExperiment,
    EvaluationIdentity,
    ProviderSpec,
)
from ehc_sn.lightning.modules.actor_critic import ActorCriticModule
from ehc_sn.tasks.seqmaze.runtime import SeqMazeRuntimeConfig
from ehc_sn.traces import resolve_capture_profile

from .config import SeqMazeHRMV2EvaluationExperimentConfig
from .model import build_seqmaze_hrm_v2_model


def build_seqmaze_hrm_v2_evaluation_experiment(
    config: SeqMazeHRMV2EvaluationExperimentConfig,
) -> EvaluationExperiment:
    """Build a resolved evaluation experiment for SeqMaze × HRM-v2."""
    executor = build_seqmaze_hrm_v2_model(
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
        )
        if config.capture.profile != "metrics_only"
        else None
    )

    return EvaluationExperiment(
        executor=executor,
        provider_spec=ProviderSpec(
            ref=config.provider.ref, settings=config.provider.settings
        ),
        regime_id=config.regime.id,
        regime_kind=config.regime.kind,
        trace_spec=trace_spec,
        capture_profile=config.capture.profile,
        capture_include=config.capture.include,
        capture_exclude=config.capture.exclude,
        identity=EvaluationIdentity(
            task="seqmaze",
            model_family="hrm-v2",
            trace_paradigm=trace_paradigm,
        ),
    )


__all__ = ["build_seqmaze_hrm_v2_evaluation_experiment"]
