"""Evaluation executor assembly for Routebind × HRM-v1."""

from __future__ import annotations

from ehc_sn.experiments._infra import (
    EvaluationExperiment,
    EvaluationIdentity,
    ProviderSpec,
)
from ehc_sn.lightning.modules.act_supervised import ACTSupervisedModule
from ehc_sn.traces import resolve_capture_profile

from .config import RoutebindHRMV1EvaluationExperimentConfig
from .model import build_routebind_hrm_v1_model


def build_routebind_hrm_v1_evaluation_experiment(
    config: RoutebindHRMV1EvaluationExperimentConfig,
) -> EvaluationExperiment:
    """Build a resolved evaluation experiment for Routebind × HRM-v1."""
    executor = build_routebind_hrm_v1_model(
        config.model,
        execution=config.execution,
    )
    trace_paradigm = getattr(executor, "_trace_paradigm", "act")

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
            task="routebind",
            model_family="hrm-v1",
            trace_paradigm=trace_paradigm,
        ),
    )


__all__ = ["build_routebind_hrm_v1_evaluation_experiment"]
