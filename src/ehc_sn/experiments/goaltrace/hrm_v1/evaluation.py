"""Evaluation executor assembly for Goaltrace × HRM-v1."""

from __future__ import annotations

from ehc_sn.experiments._infra import (
    EvaluationExperiment,
    EvaluationIdentity,
    ProviderSpec,
)
from ehc_sn.lightning.modules.act_supervised import ACTSupervisedModule
from ehc_sn.traces import resolve_capture_profile
from ehc_sn.traces.specs import CaptureParadigmBinding

from .config import GoaltraceHRMV1EvaluationExperimentConfig
from .model import build_goaltrace_hrm_v1_model


def build_goaltrace_hrm_v1_evaluation_experiment(
    config: GoaltraceHRMV1EvaluationExperimentConfig,
) -> EvaluationExperiment:
    """Build a resolved evaluation experiment for Goaltrace × HRM-v1."""
    executor = build_goaltrace_hrm_v1_model(
        config.model,
        execution=config.execution,
    )
    trace_paradigm = getattr(executor, "_trace_paradigm", "act")

    # Task-specific capture bindings for goaltrace prediction_reasoning.
    goaltrace_task_bindings = {
        "goaltrace": CaptureParadigmBinding(
            required=("goaltrace/firing_field",),
            optional=(
                "goaltrace/target_field",
                "goaltrace/node_mask",
                "goaltrace/observation_id",
            ),
        ),
    }

    trace_spec = (
        resolve_capture_profile(
            paradigm=trace_paradigm,
            profile=config.capture.profile,
            profile_version=config.capture.profile_version,
            include=config.capture.include,
            exclude=config.capture.exclude,
            task_bindings=(
                goaltrace_task_bindings
                if config.capture.profile != "metrics_only"
                else None
            ),
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
            task="goaltrace",
            model_family="hrm-v1",
            trace_paradigm=trace_paradigm,
        ),
    )


__all__ = ["build_goaltrace_hrm_v1_evaluation_experiment"]
