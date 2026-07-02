"""Evaluation executor assembly for Goaltrace × HRM-v1."""

from __future__ import annotations

from ehc_sn.adapters.hrm import GoaltraceHRMAdapterSettings
from ehc_sn.controllers.deliberation.act import ACTControllerConfig
from ehc_sn.evaluation.contracts import (
    EvaluationExperiment,
    EvaluationIdentity,
    ProviderSpec,
)
from ehc_sn.evaluation.invocation import EvaluationBuildRequest
from ehc_sn.experiments._infra import resolve_core_model_config_path
from ehc_sn.model_artifacts.assembly import ModelAssembly
from ehc_sn.objectives.composites.act import ACTSupervisedScorerConfig
from ehc_sn.objectives.task.goaltrace import GoaltraceTaskEvaluatorConfig
from ehc_sn.traces import resolve_capture_profile
from ehc_sn.traces.specs import CaptureParadigmBinding

from .config import GoaltraceHRMV1ComponentConfigs, GoaltraceHRMV1ModelConfig
from .model import build_goaltrace_hrm_v1_model

# ---------------------------------------------------------------------------
# Recipe-owned constants
# ---------------------------------------------------------------------------

_GOALTRACE_HRM_V1_PROVIDER_REF = (
    "ehp_sn.tasks.goaltrace.providers.GoaltraceReplayProvider"
)
"""Provider import path for Goaltrace.  Recipe-owned."""

_GOALTRACE_HRM_V1_REGIME_ID = "goaltrace_diagnostic"
_GOALTRACE_HRM_V1_REGIME_KIND = "diagnostic"


def _resolve_goaltrace_hrm_v1_assembly(
    build_request: EvaluationBuildRequest,
) -> tuple[GoaltraceHRMV1ComponentConfigs, Path]:
    """Resolve component configs and core model path from the artifact."""
    artifact_obj = build_request.model_artifact
    assembly = getattr(artifact_obj, "model_config", None)

    if not isinstance(assembly, ModelAssembly):
        from ehc_sn.experiments._infra import InvalidModelArtifactError

        raise InvalidModelArtifactError(
            "Model artifact does not contain a valid ModelAssembly. "
            "Assembly-format artifacts are required."
        )

    # Assembly-format artifact: read sections directly from ModelAssembly.
    adapter = GoaltraceHRMAdapterSettings.model_validate(assembly.adapter)
    controller = ACTControllerConfig.model_validate(assembly.controller or {})
    ip = assembly.inference_policy or {}
    task_evaluator = GoaltraceTaskEvaluatorConfig.model_validate(ip)

    components = GoaltraceHRMV1ComponentConfigs(
        adapter=adapter,
        controller=controller,
        task_evaluator=task_evaluator,
        scorer=ACTSupervisedScorerConfig(),
    )
    core_path = resolve_core_model_config_path(
        model_artifact=artifact_obj,
    )
    return components, core_path


def build_goaltrace_hrm_v1_evaluation_experiment(
    build_request: EvaluationBuildRequest,
) -> EvaluationExperiment:
    """Build a resolved evaluation experiment for Goaltrace × HRM-v1.

    Parameters
    ----------
    build_request:
        Narrow build request containing resolved case selection, runtime
        config, pair-specific evaluation options, and capture plan.
    """
    components, model_config_path = _resolve_goaltrace_hrm_v1_assembly(
        build_request
    )

    model_config = GoaltraceHRMV1ModelConfig(
        model_config_path=model_config_path,
        components=components,
    )

    executor = build_goaltrace_hrm_v1_model(
        model_config,
        execution=None,
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

    trace_spec = None
    if build_request.capture.profile != "metrics_only":
        trace_spec = resolve_capture_profile(
            paradigm=trace_paradigm,
            profile=build_request.capture.profile,
            profile_version=build_request.capture.profile_version,
            include=build_request.capture.fields,
            exclude=build_request.capture.exclude,
            task_bindings=goaltrace_task_bindings,
            extra_fields=executor._bindings.trace_fields,
        )

    # Build provider spec from recipe-owned provider + invocation-supplied
    # case selection and batch size.
    case_ids = list(build_request.cases.case_ids) or None
    settings: dict[str, object] = {
        "dataset_path": build_request.cases.dataset.name,
        "split": build_request.cases.split,
    }
    if case_ids:
        settings["n_cases"] = len(case_ids)
    elif build_request.cases.count is not None:
        settings["n_cases"] = build_request.cases.count
    provider_spec = ProviderSpec(
        ref=_GOALTRACE_HRM_V1_PROVIDER_REF,
        settings=settings,
        batch_size=build_request.runtime.batch_size,
    )

    return EvaluationExperiment(
        executor=executor,
        provider_spec=provider_spec,
        regime_id=_GOALTRACE_HRM_V1_REGIME_ID,
        regime_kind=_GOALTRACE_HRM_V1_REGIME_KIND,
        trace_spec=trace_spec,
        capture_profile=build_request.capture.profile,
        capture_max_cases=build_request.capture.max_cases,
        identity=EvaluationIdentity(
            task="goaltrace",
            model_family="hrm-v1",
            trace_paradigm=trace_paradigm,
        ),
    )


__all__ = ["build_goaltrace_hrm_v1_evaluation_experiment"]
