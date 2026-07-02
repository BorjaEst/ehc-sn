"""Evaluation executor assembly for SeqMaze × HRM-v1."""

from __future__ import annotations

from ehc_sn.adapters.hrm import SeqMazeAdapterSettings
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
from ehc_sn.objectives.task.seqmaze import SeqMazeTaskEvaluatorConfig
from ehc_sn.traces import resolve_capture_profile

from .config import SeqMazeHRMV1ComponentConfigs, SeqMazeHRMV1ModelConfig
from .model import build_seqmaze_hrm_v1_model

# ---------------------------------------------------------------------------
# Recipe-owned constants
# ---------------------------------------------------------------------------

# SeqMaze does not have a dedicated EvaluationSourceProvider.  The provider
# ref is set to the MazeHard replay provider as a fallback; SeqMaze evaluation
# is model-internal and does not use a provider-based pipeline.
_SEQMAZE_HRM_V1_PROVIDER_REF = (
    "ehp_sn.tasks.mazehard.providers.MazeHardReplayProvider"
)

_SEQMAZE_HRM_V1_REGIME_ID = "seqmaze_diagnostic"
_SEQMAZE_HRM_V1_REGIME_KIND = "diagnostic"


def _resolve_seqmaze_hrm_v1_assembly(
    build_request: EvaluationBuildRequest,
) -> tuple[SeqMazeHRMV1ComponentConfigs, Path]:
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
    adapter = SeqMazeAdapterSettings.model_validate(assembly.adapter)
    controller = ACTControllerConfig.model_validate(assembly.controller or {})

    components = SeqMazeHRMV1ComponentConfigs(
        adapter=adapter,
        controller=controller,
        task_evaluator=SeqMazeTaskEvaluatorConfig(),
        scorer=ACTSupervisedScorerConfig(),
    )
    core_path = resolve_core_model_config_path(
        model_artifact=artifact_obj,
    )
    return components, core_path


def build_seqmaze_hrm_v1_evaluation_experiment(
    build_request: EvaluationBuildRequest,
) -> EvaluationExperiment:
    """Build a resolved evaluation experiment for SeqMaze × HRM-v1.

    Parameters
    ----------
    build_request:
        Narrow build request containing resolved case selection, runtime
        config, pair-specific evaluation options, and capture plan.
    """
    components, model_config_path = _resolve_seqmaze_hrm_v1_assembly(
        build_request
    )

    model_config = SeqMazeHRMV1ModelConfig(
        model_config_path=model_config_path,
        components=components,
    )

    executor = build_seqmaze_hrm_v1_model(
        model_config,
        execution=None,
    )
    trace_paradigm = getattr(executor, "_trace_paradigm", "act")

    # Resolve capture profile from the resolved capture plan.
    trace_spec = None
    if build_request.capture.profile != "metrics_only":
        trace_spec = resolve_capture_profile(
            paradigm=trace_paradigm,
            profile=build_request.capture.profile,
            profile_version=build_request.capture.profile_version,
            include=build_request.capture.fields,
            exclude=build_request.capture.exclude,
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
        ref=_SEQMAZE_HRM_V1_PROVIDER_REF,
        settings=settings,
        batch_size=build_request.runtime.batch_size,
    )

    return EvaluationExperiment(
        executor=executor,
        provider_spec=provider_spec,
        regime_id=_SEQMAZE_HRM_V1_REGIME_ID,
        regime_kind=_SEQMAZE_HRM_V1_REGIME_KIND,
        trace_spec=trace_spec,
        capture_profile=build_request.capture.profile,
        capture_max_cases=build_request.capture.max_cases,
        identity=EvaluationIdentity(
            task="seqmaze",
            model_family="hrm-v1",
            trace_paradigm=trace_paradigm,
        ),
    )


__all__ = ["build_seqmaze_hrm_v1_evaluation_experiment"]
