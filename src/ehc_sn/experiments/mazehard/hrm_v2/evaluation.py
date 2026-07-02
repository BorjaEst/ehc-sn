"""Evaluation executor assembly for MazeHard × HRM-v2."""

from __future__ import annotations

from ehc_sn.adapters.hrm import MazeHardHRMAdapterSettings
from ehc_sn.controllers.deliberation.q_halting import (
    DeliberationQHaltingControllerConfig,
)
from ehc_sn.evaluation.contracts import (
    EvaluationExperiment,
    EvaluationIdentity,
    ProviderSpec,
)
from ehc_sn.evaluation.invocation import EvaluationBuildRequest
from ehc_sn.experiments._infra import resolve_core_model_config_path
from ehc_sn.model_artifacts.assembly import ModelAssembly
from ehc_sn.objectives.composites.hybrid_rl import HybridRLLossConfig
from ehc_sn.traces import resolve_capture_profile

from .config import (
    MazeHardDeliberationConfig,
    MazeHardHRMV2ComponentConfigs,
    MazeHardHRMV2ModelConfig,
)
from .model import build_mazehard_hrm_v2_model

# ---------------------------------------------------------------------------
# Recipe-owned constants
# ---------------------------------------------------------------------------

_MAZEHARD_HRM_V2_PROVIDER_REF = (
    "ehp_sn.tasks.mazehard.providers.MazeHardReplayProvider"
)
"""Provider import path for MazeHard.  Recipe-owned."""

_MAZEHARD_HRM_V2_REGIME_ID = "mazehard_diagnostic"
_MAZEHARD_HRM_V2_REGIME_KIND = "diagnostic"


def _resolve_mazehard_hrm_v2_assembly(
    build_request: EvaluationBuildRequest,
) -> tuple[MazeHardHRMV2ComponentConfigs, Path]:
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
    adapter = MazeHardHRMAdapterSettings.model_validate(assembly.adapter)
    controller = DeliberationQHaltingControllerConfig.model_validate(
        assembly.controller or {}
    )

    components = MazeHardHRMV2ComponentConfigs(
        adapter=adapter,
        controller=controller,
        objective=HybridRLLossConfig(),
    )
    core_path = resolve_core_model_config_path(
        model_artifact=artifact_obj,
    )
    return components, core_path


def build_mazehard_hrm_v2_evaluation_experiment(
    build_request: EvaluationBuildRequest,
) -> EvaluationExperiment:
    """Build a resolved evaluation experiment for MazeHard × HRM-v2.

    Parameters
    ----------
    build_request:
        Narrow build request containing resolved case selection, runtime
        config, pair-specific evaluation options, and capture plan.
    """
    components, model_config_path = _resolve_mazehard_hrm_v2_assembly(
        build_request
    )

    model_config = MazeHardHRMV2ModelConfig(
        model_config_path=model_config_path,
        components=components,
    )

    deliberation = MazeHardDeliberationConfig()

    executor = build_mazehard_hrm_v2_model(
        model_config,
        execution=deliberation.to_runtime_config(),
    )
    trace_paradigm = getattr(executor, "_trace_paradigm", "rl")

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
        ref=_MAZEHARD_HRM_V2_PROVIDER_REF,
        settings=settings,
        batch_size=build_request.runtime.batch_size,
    )

    return EvaluationExperiment(
        executor=executor,
        provider_spec=provider_spec,
        regime_id=_MAZEHARD_HRM_V2_REGIME_ID,
        regime_kind=_MAZEHARD_HRM_V2_REGIME_KIND,
        trace_spec=trace_spec,
        capture_profile=build_request.capture.profile,
        capture_max_cases=build_request.capture.max_cases,
        identity=EvaluationIdentity(
            task="mazehard",
            model_family="hrm-v2",
            trace_paradigm=trace_paradigm,
        ),
    )


__all__ = ["build_mazehard_hrm_v2_evaluation_experiment"]
