"""Evaluation executor assembly for Arena × TEM-v1."""

from __future__ import annotations

from ehc_sn.adapters.tem import ArenaTEMAdapterSettings
from ehc_sn.controllers.replay.trajectory import (
    ReplayTrajectoryControllerConfig,
)
from ehc_sn.evaluation.contracts import (
    EvaluationExperiment,
    EvaluationIdentity,
    ProviderSpec,
)
from ehc_sn.evaluation.invocation import EvaluationBuildRequest
from ehc_sn.experiments._infra import resolve_core_model_config_path
from ehc_sn.model_artifacts.assembly import ModelAssembly
from ehc_sn.objectives.composites.tem import TEMObjectiveConfig
from ehc_sn.traces import resolve_capture_profile

from .config import ArenaTEMV1ComponentConfigs, ArenaTEMV1ModelConfig
from .model import build_arena_tem_v1_model

# ---------------------------------------------------------------------------
# Recipe-owned constants
# ---------------------------------------------------------------------------

_ARENA_TEM_V1_PROVIDER_REF = "ehp_sn.tasks.arena.providers.ArenaReplayProvider"
"""Provider import path for Arena.  Recipe-owned."""


def _resolve_arena_tem_v1_assembly(
    build_request: EvaluationBuildRequest,
) -> tuple[ArenaTEMV1ComponentConfigs, Path]:
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
    adapter = ArenaTEMAdapterSettings.model_validate(assembly.adapter)
    controller = ReplayTrajectoryControllerConfig.model_validate(
        assembly.controller or {}
    )
    components = ArenaTEMV1ComponentConfigs(
        adapter=adapter,
        controller=controller,
        objective=TEMObjectiveConfig(),
    )
    core_path = resolve_core_model_config_path(
        model_artifact=artifact_obj,
    )
    return components, core_path


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_arena_tem_v1_evaluation_experiment(
    build_request: EvaluationBuildRequest,
) -> EvaluationExperiment:
    """Build a resolved evaluation experiment for Arena × TEM-v1.

    This builder constructs the executor from recipe-owned constants
    and invocation-supplied runtime/capture values.

    Parameters
    ----------
    build_request:
        Narrow build request containing resolved model reference, case
        selection, runtime config, pair-specific evaluation options, and
        capture plan.

    Returns
    -------
    EvaluationExperiment
        Fully resolved experiment ready for execution.
    """
    components, model_config_path = _resolve_arena_tem_v1_assembly(
        build_request
    )

    model_config = ArenaTEMV1ModelConfig(
        model_config_path=model_config_path,
        components=components,
    )

    executor = build_arena_tem_v1_model(model_config)
    trace_paradigm = getattr(executor, "_trace_paradigm", "tem")

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
        ref=_ARENA_TEM_V1_PROVIDER_REF,
        settings=settings,
        batch_size=build_request.runtime.batch_size,
    )

    return EvaluationExperiment(
        executor=executor,
        provider_spec=provider_spec,
        regime_id="arena_evaluation",
        regime_kind="diagnostic",
        trace_spec=trace_spec,
        capture_profile=build_request.capture.profile,
        capture_max_cases=build_request.capture.max_cases,
        identity=EvaluationIdentity(
            task="arena",
            model_family="tem-v1",
            trace_paradigm=trace_paradigm,
        ),
    )


__all__ = ["build_arena_tem_v1_evaluation_experiment"]
