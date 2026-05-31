"""Report pipeline — checkpoint → eval artifacts → report run.

Provides ``ReportPipelineSpec`` and ``build_report_run_from_checkpoint``
which compose :func:`ehc_sn.eval.offline.run_offline_eval` and
:func:`ehc_sn.reporting.builder.build_report_run` into one orchestrated
workflow.

Boundary rule:
    ``pipelines`` may import from both ``ehc_sn.eval`` and ``ehc_sn.reporting``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from ehc_sn.eval.offline import run_offline_eval
from ehc_sn.reporting.builder import build_report_run
from ehc_sn.reporting.loaders import load_eval_artifact_manifest
from ehc_sn.reporting.reader import ReportRun, load_report_run
from ehc_sn.reporting.schema import (
    EvalArtifactReference,
    ReportSpec,
)


# ---------------------------------------------------------------------------
# Pipeline specification models
# ---------------------------------------------------------------------------


class OfflineEvalJobSpec(BaseModel, extra="forbid"):
    """Describes one evaluation job for the pipeline.

    Maps directly to :func:`run_offline_eval` parameters on a per-regime
    basis.
    """

    task: str = Field(..., min_length=1)
    regime_id: str = Field(..., min_length=1)
    regime_kind: Literal["diagnostic", "benchmark"]

    provider_ref: str = Field(..., min_length=1)
    provider_settings: dict[str, Any] = Field(default_factory=dict)
    executor_config_path: Path

    trace_keys: set[str] = Field(default_factory=set)
    output_name: str | None = Field(
        default=None,
        description=(
            "Directory name for the eval artifact under _eval_artifacts/. "
            "Defaults to '{task}_{regime_kind}_{regime_id}' when not set."
        ),
    )

    def artifact_dir_name(self) -> str:
        """Return the deterministic artifact subdirectory name."""
        if self.output_name:
            return self.output_name
        return f"{self.task}_{self.regime_kind}_{self.regime_id}"


class ReportPipelineSpec(BaseModel, extra="forbid"):
    """Orchestration spec: produce eval artifacts, then assemble a report.

    Fields
    ------
    eval_jobs:
        One or more evaluation jobs executed in order.
    report:
        The :class:`ReportSpec` that describes the report to assemble.
        ``report.eval_artifacts_root`` and ``report.figures.figures`` may
        be overridden at runtime by the orchestrator.
    """

    eval_jobs: list[OfflineEvalJobSpec] = Field(..., min_length=1)
    report: ReportSpec


# =============================================================================
# Pipeline orchestrator
# =============================================================================


def build_report_run_from_checkpoint(
    spec: ReportPipelineSpec,
    *,
    overwrite: bool = False,
    skip_figures: bool = False,
    device: str = "cpu",
    max_batches: int = 0,
) -> ReportRun:
    """Evaluate each job, then assemble the report run.

    1. Creates ``<output_dir>/_eval_artifacts/``.
    2. Runs :func:`run_offline_eval` for each job in *spec.eval_jobs*.
    3. Loads each produced eval-artifact manifest.
    4. Assembles a resolved :class:`ReportSpec` with the actual
       ``eval_artifacts_root`` and optional figure override.
    5. Calls :func:`build_report_run` with the produced artifact
       references via ``eval_artifacts=``.
    6. Returns the validated :class:`ReportRun`.

    Parameters
    ----------
    spec:
        Pipeline specification with eval jobs and report assembly config.
    overwrite:
        If ``True``, remove any existing report-run directory before
        writing.  Passed through to ``build_report_run``.
    skip_figures:
        If ``True``, clear ``report.figures.figures`` before assembly.
    device:
        PyTorch device for evaluation.  Defaults to ``"cpu"``.
    max_batches:
        Maximum provider batches per eval job.  ``0`` means all.

    Returns
    -------
    ReportRun
        Validated, loaded report-run object.
    """
    report_spec = spec.report

    # ---- Compute eval artifact staging root --------------------------------
    eval_root = report_spec.output_dir / "_eval_artifacts"
    eval_root.mkdir(parents=True, exist_ok=True)

    # ---- Phase 1: produce eval artifacts ----------------------------------
    produced_artifacts: list[EvalArtifactReference] = []

    for job in spec.eval_jobs:
        artifact_dir = eval_root / job.artifact_dir_name()

        run_offline_eval(
            model_family=report_spec.model_family,
            executor_config_path=job.executor_config_path,
            checkpoint_path=report_spec.checkpoint.path,
            task=job.task,
            provider_ref=job.provider_ref,
            provider_settings=job.provider_settings,
            regime_id=job.regime_id,
            regime_kind=job.regime_kind,  # type: ignore[arg-type]
            output_dir=artifact_dir,
            device=device,
            max_batches=max_batches,
            trace_keys=list(job.trace_keys) if job.trace_keys else None,
        )

        # Load the produced manifest to build a truthful reference.
        manifest = load_eval_artifact_manifest(artifact_dir)
        produced_artifacts.append(
            EvalArtifactReference(
                task=str(manifest["task"]),
                regime_id=str(manifest["regime_id"]),
                regime_kind=str(manifest["regime_kind"]),  # type: ignore[arg-type]
                path=artifact_dir.resolve(),
            )
        )

    # ---- Phase 2: resolve report spec -------------------------------------
    # The resolved spec records the actual eval_artifacts_root so
    # config.resolved.yaml is truthful.
    resolved = report_spec.model_copy(
        update={"eval_artifacts_root": eval_root},
        deep=True,
    )

    # Apply --skip-figures override if requested.
    if skip_figures and resolved.figures.figures:
        resolved.figures.figures = []

    # ---- Phase 3: assemble report -----------------------------------------
    build_report_run(
        resolved,
        eval_artifacts=produced_artifacts,
        overwrite=overwrite,
    )

    return load_report_run(resolved.output_dir)


# ---------------------------------------------------------------------------
__all__ = [
    "OfflineEvalJobSpec",
    "ReportPipelineSpec",
    "build_report_run_from_checkpoint",
]
