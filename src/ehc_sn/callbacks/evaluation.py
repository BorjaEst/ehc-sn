"""Scheduled named replay-evaluation callback.

This callback is orchestration-only:
- resolves task-owned providers from provider_ref,
- schedules named regimes on trainer cadence,
- delegates execution through the family-owned execute_evaluation_batch seam
  via ehc_sn.eval helpers,
- logs namespaced summary metrics,
- persists returned trace/artifact payloads.

Standard report generation is offline from persisted run directories via
ehc_sn.eval.reports. Callback-side figure rendering is optional
diagnostic routing and is not the canonical report workflow.
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from numbers import Real
from pathlib import Path
from typing import Any, Literal

import lightning.pytorch as pl
import torch
from lightning.pytorch import LightningModule, Trainer
from pydantic import BaseModel, Field, field_validator, model_validator

from ehc_sn.eval import (
    EvaluationRegimeResult,
    EvaluationTraceRequest,
    iter_evaluation_regime,
)
from ehc_sn.eval.artifacts import (
    persist_regime_artifact_bundle,
    resolve_provider,
)
from ehc_sn.eval.contracts import EvaluationCaseBatch
from ehc_sn.eval.render import render_regime_preview_figures
from ehc_sn.figures import REGISTRY, FigureContext, list_figures
from ehc_sn.traces import build_trace_spec


# =============================================================================
class EvaluationScheduleSettings(BaseModel, extra="forbid"):
    """Per-regime schedule settings for callback-owned triggering."""

    every_n_epochs: int = Field(
        default=0,
        ge=0,
        description="Run this regime every N validation epochs. 0 disables.",
    )
    every_n_steps: int = Field(
        default=0,
        ge=0,
        description="Run this regime every N training steps. 0 disables.",
    )
    max_batches: int = Field(
        default=0,
        ge=0,
        description="Maximum provider batches per run. 0 means no cap.",
    )

    @model_validator(mode="after")
    def _validate_non_empty_schedule(  # -------------------------------------
        self,
    ) -> "EvaluationScheduleSettings":
        """Validate that at least one cadence is enabled."""
        if self.every_n_epochs == 0 and self.every_n_steps == 0:
            raise ValueError(
                "At least one schedule cadence must be enabled: "
                "every_n_epochs > 0 or every_n_steps > 0."
            )
        return self


# =============================================================================
class EvaluationTraceRequestSettings(BaseModel, extra="forbid"):
    """Trace request settings for one named regime."""

    enabled: bool = Field(
        default=False,
        description="Whether to request trace materialization for this regime.",
    )
    trace_keys: list[str] = Field(
        default_factory=list,
        description="Optional semantic trace keys to request from the model family.",
    )
    figure_names: list[str] = Field(
        default_factory=list,
        description=(
            "Optional registered figure names whose trace-key requirements are "
            "added to this regime request. Supports report and diagnostic "
            "figures for offline bundle production."
        ),
    )

    @model_validator(mode="after")
    def _validate_figure_names(self) -> "EvaluationTraceRequestSettings":
        """Validate declared figure names against the registry."""
        if not self.figure_names:
            return self
        available = set(list_figures())
        missing = [name for name in self.figure_names if name not in available]
        if missing:
            unknown = ", ".join(sorted(set(missing)))
            raise ValueError(
                "trace_request.figure_names contains unknown names: "
                f"{unknown}."
            )
        return self


# =============================================================================
class EvaluationFigureRequestSettings(BaseModel, extra="forbid"):
    """Optional callback-local diagnostic figure rendering settings."""

    enabled: bool = Field(
        default=False,
        description=(
            "Whether to render configured diagnostic figures during callback "
            "execution for this regime. Offline reporting from persisted "
            "bundles remains the standard report path."
        ),
    )
    figures: list[str] = Field(
        default_factory=list,
        description="Registered figure names to render from each available case trace.",
    )
    max_cases: int = Field(
        default=2,
        ge=1,
        description="Maximum number of regime case results to render per run.",
    )
    save_pdf: bool = Field(
        default=True,
        description="Persist rendered figures as PDF files.",
    )
    save_png: bool = Field(
        default=False,
        description="Persist rendered figures as PNG files.",
    )
    png_dpi: int = Field(
        default=160,
        ge=1,
        description="PNG export DPI used when save_png is enabled.",
    )
    failure_policy: Literal["warn", "raise"] = Field(
        default="warn",
        description="Figure failure handling policy: warn-and-continue or fail-fast.",
    )

    @model_validator(mode="after")
    def _validate_request(self) -> "EvaluationFigureRequestSettings":
        """Validate enabled request shape and registered figure names."""
        if not self.enabled:
            return self
        if not self.figures:
            raise ValueError(
                "figure_request.enabled=true requires a non-empty figures list."
            )
        if not self.save_pdf and not self.save_png:
            raise ValueError(
                "figure_request must enable at least one sink: save_pdf or save_png."
            )
        available = set(list_figures())
        missing = [name for name in self.figures if name not in available]
        if missing:
            unknown = ", ".join(sorted(set(missing)))
            raise ValueError(
                "figure_request.figures contains unknown names: " f"{unknown}."
            )
        non_diagnostic = [
            name
            for name in self.figures
            if REGISTRY.get(name).kind != "diagnostic"
        ]
        if non_diagnostic:
            blocked = ", ".join(sorted(set(non_diagnostic)))
            raise ValueError(
                "figure_request.figures only supports diagnostic figures for "
                f"online rendering: {blocked}."
            )

        offline_only = [
            name
            for name in self.figures
            if REGISTRY.get(name).input_contract == "offline_artifact"
        ]
        if offline_only:
            blocked = ", ".join(sorted(set(offline_only)))
            raise ValueError(
                "figure_request.figures contains offline-only figures: "
                f"{blocked}. "
                "These are for the offline report pipeline."
            )
        return self


# =============================================================================
class EvaluationRegimeSettings(BaseModel, extra="forbid"):
    """One named evaluation regime configuration."""

    task: str = Field(
        ...,
        min_length=1,
        description="Canonical task-family identifier (e.g. ``arena``).",
    )
    regime_id: str = Field(
        ...,
        min_length=1,
        description="Unique regime identifier used in metric and artifact namespaces.",
    )
    regime_kind: Literal["diagnostic", "benchmark"] = Field(
        default="diagnostic",
        description="Top-level namespace tier for logs and artifacts.",
    )
    provider_ref: str = Field(
        ...,
        min_length=1,
        description="Dotted import path to an EvaluationSourceProvider class.",
    )
    provider_settings: dict[str, Any] = Field(
        default_factory=dict,
        description="Keyword settings forwarded to the provider constructor.",
    )
    schedule: EvaluationScheduleSettings = Field(
        default_factory=EvaluationScheduleSettings,
        description="Per-regime callback cadence and max-batch cap.",
    )
    trace_request: EvaluationTraceRequestSettings = Field(
        default_factory=EvaluationTraceRequestSettings,
        description="Optional trace materialization request for this regime.",
    )
    figure_request: EvaluationFigureRequestSettings = Field(
        default_factory=EvaluationFigureRequestSettings,
        description="Optional online figure routing for this regime.",
    )

    @field_validator("regime_kind", mode="before")
    @classmethod
    def _normalize_regime_kind_value(
        cls,
        value: Any,
    ) -> Literal["diagnostic", "benchmark"]:
        """Normalize to canonical regime_kind labels."""
        if value in {"diagnostic", "benchmark"}:
            return value
        raise ValueError(
            "regime_kind must be one of: 'diagnostic', 'benchmark'."
        )


# =============================================================================
class EvaluationRegimesCallbackSettings(BaseModel, extra="forbid"):
    """Settings container for scheduled named evaluation regimes."""

    regimes: list[EvaluationRegimeSettings] = Field(
        default_factory=list,
        description="Named evaluation regimes to schedule and execute.",
    )
    rank_zero_only: bool = Field(
        default=True,
        description="Run scheduled evaluation only on global rank zero.",
    )
    persist_artifacts: bool = Field(
        default=True,
        description="Persist returned regime artifacts (trace payloads + summary).",
    )
    output_subdir: str = Field(
        default="eval_regimes",
        description="Subdirectory name under trainer log dir for regime artifacts.",
    )

    @model_validator(mode="after")
    def _validate_regimes(  # -----------------------------------------------
        self,
    ) -> "EvaluationRegimesCallbackSettings":
        """Validate that configured regimes are non-empty and unique."""
        if not self.regimes:
            raise ValueError("eval_regimes.regimes must not be empty.")
        seen: set[str] = set()
        for regime in self.regimes:
            if regime.regime_id in seen:
                raise ValueError(
                    "eval_regimes.regimes contains duplicate regime_id values: "
                    f"{regime.regime_id!r}."
                )
            seen.add(regime.regime_id)
        return self


# =============================================================================
class EvaluationRegimesCallback(pl.Callback):
    """Schedules and executes named replay-evaluation regimes."""

    def __init__(  # ----------------------------------------------------------
        self,
        settings: EvaluationRegimesCallbackSettings,
    ) -> None:
        """Store callback settings."""
        super().__init__()
        self.settings = settings

    def on_train_batch_end(  # ------------------------------------------------
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Run due regimes on step cadence after each train batch."""
        if trainer.sanity_checking:
            return
        self._run_due_regimes(
            trainer,
            pl_module,
            trigger_kind="step",
        )

    def on_validation_epoch_end(  # -------------------------------------------
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        """Run due regimes on epoch cadence after validation epoch end."""
        if trainer.sanity_checking:
            return
        self._run_due_regimes(
            trainer,
            pl_module,
            trigger_kind="epoch",
        )

    def _run_due_regimes(  # --------------------------------------------------
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        *,
        trigger_kind: Literal["step", "epoch"],
    ) -> None:
        """Execute each regime whose schedule is due for the current trigger."""
        if self.settings.rank_zero_only and not trainer.is_global_zero:
            return

        for regime in self.settings.regimes:
            if not _is_due(
                regime.schedule,
                trigger_kind=trigger_kind,
                step=trainer.global_step + 1,
                epoch=trainer.current_epoch + 1,
            ):
                continue
            regime_result = self._run_one_regime(
                trainer,
                pl_module,
                regime,
            )
            self._log_regime_result(pl_module, regime, regime_result)
            if self.settings.persist_artifacts:
                self._persist_regime_result(
                    trainer,
                    regime,
                    regime_result,
                    trigger_kind=trigger_kind,
                )
            self._render_regime_figures(
                trainer,
                pl_module,
                regime,
                regime_result,
                trigger_kind=trigger_kind,
            )

    def _render_regime_figures(  # -------------------------------------------
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        regime: EvaluationRegimeSettings,
        regime_result: EvaluationRegimeResult,
        *,
        trigger_kind: Literal["step", "epoch"],
    ) -> None:
        """Render optional callback-local diagnostic figures from run traces.

        Delegates rendering to :func:`render_regime_preview_figures` and
        owns only failure-policy and logging.
        """
        request = regime.figure_request
        if not request.enabled:
            return

        run_dir = _resolve_regime_run_dir(
            trainer,
            output_subdir=self.settings.output_subdir,
            regime=regime,
            trigger_kind=trigger_kind,
        )
        figure_dir = run_dir / "figures"

        figure_ctx = FigureContext(
            global_step=trainer.global_step,
            split_name=regime.regime_id,
        )

        previews = render_regime_preview_figures(
            case_results=regime_result.case_results,
            figure_names=request.figures,
            figure_dir=figure_dir,
            figure_ctx=figure_ctx,
            max_cases=request.max_cases,
            save_pdf=request.save_pdf,
            save_png=request.save_png,
            png_dpi=request.png_dpi,
        )

        succeeded = sum(1 for p in previews if p.error is None)
        for preview in previews:
            if preview.error is not None:
                self._handle_figure_failure(
                    request,
                    regime,
                    "Online figure rendering failed for "
                    f"regime {regime.regime_id!r}, case {preview.case_id!r}, "
                    f"figure {preview.figure_name!r}: {preview.error}",
                )

        if succeeded == 0 and previews:
            self._handle_figure_failure(
                request,
                regime,
                "figure_request is enabled for regime "
                f"{regime.regime_id!r}, but no figure was rendered from the "
                f"first {request.max_cases} case results.",
            )
        pl_module.log(
            f"{regime.regime_kind}/{regime.regime_id}/figures_rendered",
            float(succeeded),
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=False,
        )

    def _handle_figure_failure(
        self,
        request: EvaluationFigureRequestSettings,
        regime: EvaluationRegimeSettings,
        message: str,
        *,
        error: Exception | None = None,
    ) -> None:
        """Apply configured figure failure policy for this regime."""
        if request.failure_policy == "raise":
            if error is not None:
                raise RuntimeError(message) from error
            raise RuntimeError(message)

        if error is not None:
            warnings.warn(
                f"{message} ({error!r})",
                RuntimeWarning,
                stacklevel=2,
            )
            return
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    def _run_one_regime(  # ---------------------------------------------------
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        regime: EvaluationRegimeSettings,
    ) -> EvaluationRegimeResult:
        """Resolve provider and execute one configured regime end-to-end."""
        provider = resolve_provider(
            regime.provider_ref,
            regime.provider_settings,
        )
        trace_request = self._build_trace_request(pl_module, regime)

        def prepare_case_batch(case):
            return self._prepare_case_batch(trainer, pl_module, case)

        case_results = tuple(
            iter_evaluation_regime(
                provider,
                pl_module,
                max_batches=regime.schedule.max_batches,
                trace_request=trace_request,
                prepare_case_batch=prepare_case_batch,
            )
        )
        losses = [
            value
            for value in (
                _extract_loss_scalar(result.evaluated)
                for result in case_results
            )
            if value is not None
        ]
        summary: dict[str, object] = {"n_cases": len(case_results)}
        if losses:
            summary["loss"] = sum(losses) / len(losses)

        return EvaluationRegimeResult(
            regime_id=regime.regime_id,
            case_results=case_results,
            summary=summary,
        )

    def _prepare_case_batch(  # ----------------------------------------------
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        case: EvaluationCaseBatch,
    ) -> EvaluationCaseBatch:
        """Re-enter Lightning's normal batch-transfer contract for one case."""
        batch = trainer.precision_plugin.convert_input(case.batch)
        batch = pl_module._on_before_batch_transfer(batch, dataloader_idx=0)
        batch = trainer.strategy.batch_to_device(batch, dataloader_idx=0)
        return replace(case, batch=batch)

    def _build_trace_request(  # ----------------------------------------------
        self,
        pl_module: LightningModule,
        regime: EvaluationRegimeSettings,
    ) -> EvaluationTraceRequest | None:
        """Build an optional trace request from regime settings and module state.

        Trace keys are computed from configured figure names and merged with
        explicitly requested keys.  The resulting spec is used directly for the
        ``EvaluationTraceRequest``; the callback no longer calls
        ``set_eval_trace_keys()`` on the module.
        """
        trace_request = regime.trace_request
        figure_request = regime.figure_request
        if (
            not trace_request.enabled
            and not trace_request.figure_names
            and not figure_request.enabled
        ):
            return None

        # Keys needed for the persisted artifact bundle.
        artifact_keys: set[str] = set(trace_request.trace_keys)
        for name in trace_request.figure_names:
            figure = REGISTRY.get(name)
            artifact_keys.update(figure.trace_keys)
            artifact_keys.update(figure.meta_keys)

        # Keys needed for callback-local diagnostic preview rendering.
        preview_keys: set[str] = set()
        if figure_request.enabled:
            for name in figure_request.figures:
                figure = REGISTRY.get(name)
                preview_keys.update(figure.trace_keys)
                preview_keys.update(figure.meta_keys)

        all_keys = artifact_keys | preview_keys

        # Warn when preview config adds keys beyond what artifact capture
        # would request — this makes the semantic boundary observable.
        if trace_request.enabled:
            extra_preview = preview_keys - artifact_keys
            if extra_preview:
                warnings.warn(
                    f"Regime {regime.regime_id!r}: preview figures require "
                    f"trace keys {sorted(extra_preview)} that are not in the "
                    "artifact trace request. These keys will be captured for "
                    "online preview but are not persisted in the artifact bundle.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        # Build trace spec directly from the accumulated keys.
        if all_keys:
            trace_spec = _build_trace_spec_for_module(pl_module, all_keys)
        else:
            trace_spec = getattr(pl_module, "trace_spec", None)

        if trace_spec is None:
            raise RuntimeError(
                "Trace request is enabled for regime "
                f"{regime.regime_id!r}, but the active Lightning module does "
                "not expose trace_spec and no trace keys were provided via "
                "trace_request or figure_request."
            )

        return EvaluationTraceRequest(
            trace_spec=trace_spec,
        )

    def _log_regime_result(  # ------------------------------------------------
        self,
        pl_module: LightningModule,
        regime: EvaluationRegimeSettings,
        regime_result: EvaluationRegimeResult,
    ) -> None:
        """Log namespaced aggregate metrics for one completed regime run."""
        namespace = f"{regime.regime_kind}/{regime.regime_id}"
        n_cases = regime_result.summary.get("n_cases")
        if isinstance(n_cases, Real):
            n_cases_value = float(n_cases)
        else:
            n_cases_value = float(len(regime_result.case_results))
        pl_module.log(
            f"{namespace}/n_cases",
            n_cases_value,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=False,
        )
        loss = regime_result.summary.get("loss")
        if isinstance(loss, Real):
            pl_module.log(
                f"{namespace}/loss",
                float(loss),
                on_step=False,
                on_epoch=True,
                logger=True,
                sync_dist=False,
            )

    def _persist_regime_result(  # --------------------------------------------
        self,
        trainer: Trainer,
        regime: EvaluationRegimeSettings,
        regime_result: EvaluationRegimeResult,
        *,
        trigger_kind: Literal["step", "epoch"],
    ) -> None:
        """Persist regime summary and optional trace payloads under log dir."""
        run_dir = _resolve_regime_run_dir(
            trainer,
            output_subdir=self.settings.output_subdir,
            regime=regime,
            trigger_kind=trigger_kind,
        )
        persist_regime_artifact_bundle(
            run_dir=run_dir,
            task=regime.task,
            regime_kind=regime.regime_kind,
            regime_id=regime.regime_id,
            trigger_kind=trigger_kind,
            epoch=trainer.current_epoch + 1,
            step=trainer.global_step,
            regime_result=regime_result,
        )


# =============================================================================
def _resolve_output_root(  # --------------------------------------------------
    trainer: Trainer,
) -> Path:
    """Resolve the callback artifact output root from trainer/logger state."""
    log_dir = getattr(trainer, "log_dir", None)
    if isinstance(log_dir, str) and log_dir:
        return Path(log_dir)
    logger = getattr(trainer, "logger", None)
    logger_log_dir = getattr(logger, "log_dir", None)
    if isinstance(logger_log_dir, str) and logger_log_dir:
        return Path(logger_log_dir)
    return Path(trainer.default_root_dir)


# =============================================================================
def _resolve_regime_run_dir(  # ----------------------------------------------
    trainer: Trainer,
    *,
    output_subdir: str,
    regime: EvaluationRegimeSettings,
    trigger_kind: Literal["step", "epoch"],
) -> Path:
    """Return the per-regime event directory for one callback trigger event."""
    output_root = _resolve_output_root(trainer)
    event_id = (
        f"epoch-{trainer.current_epoch + 1:04d}_"
        f"step-{trainer.global_step:08d}_"
        f"{trigger_kind}"
    )
    return (
        output_root
        / output_subdir
        / regime.regime_kind
        / regime.regime_id
        / event_id
    )


# =============================================================================
def _is_due(  # ---------------------------------------------------------------
    schedule: EvaluationScheduleSettings,
    *,
    trigger_kind: Literal["step", "epoch"],
    step: int,
    epoch: int,
) -> bool:
    """Return whether a schedule is due for the current trigger and counters."""
    if trigger_kind == "step":
        return schedule.every_n_steps > 0 and step % schedule.every_n_steps == 0
    return schedule.every_n_epochs > 0 and epoch % schedule.every_n_epochs == 0


# =============================================================================
def _extract_loss_scalar(  # --------------------------------------------------
    evaluated: Any,
) -> float | None:
    """Extract a scalar loss value from an evaluated chunk payload when available."""
    loss = getattr(evaluated, "loss", None)
    if loss is None:
        return None
    if torch.is_tensor(loss):
        if loss.numel() != 1:
            return None
        return float(loss.detach().item())
    if isinstance(loss, Real):
        return float(loss)
    return None


# =============================================================================
def _build_trace_spec_for_module(  # ------------------------------------------
    module: LightningModule,
    trace_keys: set[str],
) -> Any | None:
    """Build a trace spec from explicit keys, without calling set_eval_trace_keys.

    Attempts model-family-aware trace spec construction via ``build_trace_spec``
    if a ``_trace_paradigm`` attribute is available on the module.  Falls back to
    the module's ``trace_spec`` attribute if present.

    Returns ``None`` when neither path produces a spec.
    """
    paradigm = getattr(module, "_trace_paradigm", None)
    if paradigm is not None and trace_keys:
        return build_trace_spec(paradigm, include_keys=trace_keys)

    # Fallback: use the module's existing trace_spec if available.
    return getattr(module, "trace_spec", None)


# =============================================================================
__all__ = [
    "EvaluationFigureRequestSettings",
    "EvaluationRegimeSettings",
    "EvaluationRegimesCallback",
    "EvaluationRegimesCallbackSettings",
    "EvaluationScheduleSettings",
    "EvaluationTraceRequestSettings",
]
