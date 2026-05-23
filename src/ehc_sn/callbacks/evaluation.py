"""Scheduled named replay-evaluation callback.

This callback is orchestration-only:
- resolves task-owned providers from provider_ref,
- schedules named regimes on trainer cadence,
- delegates execution through the family-owned execute_evaluation_batch seam
  via ehc_sn.eval helpers,
- logs namespaced summary metrics,
- persists returned trace/artifact payloads.
"""

from __future__ import annotations

import importlib
import json
import re
from dataclasses import asdict, is_dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Literal

import lightning.pytorch as pl
import torch
from lightning.pytorch import LightningModule, Trainer
from pydantic import BaseModel, Field, model_validator

from ehc_sn.eval import (
    EvaluationRegimeResult,
    EvaluationTraceRequest,
    iter_evaluation_regime,
)


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


# =============================================================================
class EvaluationRegimeSettings(BaseModel, extra="forbid"):
    """One named evaluation regime configuration."""

    regime_id: str = Field(
        ...,
        min_length=1,
        description="Unique regime identifier used in metric and artifact namespaces.",
    )
    phase_kind: Literal["diag", "bench"] = Field(
        default="diag",
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

    def _run_one_regime(  # ---------------------------------------------------
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        regime: EvaluationRegimeSettings,
    ) -> EvaluationRegimeResult:
        """Resolve provider and execute one configured regime end-to-end."""
        provider = _resolve_provider(
            regime.provider_ref,
            regime.provider_settings,
        )
        trace_request = self._build_trace_request(pl_module, regime)
        results = tuple(
            iter_evaluation_regime(
                provider,
                pl_module,
                max_batches=regime.schedule.max_batches,
                trace_request=trace_request,
            )
        )
        return EvaluationRegimeResult(regime_id=regime.regime_id, results=results)

    def _build_trace_request(  # ----------------------------------------------
        self,
        pl_module: LightningModule,
        regime: EvaluationRegimeSettings,
    ) -> EvaluationTraceRequest | None:
        """Build an optional trace request from regime settings and module state."""
        request = regime.trace_request
        if not request.enabled:
            return None

        if request.trace_keys:
            set_trace_keys = getattr(pl_module, "set_eval_trace_keys", None)
            if callable(set_trace_keys):
                set_trace_keys(set(request.trace_keys))

        trace_spec = getattr(pl_module, "trace_specs", None)
        if trace_spec is None:
            raise RuntimeError(
                "Trace request is enabled for regime "
                f"{regime.regime_id!r}, but the active Lightning module does "
                "not expose trace_specs."
            )

        return EvaluationTraceRequest(
            enabled=True,
            trace_spec=trace_spec,
        )

    def _log_regime_result(  # ------------------------------------------------
        self,
        pl_module: LightningModule,
        regime: EvaluationRegimeSettings,
        regime_result: EvaluationRegimeResult,
    ) -> None:
        """Log namespaced aggregate metrics for one completed regime run."""
        namespace = f"{regime.phase_kind}/{regime.regime_id}"
        losses = [
            value
            for value in (
                _extract_loss_scalar(result.evaluated) for result in regime_result.results
            )
            if value is not None
        ]
        pl_module.log(
            f"{namespace}/n_cases",
            float(len(regime_result.results)),
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=False,
        )
        if losses:
            pl_module.log(
                f"{namespace}/loss",
                sum(losses) / len(losses),
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
        output_root = _resolve_output_root(trainer)
        event_id = (
            f"epoch-{trainer.current_epoch + 1:04d}_"
            f"step-{trainer.global_step:08d}_"
            f"{trigger_kind}"
        )
        run_dir = (
            output_root
            / self.settings.output_subdir
            / regime.phase_kind
            / regime.regime_id
            / event_id
        )
        run_dir.mkdir(parents=True, exist_ok=True)

        summary_rows: list[dict[str, Any]] = []
        for idx, result in enumerate(regime_result.results):
            summary_row = {
                "case_id": result.case_id,
                "source_context": _to_jsonable(result.source_context),
                "loss": _extract_loss_scalar(result.evaluated),
                "has_trace": result.trace is not None,
            }
            summary_rows.append(summary_row)
            if result.trace is None:
                continue

            trace_payload = {
                "case_id": result.case_id,
                "source_context": result.source_context,
                "dense": result.trace.export(),
                "meta": result.trace.get_meta(),
            }
            trace_path = run_dir / (
                f"{idx:04d}-{_sanitize_filename_component(result.case_id)}.pt"
            )
            torch.save(trace_payload, trace_path)

        summary = {
            "regime_id": regime_result.regime_id,
            "phase_kind": regime.phase_kind,
            "trigger_kind": trigger_kind,
            "epoch": trainer.current_epoch + 1,
            "step": trainer.global_step,
            "n_cases": len(regime_result.results),
            "cases": summary_rows,
        }
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )


# =============================================================================
def _resolve_provider(  # -----------------------------------------------------
    provider_ref: str,
    provider_settings: dict[str, Any],
) -> Any:
    """Resolve and instantiate a provider class from dotted import path."""
    module_name, class_name = provider_ref.rsplit(".", maxsplit=1)
    module = importlib.import_module(module_name)
    provider_cls = getattr(module, class_name)
    return provider_cls(**provider_settings)


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
def _to_jsonable(  # ----------------------------------------------------------
    value: Any,
) -> Any:
    """Convert nested objects into JSON-serializable primitives recursively."""
    if value is None:
        return None
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return _to_jsonable(value.model_dump())
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    return repr(value)


# =============================================================================
def _sanitize_filename_component(  # ------------------------------------------
    value: str,
) -> str:
    """Normalize a string into a safe filename component."""
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return sanitized or "case"


# =============================================================================
__all__ = [
    "EvaluationRegimeSettings",
    "EvaluationRegimesCallback",
    "EvaluationRegimesCallbackSettings",
    "EvaluationScheduleSettings",
    "EvaluationTraceRequestSettings",
]
