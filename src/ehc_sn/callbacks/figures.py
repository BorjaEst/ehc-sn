"""Lightning callback for standalone scheduled figure generation.

Renders registered figures from validation traces at a configurable cadence
and persists results to disk (PDF/PNG) and TensorBoard.  This is a simpler
alternative to :class:`~ehc_sn.callbacks.evaluation.EvaluationRegimesCallback`
that works without requiring a provider-based regime setup.

Resource budgeting (``max_items``, ``max_cells``, ``max_retained_files``)
prevents runaway disk usage or OOM during long training runs.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Literal

import lightning.pytorch as pl
import matplotlib.pyplot as plt
from lightning.pytorch import LightningModule, Trainer
from pydantic import BaseModel, Field, model_validator

from ehc_sn.figures import REGISTRY, FigureContext, list_figures, render
from ehc_sn.figures.sinks import (
    _persist_named_figure_artifacts,
    log_tensorboard_figure,
)


# =============================================================================
class FigureGenerationSettings(BaseModel, extra="forbid"):
    """Settings for standalone scheduled figure generation.

    Context determines which training hook fires::

        validation — fires inside ``on_validation_epoch_end``.
        testing    — fires inside ``on_test_epoch_end``.

    Offline generation bypasses the callback entirely — call
    ``ehc_sn.figures.render`` directly.
    """

    enabled: bool = Field(
        default=False,
        description="Master switch; when false the callback is a no-op.",
    )
    context: Literal["validation", "testing"] = Field(
        default="validation",
        description="Training context that gates which hook fires.",
    )
    every_n_epochs: int = Field(
        default=5,
        ge=0,
        description="Render every N validation/testing epochs.  0 disables epoch-based cadence.",
    )
    every_n_steps: int = Field(
        default=0,
        ge=0,
        description="Render every N training steps.  0 disables step-based cadence.",
    )
    figures: list[str] = Field(
        default_factory=list,
        description="Registered figure names to render from available validation traces.",
    )
    max_cases: int = Field(
        default=2,
        ge=1,
        description="Maximum number of case results (batches) to render from per run.",
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
        description="PNG export DPI when save_png is enabled.",
    )
    max_items: int = Field(
        default=4,
        ge=1,
        description="Maximum items (samples) shown per multi-sample figure panel.",
    )
    max_cells: int = Field(
        default=16,
        ge=1,
        description="Maximum cells shown per cell-level figure (e.g. rate maps).",
    )
    max_retained_files: int = Field(
        default=200,
        ge=1,
        description="Maximum figure files retained on disk per figure name per run.",
    )
    failure_policy: Literal["warn", "raise"] = Field(
        default="warn",
        description=(
            "Behavior when a figure render or persistence step raises: "
            "'warn' logs a RuntimeWarning and continues; 'raise' propagates the exception."
        ),
    )

    @model_validator(mode="after")
    def _validate_figures(self) -> "FigureGenerationSettings":
        """Validate enabled request shape and registered figure names."""
        if not self.enabled:
            return self
        if not self.figures:
            raise ValueError(
                "FigureGenerationSettings.enabled=True requires a non-empty figures list."
            )
        if not self.save_pdf and not self.save_png:
            raise ValueError(
                "FigureGenerationSettings must enable at least one sink: "
                "save_pdf or save_png."
            )
        available = set(list_figures())
        missing = [name for name in self.figures if name not in available]
        if missing:
            unknown = ", ".join(sorted(set(missing)))
            raise ValueError(
                "FigureGenerationSettings.figures contains unknown names: "
                f"{unknown}."
            )
        return self


# =============================================================================
class FigureGenerationCallback(pl.Callback):
    """Standalone scheduled figure generation callback.

    Collects traces from the LightningModule's validation/test step outputs
    (expected under the ``trace`` key), gates by cadence, and renders
    registered figures via the ``ehc_sn.figures`` pipeline.

    This is a simpler alternative to
    :class:`~ehc_sn.callbacks.evaluation.EvaluationRegimesCallback` that
    requires no provider resolution — it works with any LightningModule that
    returns ``{"trace": TraceTree}`` from its step methods.
    """

    def __init__(self, settings: FigureGenerationSettings) -> None:
        super().__init__()
        self.settings = settings

    def on_validation_epoch_end(  # -------------------------------------------
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        """Render due figures after validation epoch end."""
        if self.settings.context != "validation":
            return
        self._run(trainer, pl_module, trigger_kind="epoch")

    def on_train_batch_end(  # ------------------------------------------------
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Render due figures after train batch (step-based cadence)."""
        if self.settings.context != "validation":
            return
        self._run(trainer, pl_module, trigger_kind="step")

    def on_test_epoch_end(  # -------------------------------------------------
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        """Render due figures after test epoch end."""
        if self.settings.context != "testing":
            return
        self._run(trainer, pl_module, trigger_kind="epoch")

    def _run(  # --------------------------------------------------------------
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        *,
        trigger_kind: Literal["step", "epoch"],
    ) -> None:
        """Execute figure generation if the schedule is due."""
        if not self.settings.enabled:
            return
        if trainer.sanity_checking:
            return
        if not trainer.is_global_zero:
            return

        if not self._is_due(trigger_kind, trainer):
            return

        traces = self._collect_traces(trainer, pl_module)
        if not traces:
            return

        figure_ctx = FigureContext(
            max_items=self.settings.max_items,
            max_cells=self.settings.max_cells,
            global_step=trainer.global_step,
            split_name=self.settings.context,
        )

        output_dir = self._resolve_output_dir(trainer)
        rendered_count = 0

        for figure_name in self.settings.figures:
            if figure_name not in REGISTRY:
                self._handle_failure(
                    f"Figure '{figure_name}' is not registered. Skipping."
                )
                continue

            spec = REGISTRY.get(figure_name)
            for idx, trace in enumerate(traces[: self.settings.max_cases]):
                if trace is None:
                    continue

                fig = None
                try:
                    fig = render(figure_name, trace, figure_ctx)
                    _persist_named_figure_artifacts(
                        fig,
                        output_dir=output_dir,
                        case_index=idx,
                        case_id=f"step-{trainer.global_step}",
                        default_filename=spec.default_filename,
                        save_pdf_enabled=self.settings.save_pdf,
                        save_png_enabled=self.settings.save_png,
                        png_dpi=self.settings.png_dpi,
                    )
                    log_tensorboard_figure(
                        trainer.logger,
                        f"figures/{figure_name}/{idx:04d}",
                        fig,
                        global_step=trainer.global_step,
                    )
                    rendered_count += 1
                except Exception as exc:
                    self._handle_failure(
                        f"Figure '{figure_name}' case {idx} failed: {exc}"
                    )
                finally:
                    if fig is not None:
                        plt.close(fig)

        self._enforce_disk_budget(output_dir)

        pl_module.log(
            "figures/rendered",
            float(rendered_count),
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=False,
        )

    def _is_due(  # -----------------------------------------------------------
        self,
        trigger_kind: Literal["step", "epoch"],
        trainer: Trainer,
    ) -> bool:
        """Return whether the schedule requests firing at this trigger point."""
        if trigger_kind == "step":
            return (
                self.settings.every_n_steps > 0
                and (trainer.global_step + 1) % self.settings.every_n_steps == 0
            )
        return (
            self.settings.every_n_epochs > 0
            and (trainer.current_epoch + 1) % self.settings.every_n_epochs == 0
        )

    def _collect_traces(  # ---------------------------------------------------
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> list[Any]:
        """Collect trace outputs from the LightningModule's step results.

        Expected source: ``validation_step`` / ``test_step`` outputs that
        contain a ``trace`` key.

        The callback relies on the module owning trace materialization
        (via ``observe_rollout_chunk``) and returning the trace as part of
        its step output dict.
        """
        collected: list[Any] = []
        # Traces are accumulated per batch in the module's step outputs.
        # We look for them either in trainer.callback_metrics (post-epoch)
        # or directly from the module's output cache.
        outputs_buffer = getattr(pl_module, "_validation_step_outputs", None)
        if outputs_buffer is None:
            outputs_buffer = getattr(pl_module, "_test_step_outputs", None)
        if outputs_buffer is not None and isinstance(outputs_buffer, list):
            for output in outputs_buffer:
                if isinstance(output, dict):
                    trace = output.get("trace")
                    if trace is not None:
                        collected.append(trace)
        return collected

    def _resolve_output_dir(  # -----------------------------------------------
        self,
        trainer: Trainer,
    ) -> Path:
        """Resolve the figures output directory from trainer/logger state."""
        log_dir = getattr(trainer, "log_dir", None)
        if isinstance(log_dir, str) and log_dir:
            return Path(log_dir)
        logger = getattr(trainer, "logger", None)
        logger_log_dir = getattr(logger, "log_dir", None)
        if isinstance(logger_log_dir, str) and logger_log_dir:
            return Path(logger_log_dir)
        return Path(trainer.default_root_dir) / "figures"

    def _enforce_disk_budget(  # ----------------------------------------------
        self,
        output_dir: Path,
    ) -> None:
        """Delete oldest figure files when count exceeds max_retained_files."""
        if not output_dir.is_dir():
            return
        max_files = self.settings.max_retained_files
        pdf_files = sorted(output_dir.rglob("*.pdf"))
        if len(pdf_files) <= max_files:
            return
        for old_file in pdf_files[: len(pdf_files) - max_files]:
            old_file.unlink(missing_ok=True)

    def _handle_failure(self, message: str) -> None:
        """Apply configured failure policy."""
        if self.settings.failure_policy == "raise":
            raise RuntimeError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)


# =============================================================================
__all__ = ["FigureGenerationSettings", "FigureGenerationCallback"]
