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
from ehc_sn.lightning.diagnostics import DiagnosticTraceSpec


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
    max_timesteps: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Maximum number of timesteps rendered from each bounded diagnostic "
            "trace.  This is a render-time guard — it slices diagnostic traces "
            "before figure rendering but does not reduce the amount of trace "
            "data captured by the model/family validation step."
        ),
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
        report_kind = [
            name
            for name in self.figures
            if name in available
            and "report" in REGISTRY.get(name).allowed_surfaces
        ]
        if report_kind:
            blocked = ", ".join(sorted(set(report_kind)))
            raise ValueError(
                "FigureGenerationSettings.figures contains report-surface names: "
                f"{blocked}. "
                "Use the offline report pipeline for these."
            )

        non_bounded = [
            name
            for name in self.figures
            if name in available
            and REGISTRY.get(name).input_contract != "bounded_trace"
        ]
        if non_bounded:
            lines: list[str] = [
                "FigureGenerationCallback only supports bounded_trace figures.",
            ]
            for name in sorted(set(non_bounded)):
                spec = REGISTRY.get(name)
                lines.append("")
                lines.append(f"  {name}")
                lines.append(f"    requires: {spec.input_contract}")
                if spec.input_contract == "evaluation_artifact":
                    lines.append(
                        "    Use: add this figure to an EvaluationRegimesCallback "
                        "regime to capture full eval artifacts, "
                        "then render it offline with render_report()."
                    )
                elif spec.input_contract == "offline_artifact":
                    lines.append(
                        "    Use: render this figure offline with "
                        "render_report(...) from a persisted eval artifact run."
                    )
            lines.append("")
            lines.append(
                "Valid bounded_trace figures: "
                + ", ".join(
                    sorted(
                        name
                        for name in available
                        if REGISTRY.get(name).input_contract == "bounded_trace"
                    )
                )
            )
            raise ValueError("\n".join(lines))
        return self


# =============================================================================
class FigureGenerationCallback(pl.Callback):
    """Bounded trace diagnostic figure generation callback.

    - **setup()**: Derives required trace keys from configured ``bounded_trace``
      figures via the registry and configures ``pl_module.diagnostic_trace_spec``
      to enable trace capture with the needed keys.
    - **on_validation_epoch_end** / **on_test_epoch_end**: Reads captured
      ``diagnostic_traces`` from the module, gates by cadence, and renders
      registered diagnostic figures via the ``ehc_sn.figures`` pipeline.

    The callback does not accumulate traces — it only consumes what the module
    exposes via ``diagnostic_traces``.
    """

    def __init__(self, settings: FigureGenerationSettings) -> None:
        super().__init__()
        self.settings = settings

    def setup(  # -------------------------------------------------------------
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        stage: str | None = None,
    ) -> None:
        """Derive and configure the module's ``diagnostic_trace_spec``.

        Reads required ``trace_keys`` from every configured ``bounded_trace``
        figure in the registry, merges them with any keys already present on
        the module's ``diagnostic_trace_spec``, and enables trace capture.

        This is safe to call multiple times (idempotent merge).
        """
        if not self.settings.enabled:
            return
        if not self.settings.figures:
            return
        if stage not in (None, "fit", "validate"):
            return

        required_keys: set[str] = set()
        for name in self.settings.figures:
            if name not in REGISTRY:
                continue
            spec = REGISTRY.get(name)
            if spec.input_contract != "bounded_trace":
                continue
            if not spec.trace_keys:
                raise ValueError(
                    f"Figure {name!r} has input_contract='bounded_trace' but "
                    "declares empty trace_keys; cannot derive required keys."
                )
            required_keys.update(spec.trace_keys)

        if not required_keys:
            return

        current: DiagnosticTraceSpec = getattr(
            pl_module, "diagnostic_trace_spec", None
        )
        if current is None:
            warnings.warn(
                "FigureGenerationCallback.setup(): LightningModule has no "
                "'diagnostic_trace_spec' attribute; bounded diagnostic traces "
                "will not be captured.",
                RuntimeWarning,
                stacklevel=2,
            )
            return

        merged_keys = tuple(sorted(set(current.keys) | required_keys))
        merged_max_batches = (
            current.max_batches
            if current.max_batches <= self.settings.max_cases
            else self.settings.max_cases
        )

        pl_module.diagnostic_trace_spec = DiagnosticTraceSpec(
            enabled=True,
            max_batches=merged_max_batches,
            keys=merged_keys,
        )

    def on_validation_epoch_end(  # -------------------------------------------
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        """Render due figures after validation epoch end."""
        if self.settings.context != "validation":
            return
        if not self.settings.enabled:
            return
        self._run_from_module(trainer, pl_module, trigger_kind="epoch")

    def on_test_epoch_end(  # -------------------------------------------------
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        """Render due figures after test epoch end."""
        if self.settings.context != "testing":
            return
        if not self.settings.enabled:
            return
        self._run_from_module(trainer, pl_module, trigger_kind="epoch")

    def _run_from_module(  # --------------------------------------------------
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        *,
        trigger_kind: Literal["step", "epoch"],
    ) -> None:
        """Read bounded diagnostic traces from the module and render.

        The module owns trace capture; this callback only renders and logs.
        """
        if trainer.sanity_checking:
            return
        if not trainer.is_global_zero:
            return
        if not self._is_due(trigger_kind, trainer):
            return

        traces: list[Any] = list(getattr(pl_module, "diagnostic_traces", ()))
        if not traces:
            return
        self._run(trainer, pl_module, trigger_kind="epoch", traces=traces)

    def _run(  # --------------------------------------------------------------
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        *,
        trigger_kind: Literal["step", "epoch"],
        traces: list[Any],
    ) -> None:
        """Execute figure generation if the schedule is due.

        Args:
            trainer: Lightning Trainer instance.
            pl_module: LightningModule whose traces are being rendered.
            trigger_kind: What kind of trigger invoked this run.
            traces: Module-owned ``TraceTree`` objects from the just-completed
                validation epoch.
        """
        if not self.settings.enabled:
            return
        if trainer.sanity_checking:
            return
        if not trainer.is_global_zero:
            return

        if not self._is_due(trigger_kind, trainer):
            return

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

                if self.settings.max_timesteps is not None:
                    trace = trace.slice_time(0, self.settings.max_timesteps)

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
