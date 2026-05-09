"""PyTorch Lightning callback for periodic figure generation.

This module provides :class:`~ehc_sn.callbacks.figures.FiguresCallback`, a
rank-zero-only callback that captures a single rollout trace from the selected
evaluation split and uses the figure registry to render and persist figures.

The callback is intentionally best-effort: figure generation errors are caught
and printed so training/evaluation can continue.
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Iterable, List, Literal, Optional

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from matplotlib.figure import Figure
from pydantic import BaseModel, Field

from ehc_sn.figures import register_builtin_figures, render, sinks
from ehc_sn.figures.registry import REGISTRY, FigureContext
from ehc_sn.traces.trace_tree import TraceTree


# =================================================================================================
class FigureCallbackSettings(BaseModel, extra="forbid"):
    """Settings for FiguresCallback.

    Controls when figures are generated, which figures to create, and where
    to persist them.

    Notes:
        - Figure names are resolved via :data:`ehc_sn.figures.registry.REGISTRY`.
        - When ``output_dir`` is not set, PDFs default to the TensorBoard
          logger's ``log_dir`` (if present).
    """

    enabled: bool = Field(
        default=True,
        description="Whether to enable figure generation callback.",
    )
    split: Literal["validate", "test"] = Field(
        default="validate",
        description="Split to sample for figure generation.",
    )
    figures: list[str] = Field(
        default_factory=list,
        description="Explicit figure names to generate (from registry).",
    )
    save_pdf: bool = Field(
        default=True,
        description="Save figures as PDF artifacts.",
    )
    log_tensorboard: bool = Field(
        default=True,
        description="Log figures to TensorBoard.",
    )
    output_dir: Optional[Path] = Field(
        default=None,
        description="Base directory for PDF outputs (default: logger.log_dir). PDFs go under <base_dir>/figures/.",
    )
    env_idx: int = Field(
        default=0,
        description="Environment index to visualize.",
    )
    freq_idx: int = Field(
        default=0,
        description="Frequency module index to visualize.",
    )
    regime_id: Optional[str] = Field(
        default=None,
        description=(
            "If set, figures are rendered from the named evaluation regime artifact cached by "
            "EvaluationRegimesCallback instead of from validation dataloader outputs. "
            "If absent, the legacy validation-capture path is used unchanged."
        ),
    )


# =================================================================================================
class FiguresCallback(pl.Callback):
    """Lightning callback for periodic figure generation.

    Generates figures from model rollouts at regular intervals during training,
    saving PDFs and/or logging to TensorBoard.

    This callback:
        - Runs on global rank 0 only (safe under distributed training)
        - Captures exactly one trace per eligible epoch (the first batch of the
            first dataloader)
        - Looks up figure specs from the figure registry
        - Renders each requested figure and persists via configured sinks
        - Catches exceptions to avoid interrupting training
    """

    def __init__(self, settings: FigureCallbackSettings,) -> None:  # fmt: skip  # ------------------------------------------------------------------------------
        """Initialize callback.

        Args:
            settings: Configuration for figure generation.

        Raises:
            ValueError: If any requested figure names are not registered.
        """
        super().__init__()
        self.settings = settings
        self._captured_trace: Optional[TraceTree] = None
        self._required_trace_keys: set[str] = set()
        self._required_meta_keys: set[str] = set()
        register_builtin_figures()  # Ensure built-in figure specs are registered.
        REGISTRY.validate(self.settings.figures)

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule,) -> None:  # fmt: skip  # -------------------------------------------------------------
        """Prepare to capture a validation trace for this epoch.

        When ``settings.regime_id`` is set, trace capture from validation outputs is skipped;
        figures will instead be rendered from the regime artifact. In the legacy path (no
        ``regime_id``), this behaves exactly as before.
        """
        if self.settings.regime_id is not None:
            # Regime path: do not capture validation outputs; no keys needed from the module.
            return
        if not self._should_capture(trainer, "validate"):
            return
        self._required_trace_keys = self._required_trace_keys_union(self.settings.figures)
        self._required_meta_keys = self._required_meta_keys_union(self.settings.figures)
        set_keys = getattr(pl_module, "set_eval_trace_keys", None)
        if callable(set_keys):
            set_keys(self._required_trace_keys | self._required_meta_keys)
        self._reset_capture_state()

    def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0,) -> None:  # fmt: skip  # ---------------------------------------------------------------
        """Capture a single validation trace for later figure generation.

        This captures at most one trace per epoch (first batch of the first
        validation dataloader). The trace is extracted from ``outputs`` and
        normalized to CPU-backed arrays to avoid holding GPU memory.
        """
        # Regime path: skip validation batch capture entirely; figure source is the regime artifact.
        if self.settings.regime_id is not None:
            return
        if not self._should_capture(trainer, "validate") or dataloader_idx != 0:
            return
        if self._captured_trace is not None:
            return
        if batch_idx != 0:
            return

        trace = self._extract_trace(outputs)
        if trace is None:
            return
        self._captured_trace = self._to_cpu_trace(trace)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule,) -> None:  # fmt: skip  # ---------------------------------------------------------------
        """Generate figures at validation time.

        When ``settings.regime_id`` is set, figures are rendered from the named regime
        artifact cached by :class:`~ehc_sn.callbacks.eval_regimes.EvaluationRegimesCallback`.
        When absent, the legacy validation-capture path is used unchanged.
        """
        if not self.settings.enabled or not trainer.is_global_zero or self.settings.split != "validate":
            return
        try:
            if self.settings.regime_id is not None:
                self._generate_figures_from_regime(trainer)
            else:
                self._generate_figures(trainer, split_name="validate")
        except Exception as e:
            print("FiguresCallback: Error generating figures at step " f"{trainer.global_step}: {e}")
            traceback.print_exc()

    def _generate_figures_from_regime(self, trainer: Trainer,) -> None:  # fmt: skip  # ---------------------------------------------------------
        """Generate figures from a named regime artifact (regime_id path).

        Fetches the cached :class:`~ehc_sn.lightning.eval.contracts.EvaluationBatchArtifacts`
        from the first :class:`~ehc_sn.callbacks.eval_regimes.EvaluationRegimesCallback` in the
        trainer's callback list. Does **not** fall back to validation outputs if the artifact
        is absent — it logs a skip message instead.

        Args:
            trainer: PyTorch Lightning trainer.
        """
        regime_id = self.settings.regime_id
        assert regime_id is not None  # Invariant: caller checks before dispatching.

        # Locate the EvaluationRegimesCallback among active callbacks.
        regime_callback = self._find_regime_callback(trainer)
        if regime_callback is None:
            print(
                f"FiguresCallback: regime_id='{regime_id}' is configured but no "
                "EvaluationRegimesCallback is registered. Skipping figure generation."
            )
            return

        if not regime_callback.was_refreshed_this_cycle(regime_id):
            print(
                f"FiguresCallback: Regime '{regime_id}' did not run this validation cycle. "
                "Skipping figure generation to avoid duplicate figures."
            )
            return

        artifact = regime_callback.get_latest_artifact(regime_id)
        if artifact is None:
            print(
                f"FiguresCallback: No cached artifact for regime '{regime_id}'. "
                "Skipping figure generation (will not fall back to validation outputs)."
            )
            return

        if artifact.trace is None:
            print(f"FiguresCallback: Artifact for regime '{regime_id}' has no trace. " "Skipping figure generation.")
            return

        context = self.figure_context(trainer, split_name=f"regime:{regime_id}")
        trace = self._to_cpu_trace(artifact.trace)
        self._dispatch_figures(trainer, self.settings.figures, context, trace)

    def _find_regime_callback(self, trainer: Trainer,) -> "Optional[Any]":  # fmt: skip  # -----------------------------------------------------------------
        """Return the first EvaluationRegimesCallback in the trainer's callback list.

        Import is deferred to avoid a circular dependency between callbacks modules.
        """
        try:
            from ehc_sn.callbacks.eval_regimes import EvaluationRegimesCallback
        except ImportError:
            return None
        for cb in trainer.callbacks:  # type: ignore[attr-defined]
            if isinstance(cb, EvaluationRegimesCallback):
                return cb
        return None

    def _generate_figures(self, trainer: Trainer, split_name: str,) -> None:  # fmt: skip  # ---------------------------------------------------------------------
        """Generate and persist all configured figures.

        Uses the trace captured during the epoch to render each figure in
        ``settings.figures``, saving PDFs and/or logging to TensorBoard based on
        ``settings``.

        Args:
            trainer: PyTorch Lightning trainer.
            split_name: Name of the split being visualized (e.g. ``"validate"``).
        """
        if self._captured_trace is None:
            print("FiguresCallback: No captured batch available; skipping figure generation.")
            return

        context = self.figure_context(trainer, split_name)
        self._dispatch_figures(trainer, self.settings.figures, context, self._captured_trace)
        self._reset_capture_state()

    def _should_capture(self, trainer: Trainer, split_name: str,) -> bool:  # fmt: skip  # -----------------------------------------------------------------------
        """Return whether this process should capture traces for ``split_name``.

        Capture is restricted to global rank 0 and gated by ``settings``.
        """
        return self.settings.enabled and trainer.is_global_zero and self.settings.split == split_name

    def _reset_capture_state(self,) -> None:  # fmt: skip  # ------------------------------------------------------------------
        """Clear any previously captured trace for the current epoch."""
        self._captured_trace = None

    def _dispatch_figures(self, trainer: Trainer, figure_names: Iterable[str], context: FigureContext, trace: TraceTree) -> None:  # fmt: skip  # ---------------------------------------------------------------------
        """Render and persist each figure listed in ``figure_names``."""
        for figure_name in figure_names:
            self.generate_figure(trainer, figure_name, trace, context)

    def figure_context(self, trainer: Trainer, split_name: Optional[str],) -> FigureContext:  # fmt: skip  # ------------------------------------------------------------------------
        """Build a :class:`~ehc_sn.figures.registry.FigureContext`.

        The context bundles user-configured visualization indices plus trainer
        state (e.g. global step) so figure specs can label and condition plots.
        """
        return FigureContext(
            env_idx=self.settings.env_idx,
            freq_idx=self.settings.freq_idx,
            global_step=trainer.global_step,
            split_name=split_name,
        )

    def _required_trace_keys_union(self, names: Iterable[str],) -> set[str]:  # fmt: skip  # ------------------------------------------------------------
        """ """
        keys: set[str] = set()
        for name in names:
            keys.update(REGISTRY.get(name).trace_keys)
        return keys

    def _required_meta_keys_union(self, names: Iterable[str],) -> set[str]:  # fmt: skip  # -------------------------------------------------------------
        """Return the union of metadata trace requirements for the selected figures."""
        keys: set[str] = set()
        for name in names:
            keys.update(REGISTRY.get(name).meta_keys)
        return keys

    def _extract_trace(self, outputs: Any,) -> Optional[TraceTree]:  # fmt: skip  # ------------------------------------------------------------------------
        """Extract a :class:`~ehc_sn.traces.trace_tree.TraceTree` from ``outputs``.

        Supports common Lightning return conventions:
            - Directly returning a ``TraceTree``
            - Returning a dict containing ``"trace"`` or ``"snapshot"``
            - Returning an object with a ``tree`` attribute holding a ``TraceTree``
        """
        if outputs is None:
            return None
        if isinstance(outputs, TraceTree):
            return outputs
        if isinstance(outputs, dict):
            return outputs.get("trace")

        return None

    def _to_cpu_trace(self, trace: TraceTree,) -> TraceTree:  # fmt: skip  # -------------------------------------------------------------------------
        """Normalize trace storage to CPU-backed arrays.

        The trace is finalized first (skipped if already finalized), then any dense
        tensor leaves are detached and converted to NumPy arrays on CPU. Idempotent:
        safe to call multiple times on the same trace.
        """
        if trace.dense_leaves is None:
            trace.finalize()
            return trace
        for idx, leaf in enumerate(trace.dense_leaves):
            if isinstance(leaf, torch.Tensor):
                trace.dense_leaves[idx] = leaf.detach().cpu().numpy()
        trace.meta_first = [self._normalize_meta_value(value) for value in trace.meta_first]
        trace.attached_meta = {key: self._normalize_meta_value(value) for key, value in trace.attached_meta.items()}
        return trace

    def _normalize_meta_value(self, value: object) -> object:
        """Convert metadata tensors recursively to CPU NumPy arrays."""
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
        if isinstance(value, dict):
            return {key: self._normalize_meta_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._normalize_meta_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._normalize_meta_value(item) for item in value)
        return value

    def generate_figure(self, trainer: Trainer, figure_name: str, trace: Any, ctx: FigureContext,) -> None:  # fmt: skip  # -----------------------------------------------------------------------
        """Generate a single figure and persist it using the configured sinks.

        Renders via :func:`ehc_sn.figures.render`, which validates trace
        requirements before plotting.

        Args:
            trainer: PyTorch Lightning trainer.
            figure_name: Registered figure name.
            trace: Captured rollout trace.
            ctx: Figure rendering context.
        """
        spec = REGISTRY.get(figure_name)
        fig = render(figure_name, trace, ctx)
        if self.settings.save_pdf:
            self._save_pdf(trainer, fig, spec)
        if self.settings.log_tensorboard:
            self._log_tensorboard(trainer, fig, spec)

        # Always close the figure to prevent memory growth across epochs.
        plt.close(fig)

    def _save_pdf(self, trainer: Trainer, fig: Figure, spec: Any,) -> None:  # fmt: skip  # -----------------------------------------------------------------------------
        """Save a figure as a PDF artifact.

        Resolution order for the output directory:
            1) ``settings.output_dir``
            2) TensorBoard logger ``log_dir`` (if the active logger is a
               :class:`~lightning.pytorch.loggers.TensorBoardLogger`)
        """
        base_dir = self.settings.output_dir
        if base_dir is None and isinstance(trainer.logger, TensorBoardLogger):
            base_dir = Path(trainer.logger.log_dir)

        if base_dir is not None:
            base_dir = Path(base_dir)
            pdf_path = sinks.make_figure_path(base_dir, spec.default_filename, step=trainer.global_step)
            sinks.save_pdf(fig, pdf_path)

    def _log_tensorboard(self, trainer: Trainer, fig: Figure, spec: Any,) -> None:  # fmt: skip  # ----------------------------------------------------------------------
        """Log a figure to TensorBoard (or compatible logger backends).

        Uses ``TensorBoardLogger`` when available; otherwise falls back to
        ``logger.experiment.add_figure`` when present.
        """
        tag = f"figures/{spec.name}"
        loggers = trainer.loggers if trainer.loggers is not None else []
        if not loggers and trainer.logger is not None:
            loggers = [trainer.logger]

        for logger in loggers:
            if isinstance(logger, TensorBoardLogger):
                sinks.log_tensorboard_figure(logger, tag, fig, global_step=trainer.global_step)
                continue
            experiment = getattr(logger, "experiment", None)
            if experiment is not None and hasattr(experiment, "add_figure"):
                experiment.add_figure(tag, fig, global_step=trainer.global_step)
