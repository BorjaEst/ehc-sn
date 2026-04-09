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

from ehc_sn.figures import register, sinks
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

    def __init__(  # ------------------------------------------------------------------------------
        self, settings: FigureCallbackSettings,
    ) -> None:  # fmt: skip
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
        register.register_builtin_figures()  # Ensure built-in figure specs are registered.
        REGISTRY.validate(self.settings.figures)

    def on_validation_epoch_start(  # -------------------------------------------------------------
        self, trainer: Trainer, pl_module: LightningModule,
    ) -> None:  # fmt: skip
        """Prepare to capture a validation trace for this epoch.

        Resets internal capture state if this epoch is eligible under
        ``settings``. No-op otherwise.
        """
        if not self._should_capture(trainer, "validate"):
            return
        self._required_trace_keys = self._required_trace_keys_union(self.settings.figures)
        self._required_meta_keys = self._required_meta_keys_union(self.settings.figures)
        set_keys = getattr(pl_module, "set_eval_trace_keys", None)
        if callable(set_keys):
            set_keys(self._required_trace_keys | self._required_meta_keys)
        self._reset_capture_state()

    def on_validation_batch_end(  # ---------------------------------------------------------------
        self, trainer: Trainer, pl_module: LightningModule, outputs: Any, batch: Any, batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:  # fmt: skip
        """Capture a single validation trace for later figure generation.

        This captures at most one trace per epoch (first batch of the first
        validation dataloader). The trace is extracted from ``outputs`` and
        normalized to CPU-backed arrays to avoid holding GPU memory.
        """
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

    def on_validation_epoch_end(  # ---------------------------------------------------------------
        self, trainer: Trainer, pl_module: LightningModule,
    ) -> None:  # fmt: skip
        """Generate figures at validation time."""
        if not self.settings.enabled or not trainer.is_global_zero or self.settings.split != "validate":
            return
        try:
            self._generate_figures(trainer, split_name="validate")
        except Exception as e:
            print("FiguresCallback: Error generating figures at step " f"{trainer.global_step}: {e}")
            traceback.print_exc()

    def _generate_figures(  # ---------------------------------------------------------------------
        self, trainer: Trainer, split_name: str,
    ) -> None:  # fmt: skip
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

    def _should_capture(  # -----------------------------------------------------------------------
        self, trainer: Trainer, split_name: str,
    ) -> bool:  # fmt: skip
        """Return whether this process should capture traces for ``split_name``.

        Capture is restricted to global rank 0 and gated by ``settings``.
        """
        return self.settings.enabled and trainer.is_global_zero and self.settings.split == split_name

    def _reset_capture_state(  # ------------------------------------------------------------------
        self,
    ) -> None:  # fmt: skip
        """Clear any previously captured trace for the current epoch."""
        self._captured_trace = None

    def _dispatch_figures(  # ---------------------------------------------------------------------
        self, trainer: Trainer, figure_names: Iterable[str], context: FigureContext, trace: TraceTree
    ) -> None:  # fmt: skip
        """Render and persist each figure listed in ``figure_names``."""
        for figure_name in figure_names:
            spec = REGISTRY.get(figure_name)
            self._validate_trace_keys(trace, spec.trace_keys, figure_name)
            self._validate_meta_keys(trace, spec.meta_keys, figure_name)
            self.generate_figure(trainer, trace, context, spec)

    def figure_context(  # ------------------------------------------------------------------------
        self, trainer: Trainer, split_name: Optional[str],
    ) -> FigureContext:  # fmt: skip
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

    def _required_trace_keys_union(  # ------------------------------------------------------------
        self, names: Iterable[str],
    ) -> set[str]:  # fmt: skip
        """ """
        keys: set[str] = set()
        for name in names:
            keys.update(REGISTRY.get(name).trace_keys)
        return keys

    def _required_meta_keys_union(  # -------------------------------------------------------------
        self, names: Iterable[str],
    ) -> set[str]:  # fmt: skip
        """Return the union of metadata trace requirements for the selected figures."""
        keys: set[str] = set()
        for name in names:
            keys.update(REGISTRY.get(name).meta_keys)
        return keys

    def _validate_trace_keys(  # ------------------------------------------------------------------
        self, trace: TraceTree, required: set[str], figure_name: str,
    ) -> None:  # fmt: skip
        """Validate required numeric trace keys for a figure."""
        if not required:
            return
        missing: list[str] = []
        for path in sorted(required):
            if not self._trace_has_numeric_path(trace, path):
                missing.append(path)
        if missing:
            raise ValueError(f"Figure '{figure_name}' missing required trace keys: {', '.join(missing)}")

    def _validate_meta_keys(  # -------------------------------------------------------------------
        self, trace: TraceTree, required: set[str], figure_name: str,
    ) -> None:  # fmt: skip
        """Validate required metadata trace keys for a figure."""
        if not required:
            return
        missing = [path for path in sorted(required) if not self._trace_has_meta_path(trace, path)]
        if missing:
            raise ValueError(f"Figure '{figure_name}' missing required trace metadata: {', '.join(missing)}")

    def _trace_has_numeric_path(  # ---------------------------------------------------------------
        self, trace: TraceTree, path: str,
    ) -> bool:  # fmt: skip
        """Return whether a numeric leaf or numeric subtree exists at ``path``."""
        if not trace.path_to_index:
            return False

        idx = trace.path_to_index.get(path)
        if idx is not None:
            return bool(trace.leaf_is_numeric[idx])

        prefix = f"{path}/"
        for candidate, candidate_idx in trace.path_to_index.items():
            if candidate.startswith(prefix) and trace.leaf_is_numeric[candidate_idx]:
                return True
        return False

    def _trace_has_meta_path(  # ------------------------------------------------------------------
        self, trace: TraceTree, path: str,
    ) -> bool:  # fmt: skip
        """Return whether a metadata leaf exists at ``path``."""
        if not trace.path_to_index:
            return False
        idx = trace.path_to_index.get(path)
        if idx is None:
            return False
        return (not trace.leaf_is_numeric[idx]) and (trace.get_meta_path(path) is not None)

    def _extract_trace(  # ------------------------------------------------------------------------
        self, outputs: Any,
    ) -> Optional[TraceTree]:  # fmt: skip
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

    def _to_cpu_trace(  # -------------------------------------------------------------------------
        self, trace: TraceTree,
    ) -> TraceTree:  # fmt: skip
        """Normalize trace storage to CPU-backed arrays.

        The trace is finalized first, then any dense tensor leaves are detached
        and converted to NumPy arrays on CPU.
        """
        trace.finalize()
        if trace.dense_leaves is None:
            return trace
        for idx, leaf in enumerate(trace.dense_leaves):
            if isinstance(leaf, torch.Tensor):
                trace.dense_leaves[idx] = leaf.detach().cpu().numpy()
        trace.meta_first = [self._normalize_meta_value(value) for value in trace.meta_first]
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

    def generate_figure(  # -----------------------------------------------------------------------
        self, trainer: Trainer, trace: Any, ctx: FigureContext, spec: Any,
    ) -> None:  # fmt: skip
        """Generate a single figure and persist it using the configured sinks.

        Args:
            trainer: PyTorch Lightning trainer.
            trace: Captured trace object passed to the figure spec's ``plot``.
            ctx: Figure rendering context.
            spec: Figure spec retrieved from the registry.
        """
        fig = spec.plot(trace, ctx)
        if self.settings.save_pdf:
            self._save_pdf(trainer, fig, spec)
        if self.settings.log_tensorboard:
            self._log_tensorboard(trainer, fig, spec)

        # Always close the figure to prevent memory growth across epochs.
        plt.close(fig)

    def _save_pdf(  # -----------------------------------------------------------------------------
        self, trainer: Trainer, fig: Figure, spec: Any,
    ) -> None:  # fmt: skip
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

    def _log_tensorboard(  # ----------------------------------------------------------------------
        self, trainer: Trainer, fig: Figure, spec: Any,
    ) -> None:  # fmt: skip
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
