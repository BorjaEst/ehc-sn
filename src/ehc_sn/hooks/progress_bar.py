from collections.abc import Iterable
from typing import Any, Callable, Dict, Optional, Union

import lightning.pytorch as pl
import torch
from tqdm import tqdm


class TqdmProgress(pl.Callback):
    """Callback for displaying progress with tqdm during training."""

    def __init__(self, refresh_rate: int = 1):
        """Initialize the progress bar callback.

        Args:
            refresh_rate: Determines how often the progress bar gets updated
        """
        super().__init__()
        self.refresh_rate = refresh_rate
        self._train_progress_bar = None
        self._val_progress_bar = None
        self._test_progress_bar = None
        self._predict_progress_bar = None
        self._batch_indices = {"train": 0, "val": 0, "test": 0, "predict": 0}
        self._last_metrics = {}

    def _init_progress_bar(self, total: int, desc: str) -> tqdm:
        """Initialize a progress bar with given total and description.

        Args:
            total: Total number of steps
            desc: Description for the progress bar

        Returns:
            Initialized tqdm progress bar
        """
        return tqdm(
            total=total,
            desc=desc,
            dynamic_ncols=True,
            leave=True,
            unit="batch",
        )

    def _update_progress_bar(
        self,
        progress_bar,
        batch_idx: int,
        current_epoch: int,
        total_epochs: int,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update the progress bar with the current batch index and metrics.

        Args:
            progress_bar: The progress bar to update
            batch_idx: Current batch index
            current_epoch: Current epoch number
            total_epochs: Total number of epochs
            metrics: Optional metrics to display in the progress bar
        """
        if progress_bar is not None and batch_idx % self.refresh_rate == 0:
            desc = f"Epoch {current_epoch+1}/{total_epochs}"

            # Add metrics to the description if available
            if metrics:
                metric_str = []
                for k, v in metrics.items():
                    if isinstance(v, torch.Tensor):
                        v = v.item()
                    # Format metric values to 4 decimal places
                    if isinstance(v, (float, int)):
                        metric_str.append(f"{k}: {v:.4f}")

                if metric_str:
                    desc += " [" + ", ".join(metric_str) + "]"

            progress_bar.set_description(desc)
            progress_bar.update(self.refresh_rate)

    def on_train_epoch_start(self, trainer: Any, *args: Any, **kwargs: Any) -> None:
        """Set up the progress bar at the start of a training epoch."""
        current_epoch = trainer.current_epoch
        total_batches = trainer.num_training_batches
        total_epochs = trainer.max_epochs or 1000  # Default if not specified

        if self._train_progress_bar is not None:
            self._train_progress_bar.close()

        self._batch_indices["train"] = 0
        self._train_progress_bar = self._init_progress_bar(
            total=total_batches, desc=f"Epoch {current_epoch+1}/{total_epochs} [Train]"
        )

    def on_train_batch_end(
        self, trainer: Any, pl_module: Any, outputs: Any, batch: Any, batch_idx: Any, **kwargs: Any
    ) -> None:
        """Update the progress bar after a training batch."""
        self._batch_indices["train"] += 1
        batch_idx = self._batch_indices["train"]

        # Extract metrics from outputs
        metrics = {}
        if isinstance(outputs, dict):
            for key, value in outputs.items():
                if key.startswith("train_") and "loss" in key:
                    metrics[key] = value
                    self._last_metrics[key] = value
        elif hasattr(trainer, "logged_metrics"):
            for key, value in trainer.logged_metrics.items():
                if key.startswith("train_") and "loss" in key:
                    metrics[key] = value
                    self._last_metrics[key] = value

        self._update_progress_bar(
            self._train_progress_bar, batch_idx, trainer.current_epoch, trainer.max_epochs or 1000, metrics
        )

    def on_train_epoch_end(self, trainer: Any, *args: Any, **kwargs: Any) -> None:
        """Clean up the progress bar at the end of a training epoch."""
        if self._train_progress_bar is not None:
            self._train_progress_bar.close()
            self._train_progress_bar = None

    def on_validation_epoch_start(self, trainer: Any, *args: Any, **kwargs: Any) -> None:
        """Set up the progress bar at the start of a validation epoch."""
        current_epoch = trainer.current_epoch
        # Use num_val_batches directly from the trainer
        total_batches = trainer.num_val_batches
        total_epochs = trainer.max_epochs or 1000

        if self._val_progress_bar is not None:
            self._val_progress_bar.close()

        self._batch_indices["val"] = 0
        self._val_progress_bar = self._init_progress_bar(
            total=total_batches, desc=f"Epoch {current_epoch+1}/{total_epochs} [Validation]"
        )

    def on_validation_batch_end(
        self, trainer: Any, pl_module: Any, outputs: Any, batch: Any, batch_idx: Any, **kwargs: Any
    ) -> None:
        """Update the progress bar after a validation batch."""
        self._batch_indices["val"] += 1
        batch_idx = self._batch_indices["val"]

        # Extract metrics from outputs
        metrics = {}
        if isinstance(outputs, dict):
            for key, value in outputs.items():
                if key.startswith("val_") and "loss" in key:
                    metrics[key] = value
                    self._last_metrics[key] = value
        elif hasattr(trainer, "logged_metrics"):
            for key, value in trainer.logged_metrics.items():
                if key.startswith("val_") and "loss" in key:
                    metrics[key] = value
                    self._last_metrics[key] = value

        self._update_progress_bar(
            self._val_progress_bar, batch_idx, trainer.current_epoch, trainer.max_epochs or 1000, metrics
        )

    def on_validation_epoch_end(self, trainer: Any, *args: Any, **kwargs: Any) -> None:
        """Clean up the progress bar at the end of a validation epoch."""
        if self._val_progress_bar is not None:
            # Update with final validation metrics if available
            metrics = {}
            if hasattr(trainer, "logged_metrics"):
                for key, value in trainer.logged_metrics.items():
                    if key.startswith("val_") and "loss" in key:
                        metrics[key] = value
                        self._last_metrics[key] = value

            # Display final metrics (if available)
            if metrics:
                desc = f"Epoch {trainer.current_epoch+1}/{trainer.max_epochs or 1000} [Validation] "
                metric_str = []
                for k, v in metrics.items():
                    if isinstance(v, torch.Tensor):
                        v = v.item()
                    metric_str.append(f"{k}: {v:.4f}")

                if metric_str:
                    desc += "[" + ", ".join(metric_str) + "]"
                self._val_progress_bar.set_description(desc)

            self._val_progress_bar.close()
            self._val_progress_bar = None

    def on_test_epoch_start(self, trainer: Any, *args: Any, **kwargs: Any) -> None:
        """Set up the progress bar at the start of a test epoch."""
        total_batches = trainer.num_test_batches

        if self._test_progress_bar is not None:
            self._test_progress_bar.close()

        self._batch_indices["test"] = 0
        self._test_progress_bar = self._init_progress_bar(total=total_batches, desc="Testing")

    def on_test_batch_end(
        self, trainer: Any, pl_module: Any, outputs: Any, batch: Any, batch_idx: Any, **kwargs: Any
    ) -> None:
        """Update the progress bar after a test batch."""
        self._batch_indices["test"] += 1
        batch_idx = self._batch_indices["test"]

        # Extract metrics from outputs
        metrics = {}
        if isinstance(outputs, dict):
            for key, value in outputs.items():
                if key.startswith("test_") and "loss" in key:
                    metrics[key] = value
                    self._last_metrics[key] = value
        elif hasattr(trainer, "logged_metrics"):
            for key, value in trainer.logged_metrics.items():
                if key.startswith("test_") and "loss" in key:
                    metrics[key] = value
                    self._last_metrics[key] = value

        self._update_progress_bar(self._test_progress_bar, batch_idx, 0, 1, metrics)

    def on_test_epoch_end(self, trainer: Any, *args: Any, **kwargs: Any) -> None:
        """Clean up the progress bar at the end of a test epoch."""
        if self._test_progress_bar is not None:
            # Update with final test metrics if available
            metrics = {}
            if hasattr(trainer, "logged_metrics"):
                for key, value in trainer.logged_metrics.items():
                    if key.startswith("test_") and "loss" in key:
                        metrics[key] = value

            # Display final metrics (if available)
            if metrics:
                desc = "Testing "
                metric_str = []
                for k, v in metrics.items():
                    if isinstance(v, torch.Tensor):
                        v = v.item()
                    metric_str.append(f"{k}: {v:.4f}")

                if metric_str:
                    desc += "[" + ", ".join(metric_str) + "]"
                self._test_progress_bar.set_description(desc)

            self._test_progress_bar.close()
            self._test_progress_bar = None

    def on_predict_epoch_start(self, trainer: Any, *args: Any, **kwargs: Any) -> None:
        """Set up the progress bar at the start of a predict epoch."""
        total_batches = trainer.num_predict_batches

        if self._predict_progress_bar is not None:
            self._predict_progress_bar.close()

        self._batch_indices["predict"] = 0
        self._predict_progress_bar = self._init_progress_bar(total=total_batches, desc="Predicting")

    def on_predict_batch_end(
        self, trainer: Any, pl_module: Any, outputs: Any, batch: Any, batch_idx: Any, **kwargs: Any
    ) -> None:
        """Update the progress bar after a predict batch."""
        self._batch_indices["predict"] += 1
        batch_idx = self._batch_indices["predict"]
        self._update_progress_bar(self._predict_progress_bar, batch_idx, 0, 1)

    def on_predict_epoch_end(self, trainer: Any, *args: Any, **kwargs: Any) -> None:
        """Clean up the progress bar at the end of a predict epoch."""
        if self._predict_progress_bar is not None:
            self._predict_progress_bar.close()
            self._predict_progress_bar = None

    def on_exception(self, trainer: Any, *args: Any, **kwargs: Any) -> None:
        """Handle exceptions by closing all progress bars."""
        for bar in [
            self._train_progress_bar,
            self._val_progress_bar,
            self._test_progress_bar,
            self._predict_progress_bar,
        ]:
            if bar is not None:
                bar.close()

        self._train_progress_bar = None
        self._val_progress_bar = None
        self._test_progress_bar = None
        self._predict_progress_bar = None
