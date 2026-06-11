"""Lightning callbacks for metric logging.

The primary callback here logs aggregated metric collections exposed by the
Lightning module as ``train_metrics`` / ``val_metrics``.
"""

from __future__ import annotations

from typing import Any

import lightning.pytorch as pl
from lightning.pytorch import LightningModule, Trainer
from torchmetrics import MetricCollection


# =============================================================================
class MetricsCallback(pl.Callback):
    """Logs train/val metric collections from a LightningModule.

    Expects the module to expose:
        - ``train_metrics`` with a ``compute()`` method
        - ``val_metrics`` with a ``compute()`` method
    """

    def on_train_batch_end(  # ------------------------------------------------
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Logs training metrics at the end of each training batch."""
        train_metrics = _require_metric_collection(pl_module, "train_metrics")
        if train_metrics is None:
            return
        if (trainer.global_step + 1) % trainer.log_every_n_steps != 0:
            return
        pl_module.log_dict(
            train_metrics.compute(),
            on_step=True,
            on_epoch=False,
            logger=True,
            sync_dist=True,
        )

    def on_validation_epoch_end(  # -------------------------------------------
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        """Logs validation metrics at the end of each validation epoch."""
        val_metrics = _require_metric_collection(pl_module, "val_metrics")
        if val_metrics is None:
            return
        vals = val_metrics.compute()
        pl_module.log_dict(
            vals,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        _log_val_accuracy(pl_module, vals)


# =============================================================================
def _require_metric_collection(  # --------------------------------------------
    pl_module: LightningModule,
    attr_name: str,
) -> MetricCollection | None:
    """Get MetricCollection attribute from pl_module, with error handling."""
    metrics = getattr(pl_module, attr_name, None)
    if metrics is None:
        return None
    if not isinstance(metrics, MetricCollection):
        raise TypeError(
            "MetricsCallback requires "
            f"{attr_name} to be torchmetrics.MetricCollection, got "
            f"{type(metrics).__name__}."
        )
    return metrics


# =============================================================================
def _log_val_accuracy(  # -----------------------------------------------------
    pl_module: LightningModule,
    vals: dict[str, Any],
) -> None:
    """Log validation accuracy to progress bar, using primary key or heuristic."""
    acc_key = getattr(pl_module, "primary_val_metric_key", None)
    if acc_key is not None:
        if acc_key not in vals:
            raise KeyError(
                f"primary_val_metric_key '{acc_key}' was not found in "
                f"computed validation metrics: {sorted(vals)}"
            )
        pl_module.log(
            "val/accuracy",
            vals[acc_key],
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return

    candidate_keys = [key for key in vals if key.endswith("/all/accuracy")]
    if len(candidate_keys) == 1:
        pl_module.log(
            "val/accuracy",
            vals[candidate_keys[0]],
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )


# =============================================================================
__all__ = ["MetricsCallback"]
