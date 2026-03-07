""" """

from __future__ import annotations

from typing import Any

import lightning.pytorch as pl
from lightning.pytorch import LightningModule, Trainer


# =================================================================================================
class TrainingMetricsCallback(pl.Callback):
    """ """

    def on_train_batch_end(  # --------------------------------------------------------------------
        self, trainer: Trainer, pl_module: LightningModule, outputs: Any, batch: Any, batch_idx: int,
    ) -> None:  # fmt: skip
        if not hasattr(pl_module, "train_metrics"):
            return
        if (trainer.global_step + 1) % trainer.log_every_n_steps != 0:
            return
        pl_module.log_dict(pl_module.train_metrics.compute(), on_step=True, on_epoch=False, logger=True)

    def on_validation_epoch_end(  # ---------------------------------------------------------------
        self, trainer: Trainer, pl_module: LightningModule,
    ) -> None:  # fmt: skip
        if not hasattr(pl_module, "val_metrics"):
            return
        vals = pl_module.val_metrics.compute()
        pl_module.log_dict(vals, on_step=False, on_epoch=True, logger=True, sync_dist=True)

        # Forward the primary accuracy to the progress bar.
        acc_key = next((k for k in vals if k.endswith("/all/accuracy")), None)
        if acc_key is not None:
            pl_module.log("val/accuracy", vals[acc_key], prog_bar=True, logger=True, sync_dist=True)


# =================================================================================================
__all__ = ["TrainingMetricsCallback"]
