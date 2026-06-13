"""Step-oriented progress bar for demand-driven training."""

from lightning.pytorch.callbacks import TQDMProgressBar


# =============================================================================
class StepProgressBar(TQDMProgressBar):
    """Progress bar keyed to ``global_step``, not DataLoader epoch.

    With ``max_epochs=-1`` and an infinite tick DataLoader, Lightning
    stays in a single internal epoch.  This callback swaps the default
    epoch-based counter for a step-based counter so the bar shows
    ``Training N/40000`` instead of ``Epoch 0/-2: 11/31``.
    """

    def init_train_tqdm(self):
        """Override the training tqdm bar with step-based total."""
        bar = super().init_train_tqdm()
        if self.trainer is not None and self.trainer.max_steps > 0:
            bar.total = self.trainer.max_steps
        bar.desc = "Training"
        return bar

    def on_train_batch_end(  # -------------------------------------------------
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
    ) -> None:
        """Sync the progress bar with ``global_step``."""
        super().on_train_batch_end(
            trainer, pl_module, outputs, batch, batch_idx
        )
        bar = self.train_progress_bar
        if bar is not None:
            bar.n = trainer.global_step
            if bar.total is not None and bar.n >= bar.total:
                bar.total = bar.n
            bar.refresh()


# =============================================================================
__all__ = ["StepProgressBar"]
