"""PyTorch Lightning callback for named evaluation regimes.

This module provides :class:`EvaluationRegimesCallback`, which runs named
diagnostic evaluation regimes out-of-band from fit-path validation.

Key behaviors:

- Runs **after** fit validation, not instead of it.
- Logs metrics under ``diag/<regime_id>/`` (or ``bench/<regime_id>/`` for bench regimes).
- Never writes into ``val/`` metrics.
- Exposes a read-only artifact lookup so :class:`~ehc_sn.callbacks.figures.FiguresCallback`
  can render from regime traces without touching validation dataloader outputs.
- Skips regimes silently (with a log message) if the module does not implement
  :class:`~ehc_sn.lightning.eval.contracts.SupportsEvaluationRegimes`.
"""

from __future__ import annotations

import logging
import traceback
from typing import Any, Optional

import lightning.pytorch as pl
from lightning.pytorch import LightningModule, Trainer
from pydantic import BaseModel, Field

from ehc_sn.lightning.eval.contracts import (
    EvaluationBatchArtifacts,
    EvaluationRegimeSettings,
    SupportsEvaluationRegimes,
)
from ehc_sn.lightning.eval.runner import (
    RegimeRunResult,
    load_provider,
    run_evaluation_regime,
)

log = logging.getLogger(__name__)


# =============================================================================
class EvaluationRegimesCallbackSettings(BaseModel, extra="forbid"):
    """Settings that control which named regimes run and when.

    Attributes:
        regimes: Ordered list of named evaluation regime configurations.
            An empty list disables the callback entirely.
    """

    regimes: list[EvaluationRegimeSettings] = Field(
        default_factory=list,
        description="Named evaluation regime configurations. Empty list disables regime evaluation.",
    )


# =============================================================================
class EvaluationRegimesCallback(pl.Callback):
    """Lightning callback that runs named evaluation regimes after fit validation.

    This callback:
        - Runs on global rank 0 only.
        - Executes each configured regime after ``on_validation_epoch_end`` completes,
          respecting each regime's schedule settings.
        - Logs all regime metrics via ``trainer.logger`` (if present) and ``pl_module.log_dict``.
        - Caches the latest artifact per regime for consumption by
          :class:`~ehc_sn.callbacks.figures.FiguresCallback`.
        - Never touches ``val/`` metrics or ``pl_module.val_metrics``.

    Legacy behavior (when no regimes are configured):
        The callback is a no-op.
    """

    def __init__(  # ----------------------------------------------------------
        self,
        settings: EvaluationRegimesCallbackSettings,
    ) -> None:
        """Initialize the callback.

        Args:
            settings: Regime configurations and schedule.
        """
        super().__init__()
        self.settings = settings
        # Regime id -> latest artifact from the most recent run.
        self._latest_artifacts: dict[str, EvaluationBatchArtifacts] = {}
        # Regime ids that produced a fresh artifact in the current validation cycle.
        self._refreshed_this_cycle: set[str] = set()

    def was_refreshed_this_cycle(  # ------------------------------------------
        self,
        regime_id: str,
    ) -> bool:
        """Return whether ``regime_id`` produced a fresh artifact in the current validation cycle."""
        return regime_id in self._refreshed_this_cycle

    def get_latest_artifact(  # -----------------------------------------------
        self,
        regime_id: str,
    ) -> Optional[EvaluationBatchArtifacts]:
        """Return the most recently cached artifact for a named regime.

        Args:
            regime_id: The regime identifier to look up.

        Returns:
            The latest :class:`~ehc_sn.lightning.eval.contracts.EvaluationBatchArtifacts`,
            or ``None`` if the regime has not run yet or produced no batches.
        """
        return self._latest_artifacts.get(regime_id)

    def on_validation_epoch_start(  # -----------------------------------------
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        """Clear the per-cycle freshness set before each validation run."""
        self._refreshed_this_cycle.clear()

    def on_validation_epoch_end(  # -------------------------------------------
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        """Run all scheduled named regimes after fit validation completes.

        This hook fires after fit-path validation is fully committed. Regime runs
        are entirely separate: they do not affect ``pl_module.val_metrics`` and
        do not touch the validation dataloaders.
        """
        if not trainer.is_global_zero:
            return
        if not self.settings.regimes:
            return
        if not isinstance(pl_module, SupportsEvaluationRegimes):
            log.warning(
                "EvaluationRegimesCallback: pl_module '%s' does not implement SupportsEvaluationRegimes. "
                "All regimes will be skipped.",
                type(pl_module).__name__,
            )
            return

        for regime in self.settings.regimes:
            if not self._should_run(trainer, regime):
                continue
            self._run_regime(trainer, pl_module, regime)

    def _should_run(  # -------------------------------------------------------
        self,
        trainer: Trainer,
        regime: EvaluationRegimeSettings,
    ) -> bool:
        """Return whether this regime is scheduled to run at the current trainer state.

        A regime runs if either its epoch schedule or step schedule fires.
        An every_n of 0 means that dimension is disabled.
        """
        epoch_ok = regime.schedule.every_n_epochs > 0 and (
            trainer.current_epoch % regime.schedule.every_n_epochs == 0
        )
        step_ok = regime.schedule.every_n_steps > 0 and (
            trainer.global_step % regime.schedule.every_n_steps == 0
        )
        # If both are disabled, fall back to every epoch.
        if (
            regime.schedule.every_n_epochs == 0
            and regime.schedule.every_n_steps == 0
        ):
            return True
        return epoch_ok or step_ok

    def _run_regime(  # -------------------------------------------------------
        self,
        trainer: Trainer,
        pl_module: SupportsEvaluationRegimes,
        regime: EvaluationRegimeSettings,
    ) -> None:
        """Load the provider, run the regime, log metrics, and cache the artifact.

        Failures are caught and logged; they do not interrupt training.
        """
        try:
            provider = load_provider(regime)
        except Exception as e:
            log.error(
                "EvaluationRegimesCallback: failed to load provider for regime '%s': %s",
                regime.regime_id,
                e,
            )
            traceback.print_exc()
            return

        log.info(
            "EvaluationRegimesCallback: running regime '%s' [%s] via provider '%s'.",
            regime.regime_id,
            regime.metric_namespace,
            provider.description(),
        )

        try:
            result: RegimeRunResult = run_evaluation_regime(
                pl_module=pl_module,
                regime=regime,
                provider=provider,
            )
        except Exception as e:
            log.error(
                "EvaluationRegimesCallback: regime '%s' failed: %s",
                regime.regime_id,
                e,
            )
            traceback.print_exc()
            return

        if result.n_batches == 0:
            log.warning(
                "EvaluationRegimesCallback: regime '%s' provider '%s' yielded 0 batches. "
                "No metrics logged.",
                regime.regime_id,
                result.provider_description,
            )
            return

        log.info(
            "EvaluationRegimesCallback: regime '%s' consumed %d batch(es) from '%s'.",
            regime.regime_id,
            result.n_batches,
            result.provider_description,
        )

        # Cache the latest artifact for FiguresCallback consumption.
        if result.latest_artifact is not None:
            self._latest_artifacts[regime.regime_id] = result.latest_artifact
            self._refreshed_this_cycle.add(regime.regime_id)

        # Log metrics via the module's log_dict (routed to whatever logger is active).
        if result.metrics:
            pl_module.log_dict(  # type: ignore[attr-defined]
                result.metrics,
                on_step=False,
                on_epoch=True,
                logger=True,
                add_dataloader_idx=False,
                sync_dist=False,  # Already rank-0 only.
            )


# =============================================================================
__all__ = ["EvaluationRegimesCallback", "EvaluationRegimesCallbackSettings"]
