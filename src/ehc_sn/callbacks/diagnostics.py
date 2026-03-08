"""Lightning callback for logging diagnostic signals.

Models typically return a dict from ``training_step`` containing a ``signals``
payload. This callback logs a configurable subset of those signals.
"""

from __future__ import annotations

from typing import Any, Literal

import lightning.pytorch as pl
from lightning.pytorch import LightningModule, Trainer
from pydantic import BaseModel, Field

from ehc_sn.metrics.signals import STANDARD_SIGNALS


# =================================================================================================
class DiagnosticsSettings(BaseModel, extra="forbid"):
    """Settings controlling which diagnostic signals are logged."""

    diagnostic_level: Literal["standard", "research"] = Field(
        default="standard",
        description=(
            "Diagnostic tier. "
            "'standard' logs a fixed key subset for model health monitoring. "
            "'research' logs all keys present in the signals dict."
        ),
    )
    signal_prefix: str = Field(
        default="model/",
        description=(
            "Namespace prefix applied to all signal keys when logging. "
            "Use paradigm-specific prefixes (e.g. 'rl/', 'act/') to keep "
            "TensorBoard dashboards organised."
        ),
    )


# =================================================================================================
class DiagnosticsCallback(pl.Callback):
    """Logs step-level diagnostic signals emitted by the training step."""

    def __init__(  # ------------------------------------------------------------------------------
        self, settings: DiagnosticsSettings,
    ) -> None:  # fmt: skip
        super().__init__()
        self.settings = settings

    def on_train_batch_end(  # --------------------------------------------------------------------
        self, trainer: Trainer, pl_module: LightningModule, outputs: Any, batch: Any, batch_idx: int,
    ) -> None:  # fmt: skip
        if not isinstance(outputs, dict):
            return
        signals: dict | None = outputs.get("signals")
        if not signals:
            return

        level = self.settings.diagnostic_level
        prefix = self.settings.signal_prefix
        keys = signals.keys() if level == "research" else (k for k in signals if k in STANDARD_SIGNALS)
        for key in keys:
            pl_module.log(f"{prefix}{key}", signals[key], on_step=True, on_epoch=False, logger=True)


# =================================================================================================
__all__ = ["DiagnosticsSettings", "DiagnosticsCallback"]
