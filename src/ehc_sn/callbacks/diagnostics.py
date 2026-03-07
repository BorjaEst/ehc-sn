""" """

from __future__ import annotations

from typing import Any, Literal

import lightning.pytorch as pl
from lightning.pytorch import LightningModule, Trainer
from pydantic import BaseModel, Field


# =================================================================================================
class DiagnosticsSettings(BaseModel, extra="forbid"):
    """ """

    diagnostic_level: Literal["standard", "research"] = Field(
        default="standard",
        description=(
            "Diagnostic tier. "
            "'standard' logs a fixed key subset for model health monitoring. "
            "'research' logs all keys present in the signals dict."
        ),
    )


# =================================================================================================
_STANDARD_KEYS: frozenset[str] = frozenset(
    # Keys that represent stable T2 diagnostics — always logged at 'standard' level.
    { "reward_mean", "reward_std", "q_mean", "q_std", "rpe_magnitude", "action_entropy", "steps_mean",
      "theta_cls_norm", "target_q_mean", "loss_actor", "loss_critic",
    }
)  # fmt: skip


# =================================================================================================
class DiagnosticsCallback(pl.Callback):
    """ """

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
        keys = signals.keys() if level == "research" else (k for k in signals if k in _STANDARD_KEYS)
        for key in keys:
            pl_module.log(f"model/{key}", signals[key], on_step=True, on_epoch=False, logger=True)


# =================================================================================================
__all__ = ["DiagnosticsSettings", "DiagnosticsCallback"]
