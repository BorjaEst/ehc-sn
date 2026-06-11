"""Lightning callback for logging diagnostic telemetry.

Models typically return a dict from ``training_step`` containing a ``signals``
payload. This callback is the consumer policy layer that filters and logs a
configurable subset of telemetry keys.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from typing import Any, Literal

import lightning.pytorch as pl
import torch
from lightning.pytorch import LightningModule, Trainer
from pydantic import BaseModel, Field

from ehc_sn.metrics.signals import (
    ACT_SIGNALS,
    CROSS_PARADIGM_SIGNALS,
    RL_SIGNALS,
    TEM_SIGNALS,
    VAR_SIGNALS,
)

STANDARD_SIGNALS: frozenset[str] = (
    CROSS_PARADIGM_SIGNALS
    | ACT_SIGNALS
    | RL_SIGNALS
    | VAR_SIGNALS
    | TEM_SIGNALS
)


# =============================================================================
class DiagnosticsSettings(BaseModel, extra="forbid"):
    """Settings controlling which diagnostic telemetry keys are logged."""

    diagnostic_level: Literal["standard", "research"] = Field(
        default="standard",
        description=(
            "Diagnostic tier. "
            "'standard' logs a fixed key subset for model health monitoring. "
            "'research' logs all keys present in the signals dict."
        ),
    )
    namespace: str = Field(
        default="signals",
        description=(
            "Middle tag segment used in split-aware logging names. "
            "Final tags are emitted as '<split>/<namespace>/<signal_key>'."
        ),
    )
    train_every_n_steps: int = Field(
        default=1,
        ge=1,
        description="Callback-owned train-step logging cadence.",
    )
    enable_validation: bool = Field(
        default=False,
        description="Enable diagnostics logging for validation outputs.",
    )
    enable_test: bool = Field(
        default=False,
        description="Enable diagnostics logging for test outputs.",
    )
    malformed_payload_policy: Literal["drop", "raise"] = Field(
        default="drop",
        description="Policy for malformed step outputs or signal payloads.",
    )
    unknown_signal_policy: Literal["drop", "raise"] = Field(
        default="drop",
        description="Policy for keys outside the standard set at standard tier.",
    )
    non_scalar_policy: Literal["drop", "raise"] = Field(
        default="drop",
        description="Policy for values that cannot be coerced to a scalar.",
    )
    non_finite_policy: Literal["drop", "raise"] = Field(
        default="drop",
        description="Policy for NaN/Inf scalar values.",
    )
    rank_zero_only: bool = Field(
        default=True,
        description="Only emit logs on rank zero when distributed training is active.",
    )


# =============================================================================
class DiagnosticsCallback(pl.Callback):
    """Logs step-level diagnostic telemetry emitted by the training step."""

    def __init__(  # ----------------------------------------------------------
        self,
        settings: DiagnosticsSettings,
    ) -> None:
        """Initialize the callback with the given settings."""
        super().__init__()
        self.settings = settings

    def on_train_batch_end(  # -------------------------------------------------
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Log train diagnostics emitted by the training step, if present."""
        if not self._should_log_train_step(trainer):
            return

        self._log_split_signals(
            split="train",
            trainer=trainer,
            pl_module=pl_module,
            outputs=outputs,
        )

    def on_validation_batch_end(  # -------------------------------------------
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Optionally log validation diagnostics from step outputs."""
        if not self.settings.enable_validation:
            return
        self._log_split_signals(
            split="val",
            trainer=trainer,
            pl_module=pl_module,
            outputs=outputs,
        )

    def on_test_batch_end(  # -------------------------------------------------
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Optionally log test diagnostics from step outputs."""
        if not self.settings.enable_test:
            return
        self._log_split_signals(
            split="test",
            trainer=trainer,
            pl_module=pl_module,
            outputs=outputs,
        )

    def _should_log_train_step(self, trainer: Trainer) -> bool:
        """Return whether this train step should emit diagnostics."""
        cadence = self.settings.train_every_n_steps
        if cadence <= 1:
            return True
        return (trainer.global_step + 1) % cadence == 0

    def _log_split_signals(
        self,
        split: Literal["train", "val", "test"],
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
    ) -> None:
        """Extract, filter, and log eligible scalar diagnostics for one split."""
        if self.settings.rank_zero_only and not trainer.is_global_zero:
            return

        signals = self._extract_signals(outputs)
        if signals is None:
            return

        for key, value in signals.items():
            if not self._allow_key(key):
                continue

            scalar = self._coerce_scalar(value, key=key)
            if scalar is None:
                continue

            pl_module.log(
                self._tag(split=split, key=key),
                scalar,
                on_step=True,
                on_epoch=False,
                logger=True,
            )

    def _extract_signals(self, outputs: Any) -> Mapping[str, Any] | None:
        """Return ``outputs['signals']`` when present and well-formed."""
        if not isinstance(outputs, Mapping):
            self._handle_policy(
                self.settings.malformed_payload_policy,
                "malformed outputs payload",
            )
            return None

        signals = outputs.get("signals")
        if signals is None:
            return None
        if not isinstance(signals, Mapping):
            self._handle_policy(
                self.settings.malformed_payload_policy,
                "signals payload is not a mapping",
            )
            return None
        return signals

    def _allow_key(self, key: str) -> bool:
        """Return whether the signal key should be considered for logging."""
        if self.settings.diagnostic_level == "research":
            return True
        if key in STANDARD_SIGNALS:
            return True
        self._handle_policy(
            self.settings.unknown_signal_policy,
            f"unknown standard-tier signal key '{key}'",
        )
        return False

    def _coerce_scalar(self, value: Any, key: str) -> float | int | None:
        """Convert value to a finite Python scalar or apply policy and drop."""
        scalar: float | int
        if isinstance(value, bool):
            scalar = int(value)
        elif isinstance(value, Real):
            scalar = float(value)
        elif torch.is_tensor(value):
            tensor = value.detach()
            if tensor.numel() != 1:
                self._handle_policy(
                    self.settings.non_scalar_policy,
                    f"non-scalar tensor for signal '{key}'",
                )
                return None
            scalar = float(tensor.item())
        else:
            self._handle_policy(
                self.settings.non_scalar_policy,
                f"non-scalar value for signal '{key}'",
            )
            return None

        if not math.isfinite(float(scalar)):
            self._handle_policy(
                self.settings.non_finite_policy,
                f"non-finite value for signal '{key}'",
            )
            return None
        return scalar

    def _tag(self, split: str, key: str) -> str:
        """Build split-aware diagnostics tag name."""
        namespace = self.settings.namespace.strip("/")
        return f"{split}/{namespace}/{key}"

    @staticmethod
    def _handle_policy(policy: Literal["drop", "raise"], detail: str) -> None:
        """Apply policy for malformed/unsupported diagnostics payload values."""
        if policy == "raise":
            raise ValueError(detail)


# =============================================================================
__all__ = ["DiagnosticsSettings", "DiagnosticsCallback"]
