"""EHC v1 unified Lightning training surface.

Two public modes:
    spatial_pretrain   — arena replay, EHC variational objective, recurrent TBPTT.
    reason_pretrain — MazeHard deliberation, hybrid RL, single-step + recurrent eval.

EHCV1TrainingModel owns: model, bridge_adapter, train_metrics, val_metrics, config, mode.
All mode-specific logic is delegated to the private regime object.
"""

from __future__ import annotations

from typing import Annotated, Any, Mapping, Optional, TypeAlias, Union

import lightning as L
from pydantic import Field
from torchmetrics import MetricCollection

from ehc_sn.adapters.arena.ehc import ArenaEHCV1BridgeAdapter
from ehc_sn.adapters.arena.ehc.traces import ARENA_EHC_TRACE_FIELDS
from ehc_sn.adapters.mazehard.ehc import MazeHardEHCV1BridgeAdapter
from ehc_sn.lightning.ehc.core._base import EHCMode, EHCRegime
from ehc_sn.lightning.ehc.pretrain.controller import EHCControllerPretrainConfig, EHCControllerPretrainRegime
from ehc_sn.lightning.ehc.pretrain.spatial import EHCSpatialPretrainConfig, EHCSpatialPretrainRegime
from ehc_sn.lightning.eval.contracts import EvaluationBatchArtifacts, EvaluationTraceRequest
from ehc_sn.metrics import build_train_metrics, build_val_metrics
from ehc_sn.metrics.routes.ehc import EHC_EPISODE_ROUTES, EHC_PRIMARY_VAL_ROUTE_KEY, EHC_STEP_ROUTES
from ehc_sn.metrics.routes.rl import RL_EPISODE_ROUTES, RL_STEP_ROUTES
from ehc_sn.metrics.traces import build_trace_spec
from ehc_sn.models.ehc.ehc_v1 import EHCModelV1, ModelSettingsV1
from ehc_sn.types import Batch

# =============================================================================
ModelConfig_EHC_V1: TypeAlias = Annotated[  # ---------------------------------
    Union[
        EHCSpatialPretrainConfig,
        EHCControllerPretrainConfig,
    ],
    Field(discriminator="mode"),
]
"""
EHC v1 unified config surface. Discriminated by "mode" field, which can be either:
    spatial_pretrain    — arena replay, EHC variational objective.
    reason_pretrain — MazeHard deliberation, hybrid RL.
"""


# =============================================================================
class EHCV1TrainingModel(L.LightningModule):
    """Unified EHC v1 training surface.

    Public attributes: model, bridge_adapter, train_metrics, val_metrics,
    trace_specs, config, mode.

    All mode-specific logic is delegated to the private regime object.
    """

    def __init__(  # ----------------------------------------------------------
        self,
        config: ModelConfig_EHC_V1,
    ) -> None:
        """Initialize model, bridge adapter, metrics, regime."""
        super().__init__()
        model_settings = ModelSettingsV1.from_config(config.model_config_path)
        self.model = EHCModelV1(model_settings)
        self.config = config
        self.mode = EHCMode(config.mode)

        # Regime is built after model + bridge_adapter are set, so freeze can run.
        self._regime = self._build_regime(self.model, config)
        self.automatic_optimization = False

    def _build_regime(  # ---------------------------------------------------------
        self,
        model: EHCModelV1,
        config: ModelConfig_EHC_V1,
    ) -> EHCRegime:
        """Build the appropriate regime based on the config type."""
        if isinstance(config, EHCSpatialPretrainConfig):
            self.bridge_adapter = ArenaEHCV1BridgeAdapter(model, config.adapter)
            self.train_metrics = build_train_metrics(EHC_STEP_ROUTES).clone(prefix="train/")
            self.val_metrics = build_val_metrics(EHC_EPISODE_ROUTES).clone(prefix="val/")
            self.primary_val_metric_key = f"val/{EHC_PRIMARY_VAL_ROUTE_KEY}"
            self.trace_specs = build_trace_spec("ehc", extra_fields=ARENA_EHC_TRACE_FIELDS)
            return EHCSpatialPretrainRegime(self, model, config)

        if isinstance(config, EHCControllerPretrainConfig):
            self.bridge_adapter = MazeHardEHCV1BridgeAdapter(model, config.adapter)
            self.train_metrics = build_train_metrics(RL_STEP_ROUTES).clone(prefix="train/")
            self.val_metrics = build_val_metrics(RL_EPISODE_ROUTES).clone(prefix="val/")
            self.primary_val_metric_key = None
            self.trace_specs = build_trace_spec("rl")
            return EHCControllerPretrainRegime(self, model, config)

        else:
            raise ValueError(f"Unsupported config type: {type(config)}")

    def set_eval_trace_keys(  # -----------------------------------------------
        self,
        keys: set[str],
    ) -> None:
        """Set semantic trace keys for figure capture."""
        setter = getattr(self._regime, "set_eval_trace_keys", None)
        if callable(setter):
            setter(keys)

    # -- Stable hook surface --------------------------------------------------

    def setup(  # -------------------------------------------------------------
        self,
        stage: Optional[str] = None,
    ) -> None:
        self._regime.setup(stage)

    def configure_optimizers(  # ----------------------------------------------
        self,
    ) -> tuple:
        return self._regime.configure_optimizers()

    def on_train_epoch_start(  # ----------------------------------------------
        self,
    ) -> None:
        self._regime.on_train_epoch_start()

    def on_validation_epoch_start(  # -----------------------------------------
        self,
    ) -> None:
        self._regime.on_validation_epoch_start()

    def training_step(  # -----------------------------------------------------
        self,
        batch: Batch,
        batch_idx: int,
    ) -> dict[str, Any]:
        return self._regime.training_step(batch, batch_idx)

    def validation_step(  # ---------------------------------------------------
        self,
        batch: Batch,
        batch_idx: int,
    ) -> dict[str, Any]:
        return self._regime.validation_step(batch, batch_idx)

    def build_evaluation_metrics(  # ------------------------------------------
        self,
        namespace: str,
    ) -> MetricCollection:
        return self._regime.build_evaluation_metrics(namespace)

    def execute_evaluation_batch(  # ------------------------------------------
        self,
        batch: Batch,
        trace_request: Optional[EvaluationTraceRequest],
        source_context: object | None = None,
    ) -> EvaluationBatchArtifacts:
        return self._regime.execute_evaluation_batch(batch, trace_request, source_context)


# =============================================================================
def build_ehc_v1_regime(  # ---------------------------------------------------
    training_model: EHCV1TrainingModel,
    model: EHCModelV1,
    config: ModelConfig_EHC_V1,
) -> EHCRegime:
    """Build the appropriate regime based on the config type."""
    if isinstance(config, EHCSpatialPretrainConfig):
        return EHCSpatialPretrainRegime(training_model, model, config)
    if isinstance(config, EHCControllerPretrainConfig):
        return EHCControllerPretrainRegime(training_model, model, config)
    else:
        raise ValueError(f"Unsupported config type: {type(config)}")


# =============================================================================
def parse_ehc_v1_config(  # ---------------------------------------------------
    raw: Mapping[str, Any],
) -> ModelConfig_EHC_V1:
    """Dispatch raw TOML mapping to the correct EHC v1 config class by mode."""
    mode = raw.get("mode")
    if mode == "spatial_pretrain":
        fields = set(EHCSpatialPretrainConfig.model_fields)
        return EHCSpatialPretrainConfig(**{k: v for k, v in raw.items() if k in fields})
    if mode == "reason_pretrain":
        fields = set(EHCControllerPretrainConfig.model_fields)
        return EHCControllerPretrainConfig(**{k: v for k, v in raw.items() if k in fields})
    raise ValueError(f"Unknown EHC mode: {mode!r}. Must be 'spatial_pretrain' or 'reason_pretrain'.")


# =============================================================================
__all__ = [
    "EHCV1TrainingModel",
    "ModelConfig_EHC_V1",
    "parse_ehc_v1_config",
    "EHCMode",
]
