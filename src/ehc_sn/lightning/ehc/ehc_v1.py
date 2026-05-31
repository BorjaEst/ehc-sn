"""EHC v1 unified Lightning training surface.

Two public modes:
    spatial_pretrain   — arena replay, EHC variational objective, recurrent TBPTT.
    reason_pretrain — MazeHard deliberation, hybrid RL, single-step + recurrent eval.

EHCV1TrainingModel owns: model, bridge_adapter, train_metrics, val_metrics, config, mode.
All mode-specific logic is delegated to the private regime object.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Any, Mapping, Optional, TypeAlias, Union

import lightning as L
from pydantic import Field
from torchmetrics import MetricCollection

from ehc_sn.adapters.arena.ehc import ArenaEHCV1BridgeAdapter
from ehc_sn.adapters.arena.ehc.traces import ARENA_EHC_TRACE_FIELDS
from ehc_sn.adapters.arena.tem.traces import build_arena_tem_trace_meta
from ehc_sn.adapters.mazehard.ehc import (
    MAZE_HARD_EHC_ACTOR_CRITIC_TRACE_FIELDS,
    MazeHardEHCV1BridgeAdapter,
)
from ehc_sn.eval.contracts import (
    EvaluationCaseBatch,
    EvaluationCaseResult,
    EvaluationTraceRequest,
)
from ehc_sn.lightning.ehc.core._base import EHCMode, EHCRegime
from ehc_sn.lightning.ehc.pretrain.reason import (
    EHCReasonPretrainConfig,
    EHCReasonPretrainRegime,
)
from ehc_sn.lightning.ehc.pretrain.spatial import (
    EHCSpatialPretrainConfig,
    EHCSpatialPretrainRegime,
)
from ehc_sn.metrics.builders import build_train_metrics, build_val_metrics
from ehc_sn.metrics.keys import (
    EHC_ACC_OBS_INFERENCE_ALL,
    EHC_ACC_OBS_INFERENCE_REVISIT,
)
from ehc_sn.metrics.routes.ehc import (
    EHC_EPISODE_ROUTES,
    EHC_PRIMARY_VAL_ROUTE_KEY,
    EHC_STEP_ROUTES,
)
from ehc_sn.metrics.routes.rl import RL_EPISODE_ROUTES, RL_STEP_ROUTES
from ehc_sn.models.ehc.ehc_v1 import EHCModelV1, ModelSettingsV1
from ehc_sn.traces import build_trace_spec
from ehc_sn.types import Batch

# =============================================================================
ModelConfig_EHC_V1: TypeAlias = Annotated[  # ---------------------------------
    Union[
        EHCSpatialPretrainConfig,
        EHCReasonPretrainConfig,
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
    trace_spec, config, mode.

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
        self._trace_paradigm: str | None = None
        self._extra_trace_fields: tuple = ()
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
            self.train_metrics = build_train_metrics(EHC_STEP_ROUTES).clone(
                prefix="train/"
            )
            self.val_metrics = build_val_metrics(EHC_EPISODE_ROUTES).clone(
                prefix="val/"
            )
            self.primary_val_metric_key = f"val/{EHC_PRIMARY_VAL_ROUTE_KEY}"
            self._trace_paradigm = "ehc"
            self._extra_trace_fields = ARENA_EHC_TRACE_FIELDS
            self.trace_spec = build_trace_spec(
                "ehc", extra_fields=ARENA_EHC_TRACE_FIELDS
            )
            return EHCSpatialPretrainRegime(self, model, config)

        if isinstance(config, EHCReasonPretrainConfig):
            self.bridge_adapter = MazeHardEHCV1BridgeAdapter(
                model, config.adapter
            )
            self.train_metrics = build_train_metrics(RL_STEP_ROUTES).clone(
                prefix="train/"
            )
            self.val_metrics = build_val_metrics(RL_EPISODE_ROUTES).clone(
                prefix="val/"
            )
            self.primary_val_metric_key = None
            self.trace_spec = build_trace_spec(
                "rl", extra_fields=MAZE_HARD_EHC_ACTOR_CRITIC_TRACE_FIELDS
            )
            return EHCReasonPretrainRegime(self, model, config)

        else:
            raise ValueError(f"Unsupported config type: {type(config)}")

    @property
    def diagnostic_traces(self) -> tuple[Any, ...]:
        """Delegate to regime's diagnostic_traces."""
        return getattr(self._regime, "diagnostic_traces", ())

    def reset_diagnostic_traces(  # -------------------------------------------
        self,
    ) -> None:
        """Delegate to regime's reset_diagnostic_traces."""
        resetter = getattr(self._regime, "reset_diagnostic_traces", None)
        if callable(resetter):
            resetter()

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

    def on_validation_epoch_end(  # -------------------------------------------
        self,
    ) -> None:
        self._regime.on_validation_epoch_end()

    def execute_evaluation_batch(  # ------------------------------------------
        self,
        case: EvaluationCaseBatch,
        *,
        trace_request: EvaluationTraceRequest | None = None,
    ) -> EvaluationCaseResult:
        """Execute one provider-owned replay case when the active regime supports replay."""
        # Attach trace metadata from the case batch when the trace request
        # does not already carry it (offline eval path).
        if trace_request is not None and trace_request.trace_meta is None:
            trace_request = EvaluationTraceRequest(
                trace_spec=trace_request.trace_spec,
                trace_meta=dict(build_arena_tem_trace_meta(case.batch)),
            )

        execute_batch = getattr(self._regime, "execute_evaluation_batch", None)
        if not callable(execute_batch):
            raise RuntimeError(
                "EHC v1 mode "
                f"{self.mode.value!r} does not support replay evaluation execution."
            )
        return execute_batch(case, trace_request=trace_request)

    def aggregate_evaluation_case_metrics(  # ---------------------------------
        self,
        *,
        task: str,
        regime_id: str,
        regime_kind: str,
        case_results: Sequence[EvaluationCaseResult],
    ) -> dict[str, float | int]:
        """Aggregate per-case Arena observation accuracy into a regime summary."""
        _ = task, regime_id, regime_kind

        total_correct_all = 0.0
        total_count_all = 0.0
        total_correct_revisit = 0.0
        total_count_revisit = 0.0

        for case in case_results:
            for step in case.evaluated.steps:
                metrics = step.outputs.metrics

                ratio_all = metrics.extras.get(EHC_ACC_OBS_INFERENCE_ALL)
                if ratio_all is not None:
                    total_correct_all += float(ratio_all.numerator_sum.item())
                    total_count_all += float(ratio_all.denominator_sum.item())

                ratio_revisit = metrics.extras.get(
                    EHC_ACC_OBS_INFERENCE_REVISIT
                )
                if ratio_revisit is not None:
                    total_correct_revisit += float(
                        ratio_revisit.numerator_sum.item()
                    )
                    total_count_revisit += float(
                        ratio_revisit.denominator_sum.item()
                    )

        result: dict[str, float | int] = {}
        if total_count_all > 0:
            result["accuracy_all"] = total_correct_all / total_count_all
        if total_count_revisit > 0:
            result["accuracy_revisit"] = (
                total_correct_revisit / total_count_revisit
            )
        return result


# =============================================================================
def build_ehc_v1_regime(  # ---------------------------------------------------
    training_model: EHCV1TrainingModel,
    model: EHCModelV1,
    config: ModelConfig_EHC_V1,
) -> EHCRegime:
    """Build the appropriate regime based on the config type."""
    if isinstance(config, EHCSpatialPretrainConfig):
        return EHCSpatialPretrainRegime(training_model, model, config)
    if isinstance(config, EHCReasonPretrainConfig):
        return EHCReasonPretrainRegime(training_model, model, config)
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
        return EHCSpatialPretrainConfig(
            **{k: v for k, v in raw.items() if k in fields}
        )
    if mode == "reason_pretrain":
        fields = set(EHCReasonPretrainConfig.model_fields)
        return EHCReasonPretrainConfig(
            **{k: v for k, v in raw.items() if k in fields}
        )
    raise ValueError(
        f"Unknown EHC mode: {mode!r}. Must be 'spatial_pretrain' or 'reason_pretrain'."
    )


# =============================================================================
__all__ = [
    "EHCV1TrainingModel",
    "ModelConfig_EHC_V1",
    "parse_ehc_v1_config",
    "EHCMode",
]
