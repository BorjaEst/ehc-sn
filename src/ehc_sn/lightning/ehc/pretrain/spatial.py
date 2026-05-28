"""EHC v1 spatial pretrain regime: arena replay, EHC variational objective, recurrent TBPTT."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import lightning as L
import torch
from pydantic import BaseModel, Field
from torch.optim import Optimizer
from torchmetrics import MetricCollection

from ehc_sn.adapters.arena.ehc import (
    ArenaEHCAdapterSettings,
    ArenaEHCTaskBinding,
    ArenaEHCV1BridgeAdapter,
)
from ehc_sn.controllers.replay.trajectory import (
    ReplayTrajectoryController,
    ReplayTrajectoryControllerConfig,
)
from ehc_sn.data.datamodules import Datamodule
from ehc_sn.eval.contracts import (
    EvaluationCaseBatch,
    EvaluationCaseResult,
    EvaluationTraceRequest,
)
from ehc_sn.eval.executor import execute_replay_evaluation_batch
from ehc_sn.lightning.diagnostics import DiagnosticTraceSpec
from ehc_sn.lightning.ehc.core._base import EHCMode, EHCRegime, freeze_params
from ehc_sn.lightning.ehc.core.runtime import (
    EHCRuntimeState,
    RuntimeConfig,
    resolve_ehc_runtime,
)
from ehc_sn.metrics.reducers import HiddenNormHistogram, OccupancyHistogram
from ehc_sn.metrics.renderers import (
    log_reducer_figure,
    render_hidden_norm_histogram,
    render_occupancy_histogram,
)
from ehc_sn.metrics.rollout import (
    make_observed_step_metric_observer,
    update_metric_collection_from_evaluated_chunk,
)
from ehc_sn.metrics.routes.ehc import (
    EHC_EPISODE_ROUTES,
    EHC_STEP_ROUTES,
)
from ehc_sn.models.ehc.ehc_v1 import EHCModelV1, ModelSettingsV1
from ehc_sn.objectives.ehc import (
    EHCObjective,
    EHCObjectiveConfig,
    resolve_objective_schedule,
)
from ehc_sn.rollouts.buffers import FifoBuffer
from ehc_sn.rollouts.partial_reset import PartialResetBatchAssembler
from ehc_sn.rollouts.runtime import RecurrentRunner
from ehc_sn.rollouts.sources import PartialResetSource, RepeatSource
from ehc_sn.tasks.arena.capabilities.replay import ArenaReplayCapability
from ehc_sn.tasks.arena.runtime import infer_arena_replay_batch_keys
from ehc_sn.tasks.arena.traces import (
    ArenaEvaluationSourceContext,
    apply_arena_trace_supplements,
    build_arena_trace_supplements,
)
from ehc_sn.traces import TraceTree, build_trace_spec
from ehc_sn.training.optim import Adam, AdamConfig
from ehc_sn.training.rollout import (
    score_rollout_streaming,
)
from ehc_sn.training.schedules import (
    CosineAnnealingLRWithWarmup,
    SchedulerConfig,
)
from ehc_sn.types import Batch


# =============================================================================
class EHCSpatialPretrainConfig(BaseModel, extra="forbid"):
    """Config for spatial pretrain (arena replay, EHC variational objective)."""

    mode: Literal["spatial_pretrain"] = "spatial_pretrain"

    model_config_path: Path = Field(
        ...,
        description="Path to the model config TOML file used to construct the "
        "EHCModelV1.",
    )
    adapter: ArenaEHCAdapterSettings = Field(
        ...,
        description="Adapter settings for encoding batches and decoding model "
        "outputs.",
    )
    controller: ReplayTrajectoryControllerConfig = Field(
        ...,
        description="Controller config for replay trajectory control during "
        "spatial pretrain.",
    )
    objective: EHCObjectiveConfig = Field(
        ...,
        description="EHC variational objective config for spatial pretrain.",
    )
    optimizer: AdamConfig = Field(
        default_factory=AdamConfig,
        description="Optimizer config for spatial pretrain (applied to bridge "
        "adapter parameters).",
    )
    scheduler: SchedulerConfig = Field(
        default_factory=SchedulerConfig,
        description="Learning rate scheduler config for spatial pretrain.",
    )
    runtime: RuntimeConfig = Field(
        default_factory=RuntimeConfig,
        description="Runtime settings for training and validation execution, "
        "including max rollout steps for validation.",
    )


# =============================================================================
class EHCSpatialPretrainRegime:
    """Arena replay spatial pretrain regime.

    Trains spatial_core + arena_decoder.
    Freezes controller_body, controller_heads, controller_bridge.
    """

    def __init__(  # ----------------------------------------------------------
        self,
        lm: L.LightningModule,
        model: EHCModelV1,
        config: EHCSpatialPretrainConfig,
    ) -> None:
        self._lm = lm
        self._config = config
        self._adapter = ArenaEHCV1BridgeAdapter(model, config.adapter)

        # Freeze controller params (pfc, str, bridge projections).
        freeze_params(model, "pfc", "str", "pfc_to_hpc", "hpc_to_pfc")

        self._train_controller: ReplayTrajectoryController | None = None
        self._train_objective: EHCObjective | None = None
        self._eval_controller: ReplayTrajectoryController | None = None
        self._eval_objective: EHCObjective | None = None

        self._train_runner = RecurrentRunner()
        self._eval_runner = RecurrentRunner()

        # Bounded diagnostic reducers (occupancy histogram + hidden-norm histogram).
        # These accumulate bounded sufficient statistics over validation batches
        # and are computed/reset at epoch end.
        self._val_occupancy = OccupancyHistogram(
            n_locations=config.adapter.observation_dim, max_batches=10
        )
        self._val_hidden_norms = HiddenNormHistogram(
            bin_edges=torch.linspace(0.0, 50.0, 51), max_batches=10
        )
        self._val_reducer_collection = MetricCollection(
            {
                "occupancy": self._val_occupancy,
                "hidden_norms": self._val_hidden_norms,
            }
        ).clone(prefix="val_diag/")

        self._train_carry = None
        self._diag_params_cache: dict[str, object] | None = None
        self._train_buffer: FifoBuffer | None = None
        self._train_batch_assembler: PartialResetBatchAssembler | None = None
        self._trace_paradigm: str = "ehc"
        self.diagnostic_trace_spec: DiagnosticTraceSpec = DiagnosticTraceSpec(
            enabled=False,
            max_batches=2,
            keys=(),
        )
        self._diagnostic_traces: list[Any] = []

    def _build_runtime(  # ----------------------------------------------------
        self,
    ) -> tuple[ReplayTrajectoryController, EHCObjective]:
        """ """
        lm = self._lm
        controller = ReplayTrajectoryController(
            backbone=lm.bridge_adapter,
            config=self._config.controller,
            runtime=ArenaReplayCapability(),
        )
        objective = EHCObjective(
            config=self._config.objective,
            task_binding=ArenaEHCTaskBinding(),
        )
        return controller, objective

    def _ensure_train_runtime(  # ---------------------------------------------
        self,
    ) -> None:
        """ """
        if self._train_controller is None:
            self._train_controller, self._train_objective = (
                self._build_runtime()
            )

    def _ensure_eval_runtime(  # ----------------------------------------------
        self,
    ) -> None:
        """ """
        if self._eval_controller is None:
            self._eval_controller, self._eval_objective = self._build_runtime()

    def _require_train_controller(  # -----------------------------------------
        self,
    ) -> ReplayTrajectoryController:
        """ """
        self._ensure_train_runtime()
        assert self._train_controller is not None
        return self._train_controller

    def _require_train_objective(  # ------------------------------------------
        self,
    ) -> EHCObjective:
        self._ensure_train_runtime()
        assert self._train_objective is not None
        return self._train_objective

    def _require_eval_controller(  # ------------------------------------------
        self,
    ) -> ReplayTrajectoryController:
        """ """
        self._ensure_eval_runtime()
        assert self._eval_controller is not None
        return self._eval_controller

    def _require_eval_objective(  # -------------------------------------------
        self,
    ) -> EHCObjective:
        """ """
        self._ensure_eval_runtime()
        assert self._eval_objective is not None
        return self._eval_objective

    def _train_chunk_steps(  # ------------------------------------------------
        self,
    ) -> int:
        """ """
        if self._config.controller.window_size is not None:
            return self._config.controller.window_size
        return self._config.runtime.sequence.tbptt_steps

    def _apply_runtime(  # ----------------------------------------------------
        self,
        step: int,
    ) -> EHCRuntimeState:
        """ """
        runtime = resolve_ehc_runtime(step, self._config.runtime)
        self._lm.model.set_runtime(
            runtime.eta, runtime.hebbian_decay, runtime.p2g_uncertainty_offset
        )
        return runtime

    def _ensure_train_batch_assembler(  # -------------------------------------
        self, batch: Batch
    ) -> PartialResetBatchAssembler:
        """ """
        if self._train_batch_assembler is not None:
            return self._train_batch_assembler
        keys = infer_arena_replay_batch_keys(batch)
        if "__trajectory_id__" in batch:
            keys = keys + ("__trajectory_id__",)
        local_bs = batch[next(iter(batch))].shape[0]
        capacity_rows = 4 * local_bs
        self._train_buffer = FifoBuffer(capacity_rows, keys, pin_memory=True)
        self._train_batch_assembler = PartialResetBatchAssembler(
            buffer=self._train_buffer, keys=keys
        )
        return self._train_batch_assembler

    def _reset_train_stream(  # -----------------------------------------------
        self,
    ) -> None:
        """ """
        self._train_carry = None
        if self._train_buffer is not None:
            self._train_buffer.clear()

    @property
    def diagnostic_traces(self) -> tuple[Any, ...]:
        """Return bounded diagnostic traces captured during the just-completed
        validation epoch.  Empty tuple when trace capture was disabled or no
        batches were processed.
        """
        return tuple(self._diagnostic_traces)

    def reset_diagnostic_traces(  # -------------------------------------------
        self,
    ) -> None:
        """Clear the internal diagnostic trace buffer."""
        self._diagnostic_traces.clear()

    def _build_static_params(  # ----------------------------------------------
        self,
    ) -> dict[str, object]:
        """Return static diagnostic parameters injected into the rollout carry."""
        if self._diag_params_cache is None:
            model = self._lm.model
            self._diag_params_cache = {
                "lec_alpha_sigmoid": torch.stack(
                    [
                        torch.sigmoid(a).detach().cpu()
                        for a in model.lec.filter.alpha
                    ]
                ),
                "lec_w_f_sigmoid": torch.stack(
                    [torch.sigmoid(w).detach().cpu() for w in model.lec.w_f]
                ),
            }
        return self._diag_params_cache

    # -- Regime hooks -----------------------------------------------------------------------------

    def setup(  # -------------------------------------------------------------
        self,
        stage: str | None,
    ) -> None:
        """ """
        if stage in (None, "fit"):
            self._ensure_train_runtime()
            self._ensure_eval_runtime()
        elif stage in ("validate", "test"):
            self._ensure_eval_runtime()

    def configure_optimizers(  # ----------------------------------------------
        self,
    ) -> tuple[list[Optimizer], list[dict[str, Any]]]:
        """ """
        total_steps = int(self._lm.trainer.estimated_stepping_batches)
        # spatial_pretrain: exclude controller params (already frozen via requires_grad=False).
        excluded = (
            {id(p) for p in self._lm.model.pfc.parameters()}
            | {id(p) for p in self._lm.model.str.parameters()}
            | {id(p) for p in self._lm.model.pfc_to_hpc.parameters()}
            | {id(p) for p in self._lm.model.hpc_to_pfc.parameters()}
        )
        sup_params = [
            p
            for p in self._lm.bridge_adapter.parameters()
            if p.requires_grad and id(p) not in excluded
        ]
        opt = Adam(sup_params, self._config.optimizer)
        schedulers: list[dict[str, Any]] = [
            {
                "scheduler": CosineAnnealingLRWithWarmup(
                    opt, total_steps, self._config.scheduler
                ),
                "interval": "step",
                "frequency": 1,
                "name": "optim/main",
            }
        ]
        return [opt], schedulers

    def on_train_epoch_start(  # ----------------------------------------------
        self,
    ) -> None:
        """ """
        self._reset_train_stream()
        self._lm.train_metrics.reset()

    def on_validation_epoch_start(  # -----------------------------------------
        self,
    ) -> None:
        """ """
        self._lm.val_metrics.reset()
        self._val_reducer_collection.reset()

    def training_step(  # -----------------------------------------------------
        self,
        batch: Batch,
        batch_idx: int,
    ) -> dict[str, Any]:
        """ """
        lm = self._lm
        runtime = self._apply_runtime(lm.global_step)
        lm.log(
            "train/runtime/eta", runtime.eta,
            on_step=True, on_epoch=False, logger=True,
        )  # fmt: skip
        lm.log(
            "train/runtime/hebbian_decay",
            runtime.hebbian_decay,
            on_step=True,
            on_epoch=False,
            logger=True,
        )  # fmt: skip
        lm.log(
            "train/runtime/p2g_uncertainty_offset", runtime.p2g_uncertainty_offset,
            on_step=True, on_epoch=False, logger=True,
        )  # fmt: skip
        lm.log(
            "train/runtime/p2g_trust", runtime.p2g_trust,
            on_step=True, on_epoch=False, logger=True,
        )  # fmt: skip
        ctrl = self._require_train_controller()
        obj = self._require_train_objective()
        assembler = self._ensure_train_batch_assembler(batch)

        if self._train_carry is None:
            self._train_carry = ctrl.initial_state(batch)

        source = PartialResetSource(
            incoming=batch, assembler=assembler, carry0=self._train_carry
        )
        evaluation = score_rollout_streaming(
            runner=self._train_runner,
            source=source,
            controller=ctrl,
            carry=self._train_carry,
            objective=obj,
            max_rollout_steps=self._train_chunk_steps(),
            objective_options=resolve_objective_schedule(
                lm.global_step, self._config.objective
            ),
            observed_step_observer=make_observed_step_metric_observer(
                lm.train_metrics, EHC_STEP_ROUTES
            ),
        )
        self._train_carry = evaluation.execution.final_carry.detach()
        loss = evaluation.loss / self._train_chunk_steps()

        optimizers = lm.optimizers()
        for opt in (
            optimizers if isinstance(optimizers, list) else [optimizers]
        ):
            opt.zero_grad(set_to_none=True)  # type: ignore[union-attr]
        lm.manual_backward(loss)
        for opt in (
            optimizers if isinstance(optimizers, list) else [optimizers]
        ):
            opt.step()  # type: ignore[union-attr]
        scheduler = lm.lr_schedulers()
        for sch in (scheduler if isinstance(scheduler, list) else [scheduler]):
            sch.step()  # type: ignore[union-attr]

        lm.log(
            "train/loss", loss.detach(),
            on_step=True, on_epoch=False, prog_bar=True, logger=True,
        )  # fmt: skip
        return {
            "loss": loss.detach(),
            "signals": evaluation.last_step.outputs.signals,
        }

    def validation_step(  # ---------------------------------------------------
        self,
        batch: Batch,
        batch_idx: int,
    ) -> dict[str, Any]:
        """ """
        lm = self._lm
        source_context = self._resolve_val_source_context(batch_idx, batch)
        trace_request = None
        ds = self.diagnostic_trace_spec
        if ds.enabled and batch_idx < ds.max_batches and ds.keys:
            trace_spec = build_trace_spec(
                "ehc",
                include_keys=set(ds.keys),
            )
            trace_request = EvaluationTraceRequest(
                trace_spec=trace_spec,
            )

        result = self.execute_evaluation_batch(
            EvaluationCaseBatch(
                batch=batch,
                case_id=f"val-{batch_idx:04d}",
                source_context=source_context,
            ),
            trace_request=trace_request,
        )
        update_metric_collection_from_evaluated_chunk(
            lm.val_metrics, result.evaluated, EHC_EPISODE_ROUTES
        )

        # Feed bounded diagnostic reducers.
        obs_id = batch.get("observation_id")
        if obs_id is not None:
            self._val_occupancy.update(obs_id.detach().cpu())

        # Store bounded diagnostic traces for callback consumption.
        if (
            result.trace is not None
            and len(self._diagnostic_traces) < ds.max_batches
        ):
            self._diagnostic_traces.append(result.trace)

        return {"trace": result.trace}

    def on_validation_epoch_end(  # -------------------------------------------
        self,
    ) -> None:
        """Compute, render, and reset bounded diagnostic reducers."""
        lm = self._lm
        summaries = self._val_reducer_collection.compute()

        if lm.trainer is not None and lm.trainer.is_global_zero:
            occ = summaries.get("occupancy")
            if occ is not None:
                log_reducer_figure(
                    lm.logger,
                    "val_diag/occupancy",
                    render_occupancy_histogram(occ),
                    global_step=lm.global_step,
                )

            norm = summaries.get("hidden_norms")
            if norm is not None:
                centers, density = norm
                log_reducer_figure(
                    lm.logger,
                    "val_diag/hidden_norms",
                    render_hidden_norm_histogram(centers, density),
                    global_step=lm.global_step,
                )

        self._val_reducer_collection.reset()

    def execute_evaluation_batch(  # ------------------------------------------
        self,
        case: EvaluationCaseBatch,
        *,
        trace_request: EvaluationTraceRequest | None = None,
    ) -> EvaluationCaseResult:
        """Execute one provider-owned replay case through the spatial eval path."""
        lm = self._lm
        self._apply_runtime(lm.global_step)
        ctrl = self._require_eval_controller()
        obj = self._require_eval_objective()
        carry0 = ctrl.initial_state(
            case.batch,
            static_data=self._build_static_params(),
        )

        result = execute_replay_evaluation_batch(
            case=case,
            runner=self._eval_runner,
            controller=ctrl,
            carry=carry0,
            objective=obj,
            max_rollout_steps=self._config.runtime.validation.max_rollout_steps,
            hard_max_rollout_steps=self._config.runtime.validation.hard_max_rollout_steps,
            runner_options={"allow_halt": True},
            objective_options=resolve_objective_schedule(
                lm.global_step, self._config.objective
            ),
            trace_request=trace_request,
        )
        if result.trace is not None:
            _maybe_apply_arena_supplements(result.trace, result.source_context)
        return result

    def _resolve_val_source_context(  # ---------------------------------------
        self,
        batch_idx: int,
        batch: Batch,
    ) -> object | None:
        """ """
        lm = self._lm
        trainer = getattr(lm, "trainer", None)
        if trainer is None:
            return None
        dm = getattr(trainer, "datamodule", None)
        if not isinstance(dm, Datamodule):
            return None
        first_key = next(iter(batch))
        batch_size = batch[first_key].shape[0]
        sample_ids = dm.val_sample_ids_for_batch(
            batch_idx,
            batch_size,
            rank=trainer.global_rank,
            world_size=trainer.world_size,
        )
        if not sample_ids:
            return None
        return ArenaEvaluationSourceContext(
            task_family="arena",
            dataset_path=dm.config.dataset_path,
            split="val",
            sample_ids=tuple(sample_ids),
        )


# =============================================================================
def _maybe_apply_arena_supplements(
    trace: TraceTree,
    source_context: object | None,
) -> tuple[str, ...]:
    """ """
    if not isinstance(source_context, ArenaEvaluationSourceContext):
        return ()
    supplements = build_arena_trace_supplements(source_context, trace.length)
    apply_arena_trace_supplements(trace, supplements)
    return ("arena",)


# =============================================================================
__all__ = ["EHCSpatialPretrainConfig", "EHCSpatialPretrainRegime"]
