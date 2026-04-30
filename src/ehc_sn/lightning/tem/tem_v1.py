"""TEM v1 Lightning runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import lightning as L
import torch
from pydantic import BaseModel, Field, model_validator
from torch.optim import Optimizer
from torchmetrics import MetricCollection

from ehc_sn.adapters.arena.tem import ArenaTEMAdapterSettings, ArenaTEMTaskBinding, ArenaTEMV1BridgeAdapter
from ehc_sn.adapters.arena.tem.traces import ARENA_TEM_TRACE_FIELDS, select_arena_tem_trace_fields
from ehc_sn.controllers.replay.trajectory import ReplayTrajectoryController, ReplayTrajectoryControllerConfig
from ehc_sn.lightning._rollout import (
    evaluate_rollout,
    evaluate_rollout_streaming,
    observe_rollout_chunk,
    update_metric_collection_from_evaluated_chunk,
)
from ehc_sn.lightning.eval.contracts import EvaluationBatchArtifacts, EvaluationTraceRequest
from ehc_sn.lightning.tem.core.runtime import RuntimeConfig, TEMRuntimeState, resolve_tem_runtime
from ehc_sn.metrics import build_train_metrics, build_val_metrics
from ehc_sn.metrics.routes import TEM_EPISODE_ROUTES, TEM_PRIMARY_VAL_ROUTE_KEY, TEM_STEP_ROUTES
from ehc_sn.metrics.traces import ReplayableEnvironments, build_trace_spec
from ehc_sn.models.tem.tem_v1 import ModelSettingsV1, TEMModelV1
from ehc_sn.objectives.tem import TEMObjective, TEMObjectiveConfig
from ehc_sn.rollouts import PartialResetSource, RecurrentRunner, RepeatSource
from ehc_sn.tasks.arena.capabilities.replay import ArenaReplayCapability
from ehc_sn.tasks.arena.runtime import batch_size_from_arena_batch, build_arena_trace_worlds, infer_arena_replay_batch_keys
from ehc_sn.training.buffers import FifoBuffer
from ehc_sn.training.distributed import normalize_loss_for_backward
from ehc_sn.training.optim import Adam, AdamConfig
from ehc_sn.training.partial_reset import PartialResetBatchAssembler
from ehc_sn.training.schedules import CosineAnnealingLRWithWarmup, SchedulerConfig, SequentialLR
from ehc_sn.types import Batch


# =================================================================================================
class ModelConfig_TEM_V1(BaseModel, extra="forbid"):
    """Top-level TEM v1 training config."""

    # ~~ Model architecture ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    model_config_path: Path = Field(
        ...,
        description="Path to the model configuration TOML file that specifies the TEM v1 architecture.",
    )
    adapter: ArenaTEMAdapterSettings = Field(
        ...,
        description="Settings for the arena bridge adapter that binds TEM v1 to task inputs/outputs.",
    )
    controller: ReplayTrajectoryControllerConfig = Field(
        ...,
        description="Replay trajectory controller configuration.",
    )
    objective: TEMObjectiveConfig = Field(
        ...,
        description="TEM objective configuration.",
    )

    # ~~ Optimizers & scheduling ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    optimizer: AdamConfig = Field(
        default_factory=AdamConfig,
        description="",
    )
    scheduler: SchedulerConfig = Field(
        default_factory=SchedulerConfig,
        description="",
    )
    runtime: RuntimeConfig = Field(
        default_factory=RuntimeConfig,
        description="",
    )

    # ~~ Extra ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    global_batch_size: int = Field(
        ...,
        description="",
    )  # TODO: consider moving to BufferSettings or similar

    @model_validator(mode="after")
    def validate_adapter_contract(self) -> "ModelConfig_TEM_V1":
        model_settings = ModelSettingsV1.from_config(self.model_config_path)
        if self.adapter.action_count != model_settings.transition_action_count:
            raise ValueError("adapter.action_count must match model.transition_action_count.")
        return self


# =================================================================================================
class TrainingModel(L.LightningModule):
    """ """

    def __init__(  # ------------------------------------------------------------------------------
        self, config: ModelConfig_TEM_V1,
    ) -> None:  # fmt: skip
        """ """
        super().__init__()
        model_settings = ModelSettingsV1.from_config(config.model_config_path)
        self.model = TEMModelV1(model_settings)
        self.bridge_adapter = ArenaTEMV1BridgeAdapter(self.model, config.adapter)

        self.train_controller: ReplayTrajectoryController | None = None
        self.train_objective: TEMObjective | None = None

        self.eval_controller: ReplayTrajectoryController | None = None
        self.eval_objective: TEMObjective | None = None

        self._config = config
        self._train_runner = RecurrentRunner()
        self._eval_runner = RecurrentRunner()

        # Manual optimization: explicit backward + opt step (legacy parity + dual-opt clarity).
        self.automatic_optimization = False
        self._train_carry = None

        # Metrics are cloned for train/val to allow separate logging and state management.
        self.train_metrics = build_train_metrics(TEM_STEP_ROUTES).clone(prefix="train/")
        self.val_metrics = build_val_metrics(TEM_EPISODE_ROUTES).clone(prefix="val/")
        self.primary_val_metric_key = f"val/{TEM_PRIMARY_VAL_ROUTE_KEY}"
        self.trace_specs = build_trace_spec("tem", extra_fields=ARENA_TEM_TRACE_FIELDS)
        self._eval_trace_keys: set[str] | None = None

        # Buffer + assembler implement partial-reset batching for replay training.
        self._train_buffer: FifoBuffer | None = None
        self._train_batch_assembler: PartialResetBatchAssembler | None = None

    @property
    def config(self) -> ModelConfig_TEM_V1:
        """Return the parsed configuration used by this LightningModule."""
        return self._config

    def _train_chunk_steps(self) -> int:
        """Return the TBPTT chunk length used for one optimizer update."""
        if self.config.controller.window_size is not None:
            return self.config.controller.window_size
        return self.config.runtime.sequence.tbptt_steps

    def _build_runtime(self) -> tuple[ReplayTrajectoryController, TEMObjective]:
        """Construct one phase-local replay runtime around the shared model."""
        controller = ReplayTrajectoryController(
            backbone=self.bridge_adapter,
            config=self.config.controller,
            runtime=ArenaReplayCapability(observation_dim=self.config.adapter.observation_dim),
        )
        objective = TEMObjective(self.config.objective, task_binding=ArenaTEMTaskBinding())
        return controller, objective

    def _ensure_train_runtime(self) -> None:
        """Initialize the training runtime once per process."""
        if self.train_controller is not None and self.train_objective is not None:
            return
        self.train_controller, self.train_objective = self._build_runtime()

    def _ensure_eval_runtime(self) -> None:
        """Initialize the evaluation runtime once per process."""
        if self.eval_controller is not None and self.eval_objective is not None:
            return
        self.eval_controller, self.eval_objective = self._build_runtime()

    def _require_train_controller(self) -> ReplayTrajectoryController:
        """Return the training controller, initializing the train runtime if needed."""
        self._ensure_train_runtime()
        if self.train_controller is None:
            raise RuntimeError("TEM training runtime is not initialized.")
        return self.train_controller

    def _require_train_objective(self) -> TEMObjective:
        """Return the training objective, initializing the train runtime if needed."""
        self._ensure_train_runtime()
        if self.train_objective is None:
            raise RuntimeError("TEM training runtime is not initialized.")
        return self.train_objective

    def _require_eval_controller(self) -> ReplayTrajectoryController:
        """Return the evaluation controller, initializing the eval runtime if needed."""
        self._ensure_eval_runtime()
        if self.eval_controller is None:
            raise RuntimeError("TEM evaluation runtime is not initialized.")
        return self.eval_controller

    def _require_eval_objective(self) -> TEMObjective:
        """Return the evaluation objective, initializing the eval runtime if needed."""
        self._ensure_eval_runtime()
        if self.eval_objective is None:
            raise RuntimeError("TEM evaluation runtime is not initialized.")
        return self.eval_objective

    def _build_trace_meta(self, batch: Batch) -> dict[str, object]:
        """Return out-of-band trace metadata for figure-facing evaluation traces."""
        lec_alpha = torch.stack([torch.sigmoid(alpha).detach() for alpha in self.model.lec.filter.alpha])
        lec_w_f = torch.stack([torch.sigmoid(weight).detach() for weight in self.model.lec.w_f])
        return {
            "environments": ReplayableEnvironments(build_arena_trace_worlds(batch)),
            "lec": {
                "filter": {"alpha_sigmoid": lec_alpha},
                "w_f_sigmoid": lec_w_f,
            },
        }

    def _ensure_train_batch_assembler(  # ---------------------------------------------------------
        self, batch: Batch,
    ) -> PartialResetBatchAssembler:  # fmt: skip
        """Create the partial-reset buffer lazily from the observed arena batch schema."""
        if self._train_batch_assembler is not None:
            return self._train_batch_assembler

        keys = infer_arena_replay_batch_keys(batch)
        capacity_rows = 4 * self.config.global_batch_size
        self._train_buffer = FifoBuffer(capacity_rows, keys, pin_memory=True)
        self._train_batch_assembler = PartialResetBatchAssembler(buffer=self._train_buffer, keys=keys)
        return self._train_batch_assembler

    def setup(  # --------------------------------------------------------------------------------
        self, stage: Optional[str] = None,
    ) -> None:  # fmt: skip
        """Initialize phase-local train and evaluation runtimes around the shared model."""
        if stage in (None, "fit"):
            self._ensure_train_runtime()
            self._ensure_eval_runtime()
        elif stage in ("validate", "test"):
            self._ensure_eval_runtime()

    def configure_optimizers(  # -------------------------------------------------------------------
        self,
    ) -> tuple[list[Optimizer], list[SequentialLR]]:  # fmt: skip
        """Build the optimizer and learning-rate scheduler."""
        total_steps = int(self.trainer.estimated_stepping_batches)

        # Optimizer for the full bridge surface, including observation decoding.
        sup_params = [p for p in self.bridge_adapter.parameters() if p.requires_grad]
        opt_sup = Adam(sup_params, self.config.optimizer)
        sch_sup = CosineAnnealingLRWithWarmup(opt_sup, total_steps, self.config.scheduler)
        # sch_sup = ExponentialLR(opt_sup, total_steps, self.config.scheduler)

        return [opt_sup], [sch_sup]

    def _apply_runtime(  # ------------------------------------------------------------------------
        self, step: int, *, log_values: bool,
    ) -> TEMRuntimeState:  # fmt: skip
        """Resolve and apply TEM runtime dynamics for the current global step."""
        runtime = resolve_tem_runtime(step, self.config.runtime)
        self.model.set_runtime(runtime.eta, runtime.hebbian_decay, runtime.p2g_uncertainty_offset)

        if log_values:
            self.log("train/runtime/eta", runtime.eta, on_step=True, on_epoch=False, logger=True)
            self.log("train/runtime/hebbian_decay", runtime.hebbian_decay, on_step=True, on_epoch=False, logger=True)  # fmt: skip
            self.log("train/runtime/p2g_uncertainty_offset", runtime.p2g_uncertainty_offset, on_step=True, on_epoch=False, logger=True)  # fmt: skip

        return runtime

    # -- Lifecycle --------------------------------------------------------------------------------

    def on_train_epoch_start(  # ------------------------------------------------------------------
        self,
    ) -> None:  # fmt: skip
        """ """
        self._train_carry = None
        if self._train_buffer is not None:
            self._train_buffer.clear()
        self.train_metrics.reset()

    def on_validation_epoch_start(  # ------------------------------------------------------------
        self,
    ) -> None:  # fmt: skip
        """ """
        self.val_metrics.reset()

    def set_eval_trace_keys(self, keys: set[str]) -> None:
        """Set the semantic trace keys required for evaluation-time figure capture."""
        self._eval_trace_keys = set(keys)

    # -- Training ----------------------------------------------------------------------------------

    def training_step(  # -------------------------------------------------------------------------
        self, batch: Batch, batch_idx: int,
    ) -> dict[str, object]:  # fmt: skip
        """Run one TEM chunked-TBPTT optimizer update through the recurrent runner."""
        self._apply_runtime(self.global_step, log_values=True)
        train_controller = self._require_train_controller()
        train_objective = self._require_train_objective()
        batch_assembler = self._ensure_train_batch_assembler(batch)

        # Initialize carry/state on the first batch
        if self._train_carry is None:
            self._train_carry = train_controller.initial_state(batch)

        source = PartialResetSource(incoming=batch, assembler=batch_assembler, carry0=self._train_carry)
        evaluation = evaluate_rollout_streaming(
            runner=self._train_runner,
            source=source,
            controller=train_controller,
            carry=self._train_carry,
            objective=train_objective,
            max_rollout_steps=self._train_chunk_steps(),
            metric_collection=self.train_metrics,
            metric_routes=TEM_STEP_ROUTES,
        )
        self._train_carry = evaluation.execution.final_carry.detach()

        # Normalize by local batch size; DDP averages gradients across ranks.
        local_bs = batch_size_from_arena_batch(batch)
        loss = normalize_loss_for_backward(evaluation.loss, local_bs=local_bs)
        loss = loss / self._train_chunk_steps()  # Further normalize by chunk length for stability.

        optimizers = self.optimizers()
        for opt in optimizers if isinstance(optimizers, list) else [optimizers]:
            opt.zero_grad(set_to_none=True)  # type: ignore

        self.manual_backward(loss)

        for opt in optimizers if isinstance(optimizers, list) else [optimizers]:
            opt.step()  # type: ignore

        scheduler = self.lr_schedulers()
        for sch in scheduler if isinstance(scheduler, list) else [scheduler]:
            sch.step()  # type: ignore

        # Log the accumulated chunk loss to TensorBoard.
        self.log("train/loss", loss.detach(), on_step=True, on_epoch=False, prog_bar=True, logger=True)

        return {"loss": loss.detach(), "signals": evaluation.last_step.outputs.signals}

    def validation_step(  # -----------------------------------------------------------------------
        self, batch: Batch, batch_idx: int,
    ) -> dict[str, object]:  # fmt: skip
        """Run a full TEM rollout through the recurrent runner and trace observer."""
        self._apply_runtime(self.global_step, log_values=False)
        eval_controller = self._require_eval_controller()
        eval_objective = self._require_eval_objective()
        step_options = {"allow_halt": True}
        carry0 = eval_controller.initial_state(batch)
        trace_meta = self._build_trace_meta(batch)

        evaluation = evaluate_rollout(
            runner=self._eval_runner,
            source=RepeatSource(batch),
            controller=eval_controller,
            carry=carry0,
            objective=eval_objective,
            max_rollout_steps=self.config.runtime.validation.max_rollout_steps,
            hard_max_rollout_steps=self.config.runtime.validation.hard_max_rollout_steps,
            runner_options=step_options,
        )
        update_metric_collection_from_evaluated_chunk(self.val_metrics, evaluation.evaluated, TEM_EPISODE_ROUTES)
        if self._eval_trace_keys is None:
            return {"trace": None}

        arena_extra = select_arena_tem_trace_fields(self._eval_trace_keys)
        trace_specs = build_trace_spec("tem", include_keys=self._eval_trace_keys, extra_fields=arena_extra)
        trace = observe_rollout_chunk(evaluation.chunk, trace_specs, trace_meta=trace_meta)
        return {"trace": trace}

    # -- Evaluation regime surface ---------------------------------------------------------------

    def build_evaluation_metrics(  # -------------------------------------------------------------
        self, namespace: str,
    ) -> "MetricCollection":  # fmt: skip
        """Return a fresh TEM-family metric collection with the given namespace prefix.

        The returned collection is independent of ``self.val_metrics`` and may be
        accumulated by the regime runner across multiple case batches.

        Args:
            namespace: Metric namespace prefix, e.g. ``"diag/my_probe/"``.

        Returns:
            A fresh :class:`~torchmetrics.MetricCollection` keyed by TEM episode routes.
        """
        return build_val_metrics(TEM_EPISODE_ROUTES).clone(prefix=namespace)

    def execute_evaluation_batch(  # -------------------------------------------------------------
        self,
        batch: "Batch",
        trace_request: "Optional[EvaluationTraceRequest]",
    ) -> "EvaluationBatchArtifacts":  # fmt: skip
        """Execute one TEM evaluation batch and return scored artifacts.

        Runs a full TEM rollout identical to ``validation_step`` but does **not**
        update ``self.val_metrics``. Returns a scored rollout result with an optional
        trace for regime figure consumers.

        Args:
            batch: A task batch in arena replay format.
            trace_request: Trace key request, or ``None`` for no trace.

        Returns:
            :class:`~ehc_sn.lightning.eval.contracts.EvaluationBatchArtifacts` with
            ``regime_id`` set to ``"_inline"`` (overwritten by the runner).
        """
        self._apply_runtime(self.global_step, log_values=False)
        eval_controller = self._require_eval_controller()
        eval_objective = self._require_eval_objective()
        carry0 = eval_controller.initial_state(batch)
        trace_meta = self._build_trace_meta(batch)

        evaluation = evaluate_rollout(
            runner=self._eval_runner,
            source=RepeatSource(batch),
            controller=eval_controller,
            carry=carry0,
            objective=eval_objective,
            max_rollout_steps=self.config.runtime.validation.max_rollout_steps,
            hard_max_rollout_steps=self.config.runtime.validation.hard_max_rollout_steps,
            runner_options={"allow_halt": True},
        )

        trace = None
        if trace_request is not None and trace_request.enabled:
            req_keys = trace_request.key_set()
            arena_extra = select_arena_tem_trace_fields(req_keys)
            trace_specs = build_trace_spec("tem", include_keys=req_keys, extra_fields=arena_extra)
            trace = observe_rollout_chunk(evaluation.chunk, trace_specs, trace_meta=trace_meta)

        def _apply(collection: "MetricCollection") -> None:
            update_metric_collection_from_evaluated_chunk(collection, evaluation.evaluated, TEM_EPISODE_ROUTES)

        return EvaluationBatchArtifacts(
            regime_id="_inline",
            metric_namespace="",
            evaluated=evaluation.evaluated,
            apply_to_metrics=_apply,
            trace=trace,
        )
