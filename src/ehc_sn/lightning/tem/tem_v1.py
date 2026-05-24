"""TEM v1 Lightning runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import lightning as L
from pydantic import AliasChoices, BaseModel, Field
from torch.optim import Adam, Optimizer

from ehc_sn.adapters.arena.tem import (
    ArenaTEMAdapterSettings,
    ArenaTEMTaskBinding,
    ArenaTEMV1BridgeAdapter,
)
from ehc_sn.controllers.replay.trajectory import (
    ReplayTrajectoryController,
    ReplayTrajectoryControllerConfig,
)
from ehc_sn.eval.contracts import (
    EvaluationCaseBatch,
    EvaluationCaseResult,
    EvaluationTraceRequest,
)
from ehc_sn.eval.executor import execute_replay_evaluation_batch
from ehc_sn.lightning.tem.core.runtime import (
    RuntimeConfig,
    TEMRuntimeState,
    resolve_tem_runtime,
)
from ehc_sn.metrics.builders import build_train_metrics, build_val_metrics
from ehc_sn.metrics.rollout import (
    make_observed_step_metric_observer,
    update_metric_collection_from_evaluated_chunk,
)
from ehc_sn.metrics.routes.tem import (
    TEM_EPISODE_ROUTES,
    TEM_PRIMARY_VAL_ROUTE_KEY,
    TEM_STEP_ROUTES,
)
from ehc_sn.metrics.traces import build_trace_spec
from ehc_sn.models.tem.tem_v1 import ModelSettingsV1, TEMModelV1
from ehc_sn.objectives.tem import TEMObjective, TEMObjectiveConfig
from ehc_sn.rollouts.buffers import FifoBuffer
from ehc_sn.rollouts.partial_reset import PartialResetBatchAssembler
from ehc_sn.rollouts.runtime import RecurrentRunner
from ehc_sn.rollouts.sources import PartialResetSource
from ehc_sn.tasks.arena.capabilities.replay import ArenaReplayCapability
from ehc_sn.tasks.arena.runtime import (
    batch_size_from_arena_batch,
    infer_arena_replay_batch_keys,
)
from ehc_sn.training.optim import Adam, AdamConfig
from ehc_sn.training.rollout import (
    score_rollout_streaming,
)
from ehc_sn.training.schedules import (
    CosineAnnealingLRWithWarmup,
    SchedulerConfig,
)
from ehc_sn.types import Batch

# Community-standard map-style batch: plain dict returned by DataLoader.
TEM_STATIC_REQUIRED_KEYS = ("topology", "observations", "mask_valid")
TEM_STATIC_OPTIONAL_KEYS = ("regions", "start", "goals", "landmarks")
TEM_V1_NONFINITE_DEBUG_ENV = "EHC_TEM_V1_FAILFAST_NONFINITE"
TEM_NONFINITE_DETAIL_LIMIT = 12


# =============================================================================
class TEMV1ModelConfig(BaseModel, extra="forbid"):
    """Top-level TEM v1 training config."""

    model_config_path: Path = Field(
        ...,
        description=(
            "Path to the model configuration TOML file that specifies the TEM "
            "v1 architecture."
        ),
    )
    adapter: ArenaTEMAdapterSettings = Field(
        ...,
        description="Arena bridge adapter settings (encoder kind, vocab size).",
    )
    controller: ReplayTrajectoryControllerConfig = Field(
        ...,
        description="Replay trajectory controller configuration.",
    )
    objective: TEMObjectiveConfig = Field(
        ...,
        description="",
    )

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


# =============================================================================
class TEMV1TrainingModel(L.LightningModule):
    """LightningModule encapsulating the TEM v1 model, training and evaluation
    runtimes, and optimization logic for training on arena replay data.
    """

    def __init__(  # ----------------------------------------------------------
        self,
        config: TEMV1ModelConfig,
    ) -> None:
        """Initialize the TEM v1 training model with the given configuration,
        setting up the shared model, adapter, and placeholders for the
        train/eval runtimes and objectives which will be lazily initialized on
        setup.
        """
        super().__init__()
        model_settings = ModelSettingsV1.from_config(config.model_config_path)
        self.model = TEMModelV1(model_settings)
        self.adapter = ArenaTEMV1BridgeAdapter(self.model, config.adapter)
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
        self.train_metrics = build_train_metrics(TEM_STEP_ROUTES).clone(
            prefix="train/"
        )
        self.val_metrics = build_val_metrics(TEM_EPISODE_ROUTES).clone(
            prefix="val/"
        )
        self.primary_val_metric_key = f"val/{TEM_PRIMARY_VAL_ROUTE_KEY}"
        self._eval_trace_keys: set[str] | None = None
        self.trace_specs = build_trace_spec("tem")
        # Buffer + assembler implement partial-reset batching for ACT runs.
        self._train_buffer: FifoBuffer | None = None
        self._train_batch_assembler: PartialResetBatchAssembler | None = None

    @property
    def config(self) -> TEMV1ModelConfig:
        """Return the parsed configuration used by this LightningModule."""
        return self._config

    def setup(  # -------------------------------------------------------------
        self,
        stage: Optional[str] = None,
    ) -> None:
        """Initialize phase-local train and evaluation runtimes around the shared model."""
        if stage in (None, "fit"):
            self._ensure_train_runtime()
            self._ensure_eval_runtime()
        elif stage in ("validate", "test"):
            self._ensure_eval_runtime()

    def configure_optimizers(  # ----------------------------------------------
        self,
    ) -> tuple[list[Optimizer], list[dict[str, Any]]]:
        """Build the optimizer and learning-rate scheduler."""
        total_steps = int(self.trainer.estimated_stepping_batches)

        # Optimizer for the main model parameters
        sup_params = [p for p in self.adapter.parameters() if p.requires_grad]
        opt_sup = Adam(sup_params, self.config.optimizer)
        schedulers: list[dict[str, Any]] = [
            {
                "scheduler": CosineAnnealingLRWithWarmup(
                    opt_sup, total_steps, self.config.scheduler
                ),
                "interval": "step",
                "frequency": 1,
                "name": "optim/main",
            }
        ]

        return [opt_sup], schedulers

    def on_train_epoch_start(  # ----------------------------------------------
        self,
    ) -> None:
        """
        Reset the partial-reset carry and buffer at the start of each training
        epoch to prevent cross-epoch state leakage. Metrics are also reset here
        to align with epoch-level logging.
        """
        self._train_carry = None
        if self._train_buffer is not None:
            self._train_buffer.clear()
        self.train_metrics.reset()

    def on_validation_epoch_start(  # -----------------------------------------
        self,
    ) -> None:
        """
        Reset the evaluation metrics at the start of each validation epoch to
        ensure clean aggregation of episode-level metrics across the validation set.
        Validation runtime is not reset here since it is resolved dynamically
        at each step from the global step count.
        """
        self.val_metrics.reset()

    def set_eval_trace_keys(  # -----------------------------------------------
        self,
        keys: set[str],
    ) -> None:
        """Set semantic trace keys for evaluation-regime capture."""
        self._eval_trace_keys = set(keys)
        self.trace_specs = build_trace_spec(
            "tem",
            include_keys=self._eval_trace_keys,
        )

    def _validation_seed(self, batch_idx: int) -> int:
        """Return the explicit evaluation seed for one validation batch."""
        seed = self.config.runtime.validation.seed
        if seed is None:
            raise ValueError(
                "TEM evaluation requires runtime.validation.seed to be set."
            )
        return int(seed) + int(batch_idx)

    def training_step(  # ------------------------------------------------------
        self,
        batch: Batch,
        batch_idx: int,
    ) -> dict[str, object]:
        """Run one TEM chunked-TBPTT optimizer update through the recurrent runner."""
        runtime = self._apply_runtime(self.global_step)
        self.log(
            "train/runtime/eta", runtime.eta,
            on_step=True, on_epoch=False, logger=True,
        )  # fmt: skip
        self.log(
            "train/runtime/hebbian_decay", runtime.hebbian_decay,
            on_step=True, on_epoch=False, logger=True,
        )  # fmt: skip
        self.log(
            "train/runtime/p2g_use", runtime.p2g_use,
            on_step=True, on_epoch=False, logger=True,
        )  # fmt: skip
        self.log(
            "train/runtime/p2g_uncertainty_offset", runtime.p2g_uncertainty_offset,
            on_step=True, on_epoch=False, logger=True,
        )  # fmt: skip
        train_controller = self._require_train_controller()
        train_objective = self._require_train_objective()
        batch_assembler = self._ensure_train_batch_assembler(batch)

        if self._train_carry is None:
            self._train_carry = train_controller.initial_state(batch)

        source = PartialResetSource(
            incoming=batch, assembler=batch_assembler, carry0=self._train_carry
        )
        objective_options = train_objective.runtime_loss_options(
            self.global_step, p2g_use=runtime.p2g_use
        )
        evaluation = score_rollout_streaming(
            runner=self._train_runner,
            source=source,
            controller=train_controller,
            carry=self._train_carry,
            objective=train_objective,
            max_rollout_steps=self._train_chunk_steps(),
            objective_options=objective_options,
            observed_step_observer=make_observed_step_metric_observer(
                self.train_metrics, TEM_STEP_ROUTES
            ),
        )
        next_carry = evaluation.execution.final_carry.detach()
        self._train_carry = next_carry
        loss = evaluation.loss
        loss = (
            loss / self._train_chunk_steps()
        )  # Further normalize by chunk length for stability.

        optimizers = self.optimizers()
        optimizer_handles = (
            list(optimizers) if isinstance(optimizers, list) else [optimizers]
        )
        for opt in optimizer_handles:
            opt.zero_grad(set_to_none=True)  # type: ignore

        self.manual_backward(loss)

        for opt in optimizer_handles:
            opt.step()  # type: ignore

        scheduler = self.lr_schedulers()
        for sch in scheduler if isinstance(scheduler, list) else [scheduler]:
            sch.step()  # type: ignore

        # Commit detached carry so the next chunk resumes from where this one ended.
        self._train_carry = evaluation.execution.final_carry.detach()
        final_carry = evaluation.execution.final_carry

        # Log the accumulated chunk loss to TensorBoard.
        self.log(
            "train/loss", loss.detach(),
            on_step=True, on_epoch=False, prog_bar=True, logger=True,
        )  # fmt: skip
        self.log(
            "train/fit_path/halted_fraction", final_carry.halted.float().mean(),
            on_step=True, on_epoch=False, logger=True,
        )  # fmt: skip
        self.log(
            "train/fit_path/max_cursor", final_carry.cursor.max().float(),
            on_step=True, on_epoch=False, logger=True,
        )  # fmt: skip

        return {
            "loss": loss.detach(),
            "signals": evaluation.last_step.outputs.signals,
        }

    def validation_step(  # ---------------------------------------------------
        self,
        batch: Batch,
        batch_idx: int,
    ) -> dict[str, object]:
        """Run a full TEM rollout through the recurrent runner and trace observer."""
        result = self.execute_evaluation_batch(
            EvaluationCaseBatch(
                batch=batch,
                case_id=f"val-{batch_idx:04d}",
            )
        )
        update_metric_collection_from_evaluated_chunk(
            self.val_metrics, result.evaluated, TEM_EPISODE_ROUTES
        )

        return {}

    def execute_evaluation_batch(  # ------------------------------------------
        self,
        case: EvaluationCaseBatch,
        *,
        trace_request: EvaluationTraceRequest | None = None,
    ) -> EvaluationCaseResult:
        """Execute one provider-owned replay case through the TEM eval path."""
        runtime = self._apply_runtime(self.global_step)
        eval_controller = self._require_eval_controller()
        eval_objective = self._require_eval_objective()
        carry0 = eval_controller.initial_state(case.batch)

        objective_options = eval_objective.runtime_loss_options(
            self.global_step, p2g_use=runtime.p2g_use
        )
        return execute_replay_evaluation_batch(
            case=case,
            runner=self._eval_runner,
            controller=eval_controller,
            carry=carry0,
            objective=eval_objective,
            max_rollout_steps=self.config.runtime.validation.max_rollout_steps,
            hard_max_rollout_steps=self.config.runtime.validation.hard_max_rollout_steps,
            runner_options={"allow_halt": True, "explore": False},
            objective_options=objective_options,
            trace_request=trace_request,
        )

    def _apply_runtime(  # ----------------------------------------------------
        self,
        step: int,
    ) -> TEMRuntimeState:
        """Resolve and apply TEM runtime dynamics for the current global step."""
        runtime = resolve_tem_runtime(step, self.config.runtime)
        self.model.set_runtime(
            runtime.eta, runtime.hebbian_decay, runtime.p2g_uncertainty_offset
        )
        return runtime

    def _train_chunk_steps(  # ------------------------------------------------
        self,
    ) -> int:
        """Return the TBPTT chunk length used for one optimizer update."""
        if self.config.controller.window_size is not None:
            return self.config.controller.window_size
        return self.config.runtime.sequence.tbptt_steps

    def _build_runtime(  # ----------------------------------------------------
        self,
    ) -> tuple[ReplayTrajectoryController, TEMObjective]:
        """Construct one phase-local TEM arena replay runtime around the shared
        model.
        """
        replay = ArenaReplayCapability()
        controller = ReplayTrajectoryController(
            backbone=self.adapter,
            config=self.config.controller,
            runtime=replay,
        )
        objective = TEMObjective(
            config=self.config.objective,
            task_binding=ArenaTEMTaskBinding(),
        )
        return controller, objective

    def _ensure_train_runtime(  # ---------------------------------------------
        self,
    ) -> None:
        """Initialize the training runtime once per process."""
        if (
            self.train_controller is not None
            and self.train_objective is not None
        ):
            return
        self.train_controller, self.train_objective = self._build_runtime()

    def _ensure_eval_runtime(  # ----------------------------------------------
        self,
    ) -> None:
        """Initialize the evaluation runtime once per process."""
        if self.eval_controller is not None and self.eval_objective is not None:
            return
        self.eval_controller, self.eval_objective = self._build_runtime()

    def _require_train_controller(  # -----------------------------------------
        self,
    ) -> ReplayTrajectoryController:
        """Return the training controller, initializing the train runtime if needed."""
        self._ensure_train_runtime()
        if self.train_controller is None:
            raise RuntimeError("TEM training runtime is not initialized.")
        return self.train_controller

    def _require_train_objective(  # ------------------------------------------
        self,
    ) -> TEMObjective:
        """Return the training objective, initializing the train runtime if needed."""
        self._ensure_train_runtime()
        if self.train_objective is None:
            raise RuntimeError("TEM training runtime is not initialized.")
        return self.train_objective

    def _require_eval_controller(  # ------------------------------------------
        self,
    ) -> ReplayTrajectoryController:
        """Return the evaluation controller, initializing the eval runtime if needed."""
        self._ensure_eval_runtime()
        if self.eval_controller is None:
            raise RuntimeError("TEM evaluation runtime is not initialized.")
        return self.eval_controller

    def _require_eval_objective(  # -------------------------------------------
        self,
    ) -> TEMObjective:
        """Return the evaluation objective, initializing the eval runtime if needed."""
        self._ensure_eval_runtime()
        if self.eval_objective is None:
            raise RuntimeError("TEM evaluation runtime is not initialized.")
        return self.eval_objective

    def _ensure_train_batch_assembler(  # -------------------------------------
        self,
        batch: Batch,
    ) -> PartialResetBatchAssembler:
        """Create the partial-reset buffer lazily from the observed static maze schema."""
        if self._train_batch_assembler is not None:
            return self._train_batch_assembler

        keys = infer_arena_replay_batch_keys(batch)
        capacity_rows = 4 * batch_size_from_arena_batch(batch)
        self._train_buffer = FifoBuffer(capacity_rows, keys, pin_memory=True)
        self._train_batch_assembler = PartialResetBatchAssembler(
            buffer=self._train_buffer,
            keys=keys,
        )
        return self._train_batch_assembler


# =============================================================================
__all__ = ["TEMV1ModelConfig", "TEMV1TrainingModel"]
