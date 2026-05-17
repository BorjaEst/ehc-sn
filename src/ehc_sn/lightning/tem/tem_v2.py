"""TEM v2 Lightning runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import lightning as L
from pydantic import BaseModel, Field, model_validator
from torch.optim import Optimizer

from ehc_sn.adapters.arena.tem import (
    ArenaTEMAdapterSettings,
    ArenaTEMTaskBinding,
    ArenaTEMV2BridgeAdapter,
)
from ehc_sn.controllers.replay.trajectory import (
    ReplayTrajectoryController,
    ReplayTrajectoryControllerConfig,
)
from ehc_sn.lightning._rollout import (
    evaluate_rollout,
    evaluate_rollout_streaming,
    update_metric_collection_from_evaluated_chunk,
)
from ehc_sn.lightning.tem.core.runtime import (
    RuntimeConfig,
    TEMRuntimeState,
    resolve_tem_runtime,
)
from ehc_sn.metrics import build_train_metrics, build_val_metrics
from ehc_sn.metrics.routes import (
    TEM_EPISODE_ROUTES,
    TEM_PRIMARY_VAL_ROUTE_KEY,
    TEM_STEP_ROUTES,
)
from ehc_sn.metrics.traces import build_trace_spec
from ehc_sn.models.tem.tem_v2 import Batch, ModelSettingsV2, TEMModelV2
from ehc_sn.objectives import TEMObjective, TEMObjectiveConfig
from ehc_sn.objectives.tem import (
    TEMObjective,
    TEMObjectiveConfig,
    resolve_objective_schedule,
)
from ehc_sn.rollouts import PartialResetSource, RecurrentRunner, RepeatSource
from ehc_sn.tasks.arena.capabilities.replay import ArenaReplayCapability
from ehc_sn.tasks.arena.runtime import (
    infer_arena_replay_batch_keys,
)
from ehc_sn.traces import TraceTree
from ehc_sn.training.buffers import FifoBuffer
from ehc_sn.training.optim import Adam, AdamConfig
from ehc_sn.training.partial_reset import PartialResetBatchAssembler
from ehc_sn.training.schedules import (
    CosineAnnealingLRWithWarmup,
    SchedulerConfig,
    SequentialLR,
)
from ehc_sn.types import Batch

# Community-standard map-style batch: plain dict returned by MazeDataset / DataLoader.
TEM_STATIC_REQUIRED_KEYS = ("topology", "observations", "mask_valid")
TEM_STATIC_OPTIONAL_KEYS = ("regions", "start", "goals", "landmarks")


# =============================================================================
# =============================================================================
class TEMV2ModelConfig(BaseModel, extra="forbid"):
    """Top-level TEM v2 training config."""

    # -------------------------------------------------------------------------
    # Model architecture
    model_config_path: Path = Field(
        ...,
        description="Path to the model configuration TOML file that specifies the TEM v2 architecture.",
    )
    adapter: ArenaTEMAdapterSettings = Field(
        ...,
        description="Settings for the arena bridge adapter that binds TEM v2 to task inputs/outputs.",
    )
    controller: ReplayTrajectoryControllerConfig = Field(
        ...,
        description="Replay trajectory controller configuration.",
    )
    objective: TEMObjectiveConfig = Field(
        ...,
        description="TEM objective configuration.",
    )

    # -------------------------------------------------------------------------
    # Optimizers & scheduling
    optimizer: AdamConfig = Field(
        default_factory=AdamConfig,
        description="Optimizer configuration for TEM v2 training.",
    )
    scheduler: SchedulerConfig = Field(
        default_factory=SchedulerConfig,
        description="Learning-rate scheduler configuration.",
    )
    runtime: RuntimeConfig = Field(
        default_factory=RuntimeConfig,
        description="Runtime schedules applied during training.",
    )

    # -------------------------------------------------------------------------
    # Extra
    global_batch_size: int = Field(
        ...,
        description="Global batch size across all devices.",
    )  # TODO: consider moving to BufferSettings or similar

    @model_validator(mode="after")
    def validate_adapter_contract(self) -> TEMV2ModelConfig:
        model_settings = ModelSettingsV2.from_config(self.model_config_path)
        if self.adapter.action_count != model_settings.transition_action_count:
            raise ValueError(
                "adapter.action_count must match model.transition_action_count.",
            )
        return self


# =============================================================================
class TEMV2TrainingModel(L.LightningModule):
    """Lightning wrapper for TEM v2 training and evaluation."""

    def __init__(  # ----------------------------------------------------------
        self,
        config: TEMV2ModelConfig,
    ) -> None:
        """Create the Lightning module from a parsed TEM v2 training config."""
        super().__init__()
        model_settings = ModelSettingsV2.from_config(config.model_config_path)
        self.model = TEMModelV2(model_settings)
        self.adapter = ArenaTEMV2BridgeAdapter(self.model, config.adapter)
        self.train_environment: None = None
        self.train_controller: ReplayTrajectoryController | None = None
        self.train_objective: TEMObjective | None = None
        self.eval_environment: None = None
        self.eval_controller: ReplayTrajectoryController | None = None
        self.eval_objective: TEMObjective | None = None
        self._config = config
        self._train_runner = RecurrentRunner()
        self._eval_runner = RecurrentRunner()

        # Manual optimization: explicit backward + opt step (legacy parity + dual-opt clarity).
        self.automatic_optimization = False
        self._fit_path_carry = None

        # Metrics are cloned for train/val to allow separate logging and state management.
        self.train_metrics = build_train_metrics(TEM_STEP_ROUTES).clone(
            prefix="train/"
        )
        self.val_metrics = build_val_metrics(TEM_EPISODE_ROUTES).clone(
            prefix="val/"
        )
        self.primary_val_metric_key = f"val/{TEM_PRIMARY_VAL_ROUTE_KEY}"
        # Buffer + assembler implement partial-reset batching for ACT runs.
        self._train_buffer: FifoBuffer | None = None
        self._train_batch_assembler: PartialResetBatchAssembler | None = None

    @property
    def config(self) -> TEMV2ModelConfig:
        """Return the parsed configuration used by this LightningModule."""
        return self._config

    def _train_chunk_steps(self) -> int:
        """Return the TBPTT chunk length used for one optimizer update."""
        if self.config.controller.window_size is not None:
            return self.config.controller.window_size
        return self.config.runtime.sequence.tbptt_steps

    def _build_runtime(  # ----------------------------------------------------
        self,
        *,
        batch_size: int,
    ) -> tuple[None, ReplayTrajectoryController, TEMObjective]:
        """Construct one phase-local TEM arena replay runtime around the shared model."""
        replay = ArenaReplayCapability()
        controller = ReplayTrajectoryController(
            self.adapter, self.config.controller, runtime=replay
        )
        return (
            None,
            controller,
            TEMObjective(self.config.loss, task_binding=ArenaTEMTaskBinding()),
        )

    def _ensure_train_runtime(self) -> None:
        """Initialize the training runtime once per process."""
        if (
            self.train_controller is not None
            and self.train_objective is not None
        ):
            return
        self.train_environment, self.train_controller, self.train_objective = (
            self._build_runtime(batch_size=self._local_batch_size())
        )

    def _ensure_eval_runtime(self) -> None:
        """Initialize the evaluation runtime once per process."""
        if self.eval_controller is not None and self.eval_objective is not None:
            return
        self.eval_environment, self.eval_controller, self.eval_objective = (
            self._build_runtime(batch_size=self._local_batch_size())
        )

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

    def _ensure_train_batch_assembler(  # -------------------------------------
        self,
        batch: Batch,
    ) -> PartialResetBatchAssembler:
        """Create the partial-reset buffer lazily from the observed static maze schema."""
        if self._train_batch_assembler is not None:
            return self._train_batch_assembler

        keys = (
            infer_arena_replay_batch_keys(batch)
            if self.config.environment is None
            else infer_tem_static_batch_keys(batch)
        )
        capacity_rows = 4 * self.config.global_batch_size
        self._train_buffer = FifoBuffer(capacity_rows, keys, pin_memory=True)
        self._train_batch_assembler = PartialResetBatchAssembler(
            buffer=self._train_buffer, keys=keys
        )
        return self._train_batch_assembler

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
    ) -> tuple[list[Optimizer], list[SequentialLR]]:
        """Build the optimizer and learning-rate scheduler."""
        total_steps = int(self.trainer.estimated_stepping_batches)

        # Optimizer for the main model parameters
        sup_params = [p for p in self.adapter.parameters() if p.requires_grad]
        opt_sup = Adam(sup_params, self.config.optimizer)
        sch_sup = CosineAnnealingLRWithWarmup(
            opt_sup, total_steps, self.config.scheduler
        )

        return [opt_sup], [sch_sup]

    def _apply_runtime(  # ----------------------------------------------------
        self,
        step: int,
        *,
        log_values: bool,
    ) -> TEMRuntimeState:
        """Resolve and apply TEM runtime dynamics for the current global step."""
        runtime = resolve_tem_runtime(step, self.config.runtime)
        self.model.set_runtime(
            runtime.eta, runtime.hebbian_decay, runtime.p2g_uncertainty_offset
        )

        if log_values:
            self.log( "train/runtime/eta", runtime.eta,
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

        return runtime

    def on_train_epoch_start(  # ----------------------------------------------
        self,
    ) -> None:
        """Reset training carry and metric state at the start of each epoch."""
        self._reset_fit_path_stream()
        self.train_metrics.reset()

    def on_validation_epoch_start(  # -----------------------------------------
        self,
    ) -> None:
        """Reset validation metrics at the start of each validation epoch."""
        self.val_metrics.reset()

    def _validation_seed(  # --------------------------------------------------
        self,
        batch_idx: int,
    ) -> int:
        """Return the explicit evaluation seed for one validation batch."""
        seed = self.config.runtime.validation.seed
        if seed is None:
            raise ValueError(
                "TEM evaluation requires runtime.validation.seed to be set.",
            )
        return int(seed) + int(batch_idx)

    def training_step(  # -----------------------------------------------------
        self,
        batch: Batch,
        batch_idx: int,
    ) -> Dict[str, object]:
        """Run one TEM chunked-TBPTT optimizer update through the recurrent runner."""
        runtime = self._apply_runtime(self.global_step, log_values=True)
        train_controller = self._require_train_controller()
        train_objective = self._require_train_objective()
        batch_assembler = self._ensure_fit_path_batch_assembler(batch)

        if self._fit_path_carry is None:
            self._fit_path_carry = train_controller.initial_state(batch)

        source = PartialResetSource(
            incoming=batch, assembler=batch_assembler, carry0=self._train_carry
        )
        objective_options = train_objective.runtime_loss_options(
            self.global_step, p2g_use=runtime.p2g_use
        )
        evaluation = evaluate_rollout_streaming(
            runner=self._train_runner,
            source=source,
            controller=train_controller,
            carry=self._fit_path_carry,
            objective=train_objective,
            max_rollout_steps=self._train_chunk_steps(),
            metric_collection=self.train_metrics,
            metric_routes=TEM_STEP_ROUTES,
            objective_options=objective_options,
        )

        loss = evaluation.loss
        loss = (
            loss / self._train_chunk_steps()
        )  # Average loss across the chunk for smoother gradients.

        optimizers = self.optimizers()
        optimizer_handles = (
            list(optimizers) if isinstance(optimizers, list) else [optimizers]
        )
        for opt in optimizer_handles:
            opt.zero_grad(set_to_none=True)  # type: ignore

        self.manual_backward(loss)

        for opt in optimizer_handles:
            opt.step()  # type: ignore

        self._log_train_gradient_norms()

        scheduler = self.lr_schedulers()
        for sch in scheduler if isinstance(scheduler, list) else [scheduler]:
            sch.step()  # type: ignore

        # Commit detached carry so the next chunk resumes from where this one ended.
        self._fit_path_carry = evaluation.execution.final_carry.detach()
        protocol_count = evaluation.last_step.outputs.losses.protocol_count

        # Fit-path diagnostics.
        final_carry = evaluation.execution.final_carry
        valid_prev = prev_trajectory_id >= 0
        reused = (final_carry.trajectory_id == prev_trajectory_id) & valid_prev
        self.log(
            "train/fit_path/max_cursor", final_carry.cursor.max().float(),
            on_step=True, on_epoch=False, logger=True,
        )  # fmt: skip
        self.log(
            "train/fit_path/min_cursor", final_carry.cursor.min().float(),
            on_step=True, on_epoch=False, logger=True,
        )  # fmt: skip
        self.log(
            "train/fit_path/halted_fraction", final_carry.halted.float().mean(),
            on_step=True, on_epoch=False, logger=True,
        )  # fmt: skip
        self.log(
            "train/fit_path/reused_trajectory_fraction", reused.float().mean(),
            on_step=True, on_epoch=False, logger=True,
        )  # fmt: skip
        self.log(
            "train/fit_path/protocol_count", protocol_count.float(),
            on_step=True, on_epoch=False, logger=True,
        )  # fmt: skip

        # Log the accumulated chunk loss to TensorBoard.
        self.log(
            "train/loss", loss.detach(),
            on_step=True, on_epoch=False, prog_bar=True, logger=True,
        )  # fmt: skip

        return {
            "loss": loss.detach(),
            "signals": evaluation.last_step.outputs.signals,
        }

    def validation_step(  # -----------------------------------------------------------------------
        self,
        batch: Batch,
        batch_idx: int,
    ) -> Dict[str, object]:
        """Run a full TEM rollout through the recurrent runner and trace observer."""
        runtime = self._apply_runtime(self.global_step, log_values=False)
        eval_controller = self._require_eval_controller()
        eval_objective = self._require_eval_objective()
        step_options = {"allow_halt": True, "explore": False}
        carry0 = eval_controller.initial_state(batch)
        objective_options = eval_objective.runtime_loss_options(
            self.global_step, p2g_use=runtime.p2g_use
        )

        evaluation = evaluate_rollout(
            runner=self._eval_runner,
            source=RepeatSource(batch),
            controller=eval_controller,
            carry=carry0,
            objective=eval_objective,
            max_rollout_steps=self.config.runtime.validation.max_rollout_steps,
            hard_max_rollout_steps=self.config.runtime.validation.hard_max_rollout_steps,
            runner_options=step_options,
            objective_options=objective_options,
        )
        update_metric_collection_from_evaluated_chunk(
            self.val_metrics, evaluation.evaluated, TEM_EPISODE_ROUTES
        )


#  ============================================================================
__all__ = ["TEMV2ModelConfig", "TEMV2TrainingModel"]
