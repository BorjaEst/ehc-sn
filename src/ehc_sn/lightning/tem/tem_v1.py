"""TEM v1 Lightning runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import lightning as L
from pydantic import BaseModel, Field, model_validator
from torch.optim import Optimizer

from ehc_sn.adapters.navigation.bridges.tem.objectives import NavigationTEMTaskBinding
from ehc_sn.adapters.navigation.bridges.tem.tem_v1 import NavigationTEMV1AdapterSettings, NavigationTEMV1BridgeAdapter
from ehc_sn.adapters.navigation.bridges.tem.traces import NAVIGATION_TEM_TRACE_FIELDS, select_navigation_tem_trace_fields
from ehc_sn.controllers.tem import TEMController, TEMControllerConfig
from ehc_sn.envs.dungeon_walk import DungeonWalk as Environment
from ehc_sn.envs.dungeon_walk import EnvConfig as EnvironmentConfig
from ehc_sn.lightning._rollout import (
    evaluate_rollout,
    evaluate_rollout_streaming,
    observe_rollout_chunk,
    update_metric_collection_from_evaluated_chunk,
)
from ehc_sn.lightning.tem.core.runtime import RuntimeConfig, TEMRuntimeState, resolve_tem_runtime
from ehc_sn.metrics import build_train_metrics, build_val_metrics
from ehc_sn.metrics.routes import TEM_EPISODE_ROUTES, TEM_PRIMARY_VAL_ROUTE_KEY, TEM_STEP_ROUTES
from ehc_sn.metrics.traces import ReplayableEnvironments, build_trace_spec
from ehc_sn.models.tem.tem_v1 import ModelSettingsV1, TEMModelV1
from ehc_sn.objectives.tem import TEMLossConfig, TEMLossHead
from ehc_sn.rollouts import PartialResetSource, RecurrentRunner, RepeatSource
from ehc_sn.tasks.navigation import NavigationControllerRuntime
from ehc_sn.training.buffers import FifoBuffer
from ehc_sn.training.distributed import normalize_loss_for_backward
from ehc_sn.training.optim import Adam, AdamConfig
from ehc_sn.training.partial_reset import PartialResetBatchAssembler
from ehc_sn.training.schedules import CosineAnnealingLRWithWarmup, SchedulerConfig, SequentialLR
from ehc_sn.types import Batch

# Community-standard map-style batch: plain dict returned by MazeDataset / DataLoader.
TEM_STATIC_REQUIRED_KEYS = ("topology", "observations", "mask_valid")
TEM_STATIC_OPTIONAL_KEYS = ("regions", "start", "goals", "landmarks")


# =================================================================================================
class ModelConfig_TEM_V1(BaseModel, extra="forbid"):
    """Top-level TEM v1 training config."""

    # ~~ Model architecture ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    model_config_path: Path = Field(
        ...,
        description="Path to the model configuration TOML file that specifies the TEM v1 architecture.",
    )
    adapter: NavigationTEMV1AdapterSettings = Field(
        ...,
        description="Settings for the navigation bridge adapter that binds TEM v1 to task inputs/outputs.",
    )
    environment: EnvironmentConfig = Field(
        ...,
        description="",
    )
    controller: TEMControllerConfig = Field(
        ...,
        description="",
    )
    objective: TEMLossConfig = Field(
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
    def validate_environment_contract(self) -> "ModelConfig_TEM_V1":
        model_settings = ModelSettingsV1.from_config(self.model_config_path)
        if self.environment.action_count != model_settings.transition_action_count:
            raise ValueError("environment.action_count must match model.transition_action_count.")
        if self.adapter.observation_dim != self.environment.observation_dim:
            raise ValueError("adapter.observation_dim must match environment.observation_dim.")
        if self.adapter.action_count != self.environment.action_count:
            raise ValueError("adapter.action_count must match environment.action_count.")
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
        self.bridge_adapter = NavigationTEMV1BridgeAdapter(self.model, config.adapter)
        self._controller_runtime = NavigationControllerRuntime()
        self.train_environment: Environment | None = None
        self.train_controller: TEMController | None = None
        self.train_objective: TEMLossHead | None = None
        self.eval_environment: Environment | None = None
        self.eval_controller: TEMController | None = None
        self.eval_objective: TEMLossHead | None = None
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
        self.trace_specs = build_trace_spec("tem", extra_fields=NAVIGATION_TEM_TRACE_FIELDS)
        self._eval_trace_keys: set[str] | None = None

        # Buffer + assembler implement partial-reset batching for ACT runs.
        self._train_buffer: FifoBuffer | None = None
        self._train_batch_assembler: PartialResetBatchAssembler | None = None

    @property
    def config(self) -> ModelConfig_TEM_V1:
        """Return the parsed configuration used by this LightningModule."""
        return self._config

    def _local_batch_size(self) -> int:
        """Return the per-rank batch size used by train and evaluation runtimes."""
        trainer = getattr(self, "_trainer", None)
        world_size = max(getattr(trainer, "world_size", 1), 1)
        return max(self.config.global_batch_size // world_size, 1)

    def _train_chunk_steps(self) -> int:
        """Return the TEM TBPTT chunk length used for one optimizer update."""
        return self.config.runtime.sequence.tbptt_steps

    def _build_runtime(self, *, batch_size: int) -> tuple[Environment, TEMController, TEMLossHead]:
        """Construct one phase-local TEM rollout runtime around the shared model."""
        environment = Environment(self.config.environment, batch_size=batch_size)
        controller = TEMController(self.bridge_adapter, environment, self.config.controller, self._controller_runtime)
        objective = TEMLossHead(self.config.objective, task_binding=NavigationTEMTaskBinding())
        return environment, controller, objective

    def _ensure_train_runtime(self) -> None:
        """Initialize the training runtime once per process."""
        if self.train_environment is not None and self.train_controller is not None and self.train_objective is not None:
            return
        self.train_environment, self.train_controller, self.train_objective = self._build_runtime(batch_size=self._local_batch_size())

    def _ensure_eval_runtime(self) -> None:
        """Initialize the evaluation runtime once per process."""
        if self.eval_environment is not None and self.eval_controller is not None and self.eval_objective is not None:
            return
        self.eval_environment, self.eval_controller, self.eval_objective = self._build_runtime(batch_size=self._local_batch_size())

    def _require_train_controller(self) -> TEMController:
        """Return the training controller, initializing the train runtime if needed."""
        self._ensure_train_runtime()
        if self.train_controller is None:
            raise RuntimeError("TEM training runtime is not initialized.")
        return self.train_controller

    def _require_train_objective(self) -> TEMLossHead:
        """Return the training objective, initializing the train runtime if needed."""
        self._ensure_train_runtime()
        if self.train_objective is None:
            raise RuntimeError("TEM training runtime is not initialized.")
        return self.train_objective

    def _require_eval_controller(self) -> TEMController:
        """Return the evaluation controller, initializing the eval runtime if needed."""
        self._ensure_eval_runtime()
        if self.eval_controller is None:
            raise RuntimeError("TEM evaluation runtime is not initialized.")
        return self.eval_controller

    def _require_eval_objective(self) -> TEMLossHead:
        """Return the evaluation objective, initializing the eval runtime if needed."""
        self._ensure_eval_runtime()
        if self.eval_objective is None:
            raise RuntimeError("TEM evaluation runtime is not initialized.")
        return self.eval_objective

    def _build_trace_meta(self, controller: TEMController) -> dict[str, object]:
        """Return out-of-band trace metadata for figure-facing evaluation traces."""
        return {"environments": ReplayableEnvironments(controller.environment.build_world_descriptors())}

    def _ensure_train_batch_assembler(  # ---------------------------------------------------------
        self, batch: Batch,
    ) -> PartialResetBatchAssembler:  # fmt: skip
        """Create the partial-reset buffer lazily from the observed static maze schema."""
        if self._train_batch_assembler is not None:
            return self._train_batch_assembler

        keys = infer_tem_static_batch_keys(batch)
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

    def _validation_seed(self, batch_idx: int) -> int:
        """Return the explicit evaluation seed for one validation batch."""
        seed = self.config.runtime.validation.seed
        if seed is None:
            raise ValueError("TEM evaluation requires runtime.validation.seed to be set.")
        return int(seed) + int(batch_idx)

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
        local_bs = batch_size_from_static_maze_batch(batch)
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
        eval_controller.set_evaluation_seed(self._validation_seed(batch_idx))
        step_options = {"allow_halt": True, "explore": False}
        carry0 = eval_controller.initial_state(batch)
        trace_meta = self._build_trace_meta(eval_controller)

        # Initialize carry/state on the first batch
        if self._eval_trace_keys is None:
            trace_specs = self.trace_specs
        else:
            nav_extra = select_navigation_tem_trace_fields(self._eval_trace_keys)
            trace_specs = build_trace_spec("tem", include_keys=self._eval_trace_keys, extra_fields=nav_extra)

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
        trace = observe_rollout_chunk(evaluation.chunk, trace_specs, trace_meta=trace_meta)
        update_metric_collection_from_evaluated_chunk(self.val_metrics, evaluation.evaluated, TEM_EPISODE_ROUTES)
        return {"trace": trace}


# =================================================================================================
def infer_tem_static_batch_keys(  # ---------------------------------------------------------------
    batch: Batch,
) -> tuple[str, ...]:  # fmt: skip
    """Return the static maze keys that must move together through partial reset."""
    missing = [key for key in TEM_STATIC_REQUIRED_KEYS if key not in batch]
    if missing:
        raise KeyError(f"TEM batch is missing required static maze keys: {', '.join(missing)}.")
    return TEM_STATIC_REQUIRED_KEYS + tuple(key for key in TEM_STATIC_OPTIONAL_KEYS if key in batch)


# =================================================================================================
def batch_size_from_static_maze_batch(  # ---------------------------------------------------------
    batch: Batch,
) -> int:  # fmt: skip
    """Return the leading batch dimension from the required TEM maze tensor schema."""
    return int(batch[TEM_STATIC_REQUIRED_KEYS[0]].shape[0])
