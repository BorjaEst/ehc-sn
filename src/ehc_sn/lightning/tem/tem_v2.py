"""TEM v2 Lightning runtime."""

from __future__ import annotations

from itertools import repeat
from typing import Any, Dict, Optional, TypeAlias

import lightning as L
from pydantic import BaseModel, Field, model_validator
from torch import Tensor
from torch.optim import Optimizer

from ehc_sn.controllers.tem import TEMController, TEMControllerConfig
from ehc_sn.envs.dungeon_walk import DungeonWalk as Environment
from ehc_sn.envs.dungeon_walk import EnvConfig as EnvironmentConfig
from ehc_sn.heads.tem import TEMLossConfig, TEMLossHead
from ehc_sn.lightning.tem.core.runtime import RuntimeConfig, TEMRuntimeState, resolve_tem_runtime
from ehc_sn.metrics import build_train_metrics, build_val_metrics, update_metrics_from_step
from ehc_sn.metrics.routes import TEM_EPISODE_ROUTES, TEM_STEP_ROUTES
from ehc_sn.metrics.traces import build_trace_spec
from ehc_sn.models.tem.tem_v2 import Batch, ModelSettings_V2, TEMModelV2
from ehc_sn.rollouts.collect import TraceCollector
from ehc_sn.rollouts.trace_tree import TraceTree
from ehc_sn.training.buffers import FifoBuffer
from ehc_sn.training.optim import Adam, AdamConfig
from ehc_sn.training.partial_reset import PartialResetBatchAssembler
from ehc_sn.training.schedules import CosineAnnealingLRWithWarmup, SchedulerConfig, SequentialLR
from ehc_sn.training.step_loop import StepLoop

# Community-standard map-style batch: plain dict returned by MazeDataset / DataLoader.
TEM_STATIC_REQUIRED_KEYS = ("topology", "observations", "mask_valid")
TEM_STATIC_OPTIONAL_KEYS = ("regions", "start", "goals", "landmarks")


# =================================================================================================
class ModelConfig_TEM_V2(BaseModel, extra="forbid"):
    """Top-level TEM v2 training config."""

    # ~~ Model architecture ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    model: ModelSettings_V2 = Field(
        ...,
        description="TEM v2 backbone settings.",
    )
    environment: EnvironmentConfig = Field(
        ...,
        description="Dungeon-walk environment configuration.",
    )
    controller: TEMControllerConfig = Field(
        ...,
        description="TEM rollout controller configuration.",
    )
    loss: TEMLossConfig = Field(
        ...,
        description="TEM loss-head configuration.",
    )

    # ~~ Optimizers & scheduling ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

    # ~~ Extra ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    global_batch_size: int = Field(
        ...,
        description="Global batch size across all devices.",
    )  # TODO: consider moving to BufferSettings or similar

    @model_validator(mode="after")
    def validate_environment_contract(self) -> "ModelConfig_TEM_V2":
        if self.environment.observation_dim != self.model.observation_dim:
            raise ValueError("environment.observation_dim must match model.observation_dim.")
        if self.environment.action_count != self.model.action_count:
            raise ValueError("environment.action_count must match model.action_count.")
        return self


# =================================================================================================
class TrainingModel(L.LightningModule):
    """Lightning wrapper for TEM v2 training and evaluation."""

    def __init__(  # ------------------------------------------------------------------------------
        self, config: ModelConfig_TEM_V2,
    ) -> None:  # fmt: skip
        """Create the Lightning module from a parsed TEM v2 training config."""
        super().__init__()
        self.model = TEMModelV2(config.model)
        self.environment: Environment | None = None  #  Lazy init in setup() to avoid GPU alloc issues
        self.controller: TEMController | None = None  #  Lazy init in setup() to avoid GPU alloc issues
        self.step_module: TEMLossHead | None = None  #  Lazy init in setup() to avoid GPU alloc issues
        self._config = config

        # Manual optimization: explicit backward + opt step (legacy parity + dual-opt clarity).
        self.automatic_optimization = False
        self._train_carry = None

        # Metrics are cloned for train/val to allow separate logging and state management.
        self.train_metrics = build_train_metrics(TEM_STEP_ROUTES).clone(prefix="train/")
        self.val_metrics = build_val_metrics(TEM_EPISODE_ROUTES).clone(prefix="val/")
        self.trace_specs = build_trace_spec("tem")
        self._eval_trace_keys: set[str] | None = None

        # Buffer + assembler implement partial-reset batching for ACT runs.
        self._train_buffer: FifoBuffer | None = None
        self._train_batch_assembler: PartialResetBatchAssembler | None = None

    @property
    def config(self) -> ModelConfig_TEM_V2:
        """Return the parsed configuration used by this LightningModule."""
        return self._config

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
        """Lazy initialization of the environment to avoid GPU allocation issues in DDP."""
        world_size = max(getattr(self.trainer, "world_size", 1), 1)
        local_bs = self.config.global_batch_size // world_size

        if self.environment is None:
            self.environment = Environment(self.config.environment, batch_size=local_bs)
        self.controller = TEMController(self.model, self.environment, self.config.controller)
        self.step_module = TEMLossHead(self.controller, self.config.loss)

    def configure_optimizers(  # -------------------------------------------------------------------
        self,
    ) -> tuple[list[Optimizer], list[SequentialLR]]:  # fmt: skip
        """Build the optimizer and learning-rate scheduler."""
        total_steps = int(self.trainer.estimated_stepping_batches)

        # Optimizer for the main model parameters
        sup_params = [p for p in self.model.parameters() if p.requires_grad]
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
        """Reset training carry and metric state at the start of each epoch."""
        self._train_carry = None
        if self._train_buffer is not None:
            self._train_buffer.clear()
        self.train_metrics.reset()

    def on_validation_epoch_start(  # ------------------------------------------------------------
        self,
    ) -> None:  # fmt: skip
        """Reset validation metrics at the start of each validation epoch."""
        self.val_metrics.reset()

    def set_eval_trace_keys(self, keys: set[str]) -> None:
        """Set the semantic trace keys required for evaluation-time figure capture."""
        self._eval_trace_keys = set(keys)

    # -- Training ----------------------------------------------------------------------------------

    def training_step(  # -------------------------------------------------------------------------
        self, batch: Batch, batch_idx: int,
    ) -> Dict[str, object]:  # fmt: skip
        """Run one TEM training step with manual optimization.

        Step-based checkpoints are treated as post-update recovery checkpoints.
        This loop intentionally does not snapshot pre-optimization weights inside
        ``training_step``.
        """
        self._apply_runtime(self.global_step, log_values=True)
        batch_assembler = self._ensure_train_batch_assembler(batch)

        # Initialize carry/state on the first batch
        if self._train_carry is None:
            self._train_carry = self.step_module.initial_carry(batch)

        # Assemble partial-reset step batch
        step_batch = batch_assembler.make_step_batch(
            incoming=batch,
            reset_mask=self._train_carry.halted,
        )

        # Horizon=1 step loop: run one step of the controller
        step_batches = repeat(step_batch, 1)
        step_options = {}  # FIXME after the HeadLoss and TEMController support options
        carry0 = self._train_carry

        step = None
        for _t, step in StepLoop(self.step_module, step_batches, carry0, options=step_options):
            pass  # horizon = 1; loop runs exactly once
        if step is None:
            raise ValueError("StepLoop did not yield any steps.")
        self._train_carry = step.carry.detach()

        # Normalize by local batch size; DDP averages gradients across ranks.
        local_bs = batch_size_from_static_maze_batch(batch)
        loss = _normalize_loss_for_backward(step.outputs.loss, local_bs=local_bs)

        optimizers = self.optimizers()
        for opt in optimizers if isinstance(optimizers, list) else [optimizers]:
            opt.zero_grad(set_to_none=True)  # type: ignore

        self.manual_backward(loss)

        for opt in optimizers if isinstance(optimizers, list) else [optimizers]:
            opt.step()  # type: ignore

        scheduler = self.lr_schedulers()
        for sch in scheduler if isinstance(scheduler, list) else [scheduler]:
            sch.step()  # type: ignore

        # Update metrics with unnormalized loss and log to TensorBoard.
        update_metrics_from_step(self.train_metrics, step.outputs.metrics, TEM_STEP_ROUTES)
        self.log("train/loss", loss.detach(), on_step=True, on_epoch=False, prog_bar=True, logger=True)

        return {"loss": loss.detach(), "signals": step.outputs.signals}

    def validation_step(  # -----------------------------------------------------------------------
        self, batch: Batch, batch_idx: int,
    ) -> Dict[str, object]:  # fmt: skip
        """Run a full TEM rollout until all slots reach the configured horizon.

        Validation does not persist carry across batches and accumulates episode
        metrics across the rollout, matching the ACT/RL validation pattern.
        """
        self._apply_runtime(self.global_step, log_values=False)
        step_batches = repeat(batch)  # Run until all examples halt
        step_options = {"allow_halt": False, "explore": False}
        carry0 = self.step_module.initial_carry(batch)

        # Initialize carry/state on the first batch
        if self._eval_trace_keys is None:
            trace_specs = self.trace_specs
        else:
            trace_specs = build_trace_spec("tem", include_keys=self._eval_trace_keys)

        step, collector = None, TraceCollector(TraceTree(), trace_specs)
        for t, step in StepLoop(self.step_module, step_batches, carry0, options=step_options):
            collector.append(t, step)
            update_metrics_from_step(self.val_metrics, step.outputs.metrics, TEM_EPISODE_ROUTES)
        if step is None:
            raise ValueError("Evaluation loop did not yield any steps, cannot log metrics.")

        return {"trace": collector.tree}


def _normalize_loss_for_backward(  # --------------------------------------------------------------
    total_loss: Tensor, local_bs: int,
) -> Tensor:  # fmt: skip
    """Normalize the total loss by the local batch size for distributed training."""
    if local_bs <= 0:
        raise ValueError(f"local_bs must be positive, got {local_bs}.")
    return total_loss / float(local_bs)


# =================================================================================================
def infer_tem_static_batch_keys(  # ----------------------------------------------------------------
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
