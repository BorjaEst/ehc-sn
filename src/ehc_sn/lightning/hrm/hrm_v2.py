"""HRM v2 Lightning module (hybrid RL + warmup).

This module defines a PyTorch Lightning :class:`~lightning.LightningModule` wrapper
around the HRM v2 architecture (:class:`HRModelV2`) and its training loop.

Compared to HRM v1 (ACT-supervised), HRM v2 couples a PFC-style recurrent reasoning
core with an STR actor-critic head and trains via a deliberation actor-critic pipeline:

    source -> controller.step() -> ActorCriticInteractionRecord
    -> TD0ActorCriticBatchBuilder.build_deliberation_ac_batch() -> HybridActorCriticBatch
    -> HybridRLLossHead.compute_step() -> loss -> optimizer

Key behaviors:
    - **Manual optimization**: sets ``automatic_optimization = False`` and performs
        explicit backward/optimizer/scheduler steps.
    - **Three-optimizer training**: supervised params, RL (STR) params, and vmPFC
        (``pfc.estimator``) params are optimized with separate optimizers.
    - **Warmup**: for the first ``supervised_only_warmup_steps`` global steps,
        ``allow_halt=False`` forces full deliberation and RL losses are zeroed.
    - **Partial reset batching**: halted slots are replaced with fresh rows using
        :class:`~ehc_sn.training.buffers.FifoBuffer` and
        :class:`~ehc_sn.training.partial_reset.PartialResetBatchAssembler`.

The batch structure used throughout this file is a plain ``dict[str, Tensor]``
with keys ``"input_ids"`` and ``"labels"``.
"""

from pathlib import Path
from typing import Any, Optional

import lightning as L
from pydantic import BaseModel, Field
from torch import Tensor
from torch.optim import Optimizer

from ehc_sn.adapters.mazehard.hrm import (
    MazeHardHRMAdapterSettings,
    MazeHardHRMV2BridgeAdapter,
    MazeHardHRMV2HybridTaskBinding,
    build_mazehard_hrm_trace_meta,
)
from ehc_sn.adapters.mazehard.hrm.traces import MAZE_HARD_HRM_ACTOR_CRITIC_TRACE_FIELDS
from ehc_sn.controllers.deliberation.actor_critic import DeliberationACController, DeliberationACControllerConfig
from ehc_sn.lightning._rollout import evaluate_rollout, observe_rollout_chunk, update_metric_collection_from_evaluated_chunk
from ehc_sn.lightning.hrm.core.runtime import RuntimeConfig
from ehc_sn.metrics import build_train_metrics, build_val_metrics, update_metrics_from_step
from ehc_sn.metrics.routes import RL_EPISODE_ROUTES, RL_STEP_ROUTES
from ehc_sn.metrics.traces import build_trace_spec
from ehc_sn.models.hrm.hrm_v2 import HRModelV2, ModelSettingsV2
from ehc_sn.objectives.hybrid_rl import HybridRLLossConfig, HybridRLLossHead
from ehc_sn.rollouts import PartialResetSource, RecurrentRunner, RepeatSource, SingleStepRunner
from ehc_sn.tasks.mazehard.capabilities.deliberation import MazeHardDeliberationCapability, MazeHardDeliberationConfig
from ehc_sn.tasks.mazehard.reward import MazeHardRewardProjector
from ehc_sn.training.actor_critic import TD0ActorCriticBatchBuilder, ZeroBootstrapActorCriticValidationScorer
from ehc_sn.training.buffers import FifoBuffer
from ehc_sn.training.distributed import normalize_loss_for_backward
from ehc_sn.training.optim import AdamATan2, AdamATan2Config
from ehc_sn.training.partial_reset import PartialResetBatchAssembler
from ehc_sn.training.schedules import CosineAnnealingLRWithWarmup, SchedulerConfig, SequentialLR
from ehc_sn.types import Batch


# =================================================================================================
class ModelConfig_HRM_V2(BaseModel, extra="forbid"):
    """Configuration for the HRM v2 Lightning module.

    This config wires together:
        - model settings (:class:`ModelSettingsV2`)
        - deliberation capability settings (:class:`~ehc_sn.tasks.mazehard.capabilities.deliberation.MazeHardDeliberationConfig`)
        - deliberation controller and objective configs
        - optimizer and scheduler settings

    Notes:
        - ``extra=\"forbid\"`` ensures unknown keys fail fast.
        - ``global_batch_size`` is used to derive per-rank batch size under DDP.
    """

    # ~~ Model architecture ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    model_config_path: Path = Field(
        ...,
        description="Path to the model configuration TOML file that specifies the HRM v2 architecture.",
    )
    adapter: MazeHardHRMAdapterSettings = Field(
        default_factory=MazeHardHRMAdapterSettings,
        description="Settings for the MazeHard bridge adapter that binds the HRM v2 core to task inputs/outputs.",
    )
    deliberation: MazeHardDeliberationConfig = Field(
        ...,
        description="Deliberation capability config (halt_action, episode_horizon) for the MazeHard deliberation path.",
    )
    controller: DeliberationACControllerConfig = Field(
        default_factory=DeliberationACControllerConfig,
        description="Configuration for the deliberation actor-critic controller.",
    )
    objective: HybridRLLossConfig = Field(
        ...,
        description="Configuration for the hybrid RL objective scorer.",
    )

    # ~~ Optimizers & scheduling ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    optimizer_supervised: AdamATan2Config = Field(
        default_factory=AdamATan2Config,
        description="optimizer for supervised parameters (PFC + embeddings + LM head).",
    )
    optimizer_rl: AdamATan2Config = Field(
        default_factory=AdamATan2Config,
        description="optimizer for RL parameters (STR actor-critic only).",
    )
    optimizer_qv: AdamATan2Config = Field(
        default_factory=AdamATan2Config,
        description="optimizer for vmPFC parameters (pfc.estimator only).",
    )
    scheduler: SchedulerConfig = Field(
        default_factory=SchedulerConfig,
        description="LR scheduler config applied to both optimizers.",
    )
    supervised_only_warmup_steps: int = Field(
        default=5000,
        ge=0,
        description=(
            "Number of optimizer steps during which only the supervised optimizer trains. "
            "STR and vmPFC are frozen; allow_halt=False forces full deliberation. "
            "Prevents 'halt immediately' collapse before PFC representations are informative."
        ),
    )
    runtime: RuntimeConfig = Field(
        default_factory=RuntimeConfig,
        description="HRM runtime-owned validation safety settings.",
    )

    # ~~ Extra ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    global_batch_size: int = Field(
        ...,
        description=(
            "Global batch size across all devices. " "The per-device batch size is computed as `global_batch_size // world_size`."
        ),
    )  # TODO: consider moving to BufferSettings or similar


# =================================================================================================
class TrainingModel(L.LightningModule):
    """LightningModule wrapper for HRM v2 deliberation actor-critic training.

    This wrapper manages:
        - wiring :class:`~ehc_sn.controllers.deliberation.actor_critic.DeliberationACController` and
          :class:`~ehc_sn.objectives.hybrid_rl.HybridRLLossHead`
        - partial-reset batching via FIFO buffering
        - manual optimization with three optimizers
    """

    def __init__(  # ------------------------------------------------------------------------------
        self, config: ModelConfig_HRM_V2,
    ) -> None:  # fmt: skip
        super().__init__()
        model_settings = ModelSettingsV2.from_config(config.model_config_path)
        self.model = HRModelV2(model_settings)
        self.bridge_adapter = MazeHardHRMV2BridgeAdapter(self.model, config.adapter)
        self.controller: DeliberationACController | None = None  # Initialized in setup()
        self.objective: HybridRLLossHead | None = None  # Initialized in setup()
        self.learner: TD0ActorCriticBatchBuilder | None = None  # Initialized in setup()
        self.val_scorer: ZeroBootstrapActorCriticValidationScorer | None = None  # Initialized in setup()
        self._config = config
        self._train_runner = SingleStepRunner()
        self._eval_runner = RecurrentRunner()

        # Manual optimization: explicit backward + opt step (legacy parity + dual-opt clarity).
        self.automatic_optimization = False
        self._train_carry = None

        # Metrics are cloned for train/val to allow separate logging and state management.
        self.train_metrics = build_train_metrics(RL_STEP_ROUTES).clone(prefix="train/")
        self.val_metrics = build_val_metrics(RL_EPISODE_ROUTES).clone(prefix="val/")
        self.trace_specs = build_trace_spec("rl", extra_fields=MAZE_HARD_HRM_ACTOR_CRITIC_TRACE_FIELDS)

        # Buffer + assembler implement partial-reset batching for deliberation runs.
        self._train_buffer = FifoBuffer(
            capacity_rows=4 * config.global_batch_size,
            keys=("input_ids", "labels"),
            pin_memory=True,
        )
        self._train_batch_assembler = PartialResetBatchAssembler(
            buffer=self._train_buffer,
            keys=("input_ids", "labels"),
        )

    @property
    def config(self) -> ModelConfig_HRM_V2:
        """Return the parsed configuration used by this LightningModule."""
        return self._config

    def setup(  # --------------------------------------------------------------------------------
        self, stage: Optional[str] = None,
    ) -> None:  # fmt: skip
        """Initialize the deliberation controller and wiring."""
        task_config = self.config.deliberation
        finalizer = MazeHardDeliberationCapability(task_config, MazeHardRewardProjector())
        self.controller = DeliberationACController(self.bridge_adapter, self.config.controller, finalizer)
        self.objective = HybridRLLossHead(self.config.objective)
        task_binding = MazeHardHRMV2HybridTaskBinding()
        self.learner = TD0ActorCriticBatchBuilder(self.bridge_adapter, None, gamma=self.config.objective.gamma, task_binding=task_binding)
        self.val_scorer = ZeroBootstrapActorCriticValidationScorer(self.objective, task_binding)

    def configure_optimizers(  # ------------------------------------------------------------------
        self,
    ) -> tuple[list[Optimizer], list[SequentialLR]]:  # fmt: skip
        """Build optimizers and schedulers.

        Returns:
            ``(optimizers, schedulers)`` where both lists have length 3 and share
            the same order:
                1) supervised params (everything except ``pfc.estimator``)
                2) RL params (STR actor-critic only)
                3) vmPFC params (``pfc.estimator`` only)
        """
        total_steps = int(self.trainer.estimated_stepping_batches)

        # Optimizer A: supervised — backbone + LM params only.
        vmPFC_ids = {id(p) for p in self.model.pfc.estimator.parameters()}
        str_ids = {id(p) for p in self.model.str.parameters()}
        excluded_ids = vmPFC_ids | str_ids
        sup_params = [p for p in self.bridge_adapter.parameters() if id(p) not in excluded_ids]

        # Optimizer A: supervised — backbone + LM params only ().
        opt_sup = AdamATan2(sup_params, self.config.optimizer_supervised)
        # Optimizer B: RL — STR actor-critic only (strictly isolated)
        opt_rl = AdamATan2(list(self.model.str.parameters()), self.config.optimizer_rl)
        # Optimizer C: vmPFC — pfc.estimator only (value estimator, auxiliary critic)
        opt_qv = AdamATan2(list(self.model.pfc.estimator.parameters()), self.config.optimizer_qv)

        sch_sup = CosineAnnealingLRWithWarmup(opt_sup, total_steps, self.config.scheduler)
        sch_rl = CosineAnnealingLRWithWarmup(opt_rl, total_steps, self.config.scheduler)
        sch_qv = CosineAnnealingLRWithWarmup(opt_qv, total_steps, self.config.scheduler)

        return [opt_sup, opt_rl, opt_qv], [sch_sup, sch_rl, sch_qv]

    def on_train_epoch_start(  # ------------------------------------------------------------------
        self,
    ) -> None:  # fmt: skip
        """Reset training buffer, carry, and metrics at the start of each epoch."""
        self._train_carry = None
        self._train_buffer.clear()
        self.train_metrics.reset()

    def on_validation_epoch_start(  # ------------------------------------------------------------
        self,
    ) -> None:  # fmt: skip
        """Reset validation metrics at the start of each epoch."""
        self.val_metrics.reset()

    def training_step(  # -------------------------------------------------------------------------
        self, batch: Batch, batch_idx: int,
    ) -> dict[str, Any]:  # fmt: skip
        """Run one training step (horizon = 1) with manual optimization.

        The controller emits one :class:`~ehc_sn.controllers.rl.InteractionRecord`
        per step; the objective scores it and returns a loss. The carry is the
        rollout state produced by :class:`~ehc_sn.controllers.rl.RLController`
        and is used for partial resets: rows that halted in the previous step are
        replaced with fresh examples from the current incoming batch.

        Notes:
            - ``setup()`` must have run so that ``self.controller`` and ``self.objective`` are available.
            - During warmup (``global_step < supervised_only_warmup_steps``), halting is disabled.
        """
        if self.controller is None or self.objective is None:
            raise RuntimeError("HRM v2 runtime is not initialized. Call setup() before training.")

        # Initialize carry/state on the first batch.
        if self._train_carry is None:
            self._train_carry = self.controller.initial_state(batch)

        evaluation = evaluate_rollout(
            runner=self._train_runner,
            source=PartialResetSource(incoming=batch, assembler=self._train_batch_assembler, carry0=self._train_carry),
            controller=self.controller,
            carry=self._train_carry,
            objective=self.objective,
            runner_options={"allow_halt": True, "explore": True},
            objective_options={"controller": self.controller, "td_target": True},
        )
        self._train_carry = evaluation.chunk.final_carry.detach()

        # Normalize by local batch size; DDP averages gradients across ranks.
        local_bs = int(batch["input_ids"].shape[0])
        loss = normalize_loss_for_backward(evaluation.evaluated.loss, local_bs=local_bs)

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
        update_metric_collection_from_evaluated_chunk(self.train_metrics, evaluation.evaluated, RL_STEP_ROUTES)
        self.log("train/loss", loss.detach(), on_step=True, on_epoch=False, prog_bar=True, logger=True)

        return {"loss": loss.detach(), "signals": evaluation.evaluated.last_step.outputs.signals}

    def validation_step(  # -----------------------------------------------------------------------
        self, batch: Batch, batch_idx: int,
    ) -> dict[str, Any]:  # fmt: skip
        """Run a full rollout until all slots halt, collecting traces for logging/analysis.

        Validation runs the controller to the max horizon (no exploration) and
        collects a trace tree for downstream logging/analysis.

        The :class:`~ehc_sn.training.actor_critic.ZeroBootstrapActorCriticValidationScorer` is
        used as the rollout objective.  It applies zero bootstrap values, so all RL
        loss numbers are approximate diagnostic values only.
        """
        if self.controller is None or self.val_scorer is None:
            raise RuntimeError("HRM v2 runtime is not initialized. Call setup() before validation.")

        carry0 = self.controller.initial_state(batch)
        evaluation = evaluate_rollout(
            runner=self._eval_runner,
            source=RepeatSource(batch),
            controller=self.controller,
            carry=carry0,
            objective=self.val_scorer,
            max_rollout_steps=self.config.runtime.validation.max_rollout_steps,
            hard_max_rollout_steps=self.config.runtime.validation.hard_max_rollout_steps,
            runner_options={"explore": False, "allow_halt": False},
        )
        trace = observe_rollout_chunk(evaluation.chunk, self.trace_specs, trace_meta=build_mazehard_hrm_trace_meta(batch))
        update_metric_collection_from_evaluated_chunk(self.val_metrics, evaluation.evaluated, RL_EPISODE_ROUTES)
        return {"trace": trace}


# =============================================================================
__all__ = ["ModelConfig_HRM_V2", "TrainingModel"]
