""" """

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import repeat, tee
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, TypeAlias, Union

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from adam_atan2_pytorch import AdamAtan2 as AdamATan2
from pydantic import BaseModel, ConfigDict, Field, field_validator
from torch import Tensor, nn
from torch.distributions import Normal
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ExponentialLR

from ehc_sn import utils
from ehc_sn.controllers.tem import TEMController, TEMControllerConfig, TEMOutput
from ehc_sn.data.schema import CHANNEL_SOLUTION, O_ID
from ehc_sn.data.transforms import channels_to_grid
from ehc_sn.envs.dungeon_walk import DungeonWalk as Environment
from ehc_sn.envs.dungeon_walk import EnvConfig as EnvironmentConfig
from ehc_sn.heads.tem import TEMLossConfig, TEMLossHead
from ehc_sn.metrics import build_train_metrics, build_val_metrics, update_metrics_from_step
from ehc_sn.metrics.routes import TEM_STEP_ROUTES as TEM_ROUTES
from ehc_sn.metrics.traces import build_trace_spec
from ehc_sn.modules.autoencoder import Autoencoder, AutoencoderSettings
from ehc_sn.modules.hpc import HPCModel, HPCSettings, HPCState, MemoryState
from ehc_sn.modules.lec import LECModel, LECSettings, LECState
from ehc_sn.modules.mec import MECModel, MECSettings, MECState
from ehc_sn.modules.projection import ProjectionModule, ProjectionSettings
from ehc_sn.rollouts.collect import TraceCollector
from ehc_sn.rollouts.trace_tree import TraceTree
from ehc_sn.training.buffers import FifoBuffer
from ehc_sn.training.optim import Adam, AdamConfig
from ehc_sn.training.partial_reset import PartialResetBatchAssembler
from ehc_sn.training.schedules import CosineAnnealingLRWithWarmup, SchedulerConfig, SequentialLR
from ehc_sn.training.step_loop import StepContext, StepLoop
from ehc_sn.types import Device, Dtype
from ehc_sn.utils import trunc_normal_init_
from ehc_sn.utils.detach import DetachMixin

# Community-standard map-style batch: plain dict returned by MazeDataset / DataLoader.
Batch: TypeAlias = Dict[str, Tensor]


# =================================================================================================
class ModelSettings_V1(BaseModel):
    """ """

    model_config = ConfigDict(extra="forbid", strict=False, arbitrary_types_allowed=True)

    hpc: HPCSettings = Field(
        ...,
        description="Settings for the HPC module, including Hebbian memory parameters.",
    )
    lec: LECSettings = Field(
        ...,
        description="Settings for the LEC module, including feature filtering parameters.",
    )
    mec: MECSettings = Field(
        ...,
        description="Settings for the MEC module, including path integration and correction parameters.",
    )

    @field_validator("hpc", "lec", "mec", mode="after")
    def validate_shapes(cls, v):
        """ """
        raise NotImplementedError("Module-specific validation not implemented yet.")

    vocab_size: int = Field(
        ...,
        ge=1,
        description="Number of discrete observation dimensions (input to autoencoder).",
    )

    autoencoder: AutoencoderSettings = Field(
        ...,
        description="Settings for the autoencoder module used for observation compression.",
    )

    lec_projection: ProjectionSettings = Field(
        ...,
        description=(
            "Settings for the projection module between LEC and HPC. "
            "This module projects LEC features into the format expected by HPC memory."
        ),
    )

    mec_projection: ProjectionSettings = Field(
        ...,
        description=(
            "Settings for the projection module between MEC and HPC. "
            "This module projects MEC abstract location codes into the format expected by HPC memory."
        ),
    )


# =================================================================================================
class ModelConfig_TEM_V1(BaseModel, extra="forbid"):
    """ """

    # ~~ Model architecture ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    model: ModelSettings_V1 = Field(
        ...,
        description="",
    )
    environment: EnvironmentConfig = Field(
        ...,
        description="",
    )
    controller: TEMControllerConfig = Field(
        ...,
        description="",
    )
    loss: TEMLossConfig = Field(
        ...,
        description="",
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

    # ~~ Extra ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    global_batch_size: int = Field(
        ...,
        description="",
    )  # TODO: consider moving to BufferSettings or similar


# =================================================================================================
@dataclass
class TEMState(DetachMixin):
    """Container for the full recurrent TEM state."""

    lec: LECState
    mec: MECState
    hpc: HPCState


# =================================================================================================
class TEMModelV1(nn.Module):
    """ """

    def __init__(  # ------------------------------------------------------------------------------
        self, config: ModelSettings_V1, *,
        device: Optional[Device] = None, dtype: Optional[Dtype] = None,
    ) -> None:  # fmt: skip
        """ """
        super().__init__()
        self._config = config

        # Autoencoder module for observation compression/decoding
        self.autoencoder = Autoencoder(config.autoencoder, device=device, dtype=dtype)

        # Entorhinal Hippocampal Circuit components
        self.lec = LECModel(config.lec, device=device, dtype=dtype)
        self.mec = MECModel(config.mec, device=device, dtype=dtype)
        self.hpc = HPCModel(config.hpc, device=device, dtype=dtype)

        # Projection modules
        self.lec_projection = ProjectionModule(config.lec_projection)
        self.mec_projection = ProjectionModule(config.mec_projection)

    @property
    def config(self) -> ModelSettings_V1:
        """ """
        return self._config

    def init_state(  # ----------------------------------------------------------------------------
        self, batch_size: int, *,
        device: Optional[Device] = None, dtype: Optional[Dtype] = None,
    ) -> TEMState:  # fmt: skip
        """ """
        lec_state = self.lec.init_state(batch_size, device)
        state_mec = self.mec.init_state(batch_size, device)
        hpc_state = self.hpc.init_state(batch_size, device, memory=memory)
        return TEMState(lec_state, state_mec, hpc_state)

    def reset_state(  # ---------------------------------------------------------------------------
        self, reset_flag: Tensor, state: TEMState,
    ) -> TEMState:  # fmt: skip
        """Reset halted rows to fresh recurrent state while preserving HPC memory."""
        if not torch.any(reset_flag):
            return state

        init_state = self.init_state(int(reset_flag.shape[0]), device=reset_flag.device)
        lec_state = LECState(
            cells=self._masked_blocks(state.lec.cells, init_state.lec.cells, reset_flag),
            filtered=self._masked_blocks(state.lec.filtered, init_state.lec.filtered, reset_flag),
        )
        mec_state = state.mec.new(
            cells=self._masked_blocks(state.mec.cells, init_state.mec.cells, reset_flag),
            uncertainty=self._masked_optional_blocks(
                state.mec.uncertainty, init_state.mec.uncertainty, reset_flag
            ),
        )
        hpc_state = state.hpc.new(
            cells=self._masked_blocks(state.hpc.cells, init_state.hpc.cells, reset_flag),
            uncertainty=self._masked_optional_blocks(
                state.hpc.uncertainty, init_state.hpc.uncertainty, reset_flag
            ),
        )
        return TEMState(lec_state, mec_state, hpc_state)

    def set_runtime(  # ---------------------------------------------------------------------------
        self, eta: float, hebbian_decay: float, p2g_uncertainty_offset: float,
    ) -> None:  # fmt: skip
        """ """
        self.mec.set_runtime(p2g_uncertainty_offset=p2g_uncertainty_offset)
        self.hpc.set_runtime(eta=eta, hebbian_decay=hebbian_decay)

    def forward(  # -------------------------------------------------------------------------------
        self, inputs: Batch, state: Optional[TEMState] = None,
    ) -> tuple[TEMState, TEMOutput]:  # fmt: skip
        """ """
        # state = self.reset_state(state, a_prev, observation.device)  # FIXME: this is controller logic
        features = self.autoencoder.encode(observation)  # Encode observation to compressed format

        # Observe / infer: LEC filtering + HPC retrieval + MEC correction
        x_inf, state.lec = self.lec.inference(features, state.lec)
        x_ = self.lec_projection(x_inf)  # Project to memory format
        p_xi = self.hpc.recall(x_, state.hpc, mode="full") if self.config.use_x_cued_recall else None

        # LocationBelief: MEC path integration (action-driven)
        g_gen, state.mec = self.mec.generative(action, locations, state.mec)
        g_ = self.mec_projection(g_gen)
        p_gg = self.hpc.recall(g_, state.hpc, mode="hierarchical")

        # Infer abstract location by using state and sensory experience
        g_inf, state.mec = self.mec.inference(p_xi, locations=locations, state=state.mec)
        g_ = self.mec_projection(g_inf)
        p_gi = self.hpc.recall(g_, state.hpc, mode="hierarchical")

        # Generate grounded location from inferred abstract location
        p_gen_gi, state.hpc = self.hpc.generative(p_gi, state.hpc)
        p_gen_gg, state.hpc = self.hpc.generative(p_gg, state.hpc)

        # Infer grounded location from abstract location and sensory experience
        p_inf, state.hpc = self.hpc.inference(x_, g_, state.hpc)

        # Update memory and return new state
        state.hpc = self.hpc.update(p_inf, p_gen_gi, p_xi, state.hpc)

        # Generate observation prediction from inferred grounded location
        x = self.lec_projection.inverse(p_inf)
        c_p_inf = self.lec.generative(x)
        logits_inference = self.autoencoder.decode(c_p_inf)

        # Generate observation from inferred grounded location
        x = self.lec_projection.inverse(p_gen_gi)
        c_p_gen_gi = self.lec.generative(x)
        logits_retrieved = self.autoencoder.decode(c_p_gen_gi)

        # Generate observation from generated grounded location
        x = self.lec_projection.inverse(p_gen_gg)
        c_p_gen_gg = self.lec.generative(x)
        logits_ancestral = self.autoencoder.decode(c_p_gen_gg)

        # Build full output, state and return
        new_state = state
        obs_logits = (logits_inference, logits_retrieved, logits_ancestral)
        grid = (grid_post, grid_prior)
        place = (place_post, place_prior, place_sensory)
        return new_state, obs_logits, None, grid, place


# =================================================================================================
class TrainingModel(L.LightningModule):
    """ """

    def __init__(  # ------------------------------------------------------------------------------
        self, config: ModelConfig_TEM_V1,
    ) -> None:  # fmt: skip
        """ """
        super().__init__()
        self.model = TEMModelV1(config.model)
        self.environment: Environment | None = None  #  Lazy init in setup() to avoid GPU alloc issues
        self.controller: TEMController | None = None  #  Lazy init in setup() to avoid GPU alloc issues
        self.step_module: TEMLossHead | None = None  #  Lazy init in setup() to avoid GPU alloc issues
        self._config = config

        # Manual optimization: explicit backward + opt step (legacy parity + dual-opt clarity).
        self.automatic_optimization = False
        self._train_carry = None

        # Metrics are cloned for train/val to allow separate logging and state management.
        self.train_metrics = build_train_metrics(TEM_ROUTES).clone(prefix="train/")
        self.val_metrics = build_val_metrics(TEM_ROUTES).clone(prefix="val/")
        self.trace_specs = build_trace_spec("this_prefix_name_to_define")

        # Buffer + assembler implement partial-reset batching for ACT runs.
        self._train_buffer = FifoBuffer(
            capacity_rows=4 * config.global_batch_size,  # or local batch size if you prefer
            keys=("inputs", "labels"),
            pin_memory=True,
        )
        self._train_batch_assembler = PartialResetBatchAssembler(
            buffer=self._train_buffer,
            keys=("inputs", "labels"),
        )

    @property
    def config(self) -> ModelConfig_TEM_V1:
        """Return the parsed configuration used by this LightningModule."""
        return self._config

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
    ) -> Tuple[List[Optimizer], List[SequentialLR]]:  # fmt: skip
        """ """
        total_steps = int(self.trainer.estimated_stepping_batches)

        # Optimizer for the main model parameters
        sup_params = [p for p in self.model.parameters() if p.requires_grad]
        opt_sup = Adam(sup_params, self.config.optimizer)
        sch_sup = ExponentialLR(opt_sup, total_steps, self.config.scheduler)

        return [opt_sup], [sch_sup]

    def _compute_schedule(  # ---------------------------------------------------------------------
        self, iteration: int,
    ) -> tuple[float, float, float, float]:  # fmt: skip
        # FIXME: this needs to be integrated with the LR scheduler and ideally moved to a separate ScheduleManager
        # class; it's currently a mess of hardcoded heuristics and hyperparameters scattered across the
        # codebase, but it needs to be computed at each step to update the model runtime parameters
        # (eta, hebbian decay, p2g offset, walk length center).

        walk = self.trainer_settings.walk
        hebbian = self.trainer_settings.scheduler.memory
        p2g = self.trainer_settings.scheduler.uncertainty

        # Hebbian memory parameters
        eta = min((iteration + 1) / hebbian.eta_it, 1) * hebbian.eta
        lamb = min((iteration + 1) / hebbian.lambda_it, 1) * hebbian.hebbian_decay

        # p->g uncertainty offset schedule (eta-style: schedule outputs the final runtime value)
        p2g_scale = 1 / (1 + np.exp((iteration - p2g.p2g_sig_half_it) / p2g.p2g_sig_scale_it))
        p2g_uncertainty_offset = p2g.offset_min + (p2g.offset_max - p2g.offset_min) * p2g_scale

        # Walk length center (annealing from max to min over training)
        max_steps = max(int(self.trainer_settings.max_steps), 1)
        walk_length_center = (
            walk.walk_it_max
            - walk.walk_it_window * 0.5
            - min((iteration + 1) / max_steps, 1)
            * (walk.walk_it_max - walk.walk_it_min - walk.walk_it_window)
        )

        return eta, lamb, p2g_uncertainty_offset, walk_length_center

    # -- Lifecycle --------------------------------------------------------------------------------

    def on_train_epoch_start(  # ------------------------------------------------------------------
        self,
    ) -> None:  # fmt: skip
        """ """
        self._train_carry = None
        self._train_buffer.clear()
        self.train_metrics.reset()

    def on_validation_epoch_start(  # ------------------------------------------------------------
        self,
    ) -> None:  # fmt: skip
        """ """
        self.val_metrics.reset()

    # -- Training ----------------------------------------------------------------------------------

    def training_step(  # -------------------------------------------------------------------------
        self, batch: Batch, batch_idx: int,
    ) -> Tensor:  # fmt: skip
        """ """
        # Initialize carry/state on the first batch
        if self._train_carry is None:
            self._train_carry = self.step_module.initial_carry(batch)

        # Assemble partial-reset step batch
        step_batch = self._train_batch_assembler.make_step_batch(
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
        local_bs = int(batch["inputs"].shape[0])
        loss = _normalize_loss_for_backward(step.outputs.loss, local_bs=local_bs)
        self.manual_backward(loss)

        optimizers = self.optimizers()
        for opt in optimizers if isinstance(optimizers, list) else [optimizers]:
            opt.step()  # type: ignore
            opt.zero_grad(set_to_none=True)

        scheduler = self.lr_schedulers()
        for sch in scheduler if isinstance(scheduler, list) else [scheduler]:
            sch.step()  # type: ignore

        # Update metrics with unnormalized loss and log to TensorBoard.
        update_metrics_from_step(self.train_metrics, step.outputs.metrics, TEM_ROUTES)
        self.log("train/loss", loss.detach(), on_step=True, on_epoch=False, prog_bar=True, logger=True)

        return {"loss": loss.detach(), "signals": step.outputs.signals}

    def validation_step(  # -----------------------------------------------------------------------
        self, batch: Batch, batch_idx: int,
    ) -> Dict[str, object]:  # fmt: skip
        """Run a full ACT rollout so halted-only metrics are meaningful.

        Validation uses `EvaluationLoop` (no carry is persisted across batches here) and logs
        normalized metrics.
        """
        step_batches = repeat(batch)  # Run until all examples halt
        step_options = {}  # FIXME after the HeadLoss and TEMController support options
        carry0 = self.step_module.initial_carry(batch)

        # Initialize carry/state on the first batch
        step, collector = None, TraceCollector(TraceTree(), self.trace_specs)
        for t, step in StepLoop(self.step_module, step_batches, carry0, options=step_options):
            collector.append(t, step)
        if step is None:
            raise ValueError("Evaluation loop did not yield any steps, cannot log metrics.")

        # Update metrics with the final step's metrics and log to TensorBoard.
        update_metrics_from_step(self.val_metrics, step.outputs.metrics, TEM_ROUTES)

        return {"trace": collector.tree}


# =================================================================================================
# TODO: Helpers here
