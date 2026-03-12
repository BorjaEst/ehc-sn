""" """

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import repeat, tee
from typing import Any, Dict, Literal, Optional, TypeAlias

import lightning as L
import numpy as np
import torch
import torch.nn as nn
from pydantic import BaseModel, Field, computed_field, model_validator
from torch import Tensor
from torch.distributions import Normal
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ExponentialLR

from ehc_sn import utils
from ehc_sn.controllers.tem import TEMController, TEMControllerConfig, TEMOutput
from ehc_sn.envs.dungeon_walk import DungeonWalk as Environment
from ehc_sn.envs.dungeon_walk import EnvConfig as EnvironmentConfig
from ehc_sn.heads.tem import TEMLossConfig, TEMLossHead
from ehc_sn.metrics import build_train_metrics, build_val_metrics, update_metrics_from_step
from ehc_sn.metrics.routes import TEM_EPISODE_ROUTES, TEM_STEP_ROUTES
from ehc_sn.metrics.traces import build_trace_spec
from ehc_sn.modules.autoencoder import Autoencoder, AutoencoderSettings
from ehc_sn.modules.hpc import HPCModel, HPCSettings, HPCState, MemoryState
from ehc_sn.modules.lec import LECModel, LECSettings, LECState
from ehc_sn.modules.mec import MECModel, MECSettings, MECState, resolve_mec_shape
from ehc_sn.modules.projection import ProjectionModule, ProjectionSettings
from ehc_sn.rollouts.collect import TraceCollector
from ehc_sn.rollouts.trace_tree import TraceTree
from ehc_sn.training.buffers import FifoBuffer
from ehc_sn.training.optim import Adam, AdamConfig
from ehc_sn.training.partial_reset import PartialResetBatchAssembler
from ehc_sn.training.schedules import SchedulerConfig, SequentialLR
from ehc_sn.training.step_loop import StepContext, StepLoop
from ehc_sn.types import Device, Dtype
from ehc_sn.utils.detach import DetachMixin

# Community-standard map-style batch: plain dict returned by MazeDataset / DataLoader.
Batch: TypeAlias = Dict[str, Tensor]
ObsLogits = tuple[Tensor, Tensor, Tensor]  # (inference, retrieved, ancestral)
GridCodes = tuple[Tensor, Tensor]  # (posterior, prior)
PlaceCodes = tuple[Tensor, Tensor, Optional[Tensor]]  # (posterior, prior, sensory-cued retrieval)


# =================================================================================================
class ModelSettings_V1(BaseModel, extra="forbid", strict=False):
    """Canonical TEM v1 model settings.

    This compact schema keeps the environment-facing observation/action
    contract, shared multiscale frequencies, and component-local settings in a
    single model config without reintroducing legacy aliases.
    """

    observation_dim: int = Field(
        ...,
        ge=1,
        description="Dimensionality of raw observations from the environment.",
    )
    action_count: int = Field(
        ...,
        ge=1,
        description="Number of discrete actions in the environment.",
    )

    f_initial: list[float] = Field(
        default_factory=lambda: [0.99, 0.3, 0.09, 0.5, 0.4],
        min_length=1,
        description=(
            "List of initial feature frequencies for the model's resolved modules. "
            "Its length must match the full MEC/HPC frequency count after OVC mode resolution."
        ),
    )
    use_x_cued_recall: bool = Field(
        default=True,
        description="Whether the HPC recall should be cued with LEC features (x) in addition to MEC features (g).",
    )

    @model_validator(mode="after")
    def validate_model(self) -> "ModelSettings_V1":
        self._validate_f_initial()
        self._validate_stage_alignment()
        self._validate_frequency_alignment()
        return self

    def _validate_f_initial(self) -> None:
        if any(not (0.0 < value < 1.0) for value in self.f_initial):
            raise ValueError("f_initial values must be strictly between 0 and 1.")

    def _validate_stage_alignment(self) -> None:
        expected_freq = len(self.mec_shape)
        if expected_freq != self.n_stages:
            raise ValueError("The resolved MEC frequency count must equal len(f_initial).")

    def _validate_frequency_alignment(self) -> None:
        if len(self.hpc.shape) != self.n_total_freq:
            raise ValueError("len(hpc.shape) must equal the derived total MEC frequency count.")

    @computed_field
    @property
    def n_stages(self) -> int:
        """Return the resolved number of TEM frequency modules."""
        return len(self.f_initial)

    @computed_field
    @property
    def n_freq(self) -> int:
        """Return the total number of frequency modules."""
        return len(self.f_initial)

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

    autoencoder: AutoencoderSettings = Field(
        ...,
        description="Settings for the autoencoder module used for observation compression.",
    )

    projection_lec: ProjectionSettings = Field(
        default_factory=lambda: ProjectionSettings(mode="tiling", learnable=False),
        description=(
            "Settings for the projection module between LEC and HPC. "
            "This module projects LEC features into the format expected by HPC memory."
        ),
    )
    projection_mec: ProjectionSettings = Field(
        default_factory=lambda: ProjectionSettings(mode="low_rank", learnable=False, rank=[10, 10, 8, 6, 6]),
        # default_factory=lambda: ProjectionSettings(mode="low_rank", learnable=False),
        description=(
            "Settings for the projection module between MEC and HPC. "
            "This module projects MEC abstract location codes into the format expected by HPC memory."
        ),
    )

    @computed_field
    @property
    def lec_shape(self) -> list[int]:
        """Return the resolved LEC feature shape across all frequencies."""
        return [self.lec.feature_dim] * self.n_total_freq

    @computed_field
    @property
    def mec_ovc_shape(self) -> list[int]:
        """Return the appended OVC shape implied by the configured OVC mode."""
        return list(self.mec.ovc.shape or []) if self.mec.ovc.mode == "separate" else []

    @computed_field
    @property
    def mec_shape(self) -> list[int]:
        """Return the full MEC shape including optional OVC modules."""
        return resolve_mec_shape(self.mec)

    @computed_field
    @property
    def n_total_freq(self) -> int:
        """Return the total number of MEC/HPC frequencies after OVC expansion."""
        return len(self.mec_shape)


# =================================================================================================
class ModelConfig_TEM_V1(BaseModel, extra="forbid"):
    """Top-level TEM v1 training config."""

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

    @model_validator(mode="after")
    def validate_environment_contract(self) -> "ModelConfig_TEM_V1":
        if self.environment.observation_dim != self.model.observation_dim:
            raise ValueError("environment.observation_dim must match model.observation_dim.")
        if self.environment.action_count != self.model.action_count:
            raise ValueError("environment.action_count must match model.action_count.")
        return self


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
        """Construct the TEM backbone from the resolved TEM v1 model settings."""
        super().__init__()
        self._config = config
        n_freq, n_actions = config.n_total_freq, config.action_count
        f_initial = config.f_initial

        # Autoencoder module for observation compression/decoding
        self.autoencoder = Autoencoder(config.observation_dim, config.lec.feature_dim, config.autoencoder)

        # Entorhinal Hippocampal Circuit components
        self.hpc = HPCModel(n_freq, f_initial, config.hpc, device=device, dtype=dtype)
        self.mec = MECModel(n_actions, config.hpc.shape, f_initial, config.mec, device=device, dtype=dtype)
        self.lec = LECModel(f_initial, config.lec, device=device, dtype=dtype)

        # Projection modules
        self.projection_mec = ProjectionModule(self.mec, self.hpc, config.projection_mec)
        self.projection_lec = ProjectionModule(self.lec, self.hpc, config.projection_lec)

    @property
    def config(self) -> ModelSettings_V1:
        """ """
        return self._config

    def init_state(  # ----------------------------------------------------------------------------
        self, batch_size: int, *, memory: Optional[MemoryState] = None,
        device: Optional[Device] = None, dtype: Optional[Dtype] = None,
    ) -> TEMState:  # fmt: skip
        """Create an initial recurrent TEM state."""
        memory = memory if memory is not None else self.hpc.init_memory(batch_size=batch_size, device=device)
        lec_state = self.lec.init_state(batch_size, device=device)
        state_mec = self.mec.init_state(batch_size, device=device)
        hpc_state = self.hpc.init_state(batch_size, device=device, memory=memory)
        return TEMState(lec_state, state_mec, hpc_state)

    def reset_state(  # ---------------------------------------------------------------------------
        self, reset_flag: Tensor, state: TEMState,
    ) -> TEMState:  # fmt: skip
        """Reset flagged rows to a fresh episode state while preserving active rows."""
        reset_flag = reset_flag.to(torch.bool).view(-1)
        if not torch.any(reset_flag):
            return state

        fresh = self.init_state(int(reset_flag.shape[0]), memory=None, device=reset_flag.device)
        return TEMState(
            lec=state.lec.replace_rows(reset_flag, fresh.lec),
            mec=state.mec.replace_rows(reset_flag, fresh.mec),
            hpc=state.hpc.replace_rows(reset_flag, fresh.hpc),
        )

    def set_runtime(  # ---------------------------------------------------------------------------
        self, eta: float, hebbian_decay: float, p2g_uncertainty_offset: float,
    ) -> None:  # fmt: skip
        """ """
        self.mec.set_runtime(p2g_uncertainty_offset=p2g_uncertainty_offset)
        self.hpc.set_runtime(eta=eta, hebbian_decay=hebbian_decay)

    def forward(  # -------------------------------------------------------------------------------
        self, inputs: Batch, state: Optional[TEMState] = None,
    ) -> tuple[TEMState, ObsLogits, Any, GridCodes, PlaceCodes]:  # fmt: skip
        """Run one TEM step from the current payload and recurrent state.

        ``state`` is assumed to have already been reset for any fresh episode
        rows by the caller. When ``state`` is ``None``, a fresh full-batch state
        is allocated and the normal single-step TEM transition is executed.
        """
        obs_inputs = inputs["inputs"]
        previous_action = inputs["previous_action"]
        landmark_id = inputs.get("landmark_id")
        obs_embedding = self.autoencoder.encode(obs_inputs)

        if state is None:
            state = self.init_state(int(obs_inputs.shape[0]), memory=None, device=obs_inputs.device)

        # Sensory inference: encode observations into LEC features and query place memory from them.
        lec_features_post, state.lec = self.lec.inference(obs_embedding, state.lec)
        place_query_from_obs = self.projection_lec(lec_features_post)
        place_sensory = self.hpc.recall(place_query_from_obs, state.hpc, mode="full") if self.config.use_x_cued_recall else None  # fmt: skip

        # Grid transition prior from action-driven path integration.
        grid_prior, state.mec = self.mec.generative(previous_action, landmark_id, state.mec)
        place_query_from_grid_prior = self.projection_mec(grid_prior)
        place_recall_from_grid_prior = self.hpc.recall(place_query_from_grid_prior, state.hpc, mode="hierarchical")  # fmt: skip

        # Grid posterior after correcting the prior with recalled place evidence.
        grid_post, state.mec = self.mec.inference(place_sensory, landmark_id=landmark_id, state=state.mec)  # fmt: skip
        place_query_from_grid_post = self.projection_mec(grid_post)
        place_recall_from_grid_post = self.hpc.recall(place_query_from_grid_post, state.hpc, mode="hierarchical")  # fmt: skip

        # Place prior and posterior terms used by the TEM variational losses.
        place_retrieved, state.hpc = self.hpc.generative(place_recall_from_grid_post, state.hpc)
        place_prior, state.hpc = self.hpc.generative(place_recall_from_grid_prior, state.hpc)
        place_post, state.hpc = self.hpc.inference(place_query_from_obs, place_query_from_grid_post, state.hpc)  # fmt: skip

        # Hebbian update uses the posterior place code and the sensory-cued retrieval when available.
        state.hpc = self.hpc.update(place_post, place_retrieved, place_sensory, state.hpc)

        # Decode observation logits for the three TEM pathways.
        lec_features_from_place_post = self.projection_lec.inverse(place_post)
        obs_features_inference = self.lec.generative(lec_features_from_place_post)
        logits_inference = self.autoencoder.decode(obs_features_inference)

        lec_features_from_place_retrieved = self.projection_lec.inverse(place_retrieved)
        obs_features_retrieved = self.lec.generative(lec_features_from_place_retrieved)
        logits_retrieved = self.autoencoder.decode(obs_features_retrieved)

        lec_features_from_place_prior = self.projection_lec.inverse(place_prior)
        obs_features_ancestral = self.lec.generative(lec_features_from_place_prior)
        logits_ancestral = self.autoencoder.decode(obs_features_ancestral)

        # Return controller-compatible rollout outputs for the TEM loss head.
        obs_logits = (logits_inference, logits_retrieved, logits_ancestral)
        grid = (grid_post, grid_prior)
        place = (place_post, place_prior, place_sensory)
        return state, obs_logits, None, grid, place  # Action=None as TEM provides no direct action outputs


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
        self.train_metrics = build_train_metrics(TEM_STEP_ROUTES).clone(prefix="train/")
        self.val_metrics = build_val_metrics(TEM_EPISODE_ROUTES).clone(prefix="val/")
        self.trace_specs = build_trace_spec("tem")

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
    ) -> tuple[list[Optimizer], list[SequentialLR]]:  # fmt: skip
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
        update_metrics_from_step(self.val_metrics, step.outputs.metrics, TEM_EPISODE_ROUTES)

        return {"trace": collector.tree}


# =================================================================================================
def _normalize_loss_for_backward(  # --------------------------------------------------------------
    total_loss: Tensor, local_bs: int,
) -> Tensor:  # fmt: skip
    """Normalize the total loss by the local batch size for distributed training."""
    if local_bs <= 0:
        raise ValueError(f"local_bs must be positive, got {local_bs}.")
    return total_loss / float(local_bs)
