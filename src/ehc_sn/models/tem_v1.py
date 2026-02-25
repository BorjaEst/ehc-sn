from __future__ import annotations

"""Core TEM model composition and rollout streaming.

This module defines the top-level TEM model used throughout the package.
It composes the three main circuit components:

- LEC: sensory feature processing
- MEC: abstract location dynamics (grid / optional OVC)
- HPC: associative memory over grounded location codes

The public entry points are:

- `TEMConfig`: complete configuration tree for model construction.
- `Model`: the TEM model wrapper exposing a TEM-compatible step interface.
- `RolloutStream`: an iterator that streams a `Walk` through `Model`.

Design notes:
        - The model is intentionally stateful but uses explicit `TEMState` objects
            (no hidden module state) to keep training/inference loops easy to reason
            about.
        - `RolloutStream` uses `a_prev=None` as an episode boundary signal; the
            model resets relevant parts of state at those boundaries.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import tee
from typing import List, Literal, Optional, Sequence, Tuple, Union

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor, nn

from ehc_sn import utils
from ehc_sn.modules.autoencoder import AutoencoderModule
from ehc_sn.modules.hpc import HPCModel, HPCState
from ehc_sn.modules.lec import LECModel, LECState
from ehc_sn.modules.mec import MECModel, MECState
from ehc_sn.modules.projection import ProjectionModule
from ehc_sn.types import *


class TEMConfig(BaseModel):
    """Complete settings tree for TEM model configuration."""

    model_config = ConfigDict(extra="forbid", strict=False, arbitrary_types_allowed=True)

    space_contract: SpaceContractSettings = Field(
        default_factory=SpaceContractSettings,
        description="Settings for the space contract used by the model.",
    )
    autoencoder: AutoencoderSettings = Field(
        default_factory=AutoencoderSettings,
        description="Autoencoder module settings.",
    )
    f_initial: List[float] = Field(
        default_factory=lambda: [0.99, 0.3, 0.09, 0.5, 0.4],
        frozen=True,
        description="Initial spatial frequencies for multi-scale modules.",
    )
    n_features: int = Field(
        default=10,
        frozen=True,
        description="Number of LEC context features.",
    )
    lec_settings: LECSettings = Field(
        default_factory=LECSettings,
        description="LEC module settings.",
    )
    lec_projection: LECProjectionSettings = Field(
        default_factory=LECProjectionSettings,
        description="LEC projection module settings.",
    )
    n_grids: List[int] = Field(
        default_factory=lambda: [30, 24, 18],
        frozen=True,
        description="Number of MEC neurons per frequency module.",
    )
    n_ovc: Union[Literal["off", "merged"], List[int]] = Field(
        default=[24, 18],
        frozen=True,
        description="Number of OVC neurons per frequency module. 'merged' to merge with n_grids.",
    )
    mec_settings: MECSettings = Field(
        default_factory=MECSettings,
        description="MEC module settings.",
    )
    mec_projection: MECProjectionSettings = Field(
        default_factory=MECProjectionSettings,
        description="MEC projection module settings.",
    )
    n_hippocampal: List[int] = Field(
        default_factory=lambda: [100, 100, 80, 60, 60],
        frozen=True,
        description="Number of HPC neurons per frequency module.",
    )
    hpc_settings: HPCSettings = Field(
        default_factory=HPCSettings,
        description="HPC module settings.",
    )
    use_x_cued_recall: bool = Field(
        default=True,
        description="Whether to use inferred ground location while inferring new abstract location",
    )


@dataclass
class TEMState:
    """Container for the full recurrent TEM state."""

    lec: LECState
    mec: MECState
    hpc: HPCState

    def detach(self) -> "TEMState":
        """Return a detached copy of the state."""
        states = [x.detach() for x in (self.lec, self.mec, self.hpc)]
        return TEMState(*states)

    def new(self, lec: LECState = None, mec: MECState = None, hpc: HPCState = None) -> "TEMState":
        """Return a new TEMState with updated fields."""
        return TEMState(
            lec=lec if lec is not None else self.lec.new(),  # Clear filtered features
            mec=mec if mec is not None else self.mec.new(),  # Clear abstract location
            hpc=hpc if hpc is not None else self.hpc.new(),  # Clear grounded location, transfers memory
        )


@dataclass
class TEMInference:
    """Outputs from the inference path (conditioned on observation)."""

    g_inf: AbstractLocation
    p_inf: GroundedLocation
    p_xi: GroundedLocation


@dataclass
class TEMGenerative:
    """Outputs from the generative path (conditioned on action/state)."""

    g_gen: AbstractLocation
    p_gen_gg: GroundedLocation
    p_gen_gi: GroundedLocation


@dataclass
class TEMReconstruction:
    """Reconstructed observations and logits."""

    y_p_inf: Prediction
    y_gen_gi: Prediction
    y_gen_gg: Prediction


@dataclass
class TEMOutput:
    """Top-level model output for a single step."""

    inference: TEMInference
    generative: TEMGenerative
    features: torch.Tensor
    reconstruction: TEMReconstruction


class Model(nn.Module):
    """Top-level TEM model.

    The model composes:
        - `AutoencoderModule` for observation encode/decode.
        - `LECModel` for sensory feature filtering.
        - `MECModel` for abstract location dynamics.
        - `HPCModel` for grounded location inference, recall, and Hebbian memory.
        - `ProjectionModule`s to map between circuit spaces and memory space.

    The main interaction method is `forward`, which consumes a single timestep
    (observation + metadata + previous action + previous state) and returns the
    model output and next state.
    """

    def __init__(self, config: Optional[TEMConfig] = None):
        """Initialize the TEM model.

        Args:
            config: Optional configuration tree. When `None`, defaults are used.
        """
        super().__init__()
        self._config = config or TEMConfig()
        n_observations = config.space_contract.n_observations  # Number of observation dimensions
        n_actions = config.space_contract.n_actions_move  # Number of possible discrete actions
        n_features = config.n_features  # Number of LEC features (compressed observation)
        n_grids = config.n_grids  # Number of MEC grid cells per frequency
        n_ovc = config.n_ovc if config.n_ovc != "off" else []  # Number of MEC OVC cells per frequency
        n_ovc = config.n_ovc if config.n_ovc != "merged" else None  # Merge OVC with grid cells
        n_hippocampal = config.n_hippocampal  # Number of HPC place cells per frequency
        f_initial = config.f_initial  # Initial firing rate for all cells

        # Autoencoder module for observation compression/decoding
        self.autoencoder = AutoencoderModule(n_observations, n_features, settings=config.autoencoder)

        # Entorhinal Hippocampal Circuit components
        self.lec = LECModel(n_features, f_initial, settings=config.lec_settings)
        self.mec = MECModel(n_actions, n_hippocampal, n_grids, n_ovc, f_initial, settings=config.mec_settings)
        self.hpc = HPCModel(len(n_grids), n_hippocampal, f_initial, settings=config.hpc_settings)

        # Projection modules
        self.lec_projection = ProjectionModule(self.lec, self.hpc, settings=config.lec_projection)
        self.mec_projection = ProjectionModule(self.mec, self.hpc, settings=config.mec_projection)

    def init_state(
        self, batch_size: int, device: Optional[torch.device] = None, memory: Optional[MemoryState] = None
    ) -> TEMState:
        """Create an initial TEM state.

        Args:
            batch_size: Batch size for all state tensors.
            device: Optional device for the returned tensors.

        Returns:
            An initialized `TEMState`.
        """
        lec_state = self.lec.init_state(batch_size, device)
        state_mec = self.mec.init_state(batch_size, device)
        hpc_state = self.hpc.init_state(batch_size, device, memory=memory)
        return TEMState(lec_state, state_mec, hpc_state)

    def set_runtime(self, eta: float, hebbian_decay: float, p2g_uncertainty_offset: float) -> None:
        """Set runtime hyperparameters (called by training loop each step).

        Args:
            eta: Hebbian learning rate (rate of remembering)
            hebbian_decay: Hebbian decay factor (rate of forgetting)
            p2g_uncertainty_offset: Additive uncertainty offset for p->g inference
        """
        self.mec.set_runtime(p2g_uncertainty_offset=p2g_uncertainty_offset)
        self.hpc.set_runtime(eta=eta, hebbian_decay=hebbian_decay)

    @property
    def config(self) -> TEMConfig:
        """Return TEM settings object constructed from model parameters."""
        return self._config

    @property
    def n_observations(self) -> int:
        """Return the number of observation dimensions."""
        return self.autoencoder.n_observations

    @property
    def n_features(self) -> int:
        """Return the number of LEC features (compressed observation)."""
        return self.autoencoder.n_features

    @property
    def n_actions(self) -> int:
        """Return the number of possible discrete actions."""
        return self.mec.n_actions

    @property
    def n_grids(self) -> List[int]:
        """Return the number of MEC grid cells per frequency."""
        return self.mec.n_grids

    @property
    def n_ovc(self) -> Optional[List[int]]:
        """Return the number of MEC OVC cells per frequency (or None)."""
        return self.mec.n_ovc

    @property
    def n_hippocampal(self) -> List[int]:
        """Return the number of HPC place cells per frequency."""
        return self.hpc.shape

    def forward(
        self, observation: Observation, locations: List[LocationLabel], a_prev: Action, state: TEMState
    ) -> tuple[TEMOutput, TEMState]:
        """Run one TEM step.

        Args:
            observation: Sensory observation tensor for the current timestep.
            locations: Per-environment location metadata/labels.
            a_prev: Previous actions for each environment in the batch. `None`
                indicates an episode boundary/reset for that environment.
            state: Previous TEM state.

        Returns:
            A tuple `(output, next_state)`.
        """
        state = self.setup_state(state, a_prev, observation.device)
        device = observation.device  # Get device from observation tensor
        actions = utils.one_hot_with_zero(a_prev, self.n_actions, device=device)

        features = self.autoencoder.encode(observation)  # Encode observation to compressed format
        inference, generative, state = self.step(features, locations, actions, state)
        output = self.compute_output(features, inference, generative)

        # Build full output, state and return
        return output, state

    def setup_state(self, state: TEMState, a_prev: Action, device: torch.device) -> TEMState:
        """Apply per-environment reset logic before transition.

        The batch may contain environments at different episode boundaries.
        When `a_prev[i] is None`, we treat that environment as starting a new
        episode and reset the relevant part of state before applying the MEC
        transition.

        Currently this resets the MEC abstract location belief to its learned
        prior (`mec.cells_init`). Other state is left unchanged.

        Args:
            state: Current TEM state.
            a_prev: Previous action per environment. `None` indicates reset.
            device: Device to allocate masks/tensors.

        Returns:
            Updated state with resets applied where needed.
        """
        state_lec, state_mec, state_hpc = state.lec, state.mec, state.hpc
        reset_mask = torch.tensor([a is None for a in a_prev], dtype=torch.bool, device=device)
        if torch.any(reset_mask):
            # Reset g to priors for envs with no previous action
            g_reset = [
                torch.where(reset_mask.unsqueeze(-1), self.mec.cells_init[f].unsqueeze(0), state.mec.cells[f])
                for f in range(self.mec.n_freq)
            ]
            state_mec = state.mec.new(g_reset, uncertainty=None)
        return TEMState(state_lec, state_mec, state_hpc)

    def step(
        self, features: MultiScaleCode, locations: List[LocationLabel], actions: Tensor, state: TEMState
    ) -> tuple[TEMInference, TEMGenerative, TEMState]:
        """Execute the TEM circuit dynamics for one timestep.

        This method performs the core algorithmic step:
            1) Infer sensory features with LEC.
            2) Optionally recall a grounded code from sensory cue (x-cued recall).
            3) Path integrate abstract location with MEC (generative branch).
            4) Recall grounded code from abstract code via HPC.
            5) Infer abstract location using memory correction.
            6) Infer grounded location using sensory and abstract cues.
            7) Update Hebbian memory.

        Args:
            features: Compressed observation features.
            locations: Per-environment metadata for the current timestep.
            actions: One-hot previous action tensor of shape `(B, n_actions)`.
            state: Previous TEM state.

        Returns:
            A tuple `(inference, generative, next_state)`.
        """
        # Observe / infer: LEC filtering + HPC retrieval + MEC correction
        x_inf, state.lec = self.lec.inference(features, state.lec)
        x_ = self.lec_projection(x_inf)  # Project to memory format
        p_xi = self.hpc.recall(x_, state.hpc, mode="full") if self.config.use_x_cued_recall else None

        # LocationBelief: MEC path integration (action-driven)
        g_gen, state.mec = self.mec.generative(actions, locations, state.mec)  # Updates mec state with g_path
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

        # Build tem generative and inference interfaces
        generative = TEMGenerative(g_gen=g_gen, p_gen_gg=p_gen_gg, p_gen_gi=p_gen_gi)
        inference = TEMInference(g_inf=g_inf, p_inf=p_inf, p_xi=p_xi)

        # Update memory and return new state
        state.hpc = self.hpc.update(p_inf, p_gen_gi, p_xi, state.hpc)
        return inference, generative, TEMState(state.lec, state.mec, state.hpc)

    def compute_output(
        self, features: Tensor, inference: TEMInference, generative: TEMGenerative
    ) -> TEMOutput:
        """Decode observation predictions from inferred/generated grounded codes.

        Args:
            features: LEC features for the current timestep.
            inference: Inference-branch outputs.
            generative: Generative-branch outputs.

        Returns:
            A `TEMOutput` containing reconstructions and intermediate codes.
        """
        # Generate observation prediction from inferred grounded location
        x = self.lec_projection.inverse(inference.p_inf)
        c_p_inf = self.lec.generative(x)
        o_p_inf_logits = self.autoencoder.decode(c_p_inf)
        y_p_inf = Prediction(utils.softmax(o_p_inf_logits), logits=o_p_inf_logits)

        # Generate observation from inferred grounded location
        x = self.lec_projection.inverse(generative.p_gen_gi)
        c_p_gen_gi = self.lec.generative(x)
        o_gen_gi_logits = self.autoencoder.decode(c_p_gen_gi)
        y_gen_gi = Prediction(utils.softmax(o_gen_gi_logits), logits=o_gen_gi_logits)

        # Generate observation from generated grounded location
        x = self.lec_projection.inverse(generative.p_gen_gg)
        c_p_gen_gg = self.lec.generative(x)
        x_gen_gg_logits = self.autoencoder.decode(c_p_gen_gg)
        y_gen_gg = Prediction(utils.softmax(x_gen_gg_logits), logits=x_gen_gg_logits)

        # Return all generated observations and their corresponding logits
        reconstructions = TEMReconstruction(y_p_inf, y_gen_gi, y_gen_gg)
        return TEMOutput(inference, generative, features, reconstructions)


@dataclass
class RolloutStep:
    world_step: WorldStep
    output: TEMOutput
    state: TEMState


class RolloutStream(Iterator[RolloutStep]):
    def __init__(self, model: Model, walk: Sequence[WalkBatch], initial: Optional[TEMState] = None):
        self.model = model  # TEM model to rollout
        walk_iter = iter(walk)  # Walk to iterator and peek to get batch size/device

        # Use tee to peek without consuming
        peek_iter, self.walk = tee(walk_iter, 2)
        _, first_observation, _ = next(peek_iter)  # Peek at first observation
        batch_size, device = first_observation.shape[0], first_observation.device

        # Initialize state and previous actions
        self._state = initial or model.init_state(batch_size, device)
        self._a_prev = [None for _ in range(first_observation.shape[0])]

    def __iter__(self) -> "RolloutStream":
        return self

    @property
    def state(self) -> TEMState:
        return self._state

    @property
    def previous_action(self) -> List[Optional[int]]:
        return self._a_prev

    def __next__(self) -> RolloutStep:
        locations, observation, action = next(self.walk)
        output, self._state = self.model(observation, locations, self.previous_action, self._state)
        self._a_prev = action  # Update action for next iteration
        world_step = WorldStep(locations=locations, observation=observation, action=action)
        return RolloutStep(world_step, output, self._state)


"""PyTorch Lightning training infrastructure for TEM.

Provides :class:`TrainingLoop`, a Lightning wrapper that implements the
complete TEM training loop with visit-masked loss accumulation and curriculum
scheduling.

Key Features:
    Streaming rollouts:
        Losses are accumulated incrementally during iteration (memory-efficient,
        no intermediate state storage).

    Visit masking:
        Only revisits contribute to optimization. First visits update the visited
        mask but are excluded from loss/accuracy computation.

    Stateful batches:
        Final state from each training batch is detached and reused as the
        initial state for the next batch (maintains RNN continuity).

    Curriculum scheduling:
        Hebbian parameters and walk length are scheduled dynamically during training.
        Learning rate scheduling is handled via a Lightning lr_scheduler.

Architecture Note
-----------------
Settings composition:
    - TrainerConfig composes low-level '*Settings' from settings.py
    - Prevents duplication of parameters like walk curriculum bounds
    - Instantiated in run.py from individual settings components

See Also:
    :class:`ehc_sn.losses.TEMLoss`: Loss computation
    :class:`ehc_sn.model.Model`: Core TEM model
    :class:`ehc_sn.model.RolloutStream`: Streaming rollout iterator
"""

from __future__ import annotations

from typing import Any, Optional

import lightning.pytorch as pl
import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field
from temp.tem.src.ehc_sn.tem_v1 import Model, RolloutStep, RolloutStream, TEMState
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from ehc_sn import losses, metrics, settings
from ehc_sn.losses import AccumLoss, LossG, LossOutput, LossP, LossReg, LossX, StepLoss
from ehc_sn.metrics import AccuracyO


class TrainerConfig(BaseModel):
    """Composite configuration for TEM training (Lightning trainer + schedules).

    This Config class composes low-level '*Settings' from settings.py to provide
    complete configuration for TrainingLoop and Lightning Trainer. It aggregates
    Lightning infrastructure settings with runtime schedules and PyTorch/Lightning
    optimization components (optimizer + lr_scheduler).

    Architecture:
        - Composes settings.LossSettings, settings.OptimizerSettings, and settings.SchedulerSettings.
        - Used by TrainingLoop
        - Instantiated from RunArguments in run.py (prevents parameter duplication)

    Note:
        Walk curriculum settings (walk) are shared with DataConfig as a single
        runtime curriculum: the trainer schedules the current walk length, and the
        data pipeline consumes it when generating walks.
    """

    model_config = ConfigDict(extra="allow")  # Allow extra Lightning kwargs

    # Core Lightning Trainer kwargs
    max_steps: int = Field(default=20000, description="Maximum training steps.")
    log_every_n_steps: int = Field(default=10, description="Log metrics every N steps.")
    enable_progress_bar: bool = Field(default=True, description="Show progress bar during training.")

    # Loss settings (leaf settings)
    loss: settings.LossSettings = Field(
        default_factory=settings.LossSettings,
        description="Loss settings including weights for each component.",
    )

    # Walk curriculum bounds (referenced from DataConfig for annealing schedule)
    walk: settings.CurriculumSettings = Field(
        default_factory=settings.CurriculumSettings,
        description="Walk length curriculum settings (shared with DataConfig).",
    )

    # PyTorch/Lightning optimization components
    optimizer: settings.OptimizerSettings = Field(
        default_factory=settings.OptimizerSettings,
        description="Optimizer settings (e.g., Adam hyperparameters).",
    )

    # Schedulers (LR scheduler + runtime hyperparameter schedules)
    scheduler: settings.SchedulerSettings = Field(
        default_factory=settings.SchedulerSettings,
        description="Schedulers grouped by what they control (lr/memory/uncertainty).",
    )


class TrainingLoop(pl.LightningModule):
    """PyTorch Lightning module for TEM training.

    Integrates TEM model training with PyTorch Lightning, handling:
        - Visit-masked loss accumulation during streaming rollouts
        - Dynamic hyperparameter scheduling (learning rate, Hebbian parameters)
        - Stateful batch processing for recurrent continuity
        - Curriculum control (walk length annealing)
        - Metric logging and validation

    The module maintains an internal state (``prev_state``) that carries recurrent
    memory across training batches. This state is detached after each step to
    prevent gradient accumulation across batches while preserving memory content.

    Attributes:
        tem: The wrapped TEM model.
        loss_fn: Loss computation module.
        acc_o_fn: Sensory accuracy metric.
        prev_state: Previous batch's final state (detached).
        trainer_settings: Combined trainer and schedule configuration.
    """

    def __init__(self, model: Model, training: TrainerConfig):
        """Initialize the Lightning module.

        Args:
            model: TEM model to wrap.
            training: Trainer settings including schedules for runtime hyperparameters
                and learning rate, plus walk curriculum bounds for annealing.
        """
        super().__init__()
        self.trainer_settings = training

        params = {"trainer": training.model_dump()}
        self.save_hyperparameters(params)

        self.tem: Model = model
        self.prev_state: Optional[TEMState] = None

        self.loss_fn = losses.TEMLoss(training.loss)
        self.acc_o_fn = metrics.SensoryAccuracy(reduction="none")

    def forward(
        self, batch: Any, prev_state: Optional[TEMState] = None
    ) -> tuple[LossOutput, AccuracyO, RolloutStep]:
        """Execute streaming rollout with visit-masked loss accumulation.

        Iterates through environment steps, computing and accumulating losses only
        for locations that have been previously visited (revisits). First visits
        update the visited mask but do not contribute to the loss.

        Args:
            batch: Tuple ``(chunk, visited)`` where:
                chunk: Iterable of ``(locations, observations, actions)`` tuples
                    for one rollout segment.
                visited: List of per-environment boolean masks ``[env_i][loc_id]``
                    tracking which locations have been visited. Updated in-place.
            prev_state: Initial state for the rollout. If None, a fresh state is
                initialized.

        Returns:
            Tuple of:
                loss_output: Accumulated losses over all revisits.
                accuracies: Mean sensory prediction accuracies.
                last_state: Final TEM state from the rollout.

        Raises:
            ValueError: If chunk is empty or rollout produces no states.
        """
        walk, visited = batch
        if len(walk) == 0:
            raise ValueError("forward requires a non-empty chunk (walk)")

        accum = AccumLoss.zero(device=self.device)
        acc_counts = AccuracyO.zero(device=self.device)

        for step in RolloutStream(self.tem, walk, prev_state):
            step_contrib, acc_increments = self.model_iteration(step, visited)

            # Accumulate loss and accuracies
            if step_contrib is not None:
                accum = accum + step_contrib
            acc_counts = acc_counts + acc_increments

        final_acc = acc_counts

        return accum, final_acc, step

    def model_iteration(
        self, step: RolloutStep, visited: list[list[bool]]
    ) -> tuple[Optional[StepLoss], AccuracyO]:
        """Compute visit-masked loss and accuracy for a single timestep.

        Implements the revisit gating policy: losses and accuracies are only
        accumulated for environments visiting previously-seen locations. First
        visits are excluded from optimization but update the visited mask.

        Args:
            output: TEM predictions for the current timestep.
            label: Ground truth observations for the current timestep.
            state:
            visited: Per-environment visited masks ``visited[env_i][loc_id]``.
                Updated in-place when environments visit new locations.

        Returns:
            Tuple of:
                step_loss: Mean loss over contributing environments, or None if
                    all environments are on first visits.
                accuracy_counts: :class:`AccuracyO` with weighted accuracies.
                rollout_step: The current rollout step.
        """
        output, labels, state = step.output, step.world_step, step.state
        step_losses = self.loss_fn(output, labels, state)
        step_acc = self.acc_o_fn(output, labels)

        losses_per_env: list[StepLoss] = []
        acc_total = AccuracyO.zero(device=self.device)

        for env_i, env_visited in enumerate(visited):
            loc_id = labels.locations[env_i]["id"]
            if not env_contributes_and_update(env_visited, loc_id):
                continue

            losses_per_env.append(env_step_loss(step_losses, env_i))
            acc_total = acc_total + env_acc_increments(step_acc, env_i)

        return mean_step_losses(losses_per_env), acc_total

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        """Execute one training step.

        Args:
            batch: Training batch from dataloader.
            batch_idx: Batch index (unused).

        Returns:
            Total loss for optimization.
        """
        loss_output, accuracies, step = self(batch, self.prev_state)
        self.prev_state = step.state.detach()

        self._log_step_metrics(prefix="", loss_output=loss_output)
        self._log_accuracy_metrics(prefix="", accuracies=accuracies)
        return loss_output.total

    def on_validation_epoch_start(self) -> None:
        """Reset deterministic validation streams at epoch start."""
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is None:
            return
        reset = getattr(datamodule, "reset_split", None)
        if callable(reset):
            reset("validate")

    def validation_step(self, batch: Any, batch_idx: int) -> Tensor:
        """Execute one validation step.

        Resets recurrent state (except memory) at batch boundaries.

        Args:
            batch: Validation batch from dataloader.
            batch_idx: Batch index (unused).

        Returns:
            Total loss for logging.
        """
        batch_size, device = batch[0][0][1].shape[0], batch[0][0][1].device  # Chunk, step, observations
        init_state = self.prev_state.new() if self.prev_state else self.tem.init_state(batch_size, device)
        loss_output, accuracies, _ = self(batch, init_state)

        self._log_step_metrics(prefix="val/", loss_output=loss_output)
        self._log_accuracy_metrics(prefix="val/", accuracies=accuracies)
        return loss_output.total

    def on_test_epoch_start(self) -> None:
        """Reset deterministic test streams at epoch start."""
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is None:
            return
        reset = getattr(datamodule, "reset_split", None)
        if callable(reset):
            reset("test")

    def test_step(self, batch: Any, batch_idx: int) -> Tensor:
        """Execute one test step.

        Resets recurrent state (except memory) at batch boundaries.

        Args:
            batch: Test batch from dataloader.
            batch_idx: Batch index (unused).

        Returns:
            Total loss for logging.
        """
        batch_size, device = batch[0][0][1].shape[0], batch[0][0][1].device  # Chunk, step, observations
        init_state = self.prev_state.new() if self.prev_state else self.tem.init_state(batch_size, device)
        loss_output, accuracies, _ = self(batch, init_state)

        self._log_step_metrics(prefix="test/", loss_output=loss_output)
        self._log_accuracy_metrics(prefix="test/", accuracies=accuracies)
        return loss_output.total

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        """Update runtime hyperparameters before training step.

        Applies curriculum scheduling to Hebbian parameters and walk length based on
        global step count.

        Args:
            batch: Training batch (unused).
            batch_idx: Batch index (unused).
        """
        eta, hebbian_decay, p2g_uncertainty_offset, walk_center = self._compute_schedule(self.global_step)
        self.tem.set_runtime(eta, hebbian_decay, p2g_uncertainty_offset)
        self._maybe_set_walk_length_center(walk_center)

    def on_validation_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Update runtime hyperparameters before validation step.

        Args:
            batch: Validation batch (unused).
            batch_idx: Batch index (unused).
            dataloader_idx: Dataloader index for multiple validation sets.
        """
        eta, hebbian_decay, p2g_uncertainty_offset, _ = self._compute_schedule(self.global_step)
        self.tem.set_runtime(eta, hebbian_decay, p2g_uncertainty_offset)

    def on_test_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Update runtime hyperparameters before test step.

        Args:
            batch: Test batch (unused).
            batch_idx: Batch index (unused).
            dataloader_idx: Dataloader index for multiple test sets.
        """
        eta, hebbian_decay, p2g_uncertainty_offset, _ = self._compute_schedule(self.global_step)
        self.tem.set_runtime(eta, hebbian_decay, p2g_uncertainty_offset)

    def _maybe_set_walk_length_center(self, walk_length_center: float) -> None:
        """Update datamodule walk length curriculum if available.

        Attempts to call ``datamodule.set_walk_length_center()`` if the method
        exists. Safe to call even if datamodule lacks curriculum support.

        Args:
            walk_length_center: Target mean walk length for curriculum annealing.
        """
        datamodule = getattr(self.trainer, "datamodule", None)
        setter = getattr(datamodule, "set_walk_length_center", None) if datamodule is not None else None
        if callable(setter):
            setter(walk_length_center)

    def _log_step_metrics(self, *, prefix: str, loss_output: LossOutput) -> None:
        """Log hierarchical loss components to tensorboard.

        Args:
            prefix: Metric namespace prefix (e.g., "val/", "test/", "").
            loss_output: Complete loss output to log.
        """
        self.log(f"{prefix}loss", loss_output.total, prog_bar=True)

        # Sensory reconstruction losses
        self.log(f"{prefix}Losses/lx_p_inf", loss_output.x.infer)
        self.log(f"{prefix}Losses/lx_p_gen_gi", loss_output.x.retrieved)
        self.log(f"{prefix}Losses/lx_p_gen_gg", loss_output.x.ancestral)
        self.log(f"{prefix}Losses/lx", loss_output.x.total)

        # Abstract location consistency losses
        self.log(f"{prefix}Losses/lg", loss_output.g.total)

        # Grounded location consistency losses
        self.log(f"{prefix}Losses/lp_g", loss_output.p.abstract)
        self.log(f"{prefix}Losses/lp_x", loss_output.p.sensory)
        self.log(f"{prefix}Losses/lp", loss_output.p.total)

        # Regularization losses
        self.log(f"{prefix}Losses/reg_g", loss_output.reg.g_l2)
        self.log(f"{prefix}Losses/reg_p", loss_output.reg.p_l1)

    def _log_accuracy_metrics(self, *, prefix: str, accuracies: AccuracyO) -> None:
        """Log sensory prediction accuracies to tensorboard.

        Args:
            prefix: Metric namespace prefix (e.g., "val/", "test/", "").
            accuracies: Accuracy metrics for the three prediction pathways.
        """
        self.log(f"{prefix}Accuracies/o_p_inf", accuracies.acc_p_inf)
        self.log(f"{prefix}Accuracies/o_gen_gi", accuracies.acc_gen_gi)
        self.log(f"{prefix}Accuracies/o_gen_gg", accuracies.acc_gen_gg)

    def configure_optimizers(self):
        """Configure optimizer and LR scheduler for training."""
        config = self.trainer_settings.optimizer
        optimizer = Adam(self.tem.parameters(), config.lr, config.betas, config.eps, config.weight_decay)
        scheduler = ExponentialLR(optimizer, self.trainer_settings.scheduler.lr.gamma)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def _compute_schedule(self, iteration: int) -> tuple[float, float, float, float]:
        """Compute all scheduled hyperparameters for current iteration.

        Args:
            iteration: Current global step.

        Returns:
            Tuple of (eta, hebbian_decay, p2g_uncertainty_offset, walk_length_center):
                eta: Hebbian learning rate.
                hebbian_decay: Hebbian memory decay factor.
                p2g_uncertainty_offset: Additive uncertainty offset for place-to-grid inference.
                walk_length_center: Target mean walk length for curriculum.
        """
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


def select_env(t: Tensor, env_i: int) -> Tensor:
    """Extract single environment from batch tensor.

    Handles both reduced (scalar) and unreduced (batched) tensors gracefully.

    Args:
        t: Input tensor. Either scalar (reduced) or ``(B, ...)`` (unreduced).
        env_i: Environment index in ``[0, B)``.

    Returns:
        Scalar for reduced input, or ``t[env_i]`` for batched input.
    """
    return t if t.ndim == 0 else t[env_i]


def env_contributes_and_update(visited_env: list[bool], loc_id: int) -> bool:
    """Check and update visit status for one environment.

    Implements the revisit gating policy: only revisited locations contribute
    to training. First visits update the mask in-place but are excluded.

    Args:
        visited_env: Boolean mask ``visited_env[location_id]`` for one environment.
            Modified in-place on first visit.
        loc_id: Current location identifier.

    Returns:
        True if location was previously visited (contribute to loss/accuracy).
        False if first visit (exclude from optimization, mask updated).
    """
    if visited_env[loc_id]:
        return True
    visited_env[loc_id] = True
    return False


def env_step_loss(step_losses: LossOutput, env_i: int) -> StepLoss:
    """Extract single-environment loss from batch loss output.

    Args:
        step_losses: Batch-level loss output (possibly unreduced).
        env_i: Environment index to extract.

    Returns:
        StepLoss with components for environment ``env_i``.
    """
    return StepLoss(
        x=LossX(
            infer=select_env(step_losses.x.infer, env_i),
            retrieved=select_env(step_losses.x.retrieved, env_i),
            ancestral=select_env(step_losses.x.ancestral, env_i),
        ),
        p=LossP(
            abstract=select_env(step_losses.p.abstract, env_i),
            sensory=select_env(step_losses.p.sensory, env_i),
        ),
        g=LossG(transition=select_env(step_losses.g.transition, env_i)),
        reg=LossReg(
            g_l2=select_env(step_losses.reg.g_l2, env_i),
            p_l1=select_env(step_losses.reg.p_l1, env_i),
        ),
    )


def env_acc_increments(step_acc: AccuracyO, env_i: int) -> AccuracyO:
    """Extract accuracy for one environment.

    Args:
        step_acc: Batch-level accuracy (possibly unreduced).
        env_i: Environment index to extract.

    Returns:
        :class:`AccuracyO` with per-pathway correctness and internal weight of 1.
    """
    acc_p_inf = select_env(step_acc.acc_p_inf, env_i)
    acc_gen_gi = select_env(step_acc.acc_gen_gi, env_i)
    acc_gen_gg = select_env(step_acc.acc_gen_gg, env_i)
    _total = torch.ones((), device=acc_p_inf.device, dtype=acc_p_inf.dtype)
    return AccuracyO(acc_p_inf, acc_gen_gi, acc_gen_gg, _total=_total)


def mean_step_losses(losses_per_env: list[StepLoss]) -> Optional[StepLoss]:
    """Average losses across contributing environments.

    Args:
        losses_per_env: Losses for environments that passed revisit gating.

    Returns:
        Mean loss over contributing environments, or None if list is empty.
    """
    if not losses_per_env:
        return None
    total = losses_per_env[0]
    for loss in losses_per_env[1:]:
        total = total + loss
    return total / len(losses_per_env)
    return total / len(losses_per_env)
