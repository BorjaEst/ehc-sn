"""HRM v1 Lightning module.

This module defines a PyTorch Lightning `LightningModule` wrapper around the core HRM
architecture (`HRModel`) with Adaptive Computation Time (ACT) control and loss
computation.

Key behaviors:
        - **Manual optimization**: uses `automatic_optimization = False` and explicitly performs
            backward/optimizer/scheduler steps for legacy parity.
        - **Stateful training carry**: forwards a `carry` object across mini-batches to support
            continuation/halting semantics.
        - **Partial reset batching**: halted examples are replaced with fresh rows using a FIFO buffer
            and `PartialResetBatchAssembler`.

The batch structure used throughout this file is a plain ``dict[str, Tensor]``
with keys ``"inputs"`` and ``"labels"``.
"""

import math
from dataclasses import dataclass
from itertools import repeat
from typing import Any, Dict, List, Optional, Tuple, TypeAlias

import lightning as L
import numpy as np
import torch
from adam_atan2_pytorch import AdamAtan2 as AdamATan2
from pydantic import BaseModel, Field
from torch import Tensor, nn
from torch.optim import Optimizer

from ehc_sn.data.schema import CHANNEL_SOLUTION
from ehc_sn.data.transforms import channels_to_grid
from ehc_sn.metrics import build_metrics, update_metrics_from_step
from ehc_sn.modules.pfc import PFCModel, PFCSettings, PFCState
from ehc_sn.rollouts.collect import TraceCollector, TraceField, TraceSpec, TraceValue
from ehc_sn.rollouts.trace_tree import TraceTree
from ehc_sn.training.act_controller import ACTController, ACTControllerConfig
from ehc_sn.training.act_head import ACTLossConfig, ACTLossHead
from ehc_sn.training.buffers import FifoBuffer
from ehc_sn.training.optim import AdamATan2, AdamATan2Config
from ehc_sn.training.partial_reset import PartialResetBatchAssembler
from ehc_sn.training.schedules import CosineAnnealingLRWithWarmup, SchedulerConfig, SequentialLR
from ehc_sn.training.step_loop import StepContext, StepLoop
from ehc_sn.types import Device, Dtype
from ehc_sn.utils import trunc_normal_init_

# Community-standard map-style batch: plain dict returned by MazeDataset / DataLoader.
Batch: TypeAlias = Dict[str, Tensor]

# HRM-private: solution-path token, not part of the canonical SEM vocabulary.
O_ID: int = 5


# =================================================================================================
class ModelSettings_V1(BaseModel, extra="forbid"):
    """Model-level settings composing a PFC module with embedding/LM-head parameters."""

    pfc: PFCSettings = Field(
        ...,
        description="Settings for the core PFC model architecture.",
    )

    vocab_size: int = Field(
        ...,
        ge=1,
        description="Vocabulary size for token embeddings and LM head.",
    )

    @property
    def seq_length(self) -> int:
        """Convenience property to access sequence length from the PFC settings."""
        return self.pfc.seq_length

    @property
    def hidden_size(self) -> int:
        """Convenience property to access hidden size from the PFC settings."""
        return self.pfc.reasoning_h.cortex.embedding_dim

    @property
    def embedding_scale(self) -> float:
        """Convenience property for scaling embeddings to maintain variance."""
        # scale by 1/sqrt(2) to maintain forward variance
        return 0.707106781 * math.sqrt(self.hidden_size)

    @property
    def init_std(self) -> float:
        """Convenience property for standard deviation of truncated normal initialization."""
        return 1.0 / math.sqrt(self.hidden_size)


# =================================================================================================
class ModelConfig_HRM_V1(BaseModel, extra="forbid"):
    """Configuration for the HRM v1 Lightning module.

    This config is intentionally "spec-first": it wires together the HRM core model,
    the ACT controller (adaptive computation time / halting logic), the loss head, and
    the optimizer/scheduler settings used during training.

    Notes:
        - `extra="forbid"` ensures unknown keys fail fast when parsing configs.
        - `global_batch_size` is used for scaling losses/metrics in a distributed setup.
    """

    model: ModelSettings_V1 = Field(
        ...,
        description="",
    )

    act_controller: ACTControllerConfig = Field(
        ...,
        description=(
            "Configuration for the ACT controller, which manages halting and partial resets during "
            "training. "
            "The keys in `act_controller` are passed to the ACTController constructor."
        ),
    )

    loss: ACTLossConfig = Field(
        ...,
        description="Loss config. The keys in `loss` are passed to the loss head constructor.",
    )

    optimizer: AdamATan2Config = Field(
        default_factory=AdamATan2Config,
        description=(
            "Main optimizer config for model parameters (e.g. Adam). "
            "The keys in `optim_main` are passed to the optimizer constructor."
        ),
    )

    scheduler: SchedulerConfig = Field(
        default_factory=SchedulerConfig,
        description=(
            "Learning rate scheduler config. If not set, no learning rate scheduling is applied. "
            "The keys in `scheduler` are passed to the scheduler constructor."
        ),
    )

    global_batch_size: int = Field(
        ...,
        description=(
            "Global batch size across all devices. "
            "The per-device batch size is computed as `global_batch_size // world_size`."
        ),
    )  # TODO: consider moving to BufferSettings or similar


# =================================================================================================
class TraceFields:
    """ """

    @staticmethod
    def get_model_loss(ctx: StepContext) -> TraceValue:
        loss: Tensor = ctx.outputs.loss  # Scalar tensor
        return loss.detach()  # Detach to avoid tracking gradients in the trace

    @staticmethod
    def get_model_halt_logits(ctx: StepContext) -> TraceValue:
        halt_logits: Tensor = ctx.outputs.outputs.halt_logits  # Tensor shape (B,)
        return halt_logits.detach()

    @staticmethod
    def get_model_continue_logits(ctx: StepContext) -> TraceValue:
        continue_logits: Tensor = ctx.outputs.outputs.continue_logits  # Tensor shape (B,)
        return continue_logits.detach()

    @staticmethod
    def get_model_steps(ctx: StepContext) -> TraceValue:
        steps: Tensor = ctx.carry.steps  # Tensor shape (B,)
        return steps.detach()

    @staticmethod
    def get_act_halted(ctx: StepContext) -> TraceValue:
        halted: Tensor = ctx.carry.halted
        return halted.detach()

    @staticmethod
    def get_outputs(ctx: StepContext) -> TraceValue:
        outputs: Any = ctx.outputs.outputs
        return outputs.detach()

    @staticmethod
    def get_logits(ctx: StepContext) -> TraceValue:
        logits: Tensor = ctx.outputs.outputs.logits
        return logits.detach()

    @staticmethod
    def get_pred_is_o(ctx: StepContext) -> TraceValue:
        pred: Tensor = torch.argmax(TraceFields.get_logits(ctx), dim=-1)  # type: ignore
        return (pred == O_ID).to(torch.uint8).detach()


# =================================================================================================
def trace_fields() -> List[TraceField[StepContext]]:
    return [
        TraceField(name="loss", get=TraceFields.get_model_loss),
        TraceField(name="halt_logits", get=TraceFields.get_model_halt_logits),
        TraceField(name="continue_logits", get=TraceFields.get_model_continue_logits),
        TraceField(name="steps", get=TraceFields.get_model_steps),
        TraceField(name="act/halted", get=TraceFields.get_act_halted),
        TraceField(name="pred/is_o", get=TraceFields.get_pred_is_o),
    ]


# =================================================================================================
@dataclass
class HRMState:
    pfc: PFCState

    def detach(self) -> "HRMState":
        """Return a copy with the PFC state detached from the computation graph."""
        return HRMState(pfc=self.pfc.detach())


# =================================================================================================
class HRModelV1(nn.Module):

    def __init__(  # ------------------------------------------------------------------------------
        self, config: ModelSettings_V1, *,
        device: Optional[Device]=None, dtype: Optional[Dtype]=None,
    ) -> None:  # fmt: skip
        super().__init__()
        self._config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, device=device, dtype=dtype)
        self.embed_pos = nn.Embedding(config.seq_length, config.hidden_size, device=device, dtype=dtype)
        self.pfc = PFCModel(config.pfc, device=device, dtype=dtype)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, device=device, dtype=dtype)  # fmt: skip
        self.reset_parameters()

    @property
    def config(self) -> ModelSettings_V1:
        return self._config

    def reset_parameters(  # ----------------------------------------------------------------------
        self,
    ) -> None:  # fmt: skip
        """Initialize parameters and buffers.

        Matches legacy ``CastedEmbedding`` / ``CastedLinear`` initialization so that
        the input embedding magnitude ``||x||`` and recurrent state magnitude ``||z_H||``
        are comparable (~1:1 ratio at init), which is required for multi-step reasoning
        dynamics to emerge during training.

        Std formulas (truncated normal, legacy parity):
            - Embeddings: ``std = 1 / sqrt(hidden_size)``  (= ``config.init_std``)
            - lm_head:    ``std = 1 / sqrt(hidden_size)``  (fan_in = hidden_size)
            - Reset vecs: ``std = 1``
        """
        init_std = self.config.init_std  # 1 / sqrt(hidden_size)
        trunc_normal_init_(self.embed_tokens.weight, std=init_std)
        trunc_normal_init_(self.embed_pos.weight, std=init_std)
        trunc_normal_init_(self.lm_head.weight, std=init_std)
        # self.pfc.reset_parameters()  # Already done when pfc is initialized

    def init_state(  # ---------------------------------------------------------------------------
        self, batch_size: int,
    ) -> HRMState:  # fmt: skip
        """Create a fresh recurrent state (``ACTBackbone`` protocol)."""
        return HRMState(pfc=self.pfc.init_state(batch_size))

    def reset_state(  # --------------------------------------------------------------------------
        self, reset_flag: Tensor, state: HRMState,
    ) -> HRMState:  # fmt: skip
        """Selectively reset rows of the recurrent state (``ACTBackbone`` protocol)."""
        return HRMState(pfc=self.pfc.reset_state(state.pfc, reset_flag))

    def forward(  # -------------------------------------------------------------------------------
        self, inputs: Tensor, state: Optional[HRMState] = None,
    ) -> Tuple[HRMState, Tensor, Tensor]:  # fmt: skip
        """Forward pass through the HRM (``ACTBackbone`` protocol)."""
        state = state or self.init_state(batch_size=inputs.shape[0])
        x = self.embed_inputs(inputs)  # (B, S, D) — cell tokens only
        state_pfc, z_H, q = self.pfc(x, state=state.pfc)  # z_H is (B, S+1, D)
        output = self.lm_head(z_H[:, 1:])  # Strip CLS → (B, S, vocab_size)
        return HRMState(pfc=state_pfc), output, q

    def embed_inputs(  # --------------------------------------------------------------------------
        self, input: Tensor,
    ) -> Tensor:  # fmt: skip
        """ """
        token_embeddings = self.embed_tokens(input.to(torch.int32))
        positions = torch.arange(self.config.seq_length, device=input.device)
        pos_embeddings = self.embed_pos(positions).unsqueeze(0)

        # Scale embeddings to keep activations in a reasonable range.
        return self.config.embedding_scale * (token_embeddings + pos_embeddings)


# =================================================================================================
class TrainingModel(L.LightningModule):
    """LightningModule wrapper for HRM v1 training.

    This module composes:
        - `HRModel`: the core architecture
        - `ACTController`: halting/partial-reset logic
        - A step module that computes loss and aggregates metrics

    Training uses manual optimization (`automatic_optimization = False`) to preserve
    legacy behavior (one backward pass, then explicit optimizer/scheduler steps).
    """

    def __init__(  # ------------------------------------------------------------------------------
        self, config: ModelConfig_HRM_V1,
    ) -> None:  # fmt: skip
        """Initialize the HRM v1 Lightning module.

        Args:
            config: Parsed `ModelConfig_HRM_V1` with architecture/controller/loss/optim settings.

        Notes:
            - Initializes a FIFO buffer and `PartialResetBatchAssembler` used to implement
              partial-reset semantics during training.
            - `_train_carry` is initialized lazily from the first batch via `step_module`.
        """
        super().__init__()
        self.model = HRModelV1(config.model)
        self.controller = ACTController(self.model, config.act_controller)
        self.step_module = ACTLossHead(self.controller, config.loss)
        self._config = config

        # Manual optimization: one backward, explicit opt/scheduler steps (legacy parity).
        self.automatic_optimization = False
        self._train_carry = None

        # Metrics are cloned for train/val to allow separate logging and state management.
        base_metrics = build_metrics()
        self.train_metrics = base_metrics.clone(prefix="train/")
        self.val_metrics = base_metrics.clone(prefix="val/")
        self.trace_specs = TraceSpec(fields=trace_fields())

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
    def config(self) -> ModelConfig_HRM_V1:
        """Return the parsed configuration used by this module."""
        return self._config

    def configure_optimizers(  # ------------------------------------------------------------------
        self,
    ) -> Tuple[List[Optimizer], List[SequentialLR]]:  # fmt: skip
        """Build optimizers and LR schedulers.

        Returns:
            A tuple `(optimizers, schedulers)` in the format Lightning expects.
        """
        optimizers = self.build_optimizers()
        schedulers = self.schedulers(optimizers)

        return optimizers, schedulers

    def build_optimizers(  # ----------------------------------------------------------------------
        self,
    ) -> List[Optimizer]:  # fmt: skip
        """Construct the optimizer(s) for trainable parameters.

        Returns:
            List containing a single `AdamATan2` optimizer for the HRM parameters.
        """
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = AdamATan2(params, self.config.optimizer)
        return [optimizer]

    def schedulers(  # ----------------------------------------------------------------------------
        self, optimizers: List[Optimizer],
    ) -> List[SequentialLR]:  # fmt: skip
        """Construct LR schedulers for the provided optimizers.

        Args:
            optimizers: Optimizers returned by `build_optimizers()`.

        Returns:
            A list of schedulers (one per optimizer). Currently uses cosine annealing with warmup.
        """
        total_steps = int(self.trainer.estimated_stepping_batches)
        config = self.config.scheduler
        return [CosineAnnealingLRWithWarmup(opt, total_steps, config) for opt in optimizers]

    def on_train_epoch_start(  # ------------------------------------------------------------------
        self,
    ) -> None:  # fmt: skip
        """Reset per-epoch training state.

        Clears the training carry and the FIFO buffer so that partial-reset state does not leak
        across epochs.
        """
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
    ) -> Dict[str, object]:  # fmt: skip
        """Run one training step with manual optimization.

        The training logic uses partial reset to replace halted slots with fresh examples.

        Notes:
            - Horizon is effectively 1: we run exactly one rollout step per mini-batch.
            - Loss is normalized by the *global* effective batch size for parity with legacy code.
        """
        batch_dict = batch

        # Initialize carry/state on the first batch
        if self._train_carry is None:
            self._train_carry = self.step_module.initial_carry(batch_dict)

        # Assemble a step batch using the previous carry's halted mask.
        # `reset_mask=True` means "this row is done, replace it with a fresh example".
        assembler = self._train_batch_assembler
        step_batch = assembler.make_step_batch(
            incoming=batch_dict,
            reset_mask=self._train_carry.halted,  # vectorized done flags
        )

        # Horizon=1 matches legacy behavior: exactly one ACT step per mini-batch.
        step_batches = repeat(step_batch, 1)
        act_options = {"allow_halt": True, "explore": True}  # Allow halt and exploration in training
        carry0 = self._train_carry

        step = None
        for t, step in StepLoop(self.step_module, step_batches, carry0, options=act_options):
            pass  # TODO: Sum loss across steps if horizon > 1
        if step is None:
            raise ValueError("RolloutLoop did not yield any steps, cannot proceed with training step.")
        self._train_carry = step.carry.detach()

        # Normalize by local batch size; DDP averages gradients across ranks.
        local_bs = int(batch_dict["inputs"].shape[0])
        loss = _normalize_loss_for_backward(step.outputs.loss, local_bs=local_bs)
        self.manual_backward(loss)

        optimizers = self.optimizers()
        for opt in optimizers if isinstance(optimizers, list) else [optimizers]:
            opt.step()  # type: ignore
            opt.zero_grad(set_to_none=True)

        scheduler = self.lr_schedulers()
        for sch in scheduler if isinstance(scheduler, list) else [scheduler]:
            sch.step()  # type: ignore

        update_metrics_from_step(self.train_metrics, step.outputs.metrics)
        loss_gm = step.outputs.loss / float(local_bs)
        if (self.global_step + 1) % self.trainer.log_every_n_steps == 0:  # type: ignore
            self.log_dict(self.train_metrics.compute(), on_step=True, on_epoch=False, logger=True)
        self.log("train/loss", loss.detach(), on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("train/loss_gm", loss_gm.detach(), on_step=True, on_epoch=False, logger=True)

        return {"loss": loss.detach()}

    def validation_step(  # -----------------------------------------------------------------------
        self, batch: Batch, batch_idx: int,
    ) -> Dict[str, object]:  # fmt: skip
        """Run one validation step.

        Validation uses `EvaluationLoop` (no carry is persisted across batches here) and logs
        normalized metrics.
        """
        batch_dict = batch

        # Run a full ACT rollout so halted-only metrics are meaningful.
        step_batches = repeat(batch_dict)  # Run until all examples halt
        act_options = {"allow_halt": False, "explore": False}  # No halt or exploration in validation
        carry0 = self.step_module.initial_carry(batch_dict)

        # Initialize carry/state on the first batch
        step, collector = None, TraceCollector(TraceTree(), self.trace_specs)
        for t, step in StepLoop(self.step_module, step_batches, carry0, options=act_options):
            collector.append(t, step)
        if step is None:
            raise ValueError("Evaluation loop did not yield any steps, cannot log metrics.")

        update_metrics_from_step(self.val_metrics, step.outputs.metrics)
        vals = self.val_metrics.compute()  # Compute metrics based on accumulated state
        self.log_dict(vals, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("val/accuracy", vals["val/all/accuracy"], prog_bar=True, logger=True, sync_dist=True)

        return {"trace": collector.tree}


# =================================================================================================
def _normalize_loss_for_backward(  # --------------------------------------------------------------
    total_loss: Tensor, local_bs: int,
) -> Tensor:  # fmt: skip
    """Normalize the total loss by the local batch size for distributed training.

    In distributed training (e.g. DDP), each rank computes gradients on its local mini-batch.
    To ensure that the overall gradient magnitudes are consistent regardless of the number of
    devices, we normalize the loss by the local batch size (the number of examples processed
    by this rank). DDP will then average the gradients across ranks, effectively normalizing by
    the global batch size.

    Args:
        total_loss: The unnormalized loss computed for the current mini-batch (scalar tensor).
        local_bs: The effective batch size for this mini-batch on the current rank (number of examples).

    Returns:
        The loss normalized by the local batch size, ready for backward().
    """
    if local_bs <= 0:
        raise ValueError(f"local_bs must be positive, got {local_bs}.")
    return total_loss / float(local_bs)


# =================================================================================================
def supervised_maze_tokenize(  # ------------------------------------------------------------------
    channels: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:  # fmt: skip
    """Convert raw maze channels into flattened input/label token sequences.

    Uses :func:`~ehc_sn.data.transforms.channels_to_grid` to merge topology,
    start, and goals into a canonical ``int32`` grid, then flattens to a 1-D
    token sequence.  The label sequence is a copy where solution-path cells are
    overwritten with :data:`O_ID` (HRM-private supervision token).

    Args:
        channels: Raw NPZ channel dict (as returned by ``MazeDataset``).

    Returns:
        ``{"inputs": int32 (H*W,), "labels": int32 (H*W,)}``.
    """
    grid = channels_to_grid(channels)["grid"]  # (H, W) int32
    inputs = grid.ravel()
    labels = inputs.copy()
    if CHANNEL_SOLUTION in channels:
        labels[channels[CHANNEL_SOLUTION].ravel() > 0] = O_ID
    return {"inputs": inputs, "labels": labels}
