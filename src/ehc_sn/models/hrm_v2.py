"""HRM v2 Lightning module (RL + warmup).

This module defines a PyTorch Lightning :class:`~lightning.LightningModule` wrapper
around the HRM v2 architecture (:class:`HRModelV2`) and its training loop.

Compared to HRM v1 (ACT-supervised), HRM v2 couples a PFC-style recurrent reasoning
core with an STR actor-critic head and trains with reinforcement-learning losses
computed by :class:`~ehc_sn.training.rl_head.RLLossHead` via a
:class:`~ehc_sn.training.controller.RLController`.

Key behaviors:
    - **Manual optimization**: sets ``automatic_optimization = False`` and performs
        explicit backward/optimizer/scheduler steps.
    - **Three-optimizer training**: supervised params, RL (STR) params, and vmPFC
        (``pfc.estimator``) params are optimized with separate optimizers.
    - **Warmup**: for the first ``warmup_steps`` global steps, halting is disabled
        (``allow_halt=False``) to avoid the degenerate "halt immediately" solution.
    - **Partial reset batching**: halted examples are replaced with fresh rows using
        :class:`~ehc_sn.training.buffers.FifoBuffer` and
        :class:`~ehc_sn.training.partial_reset.PartialResetBatchAssembler`.

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
from pydantic import BaseModel, Field, field_validator
from torch import Tensor, nn
from torch.optim import Optimizer

from ehc_sn.controllers.rl import RLController, RLControllerConfig
from ehc_sn.data.schema import CHANNEL_SOLUTION, O_ID
from ehc_sn.data.transforms import channels_to_grid
from ehc_sn.envs.mazehard import EnvConfig, MazeHardEnv
from ehc_sn.heads.rl import RLLossConfig, RLLossHead, RLLossStep
from ehc_sn.metrics import build_train_metrics, build_val_metrics, update_metrics_from_step
from ehc_sn.metrics.routes import RL_EPISODE_ROUTES, RL_STEP_ROUTES
from ehc_sn.metrics.traces import build_trace_spec
from ehc_sn.modules.pfc import PFCModel, PFCSettings, PFCState
from ehc_sn.modules.str import STRModelLinear, STRSettings, STRState
from ehc_sn.rollouts.collect import TraceCollector
from ehc_sn.rollouts.trace_tree import TraceTree
from ehc_sn.training.buffers import FifoBuffer
from ehc_sn.training.optim import AdamATan2, AdamATan2Config
from ehc_sn.training.partial_reset import PartialResetBatchAssembler
from ehc_sn.training.schedules import CosineAnnealingLRWithWarmup, SchedulerConfig, SequentialLR
from ehc_sn.training.step_loop import StepContext, StepLoop
from ehc_sn.types import Device, Dtype
from ehc_sn.utils import trunc_normal_init_
from ehc_sn.utils.detach import DetachMixin

# Community-standard map-style batch: plain dict returned by MazeDataset / DataLoader.
Batch: TypeAlias = Dict[str, Tensor]

# Token label ignored by supervised loss (padding / non-supervised positions).
IGNORE_LABEL_ID: int = -100


# =================================================================================================
class ModelSettings_V2(BaseModel, extra="forbid"):
    """Model-level settings for HRM v2.

    This settings object composes:
        - PFC settings (recurrent reasoning core)
        - STR settings (actor-critic / reward head)
        - token vocabulary size

    Notes:
        HRM v2 currently requires RoPE positional encodings inside the PFC modules
        for legacy parity and to match the environment tokenization.
    """

    pfc: PFCSettings = Field(
        ...,
        description="Settings for the core PFC model architecture.",
    )
    str: STRSettings = Field(
        ...,
        description="Settings for the STR actor-critic architecture.",
    )

    @field_validator("pfc", mode="after")
    def validate_pfc(cls, v: PFCSettings) -> PFCSettings:
        """Ensure that the PFC settings have a valid reasoning module configuration."""
        if v.cortex.pos_encodings != "rope":
            raise ValueError("PFC reasoning modules must use RoPE positional encodings")
        return v

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
        """Base embedding scale applied to token embeddings."""
        return math.sqrt(self.hidden_size)

    @property
    def init_std(self) -> float:
        """Convenience property for standard deviation of truncated normal initialization."""
        return 1.0 / math.sqrt(self.hidden_size)


# =================================================================================================
class ModelConfig_HRM_V2(BaseModel, extra="forbid"):
    """Configuration for the HRM v2 Lightning module.

    This config wires together:
        - model settings (:class:`ModelSettings_V2`)
        - environment settings (:class:`~ehc_sn.envs.mazehard.EnvConfig`)
        - RL controller and loss head configs
        - optimizer and scheduler settings

    Notes:
        - ``extra=\"forbid\"`` ensures unknown keys fail fast.
        - ``global_batch_size`` is used to derive per-rank batch size under DDP.
    """

    # ~~ Model architecture ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    model: ModelSettings_V2 = Field(
        ...,
        description="",
    )
    environment: EnvConfig = Field(
        ...,
        description="Environment configuration (max_steps, seq_length, vocab_size, halt_action).",
    )
    controller: RLControllerConfig = Field(
        ...,
        description="Configuration for the RL controller, which defines the forward pass and computes RL losses.",
    )
    loss: RLLossConfig = Field(
        ...,
        description="Configuration for the RL loss head, which computes losses based on the controller outputs.",
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
    warmup_steps: int = Field(
        default=5000,
        ge=0,
        description=(
            "Number of optimizer steps during which only the supervised optimizer trains. "
            "STR and vmPFC are frozen; allow_halt=False forces full deliberation. "
            "Prevents 'halt immediately' collapse before PFC representations are informative."
        ),
    )

    # ~~ Extra ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    global_batch_size: int = Field(
        ...,
        description=(
            "Global batch size across all devices. "
            "The per-device batch size is computed as `global_batch_size // world_size`."
        ),
    )  # TODO: consider moving to BufferSettings or similar


# =================================================================================================
@dataclass
class HRMState(DetachMixin):
    """Recurrent state carried across steps for HRM v2.

    Attributes:
        pfc: PFC recurrent state.
        str: STR recurrent state.
    """

    pfc: PFCState  # Prefrontal Cortex state, containing working memory and reasoning module states.
    str: STRState  # STR actor-critic state, containing any recurrent state for the STR module (if needed).


# =================================================================================================
class HRModelV2(nn.Module):
    """Core HRM v2 model.

    The model consists of:
        - token embedding table
        - PFC recurrent reasoning module producing per-token logits and a CLS summary
        - STR actor-critic module consuming the CLS summary and PFC Q logits
        - language-model head predicting per-token labels

    The forward pass returns:
        - updated recurrent state
        - tuple of logits ``(token_logits, q_logits, r_logits)``
        - CLS feature vector (used by the controller / tracing)
    """

    def __init__(  # ------------------------------------------------------------------------------
        self, config: ModelSettings_V2, *,
        device: Optional[Device] = None, dtype: Optional[Dtype] = None,
    ) -> None:  # fmt: skip
        super().__init__()
        self._config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, device=device, dtype=dtype)
        self.pfc = PFCModel(config.pfc, device=device, dtype=dtype)  # Reasoning module with embedded inputs
        self.str = STRModelLinear(config.str, device=device, dtype=dtype)  # Reward estimator
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, device=device, dtype=dtype)  # fmt: skip
        self.reset_parameters()

    @property
    def config(self) -> ModelSettings_V2:
        """Return the parsed model settings used to build this module."""
        return self._config

    def reset_parameters(self) -> None:  # -------------------------------------------------------
        """Initialize parameters.

        Uses truncated normal initialization with ``std = 1/sqrt(hidden_size)`` for
        token embeddings and the LM head to keep initial activation scales stable.
        """
        init_std = self.config.init_std
        trunc_normal_init_(self.embed_tokens.weight, std=init_std)
        trunc_normal_init_(self.lm_head.weight, std=init_std)

    def init_state(  # ---------------------------------------------------------------------------
        self, batch_size: int,
    ) -> HRMState:  # fmt: skip
        """Create a fresh recurrent state.

        Args:
            batch_size: Number of parallel environments / sequences.

        Returns:
            A new :class:`HRMState` with initialized PFC and STR states.
        """
        return HRMState(
            pfc=self.pfc.init_state(batch_size),
            str=self.str.init_state(batch_size),
        )

    def reset_state(  # --------------------------------------------------------------------------
        self, reset_flag: Tensor, state: HRMState,
    ) -> HRMState:  # fmt: skip
        """Selectively reset rows of the recurrent state.

        Args:
            reset_flag: Boolean / 0-1 tensor of shape ``(B,)`` indicating which
                batch rows should be reset.
            state: Current recurrent state.

        Returns:
            New state with flagged rows reset for both PFC and STR.
        """
        return HRMState(
            pfc=self.pfc.reset_state(state.pfc, reset_flag),
            str=self.str.reset_state(state.str, reset_flag),
        )

    def forward(  # -------------------------------------------------------------------------------
        self, batch: Batch, state: Optional[HRMState] = None,
    ) -> tuple[HRMState, tuple[Tensor, Tensor, Tensor], Tensor]:  # fmt: skip
        """Run one model step.

        Args:
            batch: Input batch containing at least ``"inputs"`` of shape ``(B, S)``.
            state: Optional recurrent state to carry across steps. If ``None``, a
                fresh state is created.

        Returns:
            ``(new_state, (logits, q_logits, r_logits), theta_cls)`` where:
                - ``logits`` is ``(B, S, vocab_size)``
                - ``q_logits`` is controller-specific (produced by PFC)
                - ``r_logits`` is reward / policy output from STR
                - ``theta_cls`` is ``(B, D)`` CLS summary vector.
        """
        state = state or self.init_state(batch_size=batch["inputs"].shape[0])
        x = self.embed_inputs(batch["inputs"])  # (B, S, D)

        state_pfc, z_H, q_logits = self.pfc(x, state=state.pfc)  # z_H: (B, S+1, D)
        logits = self.lm_head(z_H[:, 1:])  # strip CLS → (B, S, vocab)
        theta_cls = z_H[:, 0]  # (B, D) — theta/CLS summary
        state_str, r_logits = self.str(theta_cls.detach(), q_logits, state.str)

        new_state = HRMState(pfc=state_pfc, str=state_str)
        return new_state, (logits, q_logits, r_logits), theta_cls

    def embed_inputs(  # --------------------------------------------------------------------------
        self, input: Tensor,
    ) -> Tensor:  # fmt: skip
        """Embed token ids into a scaled representation.

        Args:
            input: Token ids of shape ``(B, S)``.

        Returns:
            Embedded inputs of shape ``(B, S, D)`` scaled by ``sqrt(D)``.
        """
        token_embeddings = self.embed_tokens(input.to(torch.int32))
        # Scale embeddings to keep activations in a reasonable range.
        return self.config.embedding_scale * token_embeddings


# =================================================================================================
class TrainingModel(L.LightningModule):
    """LightningModule wrapper for HRM v2 RL training.

    This wrapper manages:
        - lazy initialization of :class:`~ehc_sn.envs.mazehard.MazeHardEnv`
        - wiring :class:`~ehc_sn.training.controller.RLController` and
          :class:`~ehc_sn.training.rl_head.RLLossHead`
        - partial-reset batching via FIFO buffering
        - manual optimization with three optimizers
    """

    def __init__(  # ------------------------------------------------------------------------------
        self, config: ModelConfig_HRM_V2,
    ) -> None:  # fmt: skip
        super().__init__()
        self.model = HRModelV2(config.model)
        self.environment: MazeHardEnv | None = None  # Lazy init in setup() to avoid GPU allocation issues
        self.controller: RLController | None = None  # Initialized in setup() after environment is ready
        self.step_module: RLLossHead | None = None  # Initialized in setup() after controller is ready
        self._config = config

        # Manual optimization: explicit backward + opt step (legacy parity + dual-opt clarity).
        self.automatic_optimization = False
        self._train_carry = None

        # Metrics are cloned for train/val to allow separate logging and state management.
        self.train_metrics = build_train_metrics(RL_STEP_ROUTES).clone(prefix="train/")
        self.val_metrics = build_val_metrics(RL_EPISODE_ROUTES).clone(prefix="val/")
        self.trace_specs = build_trace_spec("rl")

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
    def config(self) -> ModelConfig_HRM_V2:
        """Return the parsed configuration used by this LightningModule."""
        return self._config

    def setup(  # --------------------------------------------------------------------------------
        self, stage: Optional[str] = None,
    ) -> None:  # fmt: skip
        """Lazy initialization of the environment to avoid GPU allocation issues in DDP."""
        world_size = max(getattr(self.trainer, "world_size", 1), 1)
        local_bs = self.config.global_batch_size // world_size

        if self.environment is None:
            self.environment = MazeHardEnv(self.config.environment, batch_size=local_bs)
        self.controller = RLController(self.model, self.environment, self.config.controller)
        self.step_module = RLLossHead(self.controller, self.config.loss)

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
        sup_params = [p for p in self.model.parameters() if id(p) not in excluded_ids]
        opt_sup = AdamATan2(sup_params, self.config.optimizer_supervised)
        # Optimizer B: RL — STR actor-critic only (strictly isolated)
        opt_rl = AdamATan2(list(self.model.str.parameters()), self.config.optimizer_rl)
        # Optimizer C: vmPFC — pfc.estimator only (auxiliary Q-predictor)
        opt_qv = AdamATan2(list(self.model.pfc.estimator.parameters()), self.config.optimizer_qv)

        sch_sup = CosineAnnealingLRWithWarmup(opt_sup, total_steps, self.config.scheduler)
        sch_rl = CosineAnnealingLRWithWarmup(opt_rl, total_steps, self.config.scheduler)
        sch_qv = CosineAnnealingLRWithWarmup(opt_qv, total_steps, self.config.scheduler)

        return [opt_sup, opt_rl, opt_qv], [sch_sup, sch_rl, sch_qv]

    # -- Lifecycle --------------------------------------------------------------------------------

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

    # -- Training ----------------------------------------------------------------------------------

    def training_step(  # -------------------------------------------------------------------------
        self, batch: Batch, batch_idx: int,
    ) -> Dict[str, Any]:  # fmt: skip
        """Run one training step (horizon = 1) with manual optimization.

        Training uses a carry object produced by :class:`~ehc_sn.training.rl_head.RLLossHead`
        to support partial resets: rows that halted in the previous step are
        replaced with fresh examples from the current incoming batch.

        Notes:
            - ``setup()`` must have run so that ``self.step_module`` is available.
            - During warmup (``global_step < warmup_steps``), halting is disabled.
        """
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
        is_warmup = self.global_step < self._config.warmup_steps
        rl_options = {"explore": True, "allow_halt": not is_warmup, "is_warmup": is_warmup}
        carry0 = self._train_carry

        step = None
        for _t, step in StepLoop(self.step_module, step_batches, carry0, options=rl_options):
            pass  # horizon = 1; loop runs exactly once
        if step is None:
            raise ValueError("StepLoop did not yield any steps.")
        self._train_carry = step.carry.detach()

        # Normalize by local batch size; DDP averages gradients across ranks.
        local_bs = int(batch["inputs"].shape[0])
        out = step.outputs
        loss = _normalize_loss_for_backward(out.loss, local_bs)

        # Zero gradients before backward so each step uses only the current batch.
        opt_sup, opt_rl, opt_qv = self.optimizers()  # type: ignore[misc]
        sch_sup, sch_rl, sch_qv = self.lr_schedulers()  # type: ignore[misc]
        opt_sup.zero_grad(set_to_none=True)
        opt_rl.zero_grad(set_to_none=True)
        opt_qv.zero_grad(set_to_none=True)

        self.manual_backward(loss)

        opt_sup.step(); sch_sup.step()  # fmt: skip
        if not is_warmup:
            opt_rl.step(); sch_rl.step()  # fmt: skip
            opt_qv.step(); sch_qv.step()  # fmt: skip

        # Update metrics with unnormalized loss and log to TensorBoard.
        update_metrics_from_step(self.train_metrics, step.outputs.metrics, RL_STEP_ROUTES)
        self.log("train/loss", loss.detach(), on_step=True, on_epoch=False, prog_bar=True, logger=True)

        signals = {**step.outputs.signals, "is_warmup": torch.tensor(float(is_warmup))}
        return {"loss": loss.detach(), "signals": signals}

    # -- Validation -------------------------------------------------------------------------------

    def validation_step(  # -----------------------------------------------------------------------
        self, batch: Batch, batch_idx: int,
    ) -> Dict[str, Any]:  # fmt: skip
        """Run a full rollout until all slots halt, collecting traces for logging/analysis.

        Validation runs the controller to the max horizon (no exploration) and
        collects a trace tree for downstream logging/analysis.
        """
        step_batches = repeat(batch)  # Run until all examples halt
        rl_options = {"explore": False, "allow_halt": False, "is_warmup": False}
        carry0 = self.step_module.initial_carry(batch)
        collector = TraceCollector(TraceTree(), self.trace_specs)

        step = None
        for t, step in StepLoop(self.step_module, step_batches, carry0, options=rl_options):
            collector.append(t, step)
            update_metrics_from_step(self.val_metrics, step.outputs.metrics, RL_EPISODE_ROUTES)
        if step is None:
            raise ValueError("Evaluation loop did not yield any steps.")

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
    token sequence.  The label sequence overwrites solution-path cells with
    :data:`O_ID` (HRM-private supervision token).

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
