""" """

import math
from dataclasses import dataclass
from itertools import repeat
from typing import Any, Dict, List, Optional, Tuple, TypeAlias

import lightning as L
import numpy as np
import torch
from pydantic import BaseModel, Field
from torch import Tensor, nn
from torch.optim import Optimizer

from ehc_sn.data.schema import CHANNEL_SOLUTION
from ehc_sn.data.transforms import channels_to_grid
from ehc_sn.envs.mazehard import EnvConfig, MazeHardEnv
from ehc_sn.metrics import build_train_metrics, build_val_metrics, update_metrics_from_step
from ehc_sn.modules.pfc import PFCModel, PFCSettings, PFCState
from ehc_sn.modules.str import STRModelLinear, STRSettings, STRState
from ehc_sn.rollouts.collect import TraceCollector, TraceField, TraceSpec, TraceValue
from ehc_sn.rollouts.trace_tree import TraceTree
from ehc_sn.training.buffers import FifoBuffer
from ehc_sn.training.optim import AdamATan2, AdamATan2Config
from ehc_sn.training.partial_reset import PartialResetBatchAssembler
from ehc_sn.training.rl_controller import RLController, RLControllerConfig, RLOutput, RLState
from ehc_sn.training.rl_head import RLLossConfig, RLLossHead, RLLossStep
from ehc_sn.training.schedules import CosineAnnealingLRWithWarmup, SchedulerConfig, SequentialLR
from ehc_sn.training.step_loop import StepContext, StepLoop
from ehc_sn.types import Device, Dtype
from ehc_sn.utils import trunc_normal_init_
from ehc_sn.utils.detach import DetachMixin

# Community-standard map-style batch: plain dict returned by MazeDataset / DataLoader.
Batch: TypeAlias = Dict[str, Tensor]

# HRM-private: solution-path token, not part of the canonical SEM vocabulary.
O_ID: int = 5

# Token label ignored by supervised loss (padding / non-supervised positions).
IGNORE_LABEL_ID: int = -100


# =================================================================================================
class ModelSettings_V2(BaseModel, extra="forbid"):
    """ """

    pfc: PFCSettings = Field(..., description="Settings for the core PFC model architecture.")
    str: STRSettings = Field(..., description="Settings for the STR actor-critic architecture.")

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
class ModelConfig_HRM_V2(BaseModel, extra="forbid"):
    """ """

    # ~~ Model architecture ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    model: ModelSettings_V2 = Field(
        ...,
        description="",
    )
    environment: EnvConfig = Field(
        ...,
        description="Environment configuration (max_steps, seq_length, vocab_size, halt_action).",
    )
    rl_controller: RLControllerConfig = Field(
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
class TraceFields:
    """ """

    @staticmethod
    def get_model_loss(ctx: StepContext) -> TraceValue:
        return ctx.outputs.loss.detach()


# =================================================================================================
def trace_fields(  # ------------------------------------------------------------------------------
) -> List[TraceField[StepContext]]:  # fmt: skip
    """ """
    return [
        TraceField(name="loss", get=TraceFields.get_model_loss),
    ]


# =================================================================================================
@dataclass
class HRMState(DetachMixin):
    """ """

    pfc: PFCState  # Prefrontal Cortex state, containing working memory and reasoning module states.
    str: STRState  # STR actor-critic state, containing any recurrent state for the STR module (if needed).


# =================================================================================================
class HRModelV2(nn.Module):
    """ """

    def __init__(  # ------------------------------------------------------------------------------
        self, config: ModelSettings_V2, *,
        device: Optional[Device] = None, dtype: Optional[Dtype] = None,
    ) -> None:  # fmt: skip
        super().__init__()
        self._config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, device=device, dtype=dtype)
        self.embed_pos = nn.Embedding(config.seq_length, config.hidden_size, device=device, dtype=dtype)
        self.pfc = PFCModel(config.pfc, device=device, dtype=dtype)  # Reasoning module with embedded inputs
        self.str = STRModelLinear(config.str, device=device, dtype=dtype)  # Reward estimator
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, device=device, dtype=dtype)  # fmt: skip
        self.reset_parameters()

    @property
    def config(self) -> ModelSettings_V2:
        """ """
        return self._config

    def reset_parameters(self) -> None:  # -------------------------------------------------------
        """ """
        init_std = self.config.init_std
        trunc_normal_init_(self.embed_tokens.weight, std=init_std)
        trunc_normal_init_(self.embed_pos.weight, std=init_std)
        trunc_normal_init_(self.lm_head.weight, std=init_std)

    def init_state(  # ---------------------------------------------------------------------------
        self, batch_size: int,
    ) -> HRMState:  # fmt: skip
        """ """
        return HRMState(
            pfc=self.pfc.init_state(batch_size),
            str=self.str.init_state(batch_size),
        )

    def reset_state(  # --------------------------------------------------------------------------
        self, reset_flag: Tensor, state: HRMState,
    ) -> HRMState:  # fmt: skip
        """ """
        return HRMState(
            pfc=self.pfc.reset_state(state.pfc, reset_flag),
            str=self.str.reset_state(state.str, reset_flag),
        )

    def forward(  # -------------------------------------------------------------------------------
        self, inputs: Tensor, state: Optional[HRMState] = None,
    ) -> Tuple[HRMState, Tuple[Tensor, Tensor, Tensor], Tensor]:  # fmt: skip
        """ """
        state = state or self.init_state(batch_size=inputs.shape[0])
        x = self.embed_inputs(inputs)  # (B, S, D)

        state_pfc, z_H, q_logits = self.pfc(x, state=state.pfc)  # z_H: (B, S+1, D)
        logits = self.lm_head(z_H[:, 1:])  # strip CLS → (B, S, vocab)
        theta_cls = z_H[:, 0]  # (B, D) — theta/CLS summary
        state_str, r_logits = self.str(theta_cls.detach(), q_logits, state.str)

        new_state = HRMState(pfc=state_pfc, str=state_str)
        return new_state, (logits, q_logits, r_logits), theta_cls

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
    """ """

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
        self.train_metrics = build_train_metrics(groups=["rl"]).clone(prefix="train/")
        self.val_metrics = build_val_metrics(groups=["rl"]).clone(prefix="val/")
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
    def config(self) -> ModelConfig_HRM_V2:
        """ """
        return self._config

    def setup(  # --------------------------------------------------------------------------------
        self, stage: Optional[str] = None,
    ) -> None:  # fmt: skip
        """Lazy initialization of the environment to avoid GPU allocation issues in DDP."""
        world_size = max(getattr(self.trainer, "world_size", 1), 1)
        local_bs = self.config.global_batch_size // world_size

        if self.environment is None:
            self.environment = MazeHardEnv(self.config.environment, batch_size=local_bs)
        self.controller = RLController(self.model, self.environment, self.config.rl_controller)
        self.step_module = RLLossHead(self.controller, self.config.loss)

    def configure_optimizers(  # ------------------------------------------------------------------
        self,
    ) -> Tuple[List[Optimizer], List[SequentialLR]]:  # fmt: skip
        """ """
        total_steps = int(self.trainer.estimated_stepping_batches)
        sch_cfg = self._config.scheduler

        # Optimizer A: supervised — all model params EXCEPT pfc.estimator (vmPFC)
        vmPFC_ids = {id(p) for p in self.model.pfc.estimator.parameters()}
        sup_params = [p for p in self.model.parameters() if id(p) not in vmPFC_ids]
        opt_sup = AdamATan2(sup_params, self._config.optimizer_supervised)

        # Optimizer B: RL — STR actor-critic only (strictly isolated)
        opt_rl = AdamATan2(list(self.model.str.parameters()), self._config.optimizer_rl)

        # Optimizer C: vmPFC — pfc.estimator only (auxiliary Q-predictor)
        opt_qv = AdamATan2(list(self.model.pfc.estimator.parameters()), self._config.optimizer_qv)

        sch_sup = CosineAnnealingLRWithWarmup(opt_sup, total_steps, sch_cfg)
        sch_rl = CosineAnnealingLRWithWarmup(opt_rl, total_steps, sch_cfg)
        sch_qv = CosineAnnealingLRWithWarmup(opt_qv, total_steps, sch_cfg)

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
        """ """
        batch_dict = batch

        # Initialize carry/state on the first batch
        if self._train_carry is None:
            self._train_carry = self.step_module.initial_carry(batch_dict)

        # Assemble partial-reset step batch
        step_batch = self._train_batch_assembler.make_step_batch(
            incoming=batch_dict,
            reset_mask=self._train_carry.halted,
        )

        # Horizon=1 step loop: run one step of the controller
        step_batches = repeat(step_batch, 1)
        is_warmup = self.global_step < self._config.warmup_steps
        step_opts = {"explore": True, "allow_halt": not is_warmup, "is_warmup": is_warmup}
        carry0 = self._train_carry

        step = None
        for _t, step in StepLoop(self.step_module, step_batches, carry0, options=step_opts):
            pass  # horizon = 1; loop runs exactly once
        if step is None:
            raise ValueError("StepLoop did not yield any steps.")
        self._train_carry = step.carry.detach()

        # Normalize by local batch size; DDP averages gradients across ranks.
        local_bs = int(batch_dict["inputs"].shape[0])
        out = step.outputs
        loss = _normalize_loss_for_backward(out.loss, local_bs)
        self.manual_backward(loss)

        # Optimizer step and reset gradient for all optimizers
        opt_sup, opt_rl, opt_qv = self.optimizers()  # type: ignore[misc]
        sch_sup, sch_rl, sch_qv = self.lr_schedulers()  # type: ignore[misc]

        opt_sup.step(); opt_sup.zero_grad(set_to_none=True); sch_sup.step()  # fmt: skip
        opt_rl.step(); opt_rl.zero_grad(set_to_none=True); sch_rl.step()  # fmt: skip
        opt_qv.step(); opt_qv.zero_grad(set_to_none=True); sch_qv.step()  # fmt: skip

        # Update metrics with unnormalized loss and log to TensorBoard.
        update_metrics_from_step(self.train_metrics, step.outputs.metrics)
        self.log("train/loss", loss.detach(), on_step=True, on_epoch=False, prog_bar=True, logger=True)

        signals = {**step.outputs.signals, "is_warmup": torch.tensor(float(is_warmup))}
        return {"loss": loss.detach(), "signals": signals}

    # -- Validation --------------------------------------------------------------------------------

    def validation_step(  # -----------------------------------------------------------------------
        self, batch: Batch, batch_idx: int,
    ) -> Dict[str, Any]:  # fmt: skip
        """ """
        batch_dict = batch

        # Run a full rollout until all slots halt, collecting traces for logging/analysis.
        step_batches = repeat(batch_dict)  # Run until all examples halt
        step_opts = {"explore": False, "allow_halt": True, "is_warmup": False}
        carry0 = self.step_module.initial_carry(batch_dict)
        collector = TraceCollector(TraceTree(), self.trace_specs)

        step = None
        for t, step in StepLoop(self.step_module, step_batches, carry0, options=step_opts):
            collector.append(t, step)
        if step is None:
            raise ValueError("Evaluation loop did not yield any steps.")

        # Update metrics with the final step's metrics and log to TensorBoard.
        update_metrics_from_step(self.val_metrics, step.outputs.metrics)

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
