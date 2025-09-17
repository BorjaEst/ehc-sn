from typing import Any, Callable, List

import lightning.pytorch as pl
import torch
from torch import Tensor
from torch.optim import Optimizer

from ehc_sn.core.trainer import BaseTrainer
from ehc_sn.hooks.registry import registry


# -------------------------------------------------------------------------------------------
class DFATrainer(BaseTrainer):

    # -----------------------------------------------------------------------------------
    def training_step(self, model: pl.LightningModule, batch: Tensor, batch_idx: int) -> None:
        raise NotImplementedError("DFA training strategy is not yet implemented.")


# -------------------------------------------------------------------------------------------
class FeedbackTainer(BaseTrainer):

    # -----------------------------------------------------------------------------------
    def training_step(self, model: pl.LightningModule, batch: Tensor, batch_idx: int) -> None:
        sensors, *_ = batch

        # Clear any previous gradients from model parameters
        optimizers = model.optimizers()
        if not isinstance(optimizers, list):
            optimizers = [optimizers]

        for optm in optimizers:
            optm.zero_grad()

        # Forward pass and compute feedback signals
        outputs = model(sensors)
        feedback = model.compute_feedback(outputs, batch)

        # Calculate gradients using feedback signals
        model.apply_feedback(feedback)

        # Optimizer step
        for optm in optimizers:
            optm.step()
