import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, Union

import lightning.pytorch as pl
from lightning import pytorch as pl
from pydantic import BaseModel, Field, field_validator
from torch import Tensor, nn, utils, zeros_like
from torch.optim import Adam, Optimizer

from ehc_sn.core import ann
from ehc_sn.core.trainer import BaseTrainer
from ehc_sn.modules import drtp
from ehc_sn.modules.loss import GramianOrthogonalityLoss as SparsityLoss


# -------------------------------------------------------------------------------------------
class ModelParams(BaseModel):
    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}

    # Encoder and decoder components
    latent_units: int = Field(default=32, gt=0, description="Dimensionality of the latent code.")
    layer2_units: int = Field(default=512, gt=0, description="Number of hidden units per layer.")
    layer1_units: int = Field(default=1024, gt=0, description="Number of hidden units per layer.")
    output_shape: List[int] = Field([25, 25], description="Dimensionality of the input and output.")

    # Gramian orthogonality loss parameters
    sparsity_weight: float = Field(5e-2, ge=0, le=1, description="Weight for Gramian orthogonality term.")
    gramian_center: bool = Field(True, description="Center activations before normalization.")

    # Training parameters
    encoder_lr: float = Field(2e-6, description="Learning rate for the encoder.")
    decoder_lr: float = Field(1e-4, description="Learning rate for the decoder.")

    def units(self) -> List[int]:
        """Return list of layer sizes from input to latent."""
        output_units = math.prod(self.output_shape)
        return [output_units, self.layer1_units, self.layer2_units, self.latent_units]

    @field_validator("output_shape")
    def check_output_shape(cls, v: List[int]) -> List[int]:
        if any(dim <= 0 for dim in v):
            raise ValueError("output_shape must be a list of positive integers; e.g. [H, W, C]")
        return v


# -------------------------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, n_inputs: int, n_h1: int, n_h2: int, n_latents: int):
        super().__init__()
        self.layer1 = ann.Layer(drtp.Linear(n_inputs, n_h1, target_features=n_inputs), nn.GELU())
        self.layer2 = ann.Layer(drtp.Linear(n_h1, n_h2, target_features=n_inputs), nn.GELU())
        self.latent = ann.Layer(drtp.Linear(n_h2, n_latents, target_features=n_inputs), nn.GELU())

    def forward(self, sensors: Tensor) -> Tensor:
        x = self.layer1(sensors)
        x = self.layer2(x)
        return self.latent(x)

    def feedback(self, target: Tensor) -> None:
        self.layer2.synapses.feedback(target, context=self.layer2.neurons)
        self.layer1.synapses.feedback(target, context=self.layer1.neurons)


# -------------------------------------------------------------------------------------------
class Decoder(nn.Module):
    def __init__(self, n_outputs: int, n_h1: int, n_h2: int, n_latents: int):
        super().__init__()
        self.layer2 = ann.Layer(drtp.Linear(n_latents, n_h2, target_features=n_outputs), nn.GELU())
        self.layer1 = ann.Layer(drtp.Linear(n_h2, n_h1, target_features=n_outputs), nn.GELU())
        self.output = ann.Layer(nn.Linear(n_h1, n_outputs), nn.Sigmoid())

    def forward(self, latent: Tensor) -> Tensor:
        x = self.layer2(latent)
        x = self.layer1(x)
        return self.output(x.detach())

    def feedback(self, target: Tensor) -> None:
        self.layer2.synapses.feedback(target, context=self.layer2.neurons)
        self.layer1.synapses.feedback(target, context=self.layer1.neurons)


# -------------------------------------------------------------------------------------------
class Autoencoder(pl.LightningModule):
    def __init__(self, params: ModelParams, trainer: BaseTrainer) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.encoder = Encoder(*params.units())
        self.decoder = Decoder(*params.units())
        self.unflatten = nn.Unflatten(1, params.output_shape)
        self.config = params
        self.automatic_optimization = False
        self.trainer_strategy = trainer

    def configure_optimizers(self) -> Optimizer:
        optm_pe = {"params": self.encoder.parameters(), "lr": self.config.encoder_lr}
        optm_pd = {"params": self.decoder.parameters(), "lr": self.config.decoder_lr}
        return Adam([optm_pe, optm_pd])

    def forward(self, sensors: Tensor) -> Tuple[Tensor, Tensor]:
        latent = self.encoder(self.flatten(sensors))
        reconstruction = self.unflatten(self.decoder(latent))
        return reconstruction, latent

    def compute_feedback(self, outputs: Tensor, batch: Tensor) -> List[Tensor]:
        reconstruction, latent = outputs
        sensors, *_ = batch
        return [sensors, reconstruction, latent]

    def apply_feedback(self, feedback: List[Tensor]) -> None:
        (sensors, reconstruction, latent) = feedback
        self.encoder.feedback(self.flatten(sensors))
        SparsityLoss(self.config.gramian_center)(latent).backward()
        self.decoder.feedback(self.flatten(sensors))
        nn.BCELoss(reduction="mean")(reconstruction, sensors).backward()

    def training_step(self, batch: Tensor, batch_idx: int) -> None:
        self.trainer_strategy.training_step(self, batch, batch_idx)

    def validation_step(self, batch: Tensor, batch_idx: int) -> List[Tensor]:
        outputs = self(batch[0])
        reconstruction_loss = nn.MSELoss(reduction="mean")(outputs[0], batch[0])
        sparsity_rate = (outputs[1] > 0.01).float().mean()
        self.log("val/sparsity_rate", sparsity_rate, prog_bar=True)
        self.log("val/reconstruction_loss", reconstruction_loss, prog_bar=True)


# -------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Usage example
    pass  # TODO
