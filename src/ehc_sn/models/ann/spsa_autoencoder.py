import math
from typing import List, Tuple

import torch
from lightning import pytorch as pl
from pydantic import BaseModel, Field, field_validator
from torch import Tensor, flatten, nn, unflatten
from torch.optim import Adam, Optimizer

from ehc_sn.core import ann
from ehc_sn.core.trainer import BaseTrainer
from ehc_sn.modules import spsa
from ehc_sn.modules.loss import GramianOrthogonalityLoss as SparsityLoss


# -------------------------------------------------------------------------------------------
class ModelParams(BaseModel):
    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}

    # Architecture parameters
    latent_units: int = Field(default=32, gt=0, description="Dimensionality of the latent code.")
    layer2_units: int = Field(default=512, gt=0, description="Number of hidden units per layer.")
    layer1_units: int = Field(default=1024, gt=0, description="Number of hidden units per layer.")
    output_shape: List[int] = Field([25, 25], description="Dimensionality of the input and output.")

    # Training parameters
    encoder_lr: float = Field(2e-6, description="Learning rate for the encoder (used by optimizer).")
    decoder_lr: float = Field(1e-4, description="Learning rate for the decoder (used by optimizer).")

    def units(self) -> List[int]:
        """Return list of layer sizes from input to latent."""
        output_units = math.prod(self.output_shape)
        return [output_units, self.layer1_units, self.layer2_units, self.latent_units]

    @field_validator("output_shape")
    def check_output_shape(cls, v: List[int]) -> List[int]:
        if len(v) != 2:
            raise ValueError("output_shape must be a list of exactly 2 integers")
        if any(dim <= 0 for dim in v):
            raise ValueError("All dimensions in output_shape must be positive")
        return v


# -------------------------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, n_inputs: int, n_h1: int, n_h2: int, n_latents: int):
        super().__init__()
        self.layer1 = ann.Layer(spsa.Linear(n_inputs, n_h1), nn.GELU())
        self.layer2 = ann.Layer(spsa.Linear(n_h1, n_h2), nn.GELU())
        self.latent = ann.Layer(spsa.Linear(n_h2, n_latents), nn.GELU())

    def forward(self, sensors: Tensor) -> Tensor:
        x = self.layer1(sensors)
        x = self.layer2(x)
        return self.latent(x)

    def prepare_perturbation(self) -> None:
        self.layer1.synapses.prepare_perturbation()
        self.layer2.synapses.prepare_perturbation()
        self.latent.synapses.prepare_perturbation()

    def apply_perturbation(self, scale: float) -> None:
        self.layer1.synapses.apply_perturbation(scale)
        self.layer2.synapses.apply_perturbation(scale)
        self.latent.synapses.apply_perturbation(scale)

    def feedback(self, loss_diff: Tensor) -> None:
        # Convert global loss difference into per-layer SPSA coefficients
        c1 = loss_diff / (2.0 * self.layer1.synapses.epsilon)
        c2 = loss_diff / (2.0 * self.layer2.synapses.epsilon)
        c3 = loss_diff / (2.0 * self.latent.synapses.epsilon)
        self.layer1.synapses.feedback(c1)
        self.layer2.synapses.feedback(c2)
        self.latent.synapses.feedback(c3)


# -------------------------------------------------------------------------------------------
class Decoder(nn.Module):
    def __init__(self, n_outputs: int, n_h1: int, n_h2: int, n_latents: int):
        super().__init__()
        self.layer2 = ann.Layer(spsa.Linear(n_latents, n_h2), nn.GELU())
        self.layer1 = ann.Layer(spsa.Linear(n_h2, n_h1), nn.GELU())
        self.output = ann.Layer(spsa.Linear(n_h1, n_outputs), nn.Sigmoid())

    def forward(self, latent: Tensor) -> Tensor:
        """Forward pass through decoder layers."""
        x = self.layer2(latent)
        x = self.layer1(x)
        return self.output(x)

    def prepare_perturbation(self) -> None:
        self.layer2.synapses.prepare_perturbation()
        self.layer1.synapses.prepare_perturbation()
        self.output.synapses.prepare_perturbation()

    def apply_perturbation(self, scale: float) -> None:
        self.layer2.synapses.apply_perturbation(scale)
        self.layer1.synapses.apply_perturbation(scale)
        self.output.synapses.apply_perturbation(scale)

    def feedback(self, loss_diff: Tensor) -> None:
        # Convert global loss difference into per-layer SPSA coefficients
        c2 = loss_diff / (2.0 * self.layer2.synapses.epsilon)
        c1 = loss_diff / (2.0 * self.layer1.synapses.epsilon)
        co = loss_diff / (2.0 * self.output.synapses.epsilon)
        self.layer2.synapses.feedback(c2)
        self.layer1.synapses.feedback(c1)
        self.output.synapses.feedback(co)


# -------------------------------------------------------------------------------------------
class Autoencoder(pl.LightningModule):
    def __init__(self, params: ModelParams, trainer: BaseTrainer) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["trainer"])
        self.config = params
        self.automatic_optimization = False
        self.trainer_module = trainer

        # Initialize encoder and decoder with DFA layers
        self.encoder = Encoder(*params.units())
        self.decoder = Decoder(*params.units())

        # Loss functions
        self.reconstruction_loss = nn.BCELoss(reduction="mean")
        self.sparsity_loss = SparsityLoss(center=True)

    # -----------------------------------------------------------------------------------
    def configure_optimizers(self) -> Optimizer:
        optm_pe = {"params": self.encoder.parameters(), "lr": self.config.encoder_lr}
        optm_pd = {"params": self.decoder.parameters(), "lr": self.config.decoder_lr}
        return Adam([optm_pe, optm_pd])

    # -----------------------------------------------------------------------------------
    def forward(self, sensors: Tensor) -> List[Tensor]:
        latent = self.encoder(flatten(sensors, start_dim=1))
        reconstruction = unflatten(self.decoder(latent), 1, sensors.shape[1:])
        return [reconstruction, latent]

    # -----------------------------------------------------------------------------------
    def compute_loss(self, outputs: List[Tensor], batch: Tensor) -> List[Tensor]:
        sensors, *_ = batch
        reconstruction, latent = outputs
        reconstruction_loss = self.reconstruction_loss(reconstruction, sensors)
        sparsity_loss = self.sparsity_loss(latent)
        return [reconstruction_loss, 2e-1 * sparsity_loss]

    # -----------------------------------------------------------------------------------
    def prepare_perturbations(self) -> None:
        """Prepare perturbations for all SPSA layers."""
        self.encoder.prepare_perturbation()
        self.decoder.prepare_perturbation()

    # -----------------------------------------------------------------------------------
    def apply_perturbations(self, scale: float) -> None:
        """Apply scaled perturbation to all SPSA layers."""
        self.encoder.apply_perturbation(scale)
        self.decoder.apply_perturbation(scale)

    # -----------------------------------------------------------------------------------
    @torch.no_grad()
    def compute_feedback(self, outputs: List[Tensor], batch: Tensor) -> List[Tensor]:
        sensors, *_ = batch
        self.prepare_perturbations()

        # +epsilon
        self.apply_perturbations(+1.0)
        outputs_plus = self(sensors)
        Lp = sum(self.compute_loss(outputs_plus, batch))

        # -epsilon
        self.apply_perturbations(-2.0)
        outputs_minus = self(sensors)
        Lm = sum(self.compute_loss(outputs_minus, batch))

        # restore
        self.apply_perturbations(+1.0)

        numerator = (Lp - Lm).detach()  # scalar
        return [numerator]

    # -----------------------------------------------------------------------------------
    def apply_feedback(self, feedback: List[Tensor]) -> None:
        (numerator,) = feedback
        self.encoder.feedback(numerator)
        self.decoder.feedback(numerator)

    # -----------------------------------------------------------------------------------
    def training_step(self, batch: Tensor, batch_idx: int) -> None:
        # Delegates to the trainer strategy (feed-forward feedback)
        self.trainer_module.training_step(self, batch, batch_idx)

    # -----------------------------------------------------------------------------------
    def validation_step(self, batch: Tensor, batch_idx: int) -> List[Tensor]:
        outputs = self(batch[0])
        reconstruction_loss = nn.MSELoss(reduction="mean")(outputs[0], batch[0])
        sparsity_rate = (outputs[1] > 0.01).float().mean()
        self.log("val/sparsity_rate", sparsity_rate, prog_bar=True)
        self.log("val/reconstruction_loss", reconstruction_loss, prog_bar=True)


# -------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
