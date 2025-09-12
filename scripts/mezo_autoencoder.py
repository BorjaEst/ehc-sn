"""
Standalone MeZO (zero-order) training example.

This script trains a tiny MLP on a synthetic regression task using a
two-point finite-difference estimator (MeZO-style) with forward-only passes.

All logic and constants are in this file. No arguments required.
"""

import math
from typing import List, Tuple

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from ehc_sn import utils
from ehc_sn.modules import zo
from ehc_sn.trainers.zeroth_order import ZOTrainer

# -------------------------------------------------------------------------------------------
# Constants (no CLI)
# -------------------------------------------------------------------------------------------

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_SAMPLES = 8192
INPUT_DIM = 128
HIDDEN_DIM = 256
LATENT_DIM = 16
BATCH_SIZE = 128
EPOCHS = 200
LR = 1e-3
RECONSTRUCTION_WEIGHT = 1.0  # Weight for reconstruction loss
SPARSITY_TARGET = 0.05  # Target sparsity level (5% activation)
SPARSITY_WEIGHT = 0.01  # Weight for sparsity loss (reduced for stability)


# -------------------------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------------------------


def create_synthetic_dataset(n: int, input_dim: int, latent_dim: int) -> Tuple[Tensor, Tensor]:
    """Generate structured synthetic data for autoencoder training as probabilities.

    Creates data with underlying low-dimensional structure that can be compressed:
    - Sinusoidal patterns with different frequencies mapped to [0, 1] using sigmoid
    - Correlated features that form meaningful probability representations
    - All values are constrained to probability range [0, 1]

    Returns input data where target = input for reconstruction task.
    """
    g = torch.Generator().manual_seed(SEED + 1)

    # Create low-dimensional latent factors (much smaller than input dim)
    latent = torch.randn(n, latent_dim, generator=g)

    # Generate structured data from latent factors
    x = torch.zeros(n, input_dim)

    # Create sinusoidal patterns with different frequencies
    for i in range(input_dim):
        freq = 0.5 + (i / input_dim) * 2.0  # Frequencies from 0.5 to 2.5
        phase = (i / input_dim) * 2 * math.pi  # Different phases

        # Combine multiple latent dimensions with different weights
        signal = torch.zeros(n)
        for j in range(latent_dim):
            weight = torch.sin(torch.tensor(i * j * math.pi / input_dim))
            signal += weight * latent[:, j]

        # Apply sigmoid to map sinusoidal patterns to [0, 1] probability range
        x[:, i] = torch.sigmoid(freq * torch.sin(signal + phase))

    # Add small amount of noise appropriate for probability values
    noise = 0.02 * torch.randn(x.shape, generator=g)
    x = torch.clamp(x + noise, 0.0, 1.0)  # Ensure values stay in [0, 1]

    return x, x


class Encoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.fc1 = zo.Linear(input_dim, HIDDEN_DIM, epsilon=1e-3)
        self.fc2 = zo.Linear(HIDDEN_DIM, latent_dim, epsilon=1e-3)

    def forward(self, x: Tensor, seed=None) -> Tensor:
        seeds = utils.seeds(2, seed, x.device) if seed else []
        x = torch.relu(self.fc1(x, seeds[0] if seed else None))
        x = torch.relu(self.fc2(x, seeds[1] if seed else None))
        return x

    def feedback(self, x: Tensor) -> Tensor:
        self.fc2.feedback(x)
        self.fc1.feedback(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, input_dim: int) -> None:
        super().__init__()
        self.fc1 = zo.Linear(latent_dim, HIDDEN_DIM, epsilon=1e-3)
        self.fc2 = zo.Linear(HIDDEN_DIM, input_dim, epsilon=1e-3)

    def forward(self, x: Tensor, seed=None) -> Tensor:
        seeds = utils.seeds(2, seed, x.device) if seed else []
        x = torch.relu(self.fc1(x, seeds[0] if seed else None))
        x = torch.sigmoid(self.fc2(x, seeds[1] if seed else None))
        return x

    def feedback(self, x: Tensor) -> Tensor:
        self.fc2.feedback(x)
        self.fc1.feedback(x)


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

    def optimizers(self) -> List[torch.optim.Optimizer]:
        return [
            torch.optim.Adam(self.encoder.parameters(), lr=LR),
            torch.optim.Adam(self.decoder.parameters(), lr=LR),
        ]

    def compute_loss(self, output, batch, log_label: str = None) -> List[Tensor]:
        x, _ = batch
        reconstructed, encoded = output

        # Reconstruction loss
        reconstruction_loss = nn.functional.binary_cross_entropy(reconstructed, x, reduction="mean")
        reconstruction_loss *= RECONSTRUCTION_WEIGHT

        # Simpler sparsity loss using L1 penalty with target sparsity
        # Apply ReLU to get non-negative activations for sparsity calculation
        avg_activation = torch.mean(encoded, dim=0)  # Average across batch

        # L1 penalty on deviation from target sparsity
        sparsity_loss = torch.mean(torch.abs(avg_activation - SPARSITY_TARGET))
        sparsity_loss *= SPARSITY_WEIGHT

        total_loss = reconstruction_loss + sparsity_loss
        return [reconstruction_loss, total_loss]  # Return total and reconstruction for logging

    def forward(self, x: Tensor, seed=None, detach_grad=None) -> List[Tensor]:
        seeds = utils.seeds(2, seed, x.device) if seed else []
        encoded = self.encoder(x, seeds[0] if seed else None)
        decoded = self.decoder(encoded, seeds[1] if seed else None)
        return [decoded, encoded]

    def feedback(self, x: Tensor) -> Tensor:
        self.decoder.feedback(x[0])
        self.encoder.feedback(x[1])


def mezo_step(model: nn.Module, batch: Tuple[Tensor, Tensor], trainer: ZOTrainer) -> None:
    """Perform one MeZO update using two forward passes."""
    model.eval()  # no autograd graph needed
    trainer.training_step(model, batch, 0)  # This should perform the actual parameter update


def mezo_epoch(model, trainer, dl, X, Y) -> List[Tensor]:
    """One epoch of MeZO training."""

    # Training loop (forward-only)
    n_batches = 0
    for xb, yb in dl:
        mezo_step(model, (xb, yb), trainer)
        n_batches += 1

    # Validation with both total and reconstruction loss
    return model.compute_loss(model(X), (X, Y))


def main() -> None:
    # Data
    X, Y = create_synthetic_dataset(N_SAMPLES, INPUT_DIM, LATENT_DIM)
    X, Y = X.to(DEVICE), Y.to(DEVICE)
    ds = TensorDataset(X, Y)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # Model
    model = Autoencoder(INPUT_DIM, LATENT_DIM).to(DEVICE)
    trainer = ZOTrainer(optimizer_init=None)

    # Training loop (forward-only)
    for epoch in range(1, EPOCHS + 1):
        losses = mezo_epoch(model, trainer, dl, X, Y)  # Perform one epoch of MeZO training

        # Print both total and reconstruction losses
        total_loss, recon_loss = losses[0].item(), losses[1].item()
        sparsity_loss = total_loss - recon_loss
        print(f"Epoch {epoch:02d} | Total: {total_loss:.6f} | Recon: {recon_loss:.6f} | Sparsity: {sparsity_loss:.6f}")


if __name__ == "__main__":
    main()
