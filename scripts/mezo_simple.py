"""
Standalone MeZO (zero-order) training example.

This script trains a tiny MLP on a synthetic regression task using a
two-point finite-difference estimator (MeZO-style) with forward-only passes.

All logic and constants are in this file. No arguments required.
"""

import math
from functools import partial
from typing import Iterable, List, Sequence, Tuple

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
IN_DIM = 16
HID_DIM = 64
OUT_DIM = 4
BATCH_SIZE = 128
EPOCHS = 200
LR = 1e-2


# -------------------------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------------------------


def create_synthetic_dataset(n: int, in_dim: int, out_dim: int) -> Tuple[Tensor, Tensor]:
    """Linear mapping with a mild nonlinearity and noise.

    y = W2 * relu(W1 x + b1) + b2 + noise
    """
    g = torch.Generator().manual_seed(SEED + 1)
    x = torch.randn(n, in_dim, generator=g)

    W1 = torch.randn(in_dim, 2 * out_dim, generator=g) / math.sqrt(in_dim)
    b1 = torch.randn(2 * out_dim, generator=g) * 0.1
    W2 = torch.randn(2 * out_dim, out_dim, generator=g) / math.sqrt(2 * out_dim)
    b2 = torch.randn(out_dim, generator=g) * 0.1

    hidden = torch.relu(x @ W1 + b1)
    y = hidden @ W2 + b2
    y += 0.05 * torch.randn(y.shape, generator=g)
    return x, y


class TinyMLP(nn.Module):
    def __init__(self, in_dim: int, hid: int, out_dim: int) -> None:
        super().__init__()
        layers = [
            zo.Linear(in_dim, hid, epsilon=1e-3),
            nn.ReLU(),
            zo.Linear(hid, out_dim, epsilon=1e-3),
        ]
        self.net = nn.ModuleList(layers)
        self.encoder = self.net[0]
        self.decoder = self.net[2]

    def optimizers(self) -> List[torch.optim.Optimizer]:
        return [
            torch.optim.SGD(self.net[0].parameters(), lr=LR),
            torch.optim.SGD(self.net[2].parameters(), lr=LR),
        ]

    def compute_loss(self, output, batch, log_label: str) -> List[Tensor]:
        _, y = batch
        loss_1 = nn.functional.mse_loss(output[0], y, reduction="mean")
        return [loss_1, loss_1]  # Dummy second loss for compatibility

    def forward(self, x: Tensor, seed=None, detach_grad=None) -> List[Tensor]:
        seeds = utils.seeds(2, seed, x.device) if seed else []
        h1 = self.net[0](x, seeds[0] if seed else None)
        h2 = self.net[1](h1)
        h3 = self.net[2](h2, seeds[1] if seed else None)
        return [h3, h1]

    def feedback(self, x: Tensor) -> Tensor:
        self.net[0].feedback(x[0])
        self.net[2].feedback(x[0])


def mezo_step(model: nn.Module, batch: Tuple[Tensor, Tensor], trainer: ZOTrainer) -> Tuple[float, float, float]:
    """Perform one MeZO update using two forward passes.

    Returns (loss_plus, loss_minus, g_scalar) for logging.
    """
    model.eval()  # no autograd graph needed
    x, y = batch
    batch_idx = 0  # Dummy batch index for trainer example
    trainer.training_step(model, batch, batch_idx)  # Example usage of the trainer


def main() -> None:
    # Data
    X, Y = create_synthetic_dataset(N_SAMPLES, IN_DIM, OUT_DIM)
    X, Y = X.to(DEVICE), Y.to(DEVICE)
    ds = TensorDataset(X, Y)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # Model
    model = TinyMLP(IN_DIM, HID_DIM, OUT_DIM).to(DEVICE)
    trainer = ZOTrainer(optimizer_init=None)

    # Training loop (forward-only)
    for epoch in range(1, EPOCHS + 1):
        n_batches = 0
        for xb, yb in dl:
            mezo_step(model, (xb, yb), trainer)
            n_batches += 1

        avg_loss = trainer.validation_step(model, (X, Y), 0).item()  # Full dataset validation
        print(f"Epoch {epoch:02d} | Avg MSE: {avg_loss:.6f}")


if __name__ == "__main__":
    main()
