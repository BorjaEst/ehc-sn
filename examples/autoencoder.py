import math
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader, TensorDataset

# -------------------------------------------------------------------------------------------
# Constants (no CLI)
# -------------------------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_SAMPLES = 320
INPUT_DIM = 256
HIDDEN_DIM = 512
LATENT_DIM = 64
BATCH_SIZE = 16
EPOCHS = 80
LR = 1e-4
SPARSITY_WEIGHT = 0.1
SPARSITY_TARGET = 0.2


# -------------------------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------------------------


def create_synthetic_dataset(n: int, input_dim: int, latent_dim: int) -> Tuple[Tensor, Tensor]:
    # Create sparse latent factors
    latent = torch.zeros(n, latent_dim)

    # Each sample activates only a subset of latent dimensions (sparsity)
    n_active_per_sample = max(1, latent_dim // 4)  # 25% sparsity in latent space

    for i in range(n):
        # Randomly select which latent dimensions to activate
        active_dims = torch.randperm(latent_dim)[:n_active_per_sample]
        latent[i, active_dims] = torch.randn(n_active_per_sample).abs()  # Positive activations

    # Create feature groups - each latent dimension controls a group of input features
    features_per_latent = input_dim // latent_dim
    x = torch.zeros(n, input_dim)

    for lat_idx in range(latent_dim):
        # Define which input features this latent dimension controls
        start_feat = lat_idx * features_per_latent
        end_feat = min(start_feat + features_per_latent, input_dim)

        # Generate patterns for this feature group
        for feat_idx in range(start_feat, end_feat):
            # Create distinct patterns with some correlation within groups
            pattern_weight = 0.8 + 0.4 * torch.sin(torch.tensor(feat_idx * math.pi / features_per_latent))
            x[:, feat_idx] = latent[:, lat_idx] * pattern_weight

    # Add small amount of noise and ensure values are in [0, 1]
    noise = 0.05 * torch.randn(x.shape)
    x = torch.clamp(x + noise, 0.0, 1.0)

    return x, x


def make_figure(original: Tensor, reconstruction: Tensor, n_samples: int = 4) -> None:
    """Plot original batch data vs reconstructions side by side."""
    # Convert to numpy for plotting
    x = original.cpu().numpy()
    y = reconstruction.cpu().numpy()

    # Limit number of samples to plot
    n_samples = min(n_samples, original.shape[0])

    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 3, 6))
    if n_samples == 1:
        axes = axes.reshape(2, 1)

    for i in range(n_samples):
        # Plot original data
        axes[0, i].plot(x[i], "b-", linewidth=1, alpha=0.7, label="Original")
        axes[0, i].set_title(f"Sample {i+1}")
        axes[0, i].set_ylim(0, 1)
        axes[0, i].grid(True, alpha=0.3)
        if i == 0:
            axes[0, i].set_ylabel("Original")

        # Plot reconstruction
        axes[1, i].plot(y[i], "r-", linewidth=1, alpha=0.7, label="Reconstruction")
        axes[1, i].set_ylim(0, 1)
        axes[1, i].set_xlabel("Feature Index")
        axes[1, i].grid(True, alpha=0.3)
        if i == 0:
            axes[1, i].set_ylabel("Reconstructed")

    # Calculate and display MSE in title
    mse = mse_loss(reconstruction, original).item()
    plt.suptitle(f"Data vs Reconstruction (MSE: {mse:.4f})")
    plt.tight_layout()
    plt.show()


def mse_loss(activations: Tensor, sensors: Tensor) -> Tensor:
    return nn.functional.mse_loss(activations, sensors, reduction="mean")


def kl_loss(embedding: Tensor) -> Tensor:
    avg_activation = torch.mean(embedding, dim=0)  # Average across batch
    return torch.mean(torch.abs(avg_activation - SPARSITY_TARGET))


# -------------------------------------------------------------------------------------------
# Model
# -------------------------------------------------------------------------------------------


class Layer(nn.Module):
    def __init__(self, units: int, synapses: Dict[str, int]):
        super().__init__()
        self.synapses = nn.ModuleDict({k: nn.Linear(v, units) for k, v in synapses.items()})
        self.register_buffer("neurons", torch.zeros(BATCH_SIZE, units))  # State of neurons
        self.activation = nn.ReLU()

    def forward(self, x: Dict[str, Tensor]) -> Tensor:
        currents = self.currents(x)
        self.neurons = self.activation(currents)
        return self.neurons

    def currents(self, x: Dict[str, Tensor]) -> Tensor:
        total_current = torch.zeros_like(self.neurons)
        for syn in x:
            # Detach inputs so grads don't flow back into the sending unit
            total_current += self.synapses[syn](x[syn].detach())
        return total_current

    def target(self, y: Dict[str, Tensor]) -> None:
        loss: Tensor = torch.tensor(0.0, device=self.neurons.device)
        for syn in y:  # Detach fb weights to prevent gradient flow to fb provider
            fb_weights = self.synapses[syn].weight.detach()
            projected_target = torch.matmul(y[syn].detach(), fb_weights.t())
            loss += mse_loss(self.neurons, projected_target)
        loss.backward(retain_graph=True)  # Backprop through this layer only

    def dfa(self, e: Dict[str, Tensor]) -> None:
        delta: Tensor = torch.zeros_like(self.neurons)
        for syn in e:  # Detach fb weights to prevent gradient flow to fb provider
            fb_weights = self.synapses[syn].weight.detach()
            delta += torch.matmul(e[syn], fb_weights.t())
        torch.autograd.backward(self.neurons, delta, retain_graph=True)


class Encoder(nn.Module):
    def __init__(self, n_sensors: int, n_hidden: List[int], n_latents: int):
        super().__init__()
        self.layer1 = Layer(n_hidden[0], {"inputs": n_sensors, "fb": n_sensors})
        self.layer2 = Layer(n_hidden[1], {"inputs": n_hidden[0], "fb": n_sensors})
        self.latent = Layer(n_latents, {"inputs": n_hidden[1], "fb": n_sensors})

    def forward(self, sensors: Tensor) -> Tensor:
        x = self.layer1({"inputs": sensors})
        x = self.layer2({"inputs": x})
        return self.latent({"inputs": x})

    def feedback(self, sensors: Tensor, decoder: "Decoder") -> Tensor:
        global_error = (decoder.output.neurons - sensors).detach()
        self.layer1.dfa({"fb": global_error})
        self.layer2.dfa({"fb": global_error})
        self.latent.dfa({"fb": global_error})


class Decoder(nn.Module):
    def __init__(self, n_sensors: int, n_hidden: List[int], n_latents: int):
        super().__init__()
        self.layer2 = Layer(n_hidden[1], {"inputs": n_latents, "fb": n_hidden[1]})
        self.layer1 = Layer(n_hidden[0], {"inputs": n_hidden[1], "fb": n_hidden[0]})
        self.output = Layer(n_sensors, {"inputs": n_hidden[0], "fb": n_sensors})

    def forward(self, latent: Tensor) -> Tensor:
        x = self.layer2({"inputs": latent})
        x = self.layer1({"inputs": x})
        return self.output({"inputs": x})

    def feedback(self, sensors: Tensor, encoder: Encoder) -> None:
        self.layer1.target({"fb": encoder.layer1.neurons})
        self.layer2.target({"fb": encoder.layer2.neurons})
        self.output.target({"fb": sensors})


class Autoencoder(nn.Module):
    def __init__(self, n_sensors: int, n_latents: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = Encoder(n_sensors, [HIDDEN_DIM, HIDDEN_DIM], n_latents)
        self.decoder = Decoder(n_sensors, [HIDDEN_DIM, HIDDEN_DIM], n_latents)
        self.sorting = nn.Linear(n_sensors, n_sensors)  # Ideally SE(n) trans

    def forward(self, sensors: Tensor) -> Tuple[Tensor, Tensor]:
        latent = self.encoder(sensors)
        reconstruction = self.decoder(latent)
        return self.sorting(reconstruction), latent

    def feedback(self, sensors: Tensor) -> List[Tensor]:
        self.encoder.feedback(sensors, self.decoder)
        self.decoder.feedback(sensors, self.encoder)
        # Allow gradients from reconstruction MSE to flow into the decoder by not detaching
        sorted_out = self.sorting(self.decoder.output.neurons)
        # Retain graph so we can also backpropagate sparsity loss afterward
        mse_loss(sorted_out, sensors).backward(retain_graph=True)
        kl_loss(self.encoder.latent.neurons).backward()


# -------------------------------------------------------------------------------------------
# Training
# -------------------------------------------------------------------------------------------


def train_step(optm: Optimizer, model: nn.Module, batch) -> List[Tensor]:
    # Single training step without autograd
    model.train()

    # Extract the sensors information
    sensors, _ = batch

    # Forward pass through the EHC model
    optm.zero_grad()
    output, embedding = model(sensors)
    model.feedback(sensors)
    optm.step()

    # Compute losses for logging
    loss_reconstruction = mse_loss(output, sensors)
    loss_sparsity = kl_loss(embedding)

    return [loss_reconstruction, loss_sparsity]


def train_epoch(optm: Optimizer, model: nn.Module, dl: DataLoader, epoch: int) -> List[Tensor]:
    # Training loop (forward-only)
    n_batches = 0
    for xb, yb in dl:
        loss_reconstruction, loss_sparsity = train_step(optm, model, (xb, yb))
        n_batches += 1

    # Validation with both total and reconstruction loss
    l1, l2 = loss_reconstruction.item(), loss_sparsity.item()
    print(f"Epoch {epoch:02d} | Recon: {l1:.6f} | Sparsity: {l2:.6f}")


def train_loop(optm: Optimizer, model: nn.Module, dl: DataLoader) -> None:
    # Training loop (forward-only)
    for epoch in range(1, EPOCHS + 1):
        train_epoch(optm, model, dl, epoch)


# -------------------------------------------------------------------------------------------
# Testing
# -------------------------------------------------------------------------------------------


def test_model(model: nn.Module, X: Tensor, Y: Tensor) -> None:
    model.eval()
    with torch.no_grad():
        xb, yb = next(iter(DataLoader(TensorDataset(X, Y), batch_size=BATCH_SIZE)))
        output, embeding = model(xb)
        l1, l2 = mse_loss(output, xb), kl_loss(embeding)
        print(f"Test | Recon: {l1:.6f} | Sparsity: {l2:.6f}")


# -------------------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------------------

# Generate synthetic dataset
X, Y = create_synthetic_dataset(N_SAMPLES, INPUT_DIM, LATENT_DIM)
X, Y = X.to(DEVICE), Y.to(DEVICE)
ds = TensorDataset(X, Y)
dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# Instantiate the EHC model
model = Autoencoder(INPUT_DIM, LATENT_DIM).to(DEVICE)
print(model)
test_model(model, X, Y)

# Training loop
optm = Adam(model.parameters(), lr=LR)
train_loop(optm, model, dl)

# Test the model
test_model(model, X, Y)

# Generate visualization
model.eval()
with torch.no_grad():
    sensors = X[:BATCH_SIZE]
    output, _ = model(sensors)

# Plot results
make_figure(sensors, output)
