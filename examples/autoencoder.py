import math
from typing import Any, Dict, Iterable, List, Tuple

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
LATENT_DIM = 32
BATCH_SIZE = 16
EPOCHS = 80


# -------------------------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------------------------


def create_synthetic_dataset(n: int, input_dim: int, latent_dim: int) -> Tuple[Tensor, Tensor]:
    # Create sparse latent factors
    g = torch.Generator().manual_seed(50)
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


class ModuleFB(nn.Module):
    def __init__(self, units: int, synapses: Dict[str, int]):
        super().__init__()
        for k, v in synapses.items():
            fb_weights_shape = torch.Size([units, v])
            self.register_buffer(k, torch.Tensor(fb_weights_shape))
        self.reset_weigths()

    def __getitem__(self, key: str) -> Tensor:
        return getattr(self, key)

    def keys(self) -> Iterable[str]:
        return self._buffers.keys()

    def items(self) -> Iterable[Tuple[str, Tensor]]:
        for k in self.keys():
            yield k, self[k]

    def reset_weigths(self):
        for k in self.keys():
            torch.nn.init.kaiming_uniform_(self[k])
            with torch.no_grad():
                self[k].div_(self[k].norm(dim=1, keepdim=True).clamp_min(1e-6))
                self[k].mul_(0.1)


def mse_loss(activations: Tensor, sensors: Tensor) -> Tensor:
    return nn.functional.mse_loss(activations, sensors, reduction="mean")


def rate_loss(activations: Tensor, rate: float = 0.1) -> Tensor:
    return ((activations.mean(dim=0) - rate) ** 2).mean()


# -------------------------------------------------------------------------------------------
# Model
# -------------------------------------------------------------------------------------------


class Layer(nn.Module):
    def __init__(self, units: int, x: Dict[str, int], fb: Dict[str, int] = None, activation: nn.Module = None):
        super().__init__()
        self.synapses = nn.ModuleDict({k: nn.Linear(v, units) for k, v in x.items()})
        self.feedback = ModuleFB(units, synapses=fb) if fb is not None else None
        self.units = units
        self.register_buffer("currents", torch.zeros(BATCH_SIZE, units))  # State of currents
        self.register_buffer("neurons", torch.zeros(BATCH_SIZE, units))  # State of neurons
        self.activation = nn.Tanh() if activation is None else activation

    def forward(self, x: Dict[str, Tensor]) -> Tensor:
        self.currents = 1e-3 * torch.ones_like(self.currents)  # Small leak to avoid dead neurons
        for syn in x:  # Loop over input synapse types to calculate total current
            self.currents += self.synapses[syn](x[syn].detach())  # Detach pre-synaptic
        self.neurons = self.activation(self.currents)  # Update neuron states
        return self.neurons

    def dfa(self, e: Dict[str, Tensor]) -> None:
        delta = torch.zeros_like(self.currents)
        for syn in e:  # Loop over feedback error types to calculate total delta
            delta += torch.matmul(e[syn], self.feedback[syn].t())
        torch.autograd.backward(self.currents, delta)

    def target(self, x: Dict[str, Tensor], target: Tensor) -> None:
        ctxt_currents = torch.zeros_like(self.currents)
        for syn in x:  # Loop over input synapse types to calculate total current
            ctxt_currents += self.synapses[syn](x[syn].detach())  # Detach pre-synaptic
        mse_loss(ctxt_currents, target.detach()).backward()


class Encoder(nn.Module):
    def __init__(self, n_sensors: int, n_hidden: List[int], n_latents: int):
        super().__init__()
        self.layer1 = Layer(n_hidden[0], {"input": n_sensors}, {"dfa": n_sensors, "fr": n_latents})
        self.layer2 = Layer(n_hidden[1], {"input": n_hidden[0]}, {"dfa": n_sensors, "fr": n_latents})
        self.latent = Layer(n_latents, {"input": n_hidden[1]}, {"dfa": n_sensors, "fr": n_latents})

    def forward(self, sensors: Tensor) -> Tensor:
        x = self.layer1({"input": sensors})
        x = self.layer2({"input": x})
        return self.latent({"input": x})

    def feedback(self, reconstruction: Tensor) -> None:
        self.latent.dfa({"dfa": reconstruction})
        self.layer2.dfa({"dfa": reconstruction})
        self.layer1.dfa({"dfa": reconstruction})


class Decoder(nn.Module):
    def __init__(self, n_sensors: int, n_hidden: List[int], n_latents: int):
        super().__init__()
        self.layer2 = Layer(n_hidden[1], {"inputs": n_latents})
        self.layer1 = Layer(n_hidden[0], {"inputs": n_hidden[1]})
        self.output = Layer(n_sensors, {"inputs": n_hidden[0]}, activation=nn.GELU())

    def forward(self, latent: Tensor) -> Tensor:
        x = self.layer2({"inputs": latent})
        x = self.layer1({"inputs": x})
        return self.output({"inputs": x})

    def feedback(self, sensors: Tensor, encoder: Encoder) -> None:
        self.layer2.target({"inputs": encoder.latent.neurons}, target=encoder.layer2.currents)
        self.layer1.target({"inputs": encoder.layer2.neurons}, target=encoder.layer1.currents)
        self.output.target({"inputs": encoder.layer1.neurons}, target=sensors)


class Autoencoder(nn.Module):
    def __init__(self, n_sensors: int, n_latents: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = Encoder(n_sensors, [HIDDEN_DIM, HIDDEN_DIM], n_latents)
        self.decoder = Decoder(n_sensors, [HIDDEN_DIM, HIDDEN_DIM], n_latents)

    def forward(self, sensors: Tensor) -> Tuple[Tensor, Tensor]:
        latent = self.encoder(sensors)
        reconstruction = self.decoder(latent)
        return reconstruction, latent

    def feedback(self, sensors: Tensor, error: Tensor) -> None:
        self.encoder.feedback(error)
        self.decoder.feedback(sensors, self.encoder)


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

    # Reconstruction error (normalize)
    error = (output - sensors).detach()
    error -= error.mean(dim=0, keepdim=True)
    error /= error.std(dim=0, keepdim=True).clamp_min(1e-3)

    # Call feedback to propagate errors
    model.feedback(sensors, error)

    # tame spikes from DFA/local targets
    nn.utils.clip_grad_norm_(model.encoder.parameters(), max_norm=1.0)
    nn.utils.clip_grad_norm_(model.decoder.parameters(), max_norm=1.0)
    optm.step()

    # Compute losses for logging
    loss_reconstruction = mse_loss(output, sensors)
    loss_sparsity = rate_loss(embedding)

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
        l1, l2 = mse_loss(output, xb), rate_loss(embeding)
        print(f"Test | Reconstruction: {l1:.6f} | Sparsity: {l2:.6f}")


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
optm_args = [
    {"params": model.encoder.parameters(), "lr": 1e-4},
    {"params": model.decoder.parameters(), "lr": 1e-3},
]
optm = Adam(optm_args)
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
