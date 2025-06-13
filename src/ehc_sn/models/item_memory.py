import torch
from torch import nn


class Encoder(nn.Module):
    """Encoder for input data to feature space."""

    def __init__(self, inputs_dim: int, hidden_dim: int, hpc_dim: int):
        super().__init__()

        # Encoder network
        self.network = nn.Sequential(
            nn.Linear(inputs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hpc_dim),
            nn.Tanh(),  # Bounded activation for normalized feature vectors
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Encode input into feature space
        return self.network(inputs)


class Decoder(nn.Module):
    """Decoder for feature space to input data."""

    def __init__(self, hpc_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()

        # Encoder network
        self.network = nn.Sequential(
            nn.Linear(hpc_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),  # Bounded activation for normalized feature vectors
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Decode feature space into input
        return self.network(inputs)


class HPCGrid(nn.Module):
    """Hippocampal Place Cells with recurrent connections"""

    def __init__(self, neurons: int, mec_dims: list[int]):
        super().__init__()

        # Linear layer to compute place cell activations
        self.register_buffer("place_cells", torch.zeros(neurons))
        self.synapses_mec = nn.ModuleList([nn.Linear(grid_dim, neurons) for grid_dim in mec_dims])
        self.synapses_features = nn.Linear(neurons, neurons)

        # Apply additional nonlinearity and normalization for place fields
        self.nonlinearity = nn.Sequential(nn.ReLU(), nn.LayerNorm(neurons))

    def forward(self, features: torch.Tensor, mec: list[torch.Tensor]) -> torch.Tensor:
        # Compute place cell activations from features and MEC grid cells
        fea_currents = self.synapses_features(features)
        mec_currents = sum(synapse(mec_i) for synapse, mec_i in zip(self.synapses_mec, mec))

        return self.nonlinearity(fea_currents + mec_currents)


class MECGrid(nn.Module):
    """Medial Entorhinal Cortex Grid Cells with attractor dynamics"""

    def __init__(self, neurons: int, hpc_dim: int, init_gain: float = 0.9):
        super().__init__()

        # Register a persistent tensor to hold the activations
        self.register_buffer("grid_cells", torch.zeros(neurons))
        self.synapses_hpc = nn.Linear(hpc_dim, neurons)
        self.synapses_rcc = nn.Linear(neurons, neurons)

        # Gain parameter for attractor dynamics
        self.gain = nn.Parameter(torch.tensor(init_gain))

    def forward(self, hpc_activations: torch.Tensor) -> torch.Tensor:
        # Update attractor state with new input
        hpc_currents = self.synapses_hpc(hpc_activations)
        rcc_currents = self.synapses_rcc(self.grid_cells)

        # Apply attractor dynamics with gain parameter
        self.grid_cells = self.gain * self.grid_cells + hpc_currents + rcc_currents
        self.grid_cells = torch.tanh(self.grid_cells)  # Bounded activation

        return self.grid_cells


class ItemMemory(nn.Module):
    """Neural network-based item memory."""

    def __init__(self):
        super().__init__()

        # Item representation network
        self.encoder = Encoder(inputs_dim=100, hidden_dim=80, hpc_dim=25)
        self.decoder = Decoder(hpc_dim=25, hidden_dim=80, output_dim=100)

        # Spatial representation networks
        self.hpc = HPCGrid(25, mec_dims=[40, 40, 40])
        self.mec = nn.ModuleList([MECGrid(40, hpc_dim=25) for _ in range(3)])

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Encode input into feature space
        hpc_activations = self.hpc(self.encoder(inputs), [mec.grid_cells for mec in self.mec])
        _mec_activations = [mec(hpc_activations) for mec in self.mec]
        # Decode feature space into output
        return hpc_activations, self.decoder(hpc_activations)

    def store_item(self, features: torch.Tensor, inputs: torch.Tensor):
        """Process input and store in memory."""
        # TODO: Implement scaffold dynamics to store item representations
        with torch.no_grad():
            # Encode input into feature space
            encoded_features = self.forward(inputs)
            err = torch.norm(encoded_features - features, dim=1)
        # TODO: Implement scaffold dynamics to store item representations

    def query(self, features: torch.Tensor) -> torch.Tensor:
        """Find items in memory most similar to the input."""
        # TODO: Implement scaffold dynamics to query item representations
        with torch.no_grad():
            # decode features into output space
            decoded_features = self.forward(features)
        # TODO: Implement scaffold dynamics to query item representations
