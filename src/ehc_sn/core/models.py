from typing import Any, Dict, List

import torch
from torch import Tensor, jit, nn
from torch.nn import Module

from ehc_sn import utils
from ehc_sn.config import parameters
from ehc_sn.core.config import Connection as ConnectionConfig
from ehc_sn.core.config import Layer as LayerConfig
from ehc_sn.core.config import Network as NetworkConfig
from ehc_sn.settings import config


class Neurons(Module):
    def __init__(self, n: int, neuron: str, **kwds: Any) -> None:
        super().__init__(**kwds)

        # Retrieve neuron parameters from configuration
        if neuron not in parameters.neurons:
            raise KeyError(f"Neuron type '{neuron}' not found in parameters")

        self.description = parameters.neurons[neuron].description

        # Initialize activation tensor with zeros
        # Shape: [batch_size=1, n_neurons]
        self.activation = torch.zeros(1, n, device=config.device)

        # Load the activation function dynamically
        activation_fname = parameters.neurons[neuron].activation_function
        try:
            self.activation_function = utils.torch_function(activation_fname)
        except AttributeError:
            raise AttributeError(f"Could not find activation function '{activation_fname}'")

    @property
    def size(self) -> int:
        # Get the number of neurons in this population.
        return self.activation.shape[1]

    def forward(self, x: Tensor) -> Tensor:
        # Update internal activation state and return it
        self.activation = self.activation_function(x)
        return self.activation


class Layer(Module):
    def __init__(self, p: LayerConfig, **kwds: Any) -> None:
        super().__init__(**kwds)

        # Store layer description or use default if not provided
        self.description = getattr(p, "description", "No description provided")

        # Create the neuron population for this layer
        self.neurons = Neurons(n=p.size, neuron=p.neuron, **kwds)

    def forward(self, x: Tensor) -> Tensor:
        # Process input through the layer's neurons.
        return self.neurons(x)


class Connection(Module):
    def __init__(self, layers: Dict[str, LayerConfig], p: ConnectionConfig, **kwds: Any) -> None:
        super().__init__(**kwds)

        # Validate synapse type and layer names
        if p.synapse not in parameters.synapses:
            raise KeyError(f"Synapse type '{p.synapse}' not found in parameters")
        if p.source not in layers:
            raise KeyError(f"Source layer '{p.source}' not found in configuration")
        if p.target not in layers:
            raise KeyError(f"Target layer '{p.target}' not found in configuration")

        self.description = parameters.synapses[p.synapse].description

        # Initialize weight matrix with dimensions: [target_neurons, source_neurons]
        # This orientation allows efficient matrix multiplication with activation vectors
        self.w = torch.ones(layers[p.target].size, layers[p.source].size, device=config.device)
        self.w *= parameters.synapses[p.synapse].w_init

        # Store parameters needed for plasticity/learning
        self.w_max = parameters.synapses[p.synapse].w_max
        self.w_min = parameters.synapses[p.synapse].w_min
        self.learning_rate = parameters.synapses[p.synapse].learning_rate
        self.source = p.source

    def forward(self, layers: nn.ModuleDict) -> Tensor:
        # Matrix multiplication: [1, n_source] @ [n_source, n_target].T -> [1, n_target]
        return layers[self.source].neurons.activation @ self.w.T


class Synapses(Module):
    def __init__(self, layers: Dict[str, LayerConfig], px: List[ConnectionConfig], **kwds: Any) -> None:
        super().__init__(**kwds)

        # Group connections by source layer
        # Each source layer should have only one connection to the target layer
        for connection in px:
            self.add_module(connection.source, Connection(layers, connection, **kwds))

    def forward(self, layers: nn.ModuleDict) -> Tensor:
        # Parallelize synapse forward pass with jit.fork and jit.wait.
        # This creates "futures" that compute in parallel and then waits for all results
        promises = [jit.fork(module, layers) for module in self.children()]
        return sum([jit.wait(p) for p in promises])


class Links(Module):
    def __init__(self, layers: Dict[str, LayerConfig], px: List[ConnectionConfig], **kwds: Any) -> None:
        super().__init__(**kwds)

        # Group connections by synapse type (e.g., "excitatory", "inhibitory")
        # Each type may have different learning rules or properties
        for synapse, connections in utils.map_by_key(px, "synapse").items():
            self.add_module(synapse, Synapses(layers, connections, **kwds))

    def forward(self, layers: nn.ModuleDict) -> Tensor:
        # Create parallel computation "promises" for each synapse type
        promises = [jit.fork(module, layers) for module in self.children()]
        return sum([jit.wait(p) for p in promises])


class Network(Module):
    def __init__(self, p: NetworkConfig, **kwds: Any) -> None:
        super().__init__(**kwds)

        # Create all layers defined in the configuration
        self.layers = nn.ModuleDict({name: Layer(p, **kwds) for name, p in p.layers.items()})

        # Group connections by target layer for efficient processing
        links: Dict[str, List[ConnectionConfig]] = utils.map_by_key(p.connections, "target")

        # Create connection modules for each target layer
        self.links = nn.ModuleDict({target: Links(p.layers, px, **kwds) for target, px in links.items()})

    def forward(self, inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        # Process connections using jit.fork and jit.wait for parallel execution
        futures = {target: jit.fork(link, self.layers) for target, link in self.links.items()}
        currents = {target: jit.wait(x) for target, x in futures.items()}

        # Initialize currents for layers that do not have incoming connections
        for name, layer in self.layers.items():
            if name not in currents:
                currents[name] = torch.zeros(1, layer.neurons.size, device=config.device)

        # Update currents with inputs
        for name, x in inputs.items():
            if name not in currents:
                raise ValueError(f"Input '{name}' not found in the network layers.")
            currents[name] = currents[name] + x

        # Process layers using jit.fork and jit.wait for parallel execution
        futures = {name: jit.fork(x, currents[name]) for name, x in self.layers.items()}
        outputs = {name: jit.wait(fut) for name, fut in futures.items()}

        # Return the outputs as a dictionary
        return outputs
