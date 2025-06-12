import torch
from torch import Tensor, jit, nn
from torch.nn import Module

from ehc_sn import utils
from ehc_sn.config import parameters
from ehc_sn.models.config import Connection as ConnectionConfig
from ehc_sn.models.config import Input as InputConfig
from ehc_sn.models.config import Layer as LayerConfig
from ehc_sn.models.config import Network as NetworkConfig
from ehc_sn.models.config import load_network
from ehc_sn.settings import config


class Neurons(Module):
    def __init__(self, n: int, neuron: str, **kwds):
        super().__init__(**kwds)
        self.description = parameters.neurons[neuron].description
        self.activation = torch.zeros(1, n, device=config.device)
        activation_fname = parameters.neurons[neuron].activation_function
        self.activation_function = utils.torch_function(activation_fname)

    @property
    def size(self) -> int:
        return self.activation.shape[1]

    def forward(self, x: Tensor) -> Tensor:
        self.activation = self.activation_function(x)
        return self.activation


class Layer(Module):
    def __init__(self, p: LayerConfig, **kwds):
        super().__init__(**kwds)
        self.description = getattr(p, "description", "No description provided")
        self.neurons = Neurons(n=p.size, neuron=p.neuron, **kwds)

    def forward(self, x: Tensor) -> Tensor:
        return self.neurons(x)


class Connection(Module):
    def __init__(self, layers: dict[str, LayerConfig], p: ConnectionConfig, **kwds):
        super().__init__(**kwds)
        self.description = parameters.synapses[p.synapse].description
        self.w = torch.ones(layers[p.source].size, layers[p.target].size, device=config.device)
        self.w *= parameters.synapses[p.synapse].w_init
        self.w_max = parameters.synapses[p.synapse].w_max
        self.w_min = parameters.synapses[p.synapse].w_min
        self.learning_rate = parameters.synapses[p.synapse].learning_rate

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.w.T


class Synapses(Module):
    def __init__(self, layers: dict[str, LayerConfig], px: list[ConnectionConfig], **kwds):
        super().__init__(**kwds)
        self.connections = nn.ModuleList([Connection(layers, p) for p in px])

    def forward(self, x: Tensor) -> Tensor:
        # Parallelize synapse forward pass with jit.fork and jit.wait.
        promises = [jit.fork(module, x) for module in self.children()]
        return sum([jit.wait(p) for p in promises])


class Links(Module):
    def __init__(self, layers: dict[str, LayerConfig], px: list[ConnectionConfig], **kwds):
        super().__init__(**kwds)
        for synapse, connections in utils.map_by_key(px, "synapse").items():
            self.add_module(synapse, Synapses(layers, connections, synapse, **kwds))

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError("Links module is not implemented yet.")


class Network(Module):
    def __init__(self, p: NetworkConfig, **kwds):
        super().__init__(**kwds)
        self.layers = nn.ModuleDict({name: Layer(p, **kwds) for name, p in p.layers.items()})
        links: dict[str, ConnectionConfig] = utils.map_by_key(p.connections, "target")
        self.links = nn.ModuleDict({target: Links(p.layers, px, **kwds) for target, px in links.items()})

    def forward(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        # Process connections using jit.fork and jit.wait for parallel execution
        futures = {target: jit.fork(link, self) for target, link in self.links.items()}
        currents = {target: jit.wait(x) for target, x in futures.items()}
        # Update currents with inputs
        for name, x in inputs.items():
            currents[name] = currents[name] + x if name in currents else x
        # Process layers using jit.fork and jit.wait for parallel execution
        futures = {name: jit.fork(x, currents[name]) for name, x in self.layers.items()}
        outputs = {name: jit.wait(fut) for name, fut in futures.items()}
        return outputs


if __name__ == "__main__":
    # Load network configuration from the test model TOML file and create an instance of Network.
    config_path = "tests/config/model_1.toml"
    net_config = load_network(config_path)
    network_instance = Network(net_config)
    print("Network instance created with layers:", list(network_instance.layers.keys()))
    output = network_instance({"input_layer": torch.randn(1, 10)})  # Example input for testing
    print("Output from the network:", output)
