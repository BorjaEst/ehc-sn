# Entorhinal-Hippocampal Complex (EHC) Spatial Navigation

[![PyPI version](https://img.shields.io/pypi/v/ehc-sn.svg)](https://pypi.org/project/ehc-sn/)
[![Python versions](https://img.shields.io/pypi/pyversions/ehc-sn.svg)](https://pypi.org/project/ehc-sn/)
[![Documentation Status](https://readthedocs.io/en/latest/?badge=latest)](https://ehc-sn.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python library for modeling and simulating the entorhinal-hippocampal complex during spatial navigation tasks.

## Overview

This library provides computational tools for research and simulations focused on the neural mechanisms underlying spatial navigation, particularly in the hippocampus and entorhinal cortex. It implements biologically plausible learning mechanisms and neural representations that support spatial cognition.

## Installation

```bash
pip install ehc-sn
```

Requires Python 3.10 or later.

## Features

- Neural representation components for place cells, grid cells, and head direction cells
- Spatial mapping functions for environment representation
- Memory encoding and retrieval mechanisms
- Implementation of spike-timing-dependent plasticity (STDP)
- Biologically plausible autoencoder components

## Usage Example

```python
import ehc_sn

# Create a simple grid cell model
grid_cell_module = ehc_sn.neural_representations.GridCellModule(
    n_cells=100, 
    scale=0.3, 
    orientation=0.0
)

# Run spatial navigation simulation
position = (0.5, 0.3)  # x, y coordinates
activity = grid_cell_module.compute_activity(position)

# Visualize grid cell activity
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
ehc_sn.utils.visualization.plot_spatial_activity(grid_cell_module, resolution=50)
plt.title("Grid Cell Spatial Activity")
plt.show()
```

## Development Setup

For development, clone the repository and install in development mode:

```bash
git clone https://github.com/username/ehc-sn.git
cd ehc-sn
pip install -e ".[dev]"
```

Run tests with:

```bash
pytest
```

## Documentation

For complete documentation, visit [https://ehc-sn.readthedocs.io](https://ehc-sn.readthedocs.io)

## Citation

If you use this library in your research, please cite:

```
@software{ehc_sn2023,
  author = {EHC Research Team},
  title = {Entorhinal-Hippocampal Complex (EHC) Spatial Navigation},
  url = {https://github.com/username/ehc-sn},
  version = {0.1.0},
  year = {2023},
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
