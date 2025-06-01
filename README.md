# EHC-SN: Entorhinal-Hippocampal Complex Spatial Navigation

A Python library for computational neuroscience modeling of spatial navigation in the entorhinal-hippocampal complex.

## Installation

```bash
pip install ehc-sn
```

## Features

- Biologically plausible neural representations of space
- Implementations of spike-timing-dependent plasticity (STDP)
- Spatial mapping functions for navigational modeling
- Memory encoding and retrieval mechanisms
- Autoencoder components aligned with biological principles

## Basic Usage

```python
import ehc_sn

# Initialize a spatial navigation model
model = ehc_sn.SpatialNavigationModel()

# Configure model parameters
model.configure(learning_rate=0.01, grid_scale=1.0)

# Train model with spatial data
model.train(trajectory_data)

# Get spatial representations
place_cells = model.get_place_cell_activity(position)
grid_cells = model.get_grid_cell_activity(position)
```

## Documentation

[Link to documentation - coming soon]

## Citation

If you use this library in your research, please cite:

[Citation information - coming soon]

## License

[License information]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
