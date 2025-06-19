# Entorhinal-Hippocampal Complex (EHC) Spatial Navigation

[![PyPI version](https://img.shields.io/pypi/v/ehc-sn.svg)](https://pypi.org/project/ehc-sn/)
[![Python versions](https://img.shields.io/pypi/pyversions/ehc-sn.svg)](https://pypi.org/project/ehc-sn/)
[![Documentation Status](https://readthedocs.io/en/latest/?badge=latest)](https://ehc-sn.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-GPLv3-yellow.svg)](https://opensource.org/license/gpl-3-0)

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
