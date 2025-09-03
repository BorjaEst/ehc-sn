# Comprehensive Instructions for Entorhinal-Hippocampal Circuit Modeling

## Overview & Overall Goal

Develop a Python library to model the entorhinal-hippocampal circuit for spatial navigation and memory functions, including:

- Pattern separation: Decoding distinct cognitive maps from overlapping inputs
- Completion: Recovering full cognitive maps from partial inputs

This library will:

- Support training using Backpropagation (BP) and Direct Random Target Projection (DRTP)
- Integrate key regions of the entorhinal-hippocampal circuit to simulate information flow
- Be distributed via PyPI (`pip install ehc-sn`)
- Provide network definition and parameters via TOML files loaded with Pydantic 2
- Focus on forward learning mechanisms and neural representations

## Project Structure

- `pyproject.toml` - Package configuration and build settings
- `src/ehc_sn/` - Main package source code
  - `VERSION` - File containing the current version number
  - `config/` - TOML files for library configuration (parameters, etc.)
  - `constants/` - Constants and types definitions (Enums, literals, types, etc.)
  - `core/` - Implementation of core library components (neurons, synapses, etc.)
  - `data/` - Data module for generating and managing Lightning data modules (cognitive maps, grid maps, etc.)
  - `figures/` - Module containing figure classes to visualize experiment results
  - `losses/` - Custom loss modules for the library (e.g., reconstruction loss, error loss)
  - `models/` - Implementation of library models (CANModel, Autoencoder, etc.)
  - `utils/` - Utility functions for the library
  - `parameters.py` - Global parameters for synapses types, neuron types, etc.
  - `settings.py` - Global settings and configurations
  - `simulations.py` - Class functions for running simulations
- `requirements-dev.txt` - Development dependencies
- `requirements.txt` - Core package dependencies
- `tests/` - Unit and integration tests

### Config Direct

- This directory should contain TOML files for library configuration
- It does not contain any code, only configuration files

### Constants Module

- This module should contain constants and types definitions
- It should include Enums, literals, and type definitions for the library
- It should be used to define constants for neuron types, synapse types, and other fixed parameters
- It should be used to define types for parameters and configurations

### Core Module

- This module should implement core library components such as neurons, synapses, and other fundamental building blocks
- It should include neuron models, synapse models, and other essential components for the entorhinal-hippocampal circuit
- It should provide a foundation for building more complex models and experiments
- It should be structured to allow easy extension and modification of core components

### Data Module

- Provide generatators and lightning data modules for obstacle and cognitive maps.
- Include plot generators for visualizing generated data (to be used on figures module)
- Plot generators should accept axes as input and generate plots on those axes
- Visualization should be implemented using Seaborn and Matplotlib

### Figures Module

- Provide functionality to visualize cognitive maps and model performance
- Implement visualization objects using Matplotlib and Seaborn
- Structure with:

1. Proper separation of plotting functions and figure classes
2. Consistent use of Pydantic for parameter management
3. Good handling of tensor conversion and visualization options
4. Comprehensive visualization capabilities for different map types

- Design figure classes that:

1. Encapsulate plotting logic and parameters into a single object
2. Initialize subplot figure and axes in the constructor
3. Have a specific and fixed title

- Plot generators should accept axes as input and generate plots on those axes

### Losses Module

- This module should contain submodules for different types of scenarios (e.g., autoencoders, reconstruction tasks)
- Each submodule should implement a specific loss function relevant to the scenario
- Loss functions should be designed to work with the entorhinal-hippocampal circuit models
- Losses should be implemented as PyTorch modules to integrate seamlessly with the training process

### Models Module

- This module should implement the library models, such as CANModel, Autoencoder, and other relevant models
- Each model should encapsulate the structure and behavior of the entorhinal-hippocampal circuit
- Models should be designed to support both training and inference modes
- Models should be modular and extensible to allow for future enhancements and variations
- Submodule autoencoder serves as a baseline for evaluation, implementing a general sparse autoencoder

### Utils Module

- This module should contain utility functions for the library
- It should include helper functions for data processing, tensor manipulation, and other common tasks
- It should provide reusable functions that can be used across different modules
- It should be structured to allow easy addition of new utility functions as needed
- It should include functions for tensor conversion, data normalization, and other common operations
- It should be designed to minimize dependencies on other modules to ensure reusability
- It should provide functions for logging, debugging, and other common tasks that are not specific to any module
- It should include functions for handling configuration files, such as loading and validating TOML files with Pydantic
- It should provide functions for managing parameters and settings across the library

### Parameters Module

- This module should define global parameters for synapse types, neuron types, and other fixed parameters
- It should use the `constants` module as a reference for defining parameter types

### Settings Module

- This module should define global settings and configurations for the library
- It should provide a centralized location for managing library-wide settings
- It should include settings for logging, debugging, and other global configurations
- It should be designed to allow easy modification of settings without changing the core library code

### Simulations Module

- This module should provide functions for running simulations of the entorhinal-hippocampal circuit
- It should include functions for initializing simulations, running experiments, and collecting results
- It should be designed to work with the library's models and trainers
- It should provide a framework for running different types of simulations

## Modeling Framework

- PyTorch with custom neuron models for the entorhinal-hippocampal circuit
- PyTorch Lightning for training, data loading and callbacks
- Norse for spiking neuron models and event-based processing
- Optuna for hyperparameter optimization
- TorchRL for reinforcement learning components, if needed
- Keep code simple and avoid extending functionality beyond requirements
- When implementing modifications, minimize the amount of code removed or added

## Circuit Components and Properties

The models are structured as autoencoders, where:

- Medial Entorhinal Cortex (MEC) layers act as encoder for extracting features from sensory inputs
- Hippocampal regions act as a decoder for reconstructing cognitive maps

### Medial Entorhinal Cortex (MEC) - Encoder

- **MEC Layer II**:

  - Receives information from MEC Layer Vb and sensory inputs
  - Calculates encoding hidden states
  - Attractor dynamics behavior
  - Projects signal to hippocampal subregions DG, CA3, CA2 and MEC Layer Vb

- **MEC Layer III**:

  - Receives information from Layer Vb and sensory inputs
  - Calculates reconstruction errors based on input from Layer Vb
  - Projects reconstruction error to CA1 and Subiculum (Sub)

- **MEC Layer Va**:

  - Lower excitability with integrator behavior
  - Calculates hidden states for MEC Layer Va
  - Projects signals to other brain areas (neocortex, etc.)
  - Optional implementation as it does not project to EHC circuit

- **MEC Layer Vb**:
  - Receives information from MEC Layer II and Subiculum (Sub)
  - Receives the reconstructed cognitive map from CA1
  - Calculates hidden states for MEC Layer Vb
  - Projects errors to MEC Layer II with fixed (no learning) weights

### Hippocampal Circuit - Decoder

- **DG (Dentate Gyrus)**:

  - Receives input from MEC Layer II
  - Calculates the features space (embeddings) for the cognitive map sensed by MEC
  - Projects the features to CA3
  - Neurogenesis; This layer dimensions can be increased after learning to allow for new cognitive maps
  - Contains mossy cells and GABAergic interneurons

- **CA3**:

  - Receives input from DG using training weights (might include recurrent connections)
  - Receives input from MEC Layer II using fixed weights
  - Calculates the decoder hidden states
  - Projects the hidden states to CA2 and CA1
  - Has extensive recurrent connections within itself
  - Association network; Pattern completion; One-shot learning; Place cells

- **CA2**:

  - Receives input from CA3 using training weights
  - Receives input from MEC Layer II using fixed weights
  - Calculates decoder hidden states
  - Projects hidden states to CA1
  - Modulates HPC dynamics; SWR generation; Unique plasticity profile

- **CA1**:

  - Receives input from CA3 and CA2 using training weights
  - Receives errors from MEC Layer III
  - Calculates the cognitive map reconstruction
  - Projects reconstructed cognitive map to MEC Layer Vb and Subiculum
  - HPC output; Contextual encoding and retrieval; Memory consolidation (via SWRs);
  - Compares "expected" (CA3-memories) with "actual" sensory information (EC-observations)

- **Subiculum (Sub)**:

  - Receives input from CA1 using training weights
  - Receives errors from MEC Layer III
  - Calculates output reconstruction
  - Projects reconstructed output to MEC Layer Vb
  - Specialized cells (head directions, grid cells, etc.)

## Models and Experiments

- First model should be a general sparse autoencoder to use as baseline
- Initial models replace the MEC by a standard sparse encoder trained a priori on cognitive maps
- Initial models ignore the Subiculum layer to work with one reconstruction at a time
- Full models will include the Subiculum layer and the MEC encoder layers

## Code Specifications

- Use type hints for function signatures
- Use Pydantic v2 for parameter validation and management
- Import in `__init__.py` from submodules to define the package/subpackage API
- Target Python 3.10 and newer (3.10, 3.11, 3.12)
- Prioritize vectorized operations over loops when applicable for performance

### Comments to split code sections

- Use comments with '-' to visually separate sections of the code
- Every function, class or method should have a section according to indentation
- Only 2 levels of separators are defined, one for the main section and one for subsections

```python
# -------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------
```

## Documentation Guidelines

- Use docstrings for all public functions and classes
- Follow PEP 257 conventions for docstrings
- `__init__.py` files should contain package-level documentation
- `__init__.py` files should only contain import statements, package metadata, and documentation strings

## Evaluation Criteria and Metrics

Evaluate model performance using these metrics:

1. Reconstruction Accuracy:

- Mean squared error (MSE) between original and reconstructed maps
- Structural similarity index (SSIM) for spatial coherence

2. Pattern Separation:

- Discriminability index between representations of similar inputs
- Hamming distance between encodings of similar patterns

3. Pattern Completion:

- Accuracy of reconstruction from partial inputs (10%, 30%, 50% missing)
- Recovery time (iterations needed for stable completion)

4. Biological Plausibility:

- Sparsity of representations (percentage of active units)
- Activity distributions compared to neurobiological data

5. Computational Efficiency:

- Training time and convergence speed
- Memory usage during training and inference

Each experiment should report these metrics in a standardized format to allow for comparison between model variants.

## Library Domain Guidelines

- Use established terminology from hippocampal and spatial navigation literature
- Include relevant citations as comments where appropriate
- Optimize computationally intensive operations with appropriate algorithms
- Implement proper error handling for mathematical and scientific calculations

## Testing Guidelines

- Use pytest framework for all tests
- Design testing and configure pytest to use importlib import mode
- Write functional tests for all public functions and classes
- Mock complex neural simulations appropriately in tests
- Verify mathematical correctness with known solutions where possible
- Test edge cases relevant to neural modeling (boundary conditions, numerical stability)

## Packaging Guidelines

- Maintain version number in src/ehc_sn/VERSION file
- Keep core dependencies in requirements.txt
- Keep development dependencies in requirements-dev.txt
- Ensure compatibility with PyPI distribution standards
- Use dynamic configuration in pyproject.toml where appropriate

## General Guidelines

- Suggest code only relevant to the current task
- Avoid redundancies by referencing unchanged code with comments
- Update instructions and README as the library evolves
- Prevent unnecessary complexity by keeping the codebase clean and maintainable
- Prevent long functions by breaking them into smaller, reusable components
- Prevent deep conditions by breaking them into smaller, manageable functions
- Prevent deep loops by using vectorized operations and comprehensions
