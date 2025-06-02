# GitHub Copilot Instructions

This file establishes guidelines for using GitHub Copilot in the "Entorhinal-Hippocampal Complex (EHC) Spatial Navigation" Python library.

## Overview
- Provides guidance for Python code suggestions related to EHC spatial navigation models
- Serves as a central reference for development standards in neuroscience modeling
- Focuses on biologically plausible learning mechanisms and neural representations
- Supports the development of a PyPI-distributed library (`pip install ehc-sn`)

## Project Structure
- `pyproject.toml` - Package configuration and build settings
- `src/ehc_sn/` - Main package source code
  - `VERSION` - File containing the current version number
  - `mec` - Medial entorhinal cortex model and components
  - `hpc` - Hippocampus model and components
  - `stdp/` - Plasticity mechanisms
  - `autoencoder/` - Autoencoder components
  - `utils/` - Utility functions
- `requirements.txt` - Core package dependencies
- `requirements-dev.txt` - Development dependencies
- `docs/` - Documentation based in md format
- `tests/` - Unit and integration tests
- `tools/` - Utility scripts for development (e.g., data generation, visualization, etc.)
- `.github/` - GitHub workflows and templates

## Python-Specific Guidelines
- Follow PEP 8 style guidelines for Python code
- Use NumPy docstring format for all functions and classes
- Target Python 3.10 and newer (3.10, 3.11, 3.12)
- Suggest scientific libraries appropriate for neural modeling (NumPy, SciPy, PyTorch, etc.)
- Prioritize vectorized operations over loops when applicable for performance
- Implement proper error handling for mathematical and scientific calculations
- Ensure backward compatibility for public API functions

## Library Domain Guidelines
- Use established terminology from hippocampal and spatial navigation literature
- Structure library components with clear separation between:
  - Neural representation components
  - Spatial mapping functions
  - Memory encoding/retrieval mechanisms
  - Plasticity mechanisms (especially STDP implementations)
  - Autoencoder components (encoding/decoding layers)
- Include relevant citations as comments where appropriate
- Optimize computationally intensive operations with appropriate algorithms
- Design public interfaces that are intuitive for neuroscience researchers

## Biological Plausibility Guidelines
- Implement spike-timing-dependent plasticity (STDP) following established neuroscience models
- Structure autoencoder components to align with biological neural network principles
- Simulate realistic neural dynamics with appropriate time constants and firing properties
- Balance computational efficiency with biological realism in model design

## Testing Guidelines
- Use pytest framework for all tests
- Design testing and configure pytest to use importlib import mode 
- Write unit tests for all public functions and classes
- Mock complex neural simulations appropriately in tests
- Verify mathematical correctness with known solutions where possible
- Test edge cases relevant to neural modeling (e.g., boundary conditions, numerical stability)

## Documentation Guidelines
- Maintain comprehensive Sphinx documentation
- Include mathematical explanations with LaTeX where appropriate
- Provide executable examples demonstrating key functionality
- Explain biological relevance and limitations of implementations

## Packaging Guidelines
- Maintain version number in src/ehc_sn/VERSION file
- Keep core dependencies in requirements.txt
- Keep development dependencies in requirements-dev.txt
- Ensure compatibility with PyPI distribution standards
- Use dynamic configuration in pyproject.toml where appropriate

## General Guidelines
- Follow existing coding styles and conventions
- Suggest code only relevant to the current task
- Use concise code blocks with file header comments
- Avoid redundancies by referencing unchanged code with comments
- Consider packaging requirements and distribution best practices

## Notes
- Update .github/copilot-instructions.md and README as the library evolves
- Confirm that all file paths match exactly during modifications
