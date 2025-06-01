# GitHub Copilot Instructions

This file establishes guidelines for using GitHub Copilot in the "Entorhinal-Hippocampal Complex (EHC) Spatial Navigation" Python library.

## Overview
- Provides guidance for Python code suggestions related to EHC spatial navigation models
- Serves as a central reference for development standards in neuroscience modeling
- Focuses on biologically plausible learning mechanisms and neural representations
- Supports the development of a PyPI-distributed library (`pip install ehc-sn`)

## Python-Specific Guidelines
- Follow PEP 8 style guidelines for Python code
- Use NumPy docstring format for all functions and classes
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

## General Guidelines
- Follow existing coding styles and conventions
- Suggest code only relevant to the current task
- Use concise code blocks with file header comments
- Avoid redundancies by referencing unchanged code with comments
- Consider packaging requirements and distribution best practices

## Notes
- Update these instructions as the library evolves
- Confirm that all file paths match exactly during modifications
- Ensure changes maintain compatibility with PyPI distribution
