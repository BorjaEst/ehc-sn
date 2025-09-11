"""Neural network models for entorhinal-hippocampal circuit spatial navigation.

This module contains neural network architectures designed to model the information
processing capabilities of the entorhinal-hippocampal circuit. The models implement
various encoder-decoder architectures, autoencoder variants, and specialized neural
components that mimic the biological functions of brain regions involved in spatial
navigation and memory.

The models are organized into artificial neural networks (ANN) and biologically-
inspired architectures, each designed to capture different aspects of spatial
information processing in the brain. All models support multiple training strategies
and can be configured for various spatial navigation tasks.

Key Features:
    - Autoencoder architectures for spatial representation learning
    - Configurable encoder-decoder components with biological constraints
    - Support for multiple training strategies (BP, DFA, DRTP, SRTP)
    - PyTorch Lightning integration for efficient training and evaluation
    - Biologically-inspired sparsity and orthogonality constraints

Model Categories:
    ann: Artificial neural network models (autoencoders, encoders, decoders)
    biological: Biologically-inspired models (CAN networks, spiking models)

Architecture Principles:
    - Sparse coding inspired by hippocampal place cell firing patterns
    - Decorrelated representations for effective pattern separation
    - Homeostatic activity regulation for stable neural dynamics
    - Configurable activation functions for different learning algorithms

Examples:
    >>> from ehc_sn.models.ann.autoencoders import Autoencoder, AutoencoderParams
    >>> from ehc_sn.trainers import ClassicTrainer
    >>>
    >>> # Create and configure an autoencoder model
    >>> params = AutoencoderParams(...)
    >>> trainer = ClassicTrainer(...)
    >>> model = Autoencoder(params, trainer)

References:
    - O'Keefe, J., & Nadel, L. (1978). The hippocampus as a cognitive map.
    - Hafting, T., et al. (2005). Microstructure of a spatial map in the entorhinal cortex.
    - Dordek, Y., et al. (2016). Extracting grid cell characteristics from place cell inputs.
"""
