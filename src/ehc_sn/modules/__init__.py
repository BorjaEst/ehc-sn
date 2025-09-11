"""Core neural network modules and components for entorhinal-hippocampal circuit modeling.

This module provides fundamental building blocks for constructing neural network models
of the entorhinal-hippocampal circuit. It includes specialized loss functions, custom
autograd functions for biologically plausible learning, and neural network layers
that implement various learning algorithms.

The modules are designed to support both standard artificial neural networks and
biologically-inspired architectures, with particular emphasis on addressing the
biological constraints that limit the plausibility of standard backpropagation
in real neural systems.

Key Features:
    - Biologically-inspired loss functions for sparse coding and homeostasis
    - Custom autograd functions for alternative learning algorithms
    - Specialized neural network layers (DFA, DRTP, SRTP)
    - Mathematical formulations based on neuroscience principles
    - Integration with PyTorch's automatic differentiation system

Components:
    loss: Loss functions for biological constraints (sparsity, orthogonality, homeostasis)
    dfa: Direct Feedback Alignment implementation with custom autograd
    drtp: Direct Random Target Projection learning mechanisms
    srtp: Symmetric Random Target Projection algorithms
    zo: Zeroth-order optimization methods

Biological Motivation:
    The modules address fundamental questions about how biological neural networks
    can learn effectively without the symmetric weight transport required by
    backpropagation. Alternative learning mechanisms like DFA and target projection
    provide biologically plausible solutions to this problem.

Examples:
    >>> from ehc_sn.modules.loss import GramianOrthogonalityLoss, HomeostaticActivityLoss
    >>> from ehc_sn.modules.dfa import DFALayer
    >>>
    >>> # Create biologically-inspired loss functions
    >>> gramian_loss = GramianOrthogonalityLoss(center=True)
    >>> homeo_loss = HomeostaticActivityLoss(target_rate=0.1)
    >>>
    >>> # Add DFA learning to a network
    >>> dfa_layer = DFALayer(output_dim=10, hidden_dim=128)

References:
    - Lillicrap, T. P., et al. (2016). Random synaptic feedback weights support
      error backpropagation for deep learning. Nature Communications, 7, 13276.
    - Nøkland, A. (2016). Direct feedback alignment provides learning in deep
      neural networks without loss gradients. arXiv preprint arXiv:1609.01596.
"""
