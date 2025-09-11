"""Training strategies for entorhinal-hippocampal circuit models.

This module provides various training strategies for neural network models of the
entorhinal-hippocampal circuit. The training strategies implement different learning
algorithms ranging from standard backpropagation to biologically plausible alternatives
that address fundamental constraints of biological neural networks.

The trainer module follows the strategy pattern, allowing the same model architecture
to be trained with different algorithms while maintaining a consistent interface. This
design enables easy experimentation with various learning mechanisms and comparison
of their effectiveness for spatial navigation tasks.

Key Features:
    - Strategy pattern for pluggable training algorithms
    - Support for both standard and biologically plausible learning methods
    - PyTorch Lightning integration for efficient training workflows
    - Consistent interface across all training strategies
    - Manual and automatic optimization support

Training Strategies:
    - Backpropagation: Standard and detached gradient variants
    - Direct Feedback Alignment (DFA): Biologically plausible random feedback
    - Target Projection Methods: DRTP and SRTP for direct target learning
    - Zeroth-order methods: Gradient-free optimization approaches

Classes:
    BaseTrainer: Abstract base class defining the trainer interface
    ClassicTrainer: Standard backpropagation with full gradient flow
    DetachedTrainer: Split training with independent component optimization
    DFATrainer: Direct Feedback Alignment with random feedback weights
    DRTPTrainer: Direct Random Target Projection learning
    SRTPTrainer: Symmetric Random Target Projection learning

Examples:
    >>> from functools import partial
    >>> import torch.optim as optim
    >>> from ehc_sn.trainers import ClassicTrainer, DFATrainer
    >>>
    >>> # Standard backpropagation training
    >>> bp_trainer = ClassicTrainer(
    ...     optimizer_init=partial(optim.Adam, lr=1e-3)
    ... )
    >>>
    >>> # Biologically plausible DFA training
    >>> dfa_trainer = DFATrainer(
    ...     optimizer_init=partial(optim.Adam, lr=1e-3)
    ... )

References:
    - Lillicrap, T. P., et al. (2016). Random synaptic feedback weights support
      error backpropagation for deep learning. Nature Communications, 7, 13276.
    - Nøkland, A. (2016). Direct feedback alignment provides learning in deep
      neural networks without loss gradients. arXiv preprint arXiv:1609.01596.
"""
