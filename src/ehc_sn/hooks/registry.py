"""
Unified hooks and activation management system for EHC neural networks.

This module provides a centralized registry for capturing and accessing activations
and error signals across encoders, decoders, and training algorithms. It replaces
module-specific global state with a thread-safe, device-aware storage system.

Key Features:
- Thread-local storage to prevent cross-thread leakage
- Device and dtype preservation for tensor storage
- Automatic cleanup with context managers and lifecycle hooks
- Support for DFA, SRTP, and custom training algorithms
- Minimal overhead with detached tensor storage

Example:
    >>> from ehc_sn.hooks.registry import registry
    >>> 
    >>> # Store an activation for later retrieval
    >>> registry.set_activation("encoder.layer1", hidden_tensor)
    >>> 
    >>> # Register error capture on network output
    >>> remover = registry.register_output_error_hook(output_tensor)
    >>> 
    >>> # Access stored values
    >>> hidden = registry.get_activation("encoder.layer1")
    >>> error = registry.get_error("global")
    >>> 
    >>> # Clean up
    >>> registry.clear("batch")
    >>> remover()
"""

import threading
from contextlib import contextmanager
from typing import Callable, Dict, Generator, Literal, Optional

import torch
from torch import Tensor


class ActivationRegistry:
    """Thread-safe registry for managing activations and error signals.
    
    This class provides centralized storage for neural network activations
    and error gradients, supporting both forward activations capture and
    backward error signal propagation for biologically-inspired training
    algorithms like DFA and SRTP.
    
    The registry uses thread-local storage to ensure isolation between
    concurrent training processes and properly handles tensor device
    placement and memory management.
    
    Attributes:
        _local: Thread-local storage containing activation and error dictionaries.
    """
    
    def __init__(self):
        self._local = threading.local()
    
    def _ensure_storage(self) -> None:
        """Ensure thread-local storage is initialized."""
        if not hasattr(self._local, 'activations'):
            self._local.activations = {}
            self._local.errors = {}
            self._local.hooks = {}
    
    def set_activation(self, key: str, tensor: Tensor, *, detach: bool = True) -> None:
        """Store an activation tensor for later retrieval.
        
        Args:
            key: Unique identifier for the activation (e.g., "encoder.layer1.post").
            tensor: Activation tensor to store.
            detach: Whether to detach the tensor from the computation graph.
                   Defaults to True for memory efficiency.
        
        Note:
            Tensors are stored with preserved device and dtype. Detaching is
            recommended to prevent memory leaks in most use cases.
        """
        self._ensure_storage()
        stored_tensor = tensor.detach() if detach else tensor
        self._local.activations[key] = stored_tensor
    
    def get_activation(self, key: str) -> Tensor:
        """Retrieve a stored activation tensor.
        
        Args:
            key: Unique identifier for the activation.
            
        Returns:
            Stored activation tensor.
            
        Raises:
            KeyError: If the activation key is not found.
        """
        self._ensure_storage()
        if key not in self._local.activations:
            raise KeyError(f"Activation '{key}' not found. Available keys: {list(self._local.activations.keys())}")
        return self._local.activations[key]
    
    def set_error(self, key: str, tensor: Tensor) -> None:
        """Store an error signal tensor for later retrieval.
        
        Args:
            key: Unique identifier for the error signal (e.g., "global", "layer_specific").
            tensor: Error gradient tensor to store.
        
        Note:
            Error tensors are stored as-is without detaching since they're
            typically already detached gradient tensors from hooks.
        """
        self._ensure_storage()
        self._local.errors[key] = tensor
    
    def get_error(self, key: str) -> Tensor:
        """Retrieve a stored error signal tensor.
        
        Args:
            key: Unique identifier for the error signal.
            
        Returns:
            Stored error tensor.
            
        Raises:
            KeyError: If the error key is not found.
        """
        self._ensure_storage()
        if key not in self._local.errors:
            raise KeyError(f"Error signal '{key}' not found. Available keys: {list(self._local.errors.keys())}")
        return self._local.errors[key]
    
    def clear(self, scope: Literal["batch", "all"] = "batch") -> None:
        """Clear stored activations and/or error signals.
        
        Args:
            scope: Scope of clearing - "batch" clears both activations and errors,
                  "all" is identical for now but reserved for future extensions.
        
        Note:
            This method should be called at the beginning of each training batch
            to prevent stale activations from affecting current computations.
        """
        self._ensure_storage()
        if scope in ("batch", "all"):
            self._local.activations.clear()
            self._local.errors.clear()
    
    def register_output_error_hook(self, tensor: Tensor, key: str = "global") -> Callable[[], None]:
        """Register a backward hook to capture output error gradients.
        
        This method registers a hook on the provided tensor that will capture
        the gradient during backpropagation and store it in the error registry.
        This is essential for DFA training where global error signals need to
        be propagated to all hidden layers.
        
        Args:
            tensor: Output tensor on which to register the gradient hook.
            key: Identifier for storing the captured error signal.
            
        Returns:
            Function to remove the registered hook. Call this after training
            to clean up hook registrations.
        
        Example:
            >>> output = model(input)
            >>> remover = registry.register_output_error_hook(output)
            >>> loss = criterion(output, target)
            >>> loss.backward()  # Hook captures gradients
            >>> error = registry.get_error("global")  # Access captured error
            >>> remover()  # Clean up hook
        """
        self._ensure_storage()
        
        def capture_grad_hook(grad: Tensor) -> Tensor:
            """Hook function to capture and store gradient."""
            if grad is not None:
                self.set_error(key, grad.detach())
            return grad
        
        # Register the hook and store the handle
        hook_handle = tensor.register_hook(capture_grad_hook)
        hook_key = f"output_error_{key}"
        self._local.hooks[hook_key] = hook_handle
        
        def remove_hook() -> None:
            """Remove the registered hook."""
            if hook_key in self._local.hooks:
                self._local.hooks[hook_key].remove()
                del self._local.hooks[hook_key]
        
        return remove_hook
    
    @contextmanager
    def batch_scope(self) -> Generator[None, None, None]:
        """Context manager for automatic batch lifecycle management.
        
        This context manager automatically clears the registry at entry
        and ensures cleanup on exit, providing a convenient way to manage
        registry state during training batches.
        
        Example:
            >>> with registry.batch_scope():
            ...     output = model(input)
            ...     registry.register_output_error_hook(output)
            ...     loss.backward()
            ...     # Registry automatically cleared on exit
        """
        self.clear("batch")
        try:
            yield
        finally:
            # Clean up any remaining hooks
            self._ensure_storage()
            for hook_handle in self._local.hooks.values():
                hook_handle.remove()
            self._local.hooks.clear()
    
    def get_activation_keys(self) -> list[str]:
        """Get list of currently stored activation keys."""
        self._ensure_storage()
        return list(self._local.activations.keys())
    
    def get_error_keys(self) -> list[str]:
        """Get list of currently stored error keys."""
        self._ensure_storage()
        return list(self._local.errors.keys())


# Global registry instance
registry = ActivationRegistry()

# Module-level convenience functions
def set_activation(key: str, tensor: Tensor, *, detach: bool = True) -> None:
    """Store an activation tensor. See ActivationRegistry.set_activation."""
    registry.set_activation(key, tensor, detach=detach)

def get_activation(key: str) -> Tensor:
    """Retrieve an activation tensor. See ActivationRegistry.get_activation."""
    return registry.get_activation(key)

def set_error(key: str, tensor: Tensor) -> None:
    """Store an error tensor. See ActivationRegistry.set_error."""
    registry.set_error(key, tensor)

def get_error(key: str) -> Tensor:
    """Retrieve an error tensor. See ActivationRegistry.get_error."""
    return registry.get_error(key)

def clear(scope: Literal["batch", "all"] = "batch") -> None:
    """Clear stored tensors. See ActivationRegistry.clear."""
    registry.clear(scope)

def register_output_error_hook(tensor: Tensor, key: str = "global") -> Callable[[], None]:
    """Register output error hook. See ActivationRegistry.register_output_error_hook."""
    return registry.register_output_error_hook(tensor, key)


# Backward compatibility aliases for DFA
def register_dfa_hook(output_tensor: Tensor) -> Callable[[], None]:
    """Backward compatibility alias for register_output_error_hook.
    
    Args:
        output_tensor: The final output tensor of the network.
        
    Returns:
        Function to remove the registered hook.
    """
    return register_output_error_hook(output_tensor, key="global")

def get_dfa_error() -> Tensor:
    """Backward compatibility alias for get_error("global").
    
    Returns:
        The global error signal for DFA training.
        
    Raises:
        KeyError: If no DFA error signal is available.
    """
    try:
        return get_error("global")
    except KeyError:
        raise RuntimeError("No DFA error signal available. DFA hook not registered properly.")

def set_dfa_error(error: Tensor) -> None:
    """Backward compatibility alias for set_error("global").
    
    Args:
        error: The global error signal to store.
    """
    set_error("global", error)

def clear_dfa_error() -> None:
    """Backward compatibility alias for clearing DFA error."""
    registry._ensure_storage()
    if "global" in registry._local.errors:
        del registry._local.errors["global"]


if __name__ == "__main__":
    # Example usage and testing
    print("=== Activation Registry Example ===")
    
    # Create some test tensors
    activation = torch.randn(4, 128)
    output = torch.randn(4, 10, requires_grad=True)
    
    # Store activation
    set_activation("encoder.layer1", activation)
    print(f"Stored activation with shape: {get_activation('encoder.layer1').shape}")
    
    # Register error hook
    remover = register_output_error_hook(output)
    
    # Simulate backward pass
    loss = torch.sum(output)
    loss.backward()
    
    # Check if error was captured
    try:
        error = get_error("global")
        print(f"Captured error with shape: {error.shape}")
    except KeyError:
        print("No error captured")
    
    # Test batch scope
    with registry.batch_scope():
        set_activation("test", torch.randn(2, 64))
        print(f"In scope - activations: {registry.get_activation_keys()}")
    
    print(f"After scope - activations: {registry.get_activation_keys()}")
    
    # Clean up
    remover()
    clear("all")
    
    print("Registry example completed successfully!")
