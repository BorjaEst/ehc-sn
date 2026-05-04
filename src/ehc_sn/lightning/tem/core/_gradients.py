"""TEM Lightning gradient diagnostic helpers."""

from __future__ import annotations

import torch
from torch import Tensor, nn


# =================================================================================================
def grad_l2_norm(module: nn.Module, *, device: torch.device | None = None) -> Tensor:
    """Return the global L2 norm of all available parameter gradients.

    Args:
        module: Module whose parameter gradients should be aggregated.
        device: Optional device for the returned zero tensor when no gradients
            are present.

    Returns:
        Scalar tensor containing the aggregated L2 norm. Returns zero when the
        module has no populated gradients.
    """
    grad_norms: list[Tensor] = []
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().to(dtype=torch.float32)
        grad_norms.append(torch.linalg.vector_norm(grad))
        if device is None:
            device = grad.device

    if grad_norms:
        return torch.linalg.vector_norm(torch.stack(grad_norms))
    if device is None:
        device = torch.device("cpu")
    return torch.zeros((), dtype=torch.float32, device=device)


# =================================================================================================
__all__ = ["grad_l2_norm"]