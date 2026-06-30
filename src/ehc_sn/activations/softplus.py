import torch
from torch import Tensor, nn


# =============================================================================
def bounded_positive_scale(  # ------------------------------------------------
    x: Tensor,
    *,
    min_sigma: float = 1e-4,
    max_sigma: float = 1e2,
) -> Tensor:
    """Softplus activation clamped to ``[min_sigma, max_sigma]``.

    Args:
        x: Input tensor of any shape.
        min_sigma: Minimum output value.
        max_sigma: Maximum output value.

    Returns:
        Tensor of same shape as ``x`` with values in ``[min_sigma, max_sigma]``.
    """
    sigma = nn.functional.softplus(x) + min_sigma
    return torch.clamp(sigma, min=min_sigma, max=max_sigma)


# =============================================================================
__all__ = [
    "bounded_positive_scale",
]
