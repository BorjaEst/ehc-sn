import torch


def w_gh(g: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Compute the weight matrix from hippocampal cells to grid cells.

    Args:
        g (torch.Tensor): Grid cell activations of shape (n_attractors, n_grid_cells).
        h (torch.Tensor): Hippocampal cell activations of shape (n_attractors, n_hpc_cells).

    Returns:
        torch.Tensor: Weight matrix of shape (n_grid_cells, n_hpc_cells).
    """
    hpc_dim = h.shape[1]  # n_hpc_cells
    return torch.einsum("ni, nj -> ij", g, h) / hpc_dim  # (3)


def w_gg(g: torch.Tensor) -> torch.Tensor:
    """Compute the recurrent weight matrix for grid cells.

    Args:
        g (torch.Tensor): Grid cell activations of shape (n_attractors, n_grid_cells).

    Returns:
        torch.Tensor: Recurrent weight matrix of shape (n_grid_cells, n_grid_cells).
    """
    grid_dim = g.shape[1]  # n_grid_cells
    return torch.einsum("ni, nj -> ij", g, g) / grid_dim  # (3*)
