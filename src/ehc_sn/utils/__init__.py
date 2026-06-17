import datetime
import logging
import math
import os
import warnings
from dataclasses import fields, is_dataclass, replace
from itertools import combinations
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import comb
from torch import Tensor
from torch import device as Device
from torch import dtype as Dtype

from ehc_sn.types import LocationBelief, Matrix, Reduction, Vector


def to_cpu(
    value: object,
) -> object:
    """Move a tensor to CPU if needed; return other values unchanged."""
    return value.cpu() if hasattr(value, "cpu") else value  # type: ignore[union-attr]


def inv_var_weight(
    mus,
    sigmas,
    *,
    eps: float = 1e-4,
) -> tuple[Tensor, Tensor]:
    """
    Accepts lists batches of row vectors of means and standard deviations, with batches along dim 0.
    Returns inverse-variance weighted averages and standard deviations.
    """
    mus = torch.stack(mus, dim=0)
    sigmas = torch.stack(sigmas, dim=0)

    max_sigma = 1.0 / eps
    safe_sigmas = torch.nan_to_num(
        sigmas, nan=eps, posinf=max_sigma, neginf=eps
    )
    safe_sigmas = torch.clamp(safe_sigmas, min=eps, max=max_sigma)
    precision = 1.0 / safe_sigmas.square()
    inv_var_var = 1.0 / precision.sum(dim=0).clamp_min(eps)
    inv_var_avg = (mus * precision).sum(dim=0) * inv_var_var
    inv_var_sigma = torch.sqrt(inv_var_var.clamp_min(eps))

    return inv_var_avg, inv_var_sigma


def has_any_grad(
    opt: Any,
) -> bool:
    warnings.warn(
        "has_any_grad is deprecated for training control-plane logic; "
        "use only for diagnostics and assertions. "
        "Remove the guard: stepping an active optimizer with zero gradients "
        "is harmless and avoids scheduler drift.",
        FutureWarning,
        stacklevel=2,
    )
    raw_opt = getattr(opt, "optimizer", opt)
    return any(
        param.grad is not None
        for group in raw_opt.param_groups
        for param in group["params"]
    )


def find_multiple(
    n: int,
    k: int,
) -> int:
    """Round ``n`` up to the nearest multiple of ``k``."""
    return n if n % k == 0 else n + k - (n % k)


def trunc_normal_init_(
    tensor: Tensor,
    std: float = 1.0,
    lower: float = -2.0,
    upper: float = 2.0,
) -> Tensor:
    # NOTE: PyTorch nn.init.trunc_normal_ is not mathematically correct, the std dev is not actually the std dev of initialized tensor
    # This function is a PyTorch version of jax truncated normal init (default init method in flax)
    # https://github.com/jax-ml/jax/blob/main/jax/_src/random.py#L807-L848
    # https://github.com/jax-ml/jax/blob/main/jax/_src/nn/initializers.py#L162-L199

    with torch.no_grad():
        if std == 0:
            tensor.zero_()
        else:
            sqrt2 = math.sqrt(2)
            a = math.erf(lower / sqrt2)
            b = math.erf(upper / sqrt2)
            z = (b - a) / 2

            c = (2 * math.pi) ** -0.5
            pdf_u = c * math.exp(-0.5 * lower**2)
            pdf_l = c * math.exp(-0.5 * upper**2)
            comp_std = std / math.sqrt(
                1
                - (upper * pdf_u - lower * pdf_l) / z
                - ((pdf_u - pdf_l) / z) ** 2
            )

            tensor.uniform_(a, b)
            tensor.erfinv_()
            tensor.mul_(sqrt2 * comp_std)
            tensor.clip_(lower * comp_std, upper * comp_std)

    return tensor


def sample_diag_gaussian(
    transition: LocationBelief,
    *,
    scale: float = 1.0,
) -> list[Tensor]:
    """Sample a diagonal Gaussian distribution.

    Args:
        transition: LocationBelief object with mean and uncertainty.
        scale: Optional noise scale multiplier.

    Returns:
        Sampled activations if enabled, otherwise the means.
    """
    mu, sigma = transition.mean, transition.uncertainty
    return [
        mu_f + float(scale) * sigma_f * torch.randn_like(mu_f)
        for mu_f, sigma_f in zip(mu, sigma)
    ]


def expand_row_mask(
    flag: Tensor,
    ref: Tensor,
) -> Tensor:
    """Expand a boolean row mask to match a batched tensor."""
    return flag.view((-1,) + (1,) * (ref.ndim - 1))


def merge_rows(
    flag: Tensor,
    current: Tensor,
    fresh: Tensor,
) -> Tensor:
    """Select fresh rows where ``flag`` is true and keep current rows otherwise."""
    return torch.where(expand_row_mask(flag, current), fresh, current)


def merge_multiscale_rows(
    flag: Tensor,
    current: Sequence[Tensor],
    fresh: Sequence[Tensor],
) -> list[Tensor]:
    """Apply row-wise replacement over a multiscale latent code."""
    return [
        merge_rows(flag, current_f, fresh_f)
        for current_f, fresh_f in zip(current, fresh, strict=True)
    ]


def merge_tree_rows(
    flag: Tensor,
    current: Any,
    fresh: Any,
) -> Any:
    """Recursively apply row-wise replacement to tensor-bearing structures.

    Supported inputs are tensors, dataclass instances, dictionaries, and
    nested lists/tuples composed from those leaves. Both inputs must share the
    same structure.
    """
    if isinstance(current, torch.Tensor) and isinstance(fresh, torch.Tensor):
        return merge_rows(flag, current, fresh)

    if type(current) is not type(fresh):
        raise TypeError(
            f"merge_tree_rows requires matching types, got {type(current).__name__} and {type(fresh).__name__}"
        )

    if isinstance(current, list):
        return [
            merge_tree_rows(flag, current_v, fresh_v)
            for current_v, fresh_v in zip(current, fresh, strict=True)
        ]

    if isinstance(current, tuple):
        return tuple(
            merge_tree_rows(flag, current_v, fresh_v)
            for current_v, fresh_v in zip(current, fresh, strict=True)
        )

    if isinstance(current, dict):
        if current.keys() != fresh.keys():
            raise TypeError("merge_tree_rows requires matching dict keys.")
        return {
            key: merge_tree_rows(flag, current[key], fresh[key])
            for key in current
        }

    if is_dataclass(current) and is_dataclass(fresh):
        updates = {
            field.name: merge_tree_rows(
                flag, getattr(current, field.name), getattr(fresh, field.name)
            )
            for field in fields(current)
        }
        return replace(current, **updates)

    raise TypeError(
        f"Unsupported merge_tree_rows input type: {type(current).__name__}"
    )


def inv_var_trans(
    base: LocationBelief,
    corr: LocationBelief,
    mask: Optional[Tensor] = None,
    freqs: Optional[range] = None,
) -> LocationBelief:
    """Fuse correction into base using inverse-variance weighting.

    Supports selective fusion by frequency range and batch masking.
    Uses precision weighting to combine two multi-scale estimates.
    """
    freqs = range(len(base.mean)) if freqs is None else freqs
    mu_out, sigma_out = list(base.mean), list(base.uncertainty)
    idx = slice(None) if mask is None else mask

    for i, f in enumerate(freqs):
        mu_f, sigma_f = base.mean[f].clone(), base.uncertainty[f].clone()
        mu_f[idx], sigma_f[idx] = inv_var_weight(
            [base.mean[f][idx], corr.mean[i]],
            [base.uncertainty[f][idx], corr.uncertainty[i]],
        )
        mu_out[f], sigma_out[f] = mu_f, sigma_f

    return LocationBelief(mean=mu_out, uncertainty=sigma_out)


def softmax(o):
    """
    Applies softmax to tensors of inputs, using torch softmax function
    Assumes o is a 1D vector, or batches of row vectors with the batches along dim 0
    """
    # Return torch softmax
    return torch.nn.Softmax(dim=-1)(o)


def normalize(o):
    """
    normalizes vector of input to unit norm, using torch normalize function
    Assumes o is a 1D vector, or batches of row vectors with the batches along dim 0
    """
    # Return torch normalize with p=2 for L2 norm
    return torch.nn.functional.normalize(o, p=2, dim=-1)


def relu(o):
    """
    Applies rectified linear activation unit to tensors of inputs, using torch relu function
    """
    # Return torch relu
    return torch.nn.functional.relu(o)


def leaky_relu(o):
    """
    Applies leaky (meaning small negative slope instead of zeros) rectified linear activation unit to tensors of inputs, using torch leaky relu function
    """
    # Return torch leaky relu [torch.nn.functional.leaky_relu(val) for val in o] if type(o) is list else
    return torch.nn.functional.leaky_relu(o)


def activation_from_str(name: str):
    """Return activation function from string name."""
    if name == "none":
        return lambda x: x
    if name == "relu":
        return torch.nn.functional.relu
    name = name.lower()
    if name == "leaky_relu":
        return torch.nn.functional.leaky_relu
    if name == "sigmoid":
        return torch.sigmoid
    if name == "softmax":
        return torch.nn.Softmax(dim=-1)
    if name == "tanh":
        return torch.tanh
    if name == "exp":
        return torch.exp
    raise ValueError(f"Unknown activation function: {name}")


def squared_error(value, target):
    """
    Calculates mean squared error (L2 norm) between (list of) tensors value and target by using torch MSE loss
    Include a factor 0.5 to squared error by convention
    Set reduction to none, then get mean over last dimension to keep losses of different batches separate
    """
    # Return torch MSE loss
    if type(value) is list:
        loss = [
            0.5
            * torch.sum(
                torch.nn.MSELoss(reduction="none")(value[i], target[i]), dim=-1
            )
            for i in range(len(value))
        ]
    else:
        loss = 0.5 * torch.sum(
            torch.nn.MSELoss(reduction="none")(value, target), dim=-1
        )
    return loss


def cross_entropy(value, target):
    """
    Calculates binary cross entropy between tensors value and target by using torch cross entropy loss
    Set reduction to none, then get mean over last dimension to keep losses of different batches separate
    """
    # Return torch BCE loss
    if type(value) is list:
        loss = [
            torch.nn.CrossEntropyLoss(reduction="none")(v, t)
            for v, t in zip(value, target)
        ]
    else:
        loss = torch.nn.CrossEntropyLoss(reduction="none")(value, target)
    return loss


def downsample(value, target_dim):
    """
    Does downsampling by taking the an input vector, then averaging chunks to make it of requested dimension
    Assumes o is a 1D vector, or batches of row vectors with the batches along dim 0
    """
    # Get input dimension
    value_dim = value.size()[-1]
    # Set places to break up input vector into chunks
    edges = np.append(
        np.round(np.arange(0, value_dim, float(value_dim) / target_dim)),
        value_dim,
    ).astype(int)
    # Create downsampling matrix
    downsample = torch.zeros((value_dim, target_dim), dtype=torch.float)
    # Fill downsampling matrix with chunks
    for curr_entry in range(target_dim):
        downsample[edges[curr_entry] : edges[curr_entry + 1], curr_entry] = (
            torch.tensor(
                1.0 / (edges[curr_entry + 1] - edges[curr_entry]),
                dtype=torch.float,
            )
        )
    # Do downsampling by matrix multiplication
    return torch.matmul(value, downsample)


def make_directories():
    """
    Creates directories for storing data during a model training run
    """
    # Get current date for saving folder
    date = datetime.datetime.today().strftime("%Y-%m-%d")
    # Initialise the run and dir_check to create a new run folder within the current date
    run = 0
    dir_check = True
    # Initialise all paths
    train_path, model_path, save_path, script_path, run_path = (
        None,
        None,
        None,
        None,
        None,
    )
    # Find the current run: the first run that doesn't exist yet
    while dir_check:
        # Construct new paths
        run_path = "../Summaries/" + date + "/run" + str(run) + "/"
        train_path = run_path + "train"
        model_path = run_path + "model"
        save_path = run_path + "save"
        script_path = run_path + "script"
        envs_path = script_path + "/envs"
        run += 1
        # And once a path doesn't exist yet: create new folders
        if (
            not os.path.exists(train_path)
            and not os.path.exists(model_path)
            and not os.path.exists(save_path)
        ):
            os.makedirs(train_path)
            os.makedirs(model_path)
            os.makedirs(save_path)
            os.makedirs(script_path)
            os.makedirs(envs_path)
            dir_check = False
    # Return folders to new path
    return run_path, train_path, model_path, save_path, script_path, envs_path


def set_directories(date, run):
    """
    Returns directories for storing data during a model training run from a given previous training run
    """
    # Initialise all paths
    train_path, model_path, save_path, script_path, run_path = (
        None,
        None,
        None,
        None,
        None,
    )
    # Find the current run: the first run that doesn't exist yet
    run_path = "../Summaries/" + date + "/run" + str(run) + "/"
    train_path = run_path + "train"
    model_path = run_path + "model"
    save_path = run_path + "save"
    script_path = run_path + "script"
    envs_path = script_path + "/envs"
    # Return folders to new path
    return run_path, train_path, model_path, save_path, script_path, envs_path


def make_logger(run_path):
    """
    Creates logger so output during training can be stored to file in a consistent way
    """
    # Create new logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # Remove any existing handlers so you don't output to old files, or to new files twice
    logger.handlers = []
    # Create a file handler, but only if the handler does
    handler = logging.FileHandler(run_path + "report.log")
    handler.setLevel(logging.INFO)
    # Create a logging format
    formatter = logging.Formatter("%(asctime)s: %(message)s")
    handler.setFormatter(formatter)
    # Add the handlers to the logger
    logger.addHandler(handler)
    # Return the logger object
    return logger


def as_dir_str(
    path: Path,
) -> str:
    """Return a directory path as a string with trailing separator for legacy utils."""
    s = str(path)
    return s if s.endswith(os.sep) else s + os.sep


def resolve_envs_path(
    run_path: Path,
) -> Path:
    """Find envs directory for a run (prefer script/envs, fall back to envs)."""
    candidate = run_path / "script" / "envs"
    if candidate.exists():
        return candidate
    candidate = run_path / "envs"
    return candidate


def parse_iter_from_stem(
    stem: str,
) -> Optional[int]:
    """Parse iteration number from checkpoint stem like 'tem_4000' or 'params_4000'"""
    parts = stem.split("_")
    if len(parts) < 2:
        return None
    try:
        return int(parts[-1])
    except ValueError:
        return None


def apply_overrides(
    params: dict[str, Any],
    override: dict[str, Any],
) -> dict[str, Any]:
    """Return params with override applied (shallow merge)."""
    if not override:
        return params
    for k, v in override.items():
        params[k] = v
    return params


def require_exists(
    path: Path,
    what: str,
) -> None:
    """Raise friendly error if path doesn't exist."""
    if not path.exists():
        raise FileNotFoundError(f"{what} not found: {path}")


def reduce_per_env(
    loss_per_env: Tensor,
    reduction: Reduction,
) -> Tensor:
    """Reduce a per-environment loss vector according to `reduction`."""
    if reduction == "sum":
        return loss_per_env.sum()
    if reduction == "mean":
        return loss_per_env.mean()
    if reduction == "none":
        return loss_per_env
    raise ValueError(f"Unknown reduction: {reduction}")


def create_downsample_matrix(
    n: list[int],
    n_subsampled: list[int],
) -> list[Matrix]:
    """Create downsampling matrices.

    Downsampling matrix to go from cells to compressed cells for
    indexing memories by simply taking only the first n_subsampled cells.

    Args:
        n: Full input dimensions per frequency
        n_subsampled: Subsampled output dimensions per frequency

    Returns:
        List of downsampling matrices, one per frequency module
    """
    # Matrix shape: [n_in, n_out] where we select first n_out columns
    # For input o: [B, n_in], result is o @ W_down = [B, n_out]
    return [
        torch.cat(
            [
                torch.eye(dim_out, dtype=torch.float),
                torch.zeros((dim_in - dim_out, dim_out), dtype=torch.float),
            ]
        )
        for dim_in, dim_out in zip(n, n_subsampled)
    ]


def create_repeat_matrices(
    n_subsampled: list[int],
    n: list[int],
) -> list[Matrix]:
    """Create repeat matrices.

    Matrix for repeating cells information using element wise product
    after matrix multiplication.

    Args:
        n_subsampled: Subsampled input dimensions per frequency
        n: Full output dimensions per frequency

    Returns:
        List of repeat matrices, one per frequency module
    """
    # Matrix shape: [n_subsampled, n_p] where each row is repeated
    # For input g: [B, n_subsampled], result is g @ W_repeat = [B, n_p]
    # Uses Kronecker product: eye(n_subsampled) ⊗ ones(1, n_p/n_subsampled)
    return [
        torch.tensor(
            np.kron(np.eye(dim_in), np.ones((1, dim_out // dim_in))),
            dtype=torch.float,
        )
        for dim_in, dim_out in zip(n_subsampled, n)
    ]


def create_tiling_matrices(
    n_in: list[int],
    n_out: list[int],
) -> list[Matrix]:
    """Create tile matrices.

    Tiling matrix to project from one cortical region to another by repeating
    the input representation multiple times.

    Args:
        n_in: Input dimensions per frequency module
        n_out: Output dimensions per frequency module

    Returns:
        List of tile matrices, one per frequency module.
    """
    # Matrix shape: [n_in, n_out] where each input is tiled
    # For input o_c: [B, n_in], result is o_c @ W_tile = [B, n_out]
    # Uses Kronecker product: ones(1, n_tiles) ⊗ eye(n_in)
    # where n_tiles = n_out / n_in

    # Validate divisibility
    if any(out % inp != 0 for out, inp in zip(n_out, n_in)):
        raise ValueError(
            f"n_out must be divisible by n_in. Got n_out={n_out}, n_in={n_in}"
        )

    return [
        torch.tensor(
            np.kron(np.ones((1, out // inp)), np.eye(inp)), dtype=torch.float
        )
        for inp, out in zip(n_in, n_out)
    ]


def create_random_projection(
    n_in: list[int],
    n_out: list[int],
    sparsity: float = 1.0,
    seed: Optional[int] = None,
) -> list[Matrix]:
    """Create random fixed projection matrices.

    Biologically-inspired alternative to downsampling + W_repeat expansion.
    Models the random connectivity from EC (grid cells) to HPC (place cells)
    as observed in experimental data and used in CAN models (Chandra et al. 2025).

    This replaces the two-step structured transformation:
        g → downsample → g_ → W_repeat expansion → hippocampal space
    With a single direct random projection:
        g → W_random → hippocampal space

    The random projection is fixed (non-learnable) and can be sparse to match
    biological connectivity patterns (~10-20% in real circuits).

    Args:
        n_in: Input dimensions per frequency module
        n_out: Output dimensions per frequency module
        sparsity: Connection probability (1.0 = fully connected, 0.1 = 10% connectivity)
        seed: Random seed for reproducibility (optional)

    Returns:
        List of random projection matrices [n_g[f], n_p[f]], one per frequency module

    Example:
        >>> n_g = [36, 30, 24]  # Grid cell dimensions
        >>> n_p = [96, 80, 64]  # Place cell dimensions
        >>> W_random = create_random_projection(n_g, n_p, sparsity=0.15)
        >>> # Use in projection head:
        >>> g_ = [g[f] @ W_random[f] for f in range(n_f)]

    References:
        Chandra et al. (2025). "Episodic and associative memory from spatial
        scaffolds in the hippocampus." CAN model architecture.
    """
    if seed is not None:
        torch.manual_seed(seed)

    matrices = []
    for g_dim, p_dim in zip(n_in, n_out):
        # Random Gaussian initialization scaled by input dimension
        # This ensures variance is maintained across the projection
        W = torch.randn(g_dim, p_dim, dtype=torch.float) / np.sqrt(g_dim)

        # Apply sparsity mask if requested
        if sparsity < 1.0:
            mask = torch.rand(g_dim, p_dim) < sparsity
            W = W * mask.float()
            # Rescale to maintain expected magnitude after sparsification
            W = W / np.sqrt(sparsity)

        matrices.append(W)

    return matrices


def create_encoding_table(
    n_in: int,
    n_out: int,
    n_hot: int = 2,
) -> list[Vector]:
    """Create n-hot encoding lookup table.

    Generates a lookup table for converting one-hot observations to n-hot
    compressed representations. Each observation is encoded using exactly
    `n_hot` active units from `n_out` dimensions.

    Args:
        n_in: Number of possible observations (must be <= C(n_out, n_hot))
        n_out: Compressed sensory dimension
        n_hot: Number of active units per code (1, 2, 3, etc.)
            - n_hot=1: One-hot (identity, no compression unless n_in > n_out)
            - n_hot=2: Two-hot (default, typically 45 → 10)
            - n_hot=3: Three-hot (more distributed, e.g., 220 → 12)

    Returns:
        List of n-hot code tensors, one per possible observation [n_in, n_out]

    Raises:
        ValueError: If n_in > C(n_out, n_hot) (too many observations for compression)

    Example:
        >>> # Two-hot encoding: 45 observations → 10 dimensions
        >>> table = create_encoding_table(n_in=45, n_out=10, n_hot=2)
        >>> len(table)
        45
        >>> table[0].sum()
        2.0

        >>> # Three-hot encoding: 220 observations → 12 dimensions
        >>> table = create_encoding_table(n_in=220, n_out=12, n_hot=3)
        >>> len(table)
        220
        >>> table[0].sum()
        3.0
    """

    # Validate: number of observations must not exceed possible n-hot codes
    max_codes = int(comb(n_out, n_hot))
    if n_in > max_codes:
        raise ValueError(
            f"Cannot encode {n_in} observations with {n_hot}-hot codes in {n_out} dimensions. "
            f"Maximum possible codes: C({n_out}, {n_hot}) = {max_codes}"
        )

    # Generate all possible n-hot codes using combinations
    # combinations(range(n_out), n_hot) gives all ways to choose n_hot positions
    encoding_table = []
    for active_positions in combinations(range(n_out), n_hot):
        # Create zero vector
        code = [0] * n_out
        # Activate n_hot positions
        for pos in active_positions:
            code[pos] = 1
        # Add to table
        encoding_table.append(torch.tensor(code, dtype=torch.float))

        # Stop when we have enough codes for all observations
        if len(encoding_table) >= n_in:
            break

    return torch.stack(encoding_table, dim=0)


def uncat_to_list(
    x: Tensor,
    dims: list[int],
) -> list[Tensor]:
    """Split a concatenated tensor into a list of tensors with given last-dim sizes.

    Args:
        x: Tensor shaped (B, sum(dims)).
        dims: List of segment sizes.

    Returns:
        List of tensors [x0, x1, ...] where xi.shape == (B, dims[i]).
    """
    return list(torch.split(x, dims, dim=1))


def one_hot_with_zero(
    action: list[int | None],
    num_actions: int,
    device: Device | None = None,
) -> torch.Tensor:
    """
    Convert actions to one-hot encoding where action 0/None = all-zeros (static action).

    action: list of int or None
        0|None = static / no-op -> all zeros
        1..num_actions = discrete actions -> one-hot at index (action - 1)
    returns: FloatTensor, shape (len(action), num_actions)
    """
    action_t = torch.tensor(
        [a if a is not None else 0 for a in action],
        dtype=torch.long,
        device=device,
    )
    mask = action_t > 0
    out = torch.zeros(
        (len(action), num_actions), dtype=torch.float32, device=device
    )
    if mask.any():
        out[mask] = F.one_hot(
            action_t[mask] - 1, num_classes=num_actions
        ).float()
    return out


def connections(
    f_grid: list[float],
) -> list[list[bool]]:
    """Compute hierarchical connection matrix from frequency list.

    Entry [f_to][f_from] is True if f_from connects to f_to.
    Connection rule: lower-frequency modules connect to higher-frequency.

    Args:
        f_grid: Frequency values per module (higher value = higher frequency)

    Returns:
        Connection matrix where [f_to][f_from] indicates connectivity
    """
    n = len(f_grid)
    return [[f_grid[f1] <= f_grid[f2] for f1 in range(n)] for f2 in range(n)]


def resolve_ovc_slice(
    n_freq_total: int,
    n_freq_ovc: Optional[int],
) -> tuple[int, int]:
    """Determine which MEC modules are OVC (receive shiny landmark correction).

    Args:
        n_freq_total: Total number of MEC modules
        n_freq_ovc: User setting for OVC module count:
            - None: all modules are OVC (legacy separate_ovc=False)
            - 0: no OVC modules (disable shiny correction)
            - k>0: last k modules are OVC (legacy separate_ovc=True, n_f_ovc=k)

    Returns:
        (ovc_start, ovc_count): slice range [ovc_start:ovc_start+ovc_count] for OVC modules

    Examples:
        >>> resolve_ovc_slice(5, None)  # All modules are OVC
        (0, 5)
        >>> resolve_ovc_slice(5, 0)     # No OVC modules
        (5, 0)
        >>> resolve_ovc_slice(5, 2)     # Last 2 modules are OVC
        (3, 2)
    """
    if n_freq_ovc is None:
        # Apply to all modules (legacy separate_ovc=False)
        return 0, n_freq_total

    n_freq_ovc = int(n_freq_ovc)
    if n_freq_ovc < 0 or n_freq_ovc > n_freq_total:
        raise ValueError(
            f"MECSettings.n_freq_ovc must be in [0, {n_freq_total}] or None; got {n_freq_ovc}"
        )

    if n_freq_ovc == 0:
        # No OVC correction
        return n_freq_total, 0

    # Apply to last k modules (legacy separate_ovc=True)
    return n_freq_total - n_freq_ovc, n_freq_ovc


def update_to_masks(
    shape: list[int],
    *,
    update: torch.Tensor,
) -> torch.Tensor:
    """Expand a stage×freq update matrix to a stage×sum(shape) mask tensor.

    Args:
        shape: Feature dims per frequency module.
        update: Bool/0-1 tensor of shape (n_stages, n_freq).

    Returns:
        masks: Float tensor of shape (n_stages, sum(hpc_shape)).
    """
    n_freq = len(shape)
    if update.ndim != 2 or update.shape[1] != n_freq:
        raise ValueError(
            f"Expected update shape (n_stages, {n_freq}), got {tuple(update.shape)}"
        )

    widths = torch.tensor(shape, device=update.device)
    masks = update.to(dtype=torch.float).repeat_interleave(widths, dim=1)
    return masks


def make_update_full(
    n_stages: int,
    n_freq: int,
    *,
    device=None,
) -> torch.Tensor:
    """Full update matrix (all True).

    Args:
        n_stages: Number of attractor stages.
        n_freq: Number of frequency modules.

    Returns:
        Full update matrix of shape (n_stages, n_freq).
    """
    return torch.ones((n_stages, n_freq), dtype=torch.bool, device=device)


def make_update_hierarchical(
    n_stages: int,
    n_freq: int,
    *,
    ramp_len: int | None = None,
    device=None,
) -> torch.Tensor:
    """Hierarchical update matrix.

    ramp_len controls how many frequency modules participate in the triangular ramp.
    If None, uses min(n_stages, n_freq).
    """
    ramp_len = min(n_stages, n_freq) if ramp_len is None else ramp_len
    if not (0 <= ramp_len <= n_freq):
        raise ValueError("ramp_len must be in [0, n_freq]")

    i = torch.arange(n_stages, device=device)[:, None]  # (n_stages, 1)
    f = torch.arange(ramp_len, device=device)[None, :]  # (1, ramp_len)

    # Triangular ramp: freq 0 updates n_stages times, freq 1 updates n_stages-1, etc.
    update_ramp = i < (n_stages - f)  # (n_stages, ramp_len)

    if ramp_len < n_freq:
        update_tail = torch.ones(
            (n_stages, n_freq - ramp_len), dtype=torch.bool, device=device
        )
        return torch.cat([update_ramp, update_tail], dim=1)

    return update_ramp


def make_hebbian_write_mask(
    n_stages: int,
    shape: list[int],
    f_initial: list[float],
) -> torch.Tensor:
    """Create a Hebbian write-connectivity mask for a block-structured memory matrix.

    This mask gates which synapses in the hippocampal memory matrix are allowed
    to be updated by the Hebbian outer product during memory writes.

    The memory matrix is block-structured by frequency module, with blocks of
    size ``shape[f]``. The rule is:

    - Within *stages* modules (``f < n_stages``): allow only low→high frequency
      connections based on ``f_initial[from] <= f_initial[to]``.
    - Within *attractor* modules (``f >= n_stages``): allow all connections (fully recurrent).
    - Between modules: allow all connections (bidirectional).

    Args:
        n_stages: Number of attractor stages (i.e., constrained modules).
        shape: Feature dims per frequency module.
        f_initial: Frequency values per module (length must match ``len(shape)``).

    Returns:
        A tensor of shape ``(sum(shape), sum(shape))`` with 1.0 for allowed
        connections and 0.0 otherwise.
    """
    n_freq = len(shape)
    if n_freq != len(f_initial):
        raise ValueError(
            f"Expected f_initial length {n_freq}, got {len(f_initial)}"
        )
    if not (0 <= int(n_stages) <= n_freq):
        raise ValueError(f"n_stages must be in [0, {n_freq}], got {n_stages}")

    # Module-level adjacency A[from, to]
    module = torch.arange(n_freq)
    is_constrained = module < int(n_stages)
    same_type = is_constrained[:, None] == is_constrained[None, :]

    f = torch.as_tensor(f_initial)
    low_to_high = f[:, None] <= f[None, :]
    allow = (~same_type) | low_to_high

    # Expand module-level adjacency to unit-level connectivity
    widths = torch.as_tensor(shape, dtype=torch.long)
    module_id = module.repeat_interleave(widths)
    mask = allow[module_id[:, None], module_id[None, :]]
    return mask
