"""Private shared helpers for HRM training entrypoints."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch import nn

# Mapping from public semantic group name to HRM model state-dict prefixes.
_GROUP_TO_PREFIXES: dict[str, tuple[str, ...]] = {
    "pfc_core": ("pfc",),
    "striatum": ("str",),
    "all": (),
}

VALID_INIT_GROUPS: frozenset[str] = frozenset(_GROUP_TO_PREFIXES)
"""Set of valid semantic group names for HRM init-only hydration."""


# =============================================================================
def load_weights_from_checkpoint(  # ------------------------------------------
    model: nn.Module,
    checkpoint_path: str | Path,
    groups: Sequence[str],
) -> list[str]:
    """Hydrate named HRM semantic-group weights from a checkpoint into *model*.

    Loads only model parameter subsets for the specified groups. Optimizer,
    scheduler, and trainer-progress state in the checkpoint are not touched.
    This is distinct from ``Trainer.fit(ckpt_path=...)`` full-state resume.

    Lightning checkpoint keys prefixed with ``"model."`` are supported and
    stripped before matching.

    Args:
        model: Target HRM model to hydrate in place.
        checkpoint_path: Path to a Lightning or raw-model checkpoint.
        groups: Non-empty sequence of semantic group names.

    Returns:
        Sorted list of model state-dict keys loaded from the checkpoint.

    Raises:
        ValueError: If *groups* is empty, contains unknown names, or no keys
            matched the requested groups.
    """
    if not groups:
        raise ValueError("groups must be non-empty.")
    unknown = [g for g in groups if g not in _GROUP_TO_PREFIXES]
    if unknown:
        raise ValueError(
            f"Unknown semantic groups: {unknown!r}. "
            f"Valid groups: {sorted(_GROUP_TO_PREFIXES)!r}."
        )

    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw_sd: dict = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    stripped: dict = {
        (k[len("model.") :] if k.startswith("model.") else k): v
        for k, v in raw_sd.items()
    }
    model_keys = set(model.state_dict().keys())

    use_all = "all" in groups
    prefixes = tuple(p for g in groups for p in _GROUP_TO_PREFIXES[g])
    filtered = {
        k: v
        for k, v in stripped.items()
        if k in model_keys
        and (use_all or any(k == p or k.startswith(f"{p}.") for p in prefixes))
    }
    if not filtered:
        raise ValueError(
            f"No model keys matched for groups {list(groups)!r} "
            f"in checkpoint {str(checkpoint_path)!r}."
        )

    model.load_state_dict(filtered, strict=False)
    return sorted(filtered)


# =============================================================================
__all__ = ["VALID_INIT_GROUPS", "load_weights_from_checkpoint"]
