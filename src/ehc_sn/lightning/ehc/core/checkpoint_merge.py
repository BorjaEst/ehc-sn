"""Checkpoint-merge utility for the EHC v1 phase-3 alignment trainer.

Phase-3 initial alignment assembles a shared backbone from two pre-trained
checkpoints instead of assuming one checkpoint already absorbed both:

    arena checkpoint  → spatial substrate weights + arena decoder
    mazehard checkpoint → controller weights + mazehard encoder/decoder

``pfc_to_hpc`` and ``hpc_to_pfc`` are intentionally left at fresh initialization
in both cases: aligning these projections is the explicit training objective of
the initial alignment phase.

Public API
----------
merge_ehc_alignment_checkpoints(arena_ckpt_path, mazehard_ckpt_path, *, ...)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)


# =============================================================================
@dataclass(frozen=True)
class MergeReport:
    """Summary of keys successfully loaded from each checkpoint source."""

    spatial_keys_from_arena: list[str]
    """Model keys loaded from the arena checkpoint (lec/mec/hpc/lec_to_hpc/mec_to_hpc)."""

    decoder_keys_from_arena: list[str]
    """Arena decoder keys loaded from the arena checkpoint."""

    controller_keys_from_mazehard: list[str]
    """Model keys loaded from the mazehard checkpoint (pfc/str)."""

    encoder_keys_from_mazehard: list[str]
    """Mazehard encoder keys loaded from the mazehard checkpoint."""

    decoder_keys_from_mazehard: list[str]
    """Mazehard decoder keys loaded from the mazehard checkpoint."""

    fresh_init_keys: list[str]
    """Keys deliberately left at fresh initialization (pfc_to_hpc, hpc_to_pfc)."""


# =============================================================================
def _load_state_dict_from_checkpoint(path: Path) -> dict[str, Any]:
    """Load the ``state_dict`` payload from a Lightning checkpoint file."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "state_dict" not in ckpt:
        raise KeyError(f"Checkpoint at '{path}' has no 'state_dict' key. Found: {list(ckpt.keys())}")
    return ckpt["state_dict"]


def _load_prefix(
    source_sd: dict[str, Any],
    prefix: str,
    target: nn.Module,
    *,
    strict: bool = False,
) -> list[str]:
    """Load matching keys from *source_sd* into *target*, stripping *prefix*.

    Returns the list of successfully matched source keys.
    """
    stripped: dict[str, Any] = {}
    for k, v in source_sd.items():
        if k.startswith(prefix):
            stripped[k[len(prefix):]] = v

    if not stripped:
        logger.warning("merge: no keys found under prefix '%s' in source checkpoint.", prefix)
        return []

    missing, unexpected = target.load_state_dict(stripped, strict=strict)
    if missing:
        logger.warning("merge: missing keys in target for prefix '%s': %s", prefix, missing)
    if unexpected:
        logger.warning("merge: unexpected keys from source for prefix '%s': %s", prefix, unexpected)

    return [prefix + k for k in stripped]


# =============================================================================
def merge_ehc_alignment_checkpoints(
    arena_ckpt_path: Path | str,
    mazehard_ckpt_path: Path | str,
    *,
    model: nn.Module,
    arena_decoder: nn.Module,
    mazehard_encoder: nn.Module,
    mazehard_decoder: nn.Module,
) -> MergeReport:
    """Merge two task-specific checkpoints into a shared alignment model.

    Spatial substrate weights (lec, mec, hpc, lec_to_hpc, mec_to_hpc) come
    from the arena checkpoint; controller weights (pfc, str) and both mazehard
    adapter components come from the mazehard checkpoint.

    ``model.projections.pfc_to_hpc`` and ``model.projections.hpc_to_pfc`` are
    explicitly left at fresh initialization — training these projections is the
    objective of phase-3 initial alignment.

    Args:
        arena_ckpt_path: Path to the phase-1 arena training checkpoint (.ckpt).
        mazehard_ckpt_path: Path to the phase-2 mazehard training checkpoint (.ckpt).
        model: Freshly constructed shared ``EHCModelV1`` instance.
        arena_decoder: ``arena_adapter._decoder`` module of the alignment model.
        mazehard_encoder: ``mazehard_adapter._encoder`` module of the alignment model.
        mazehard_decoder: ``mazehard_adapter._decoder`` module of the alignment model.

    Returns:
        :class:`MergeReport` summarizing which keys came from which source.

    Raises:
        KeyError: If either checkpoint file does not contain a ``state_dict`` key.
        FileNotFoundError: If either checkpoint file does not exist.
    """
    arena_ckpt_path = Path(arena_ckpt_path)
    mazehard_ckpt_path = Path(mazehard_ckpt_path)

    logger.info("merge: loading arena checkpoint from '%s'", arena_ckpt_path)
    arena_sd = _load_state_dict_from_checkpoint(arena_ckpt_path)

    logger.info("merge: loading mazehard checkpoint from '%s'", mazehard_ckpt_path)
    mazehard_sd = _load_state_dict_from_checkpoint(mazehard_ckpt_path)

    # ---- Spatial substrate: from arena checkpoint ---------------------------
    # Load region modules and spatial projections.  Skip pfc, str, pfc_to_hpc,
    # hpc_to_pfc — they come from the mazehard checkpoint or fresh init.
    spatial_keys: list[str] = []
    for sub in ("lec", "mec", "hpc"):
        spatial_keys += _load_prefix(arena_sd, f"model.{sub}.", getattr(model, sub))
    for proj in ("lec_to_hpc", "mec_to_hpc"):
        spatial_keys += _load_prefix(arena_sd, f"model.projections.edges.{proj}.", getattr(model, proj))

    # ---- Arena decoder: from arena checkpoint --------------------------------
    decoder_arena_keys = _load_prefix(arena_sd, "bridge_adapter._decoder.", arena_decoder)

    # ---- Controller: from mazehard checkpoint --------------------------------
    ctrl_keys: list[str] = []
    for sub in ("pfc", "str"):
        ctrl_keys += _load_prefix(mazehard_sd, f"model.{sub}.", getattr(model, sub))

    # ---- MazeHard adapter: from mazehard checkpoint -------------------------
    encoder_mh_keys = _load_prefix(mazehard_sd, "bridge_adapter._encoder.", mazehard_encoder)
    decoder_mh_keys = _load_prefix(mazehard_sd, "bridge_adapter._decoder.", mazehard_decoder)

    # ---- Fresh initialization notice ----------------------------------------
    fresh_prefixes = ["model.projections.edges.pfc_to_hpc.", "model.projections.edges.hpc_to_pfc."]
    fresh_keys: list[str] = [
        k for k in {**arena_sd, **mazehard_sd}
        if any(k.startswith(p) for p in fresh_prefixes)
    ]
    logger.info(
        "merge: pfc_to_hpc and hpc_to_pfc are left at fresh initialization "
        "(%d keys not loaded: %s). These projections are the training objective of phase-3.",
        len(fresh_keys),
        fresh_keys[:6],
    )

    return MergeReport(
        spatial_keys_from_arena=spatial_keys,
        decoder_keys_from_arena=decoder_arena_keys,
        controller_keys_from_mazehard=ctrl_keys,
        encoder_keys_from_mazehard=encoder_mh_keys,
        decoder_keys_from_mazehard=decoder_mh_keys,
        fresh_init_keys=fresh_keys,
    )


# =============================================================================
__all__ = ["MergeReport", "merge_ehc_alignment_checkpoints"]
