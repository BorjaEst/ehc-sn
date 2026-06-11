"""TEM v1 Hebbian memory self-retrieval diagnostic probe.

Determines whether the learned Hebbian attractor memory *M* can retrieve its
own stored posterior place codes (``p_post``).  Answers the single blocking
question — *"Can M self-retrieve?"* — before any decoder, inverse-projection,
or single-frequency debugging begins.

Public API
---------
- :class:`MemoryProbeResult` — Pydantic schema for the probe output.
- :func:`produce_tem_memory_probe` — run the probe against a loaded model and
  eval batch.
- :func:`persist_tem_memory_probe` — write the probe result to a JSON artifact.

Usage::

    from ehc_sn.diagnostics.tem_memory_probe import (
        produce_tem_memory_probe,
        persist_tem_memory_probe,
    )

    result = produce_tem_memory_probe(model, eval_batch)
    path = persist_tem_memory_probe(result, Path("probes/"))
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from pydantic import BaseModel, Field
from torch import Tensor

from ehc_sn.models.tem.tem_v1 import TEMModelV1, TEMInputV1
from ehc_sn.types import DenseMemoryStore


# =============================================================================
# Probe result schema
# =============================================================================


class MemoryStats(BaseModel, extra="forbid"):
    """Health statistics for the Hebbian memory matrix *M*.

    Attributes:
        memory_norm: Frobenius norm of *M* (mean over batch dim).
        memory_std: Standard deviation of *M* entries (mean over batch dim).
        memory_max: Maximum entry value across *M*.
        memory_min: Minimum entry value across *M*.
        memory_effective_rank: Effective (nuclear) rank of *M*,
            computed as *exp(H)* where *H* is the entropy of the normalized
            singular value distribution.  Mean over batch dim.
        memory_saturation_fraction: Fraction of *M* entries at the clamp
            boundary (within 1 % of ``clamp_min`` or ``clamp_max``).
    """

    memory_norm: float = Field(
        default=float("nan"), description="Frobenius norm of M (mean batch)."
    )
    memory_std: float = Field(
        default=float("nan"),
        description="Entry standard deviation of M (mean batch).",
    )
    memory_max: float = Field(
        default=float("nan"), description="Maximum entry of M."
    )
    memory_min: float = Field(
        default=float("nan"), description="Minimum entry of M."
    )
    memory_effective_rank: float = Field(
        default=float("nan"),
        description="Effective (nuclear) rank of M (mean batch).",
    )
    memory_saturation_fraction: float = Field(
        default=float("nan"),
        description="Fraction of M entries within 1 % of clamp boundary.",
    )


class PlaceCodeStats(BaseModel, extra="forbid"):
    """Place-code diversity statistics for one pathway.

    Attributes:
        pairwise_cosine_mean: Mean pairwise cosine similarity across all
            timestep pairs in the flattened code.  1.0 = all identical,
            0.0 = orthogonal.
        pairwise_cosine_std: Standard deviation of pairwise similarities.
        norm_mean: Mean L2 norm per timestep.
    """

    pairwise_cosine_mean: float = Field(default=float("nan"))
    pairwise_cosine_std: float = Field(default=float("nan"))
    norm_mean: float = Field(default=float("nan"))


class RetrievalResult(BaseModel, extra="forbid"):
    """Self- or grid-query retrieval performance.

    Attributes:
        cosine_mean: Mean cosine similarity between query and retrieved code.
        cosine_std: Standard deviation of retrieval cosines.
        nn_accuracy: Fraction of timesteps where the nearest neighbour of
            the retrieved code among all *candidates* is the same timestep.
        nn_temporal_distance_mean: Mean absolute temporal offset between
            the queried timestep and the nearest-neighbour timestep.
        nn_temporal_distance_median: Median absolute temporal offset.
        nn_same_observation_fraction: Fraction of nearest-neighbour pairs
            where query and NN have the same observation ID (requires
            ``observation_ids`` at probe time).
        nn_gt_observation_accuracy: Fraction where the *retrieved* code's
            decoded observation ID matches ground truth.
        top5_accuracy: Fraction where the true timestep is in the top-5
            nearest neighbours.
        top10_accuracy: Fraction where the true timestep is in the top-10
            nearest neighbours.
        nn_same_position_fraction: Fraction where query and NN share the
            same grid position (requires ``position_ids`` at probe time).
        nn_spatial_distance_mean: Mean Euclidean distance between query
            and NN positions (requires ``position_ids``).
    """

    cosine_mean: float = Field(default=float("nan"))
    cosine_std: float = Field(default=float("nan"))
    nn_accuracy: Optional[float] = Field(
        default=None, ge=0.0, le=1.0
    )
    nn_temporal_distance_mean: Optional[float] = Field(default=None)
    nn_temporal_distance_median: Optional[float] = Field(default=None)
    nn_same_observation_fraction: Optional[float] = Field(default=None)
    nn_gt_observation_accuracy: Optional[float] = Field(default=None)
    top5_accuracy: Optional[float] = Field(default=None)
    top10_accuracy: Optional[float] = Field(default=None)
    nn_same_position_fraction: Optional[float] = Field(default=None)
    nn_spatial_distance_mean: Optional[float] = Field(default=None)


class MemoryProbeResult(BaseModel, extra="forbid"):
    """Complete diagnostic output from one TEM memory probe run.

    Attributes:
        model_family: Always ``"tem-v1"``.
        checkpoint_hint: Optional label identifying the checkpoint.
        episode_steps: Number of timesteps in the probed episode.
        n_freq: Number of HPC frequency modules.
        feature_dim: Flattened memory-code dimension *S*.
        memory_stats: Hebbian memory matrix health.
        p_post: Place-code diversity for the posterior (inference) pathway.
        p_retrieved: Place-code diversity for the generative (retrieved)
            pathway.
        p_prior: Place-code diversity for the ancestral (prior) pathway.
        self_retrieval: Self-retrieval result — query *M* with ``p_post``.
        grid_retrieval: Optional grid-query retrieval result.
        retrieved_vs_ancestral_delta: Mean L2 distance between ``p_retrieved``
            and ``p_prior`` in flattened code space.  Near zero indicates
            that sensory correction is functionally inactive.
    """

    model_family: str = "tem-v1"
    checkpoint_hint: str = ""
    episode_steps: int = 0
    n_freq: int = 0
    feature_dim: int = 0

    memory_stats: MemoryStats = Field(default_factory=MemoryStats)
    p_post: PlaceCodeStats = Field(default_factory=PlaceCodeStats)
    p_retrieved: PlaceCodeStats = Field(default_factory=PlaceCodeStats)
    p_prior: PlaceCodeStats = Field(default_factory=PlaceCodeStats)
    self_retrieval: RetrievalResult = Field(default_factory=RetrievalResult)
    grid_retrieval: Optional[RetrievalResult] = None
    retrieved_vs_ancestral_delta: float = Field(default=float("nan"))


# =============================================================================
# Helper: pairwise cosine similarity over ``(T, D)``
# =============================================================================


def _pairwise_cosine(z: Tensor) -> Tensor:
    """Compute pairwise cosine similarity over the first dimension of *z*.

    Args:
        z: Tensor with shape ``(T, D)``.

    Returns:
        Tensor with shape ``(T, T)`` of cosine similarities.
    """
    z_norm = z / (z.norm(dim=-1, keepdim=True) + 1e-8)
    return z_norm @ z_norm.T


def _place_code_stats(code_bundle: list[Tensor]) -> PlaceCodeStats:
    """Compute diversity statistics from a multi-frequency place-code bundle.

    Args:
        code_bundle: Multi-frequency code, each tensor ``(T, D_f)``.

    Returns:
        Filled ``PlaceCodeStats``.
    """
    # Determine temporal dimension
    T = int(code_bundle[0].shape[0])
    if T <= 1:
        return PlaceCodeStats()

    z = torch.cat(code_bundle, dim=-1)  # (T, S)
    norms = z.norm(dim=-1)
    sim = _pairwise_cosine(z)
    # Exclude self-pairs (diagonal) for the mean
    triu_indices = torch.triu_indices(T, T, offset=1, device=sim.device)
    off_diag = sim[triu_indices[0], triu_indices[1]]
    return PlaceCodeStats(
        pairwise_cosine_mean=float(off_diag.mean().item()),
        pairwise_cosine_std=float(off_diag.std().item()),
        norm_mean=float(norms.mean().item()),
    )


def _memory_stats(m: Tensor, clamp_min: float = -1.0, clamp_max: float = 1.0) -> MemoryStats:
    """Compute memory matrix health statistics.

    Args:
        m: Memory matrix with shape ``(B, S, S)``.
        clamp_min: Lower clamp bound (used for saturation detection).
        clamp_max: Upper clamp bound (used for saturation detection).

    Returns:
        Filled ``MemoryStats``.
    """
    B = int(m.shape[0])
    m_flat = m.view(B, -1)
    fro_norm = float(m.norm(p="fro", dim=(1, 2)).mean().item())
    std_val = float(m_flat.std(dim=1).mean().item())

    # Effective rank via SV entropy
    ranks = []
    for b in range(min(B, 8)):
        s = torch.linalg.svdvals(m[b].to(dtype=torch.float64))
        s = s[s > 1e-12]
        if s.numel() > 1:
            p = s / s.sum()
            entropy = -(p * p.log()).sum()
            ranks.append(float(torch.exp(entropy).item()))
        elif s.numel() == 1:
            ranks.append(1.0)
        else:
            ranks.append(0.0)
    eff_rank = float(np.mean(ranks)) if ranks else float("nan")

    # Saturation: entries within 1% of clamp boundary
    half_range = (clamp_max - clamp_min) * 0.01
    near_min = (m_flat >= clamp_min) & (m_flat <= clamp_min + half_range)
    near_max = (m_flat <= clamp_max) & (m_flat >= clamp_max - half_range)
    saturation = float((near_min | near_max).float().mean().item())

    return MemoryStats(
        memory_norm=fro_norm,
        memory_std=std_val,
        memory_max=float(m.max().item()),
        memory_min=float(m.min().item()),
        memory_effective_rank=eff_rank,
        memory_saturation_fraction=saturation,
    )


def _retrieval_result(
    query: Tensor,
    retrieved: Tensor,
    candidates: Tensor,
    *,
    observation_ids: Tensor | None = None,
    position_ids: Tensor | None = None,
) -> RetrievalResult:
    """Evaluate retrieval quality for one set of queries.

    Args:
        query: Original code used as query, ``(T, S)``.
        retrieved: Code returned by the retrieval system, ``(T, S)``.
        candidates: Bank of candidate codes for nearest-neighbour test,
            ``(T, S)``.  Typically the same as *query* for self-retrieval.
        observation_ids: Optional ``(T,)`` observation IDs at each timestep.
        position_ids: Optional ``(T, 2)`` spatial (row, col) coordinates.

    Returns:
        Filled ``RetrievalResult``.
    """
    T = int(query.shape[0])
    device = query.device
    cos = torch.nn.functional.cosine_similarity(query, retrieved, dim=-1)
    cos_mean = float(cos.mean().item())
    cos_std = float(cos.std().item())

    if T <= 1:
        return RetrievalResult(
            cosine_mean=cos_mean, cosine_std=cos_std, nn_accuracy=None
        )

    # Nearest-neighbour indices from retrieved to candidates.
    retrieved_norm = retrieved / (retrieved.norm(dim=-1, keepdim=True) + 1e-8)
    candidates_norm = candidates / (candidates.norm(dim=-1, keepdim=True) + 1e-8)
    sim_matrix = retrieved_norm @ candidates_norm.T  # (T, T)
    # Exclude self-pair for NN depth analysis (the diagonal is always 1.0).
    arange = torch.arange(T, device=device)
    nn_indices = sim_matrix.argmax(dim=-1)  # (T,)  — includes self when self is in candidates
    nn_acc = float((nn_indices == arange).float().mean().item())

    # NN temporal distance.
    temporal_dist = (nn_indices - arange).abs().float()
    td_mean = float(temporal_dist.mean().item())
    td_median = float(temporal_dist.median().item())

    # Top-k accuracy.
    topk = sim_matrix.topk(k=min(10, T), dim=-1)
    true_in_top5 = (topk.indices[:, :5] == arange.unsqueeze(1)).any(dim=1)
    true_in_top10 = (topk.indices[:, :10] == arange.unsqueeze(1)).any(dim=1)
    top5_acc = float(true_in_top5.float().mean().item())
    top10_acc = float(true_in_top10.float().mean().item())

    # Observation-ID analysis.
    same_obs_frac: float | None = None
    gt_obs_acc: float | None = None
    if observation_ids is not None:
        same_obs_frac = float(
            (observation_ids[nn_indices] == observation_ids).float().mean().item()
        )
        # "GT observation accuracy": does the retrieved code's nearest
        # candidate share the same observation ID as the query?
        gt_obs_acc = same_obs_frac  # same metric for NN-level analysis

    # Spatial-position analysis.
    same_pos_frac: float | None = None
    spatial_dist_mean: float | None = None
    if position_ids is not None and position_ids.shape[-1] >= 2:
        pos_q = position_ids.float()
        pos_nn = position_ids[nn_indices].float()
        spatial_dist = (pos_q - pos_nn).norm(dim=-1)
        spatial_dist_mean = float(spatial_dist.mean().item())
        same_pos_frac = float((spatial_dist < 0.5).float().mean().item())

    return RetrievalResult(
        cosine_mean=cos_mean,
        cosine_std=cos_std,
        nn_accuracy=nn_acc,
        nn_temporal_distance_mean=td_mean,
        nn_temporal_distance_median=td_median,
        nn_same_observation_fraction=same_obs_frac,
        nn_gt_observation_accuracy=gt_obs_acc,
        top5_accuracy=top5_acc,
        top10_accuracy=top10_acc,
        nn_same_position_fraction=same_pos_frac,
        nn_spatial_distance_mean=spatial_dist_mean,
    )


# =============================================================================
# Public API
# =============================================================================


def produce_tem_memory_probe(  # -------------------------------------------------
    model: TEMModelV1,
    input_batch: TEMInputV1,
    *,
    mask_kind: str = "full",
    observation_ids: Tensor | None = None,
    position_ids: Tensor | None = None,
) -> MemoryProbeResult:
    """Run the Hebbian memory self-retrieval probe for one TEM v1 episode.

    The probe runs a full forward pass to populate the HPC memory, then
    extracts the accumulated Hebbian matrix *M* and per-step place codes
    to evaluate memory health and self-retrieval quality.

    Args:
        model: A TEM v1 model in evaluation mode (``model.eval()``).
        input_batch: A single-step or multi-step ``TEMInputV1``.  When
            ``input_batch.seq_len > 1``, all steps are processed
            sequentially.
        mask_kind: Which attractor masks to use for self-retrieval.
            ``"full"`` or ``"hierarchical"``.
        observation_ids: Optional ``(T,)`` integer observation IDs for NN
            error characterisation.
        position_ids: Optional ``(T, 2)`` (row, col) spatial coordinates
            for NN error characterisation.

    Returns:
        Filled ``MemoryProbeResult`` containing memory statistics and
        retrieval diagnostics.

    Raises:
        TypeError: If the HPC backend is not ``HPCAttractor`` (i.e., the
            memory store is not ``DenseMemoryStore``).
        RuntimeError: If the model is in training mode.
    """
    if model.training:
        raise RuntimeError(
            "TEM memory probe requires the model in evaluation mode. "
            "Call model.eval() before probing."
        )

    n_freq = len(model._config.hpc.shape)
    feature_dim = sum(model._config.hpc.shape)
    device = next(model.parameters()).device

    # Determine episode length from the observation-embedding's first dimension
    # (batch dimension = 1 for a single-episode probe, seq dimension = number
    # of timesteps to unroll).
    T = int(input_batch.observation_embedding[0].shape[0])

    # Gather masks from the HPC module.
    hpc = model.hpc

    if mask_kind == "full":
        masks = hpc.masks_full
    elif mask_kind == "hierarchical":
        masks = hpc.masks_hierarchical
    else:
        raise ValueError(f"Unknown mask_kind: {mask_kind!r}. Use 'full' or 'hierarchical'.")

    # Run the forward pass one step at a time (TEM is recurrent).  Each step
    # gets the same batch-size-1 state so memory accumulates across the episode.
    state = model.init_state(1, device=device)

    place_codes_post: list[list[Tensor]] = []
    place_codes_prior: list[list[Tensor]] = []
    place_codes_retrieved: list[list[Tensor]] = []

    with torch.no_grad():
        for t in range(T):
            # Build a single-step input for this timestep.
            step_input = TEMInputV1(
                observation_embedding=[
                    freq_t[t : t + 1] for freq_t in input_batch.observation_embedding
                ],
                previous_action=input_batch.previous_action[t : t + 1],
                episode_start=(
                    input_batch.episode_start[t : t + 1]
                    if input_batch.episode_start is not None
                    else None
                ),
                landmark_id=(
                    input_batch.landmark_id[t : t + 1]
                    if input_batch.landmark_id is not None
                    else None
                ),
            )
            output, state = model(step_input, state=state)
            place_codes_post.append(
                [p.clone().cpu() for p in output.place_codes.posterior]
            )
            place_codes_prior.append(
                [p.clone().cpu() for p in output.place_codes.prior]
            )
            if output.place_codes.retrieved is not None:
                place_codes_retrieved.append(
                    [p.clone().cpu() for p in output.place_codes.retrieved]
                )

    # Stack per-step codes into bundles of (T, D_f).
    def _stack_bundle(bundle: list[list[Tensor]]) -> list[Tensor]:
        return [torch.cat([step[f] for step in bundle], dim=0) for f in range(n_freq)]

    p_post_bundle = _stack_bundle(place_codes_post)
    p_prior_bundle = _stack_bundle(place_codes_prior)

    has_retrieved = len(place_codes_retrieved) == T
    p_retrieved_bundle = _stack_bundle(place_codes_retrieved) if has_retrieved else p_post_bundle

    # Flatten to (T, S) for self-retrieval.
    p_post_flat = torch.cat(p_post_bundle, dim=-1)  # (T, S)
    p_prior_flat = torch.cat(p_prior_bundle, dim=-1)
    p_retrieved_flat = torch.cat(p_retrieved_bundle, dim=-1)

    # Extract Hebbian memory matrix from the HPC state.
    memory_entry = state.hpc.memory.g_cued
    if not isinstance(memory_entry, DenseMemoryStore):
        raise TypeError(
            "TEM memory probe requires DenseMemoryStore. "
            f"Got {type(memory_entry).__name__}."
        )
    M = memory_entry.matrix.detach().cpu().to(torch.float64)  # (B, S, S)
    clamp_min = float(hpc.write_module.config.clamp_min)
    clamp_max = float(hpc.write_module.config.clamp_max)

    # Use batch index 0 for probe statistics.
    M0 = M[0]  # (S, S)
    p_post_flat = p_post_flat.to(dtype=M0.dtype)
    p_prior_flat = p_prior_flat.to(dtype=M0.dtype)
    p_retrieved_flat = p_retrieved_flat.to(dtype=M0.dtype)

    # ---- Memory stats ----
    mem_stats = _memory_stats(M, clamp_min=clamp_min, clamp_max=clamp_max)

    # ---- Place-code diversity ----
    p_post_stats = _place_code_stats(p_post_bundle)
    p_prior_stats = _place_code_stats(p_prior_bundle)
    p_retrieved_stats = _place_code_stats(p_retrieved_bundle)

    # ---- Retrieved vs ancestral delta ----
    retrieved_ancestral_delta = float(
        (p_retrieved_flat - p_prior_flat).norm(dim=-1).mean().item()
    )

    # ---- Self-retrieval: query M with p_post ----
    # Build the linear memory view and run attractor dynamics.
    memory_view = memory_entry.as_linear_view()

    # Run attractor read: M0 @ p_post_flat
    # The attractor uses activation clamping + staged masks.
    kappa = float(hpc.retrieval_module.config.kappa)
    state_attractor = hpc.retrieval_module.activation(p_post_flat)
    for stage_mask in masks:
        field = kappa * state_attractor + (p_post_flat.unsqueeze(1) @ M0.to(dtype=p_post_flat.dtype)).squeeze(1)
        # stage_mask is (S,) — unsqueeze to (1, S) for broadcasting over (T, S).
        mask = stage_mask.to(dtype=state_attractor.dtype).unsqueeze(0)
        state_attractor = (1 - mask) * state_attractor + mask * hpc.retrieval_module.activation(field)

    p_self_retrieved = state_attractor  # (T, S)

    # Move optional metadata to CPU for the retrieval analysis.
    obs_ids_cpu = (
        observation_ids.detach().cpu()
        if observation_ids is not None else None
    )
    pos_ids_cpu = (
        position_ids.detach().cpu()
        if position_ids is not None else None
    )

    self_retrieval_res = _retrieval_result(
        query=p_post_flat,
        retrieved=p_self_retrieved,
        candidates=p_post_flat,
        observation_ids=obs_ids_cpu,
        position_ids=pos_ids_cpu,
    )

    # ---- Grid-query retrieval (optional placeholder, not computed here) ----

    return MemoryProbeResult(
        model_family="tem-v1",
        episode_steps=T,
        n_freq=n_freq,
        feature_dim=feature_dim,
        memory_stats=mem_stats,
        p_post=p_post_stats,
        p_retrieved=p_retrieved_stats,
        p_prior=p_prior_stats,
        self_retrieval=self_retrieval_res,
        grid_retrieval=None,
        retrieved_vs_ancestral_delta=retrieved_ancestral_delta,
    )


# =============================================================================
def persist_tem_memory_probe(  # -------------------------------------------------
    result: MemoryProbeResult,
    output_dir: Path,
    *,
    filename: str = "memory_probe.json",
) -> Path:
    """Write a memory probe result to a JSON file.

    Args:
        result: The probe result to persist.
        output_dir: Directory for the output file.  Created if needed.
        filename: Output filename within *output_dir*.

    Returns:
        Absolute path to the written JSON file.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    # Use json.dumps with allow_nan=True to serialize NaN/Inf correctly.
    out_path.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, allow_nan=True),
        encoding="utf-8",
    )
    return out_path


def load_tem_memory_probe(  # ---------------------------------------------------
    path: str | Path,
) -> MemoryProbeResult:
    """Load a memory probe result from a persisted JSON file.

    Args:
        path: Path to a ``memory_probe.json`` file produced by
            :func:`persist_tem_memory_probe`.

    Returns:
        Deserialised ``MemoryProbeResult``.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    # Pydantic rejects NaN/Inf from JSON; convert back to Python floats.
    return MemoryProbeResult.model_validate(raw)


# =============================================================================
def _fmt(v: float | None, decimals: int = 6) -> str:
    """Format a value for table output, handling NaN/None."""
    if v is None:
        return "—"
    if v != v:  # NaN check
        return "—"
    return f"{v:.{decimals}f}"


def format_memory_probe_table(  # ------------------------------------------------
    result: MemoryProbeResult,
) -> str:
    """Format a memory probe result as a human-readable markdown table.

    Args:
        result: The probe result to format.

    Returns:
        A markdown string with the diagnostic table.
    """
    sr = result.self_retrieval
    lines = [
        "| metric | value |",
        "| ------ | ----- |",
        f"| memory_norm | {_fmt(result.memory_stats.memory_norm)} |",
        f"| memory_std | {_fmt(result.memory_stats.memory_std)} |",
        f"| memory_effective_rank | {_fmt(result.memory_stats.memory_effective_rank, 4)} |",
        f"| memory_saturation_fraction | {_fmt(result.memory_stats.memory_saturation_fraction)} |",
        f"| p_post_pairwise_cosine_mean | {_fmt(result.p_post.pairwise_cosine_mean)} |",
        f"| p_retrieved_pairwise_cosine_mean | {_fmt(result.p_retrieved.pairwise_cosine_mean)} |",
        f"| p_prior_pairwise_cosine_mean | {_fmt(result.p_prior.pairwise_cosine_mean)} |",
        f"| self_retrieval_cosine_mean | {_fmt(sr.cosine_mean)} |",
        f"| self_retrieval_nn_accuracy | {_fmt(sr.nn_accuracy)} |",
        f"| self_retrieval_nn_temp_dist_mean | {_fmt(sr.nn_temporal_distance_mean)} |",
        f"| self_retrieval_nn_temp_dist_median | {_fmt(sr.nn_temporal_distance_median)} |",
        f"| self_retrieval_top5_accuracy | {_fmt(sr.top5_accuracy)} |",
        f"| self_retrieval_top10_accuracy | {_fmt(sr.top10_accuracy)} |",
        f"| self_retrieval_same_obs_frac | {_fmt(sr.nn_same_observation_fraction)} |",
        f"| retrieved_vs_ancestral_delta | {_fmt(result.retrieved_vs_ancestral_delta)} |",
        "",
    ]
    return "\n".join(lines)


# Rebuild forward-reference Pydantic models (required by `from __future__ import annotations`).
MemoryProbeResult.model_rebuild()


# =============================================================================
__all__ = [
    "MemoryProbeResult",
    "MemoryStats",
    "PlaceCodeStats",
    "RetrievalResult",
    "produce_tem_memory_probe",
    "persist_tem_memory_probe",
    "load_tem_memory_probe",
    "format_memory_probe_table",
]
