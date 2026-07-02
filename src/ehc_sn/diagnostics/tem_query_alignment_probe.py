"""TEM v1 query-alignment diagnostic probe.

Compares Hebbian AttractorRead retrieval from posterior, retrieved, prior,
grid-projected, sensory-projected, and random queries across multiple attractor
iteration depths.

Answers the paper-to-code question: *"Does mec_to_hpc(g) produce a valid
query into the Hebbian attractor memory?"*

Public API
----------
- :class:`QueryAlignmentResult` — Pydantic schema for the probe output.
- :func:`produce_tem_query_alignment_probe` — run the probe against a loaded
  model and evaluation batch.
- :func:`persist_query_alignment_probe` — write the result to a JSON artifact.
- :func:`format_query_alignment_table` — format as a markdown table.

Usage::

    from ehc_sn.diagnostics.tem_query_alignment_probe import (
        produce_tem_query_alignment_probe,
        persist_query_alignment_probe,
    )

    result = produce_tem_query_alignment_probe(
        model, input_batch, observation_ids=obs_ids,
    )
    path = persist_query_alignment_probe(result, Path("probes/"))
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from pydantic import BaseModel, Field
from torch import Tensor

from ehc_sn.models.tem.tem_v1 import TEMInputV1, TEMModelV1
from ehc_sn.types import DenseMemoryStore

# =============================================================================
# Probe result schema
# =============================================================================


class QueryRetrievalMetrics(BaseModel, extra="forbid"):
    """Retrieval metrics for one query type at one attractor iteration depth.

    Attributes:
        cosine_to_matching: Mean cosine between retrieved[t] and p_post[t].
        cosine_to_nearest: Mean cosine between retrieved[t] and nearest p_post[t'].
        nn_accuracy: Fraction where nearest p_post is the exact timestep.
        top5_accuracy: Fraction where true timestep is in top-5 nearest neighbours.
        top10_accuracy: Fraction where true timestep is in top-10 nearest neighbours.
        same_observation_fraction: Fraction where nearest p_post shares the query's
            observation ID.
        same_position_fraction: Fraction where nearest p_post shares the query's
            spatial position.
        temporal_distance_mean: Mean absolute temporal offset to nearest p_post.
        temporal_distance_median: Median absolute temporal offset.
        retrieval_sharpening: Mean cosine improvement over the initial query
            (cos(retrieved, p_post) - cos(initial_query, p_post)).
        fixed_point_delta_norm: Mean L2 change from the previous iteration count
            (NaN for the first iteration).
    """

    cosine_to_matching: float = Field(default=float("nan"))
    cosine_to_nearest: float = Field(default=float("nan"))
    nn_accuracy: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    top5_accuracy: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    top10_accuracy: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    same_observation_fraction: Optional[float] = Field(default=None)
    same_position_fraction: Optional[float] = Field(default=None)
    temporal_distance_mean: Optional[float] = Field(default=None)
    temporal_distance_median: Optional[float] = Field(default=None)
    retrieval_sharpening: float = Field(default=float("nan"))
    fixed_point_delta_norm: float = Field(default=float("nan"))


class InitialQueryAlignment(BaseModel, extra="forbid"):
    """Pre-attractor alignment of the bare query against the p_post bank.

    Attributes:
        cosine_to_matching: Mean cosine between query[t] and p_post[t].
        cosine_to_nearest: Mean cosine between query[t] and nearest p_post[t'].
        nn_same_observation_fraction: Fraction where nearest p_post shares the
            query's observation ID.
        nn_accuracy: Fraction where nearest p_post is the exact timestep.
    """

    cosine_to_matching: float = Field(default=float("nan"))
    cosine_to_nearest: float = Field(default=float("nan"))
    nn_same_observation_fraction: Optional[float] = Field(default=None)
    nn_accuracy: Optional[float] = Field(default=None)


class QueryTypeMetrics(BaseModel, extra="forbid"):
    """All metrics for one query type across iteration depths.

    Attributes:
        label: Human-readable label for this query type (e.g. "p_post",
            "mec_to_hpc(g_post)").
        initial_alignment: Pre-attractor query-to-p_post alignment.
        per_iter: Metrics keyed by iteration count (1, 2, 3, 5, 10, 20).
    """

    label: str = Field(...)
    initial_alignment: InitialQueryAlignment = Field(
        default_factory=InitialQueryAlignment,
    )
    per_iter: dict[str, QueryRetrievalMetrics] = Field(default_factory=dict)


class QueryAlignmentResult(BaseModel, extra="forbid"):
    """Complete diagnostic output from one TEM query-alignment probe run.

    Attributes:
        model_family: Always ``"tem-v1"``.
        checkpoint_hint: Optional label identifying the checkpoint.
        episode_steps: Number of timesteps in the probed episode.
        n_freq: Number of HPC frequency modules.
        feature_dim: Flattened memory-code dimension *S*.
        n_iters: Attractor iteration counts tested.
        query_types: Metrics for each query type, keyed by query key (e.g.
            ``"p_post"``, ``"mec_to_hpc_g_post"``).
        diagnosis: Brief classification of the dominant bottleneck pattern.
    """

    model_family: str = "tem-v1"
    checkpoint_hint: str = ""
    episode_steps: int = 0
    n_freq: int = 0
    feature_dim: int = 0
    n_iters: list[int] = Field(default_factory=list)
    query_types: dict[str, QueryTypeMetrics] = Field(default_factory=dict)
    diagnosis: str = "not evaluated"


# =============================================================================
# Helpers
# =============================================================================


def _pairwise_cosine_similarity(
    a: Tensor,
    b: Tensor,
) -> Tensor:
    """Compute pairwise cosine similarity between rows of *a* and *b*.

    Args:
        a: Tensor with shape ``(T, D)``.
        b: Tensor with shape ``(T, D)``.

    Returns:
        Tensor with shape ``(T,)`` of cosine similarities.
    """
    a_norm = a / (a.norm(dim=-1, keepdim=True) + 1e-8)
    b_norm = b / (b.norm(dim=-1, keepdim=True) + 1e-8)
    return (a_norm * b_norm).sum(dim=-1)


def _nns_to_p_post(
    query_flat: Tensor,
    p_post_flat: Tensor,
) -> tuple[Tensor, Tensor]:
    """Find nearest neighbours from *query_flat* to *p_post_flat*.

    Args:
        query_flat: ``(T, S)`` query codes.
        p_post_flat: ``(T, S)`` posterior place codes (the reference bank).

    Returns:
        Tuple of (nn_indices, sim_matrix) where nn_indices has shape ``(T,)``
        and sim_matrix has shape ``(T, T)``.
    """
    q_norm = query_flat / (query_flat.norm(dim=-1, keepdim=True) + 1e-8)
    c_norm = p_post_flat / (p_post_flat.norm(dim=-1, keepdim=True) + 1e-8)
    sim = q_norm @ c_norm.T  # (T, T)
    nn_idx = sim.argmax(dim=-1)  # (T,)
    return nn_idx, sim


def _initial_alignment(
    query_flat: Tensor,
    p_post_flat: Tensor,
    *,
    observation_ids: Tensor | None = None,
) -> InitialQueryAlignment:
    """Compute pre-attractor query-to-p_post alignment.

    Args:
        query_flat: ``(T, S)`` raw query (not yet passed through attractor).
        p_post_flat: ``(T, S)`` reference posterior bank.
        observation_ids: Optional ``(T,)`` observation IDs.

    Returns:
        Filled ``InitialQueryAlignment``.
    """
    T = int(query_flat.shape[0])
    cos_match = _pairwise_cosine_similarity(query_flat, p_post_flat)
    cos_match_mean = float(cos_match.mean().item())

    nn_idx, sim = _nns_to_p_post(query_flat, p_post_flat)
    arange = torch.arange(T, device=query_flat.device)
    nn_acc = float((nn_idx == arange).float().mean().item())

    # Cosine to nearest (which might differ from matching).
    nearest_cos = sim[arange, nn_idx]  # (T,)
    nearest_cos_mean = float(nearest_cos.mean().item())

    same_obs = None
    if observation_ids is not None:
        obs = observation_ids.to(device=query_flat.device)
        same_obs = float((obs[nn_idx] == obs).float().mean().item())

    return InitialQueryAlignment(
        cosine_to_matching=cos_match_mean,
        cosine_to_nearest=nearest_cos_mean,
        nn_same_observation_fraction=same_obs,
        nn_accuracy=nn_acc,
    )


def _retrieval_metrics(
    query_flat: Tensor,
    initial_query_flat: Tensor,
    retrieved_flat: Tensor,
    previous_recall_flat: Tensor | None,
    p_post_flat: Tensor,
    *,
    observation_ids: Tensor | None = None,
    position_ids: Tensor | None = None,
) -> QueryRetrievalMetrics:
    """Compute retrieval metrics for one attractor invocation.

    Args:
        query_flat: The raw query before the attractor ``(T, S)``.
        initial_query_flat: Same as query_flat (for sharpening calc).
        retrieved_flat: Attractor output ``(T, S)``.
        previous_recall_flat: Attractor output from previous iteration count,
            or ``None`` for the first iteration.
        p_post_flat: Reference posterior bank ``(T, S)``.
        observation_ids: Optional ``(T,)`` observation IDs.
        position_ids: Optional ``(T, 2)`` spatial coordinates.

    Returns:
        Filled ``QueryRetrievalMetrics``.
    """
    T = int(retrieved_flat.shape[0])
    device = retrieved_flat.device

    # Cosine to matching p_post[t].
    cos_match = _pairwise_cosine_similarity(retrieved_flat, p_post_flat)
    cos_match_mean = float(cos_match.mean().item())

    # Nearest neighbour in p_post bank.
    nn_idx, sim = _nns_to_p_post(retrieved_flat, p_post_flat)
    arange = torch.arange(T, device=device)
    nn_acc = float((nn_idx == arange).float().mean().item())

    # Cosine to nearest.
    nearest_cos = sim[arange, nn_idx]
    nearest_cos_mean = float(nearest_cos.mean().item())

    # Temporal distance.
    temporal_dist = (nn_idx - arange).abs().float()
    td_mean = float(temporal_dist.mean().item())
    td_median = float(temporal_dist.median().item())

    # Top-k.
    topk = sim.topk(k=min(10, T), dim=-1)
    true_in_top5 = (topk.indices[:, :5] == arange.unsqueeze(1)).any(dim=1)
    true_in_top10 = (topk.indices[:, :10] == arange.unsqueeze(1)).any(dim=1)
    top5_acc = float(true_in_top5.float().mean().item())
    top10_acc = float(true_in_top10.float().mean().item())

    # Observation and position.
    same_obs = None
    same_pos = None
    if observation_ids is not None:
        obs = observation_ids.to(device=device)
        same_obs = float((obs[nn_idx] == obs).float().mean().item())
    if position_ids is not None and position_ids.shape[-1] >= 2:
        pos = position_ids.float().to(device=device)
        spatial_dist = (pos - pos[nn_idx]).norm(dim=-1)
        same_pos = float((spatial_dist < 0.5).float().mean().item())

    # Retrieval sharpening.
    cos_initial = _pairwise_cosine_similarity(initial_query_flat, p_post_flat)
    sharpening = float((cos_match - cos_initial).mean().item())

    # Fixed-point delta.
    fpd = float("nan")
    if previous_recall_flat is not None:
        fpd = float(
            (retrieved_flat - previous_recall_flat).norm(dim=-1).mean().item()
        )

    return QueryRetrievalMetrics(
        cosine_to_matching=cos_match_mean,
        cosine_to_nearest=nearest_cos_mean,
        nn_accuracy=nn_acc,
        top5_accuracy=top5_acc,
        top10_accuracy=top10_acc,
        same_observation_fraction=same_obs,
        same_position_fraction=same_pos,
        temporal_distance_mean=td_mean,
        temporal_distance_median=td_median,
        retrieval_sharpening=sharpening,
        fixed_point_delta_norm=fpd,
    )


def _generate_random_query(
    p_post_flat: Tensor,
    seed: int = 42,
) -> Tensor:
    """Generate a random same-norm baseline query.

    Each row is a random vector normalized to match the norm of the
    corresponding ``p_post`` row, so the attractor sees inputs with
    comparable magnitude.

    Args:
        p_post_flat: ``(T, S)`` reference — used for per-row norm scaling.
        seed: Random seed for reproducibility.

    Returns:
        ``(T, S)`` tensor with random directions scaled to match p_post norms.
    """
    T, S = p_post_flat.shape
    g = torch.Generator(device=p_post_flat.device)
    g.manual_seed(seed)
    rand = torch.randn(T, S, device=p_post_flat.device, generator=g)
    rand_normed = rand / (rand.norm(dim=-1, keepdim=True) + 1e-8)
    target_norms = p_post_flat.norm(dim=-1, keepdim=True)
    return rand_normed * target_norms


def _classify_diagnosis(
    result: QueryAlignmentResult,
) -> str:
    """Classify the dominant bottleneck from probe metrics.

    Examines the per-iteration metrics for each query type and returns a
    short diagnosis string.
    """
    q = result.query_types
    if not q:
        return "insufficient data"

    # Helper to get a metric at a given iteration.
    def _get(
        key: str, metric: str, iters: list[int] | None = None
    ) -> float | None:
        qt = q.get(key)
        if qt is None:
            return None
        targets = iters if iters is not None else result.n_iters
        for it in targets:
            pm = qt.per_iter.get(str(it))
            if pm is not None:
                v = getattr(pm, metric, None)
                if v is not None and not (isinstance(v, float) and v != v):
                    return v
        return None

    # Check memory/dynamics: does p_post self-retrieval work?
    p_post_nn = _get("p_post", "nn_accuracy", iters=[1, 2])
    if p_post_nn is not None and p_post_nn < 0.3:
        return "memory_dynamics: p_post self-retrieval fails — revisit attractor/memory storage"

    # Check attractor dynamics: do additional iterations improve or degrade?
    p_post_cos_1 = _get("p_post", "cosine_to_matching", iters=[1])
    p_post_cos_5 = _get("p_post", "cosine_to_matching", iters=[5])
    g_post_nn_1 = _get("mec_to_hpc_g_post", "nn_accuracy", iters=[1])
    g_post_same_obs_1 = _get(
        "mec_to_hpc_g_post", "same_observation_fraction", iters=[1]
    )
    g_post_nn_5 = _get("mec_to_hpc_g_post", "nn_accuracy", iters=[5])

    if (
        p_post_cos_1 is not None
        and p_post_cos_5 is not None
        and p_post_cos_5 < p_post_cos_1 - 0.1
    ):
        # All queries degrade — attractor pushes away from p_post manifold.
        base = (
            "attractor_divergence: repeated attractor steps push codes away from "
            f"the posterior manifold (p_post match_cos: {p_post_cos_1:.3f} → "
            f"{p_post_cos_5:.3f}). The attractor dynamics as probed do not "
            "converge to content-bearing fixed points."
        )
        # Check if grid queries have good initial alignment despite attractor.
        if (
            g_post_nn_1 is not None
            and g_post_nn_1 > 0.5
            and g_post_same_obs_1 is not None
            and g_post_same_obs_1 > 0.5
        ):
            base += (
                f" mec_to_hpc(g_post) has excellent initial alignment "
                f"(nn_acc={g_post_nn_1:.3f}, same_obs={g_post_same_obs_1:.3f}) "
                f"but attractor dynamics degrade it to nn_acc={g_post_nn_5:.3f}. "
                "The frozen projection preserves content-indexing; the attractor "
                "dynamics are the primary bottleneck."
            )
        return base

    # Check query alignment: does g_post query approach p_post performance?
    p_post_same_obs = _get("p_post", "same_observation_fraction", iters=[5])
    g_post_same_obs = _get(
        "mec_to_hpc_g_post", "same_observation_fraction", iters=[5]
    )

    # Compare g_post across iterations.
    g_post_nn_1 = _get("mec_to_hpc_g_post", "nn_accuracy", iters=[1])
    g_post_nn_20 = _get("mec_to_hpc_g_post", "nn_accuracy", iters=[20])
    g_post_nn_5 = _get("mec_to_hpc_g_post", "nn_accuracy", iters=[5])
    g_post_cos_1 = _get("mec_to_hpc_g_post", "cosine_to_matching", iters=[1])
    g_post_same_obs_1 = _get(
        "mec_to_hpc_g_post", "same_observation_fraction", iters=[1]
    )

    # Check sensory query.
    x_nn_1 = _get("lec_to_hpc_x", "nn_accuracy", iters=[1])
    x_nn_5 = _get("lec_to_hpc_x", "nn_accuracy", iters=[5])

    # Check correction: g_post vs g_prior.
    g_prior_same_obs_5 = _get(
        "mec_to_hpc_g_prior", "same_observation_fraction", iters=[5]
    )

    # Case E: good initial NN alignment for grid queries but attractor degrades.
    if (
        g_post_nn_1 is not None
        and g_post_nn_1 > 0.5
        and g_post_same_obs_1 is not None
        and g_post_same_obs_1 > 0.5
    ):
        if g_post_nn_5 is not None and g_post_nn_5 < 0.3:
            return (
                "attractor_divergence: mec_to_hpc(g_post) has excellent initial "
                f"alignment (nn_acc={g_post_nn_1:.3f}, same_obs={g_post_same_obs_1:.3f}) "
                f"but attractor dynamics degrade it to nn_acc={g_post_nn_5:.3f}. "
                "The frozen projection preserves content basins; the attractor "
                "dynamics are the bottleneck."
            )
        if x_nn_5 is not None and x_nn_5 > g_post_nn_5 + 0.05:
            # Sensory query survives attractor better.
            return (
                "query_alignment_failure: mec_to_hpc(g_post) starts well "
                f"(nn_acc={g_post_nn_1:.3f}) but degrades faster than lec_to_hpc(x)"
                f" (nn_acc={x_nn_5:.3f}) under attractor dynamics. The grid projection "
                "produces a fragile query that does not survive iterative retrieval."
            )

    # Case A: query-alignment failure.
    if (
        p_post_same_obs is not None
        and p_post_same_obs > 0.5
        and x_nn_5 is not None
        and x_nn_5 > 0.3
        and g_post_same_obs is not None
        and g_post_same_obs < 0.3
    ):
        return (
            "query_alignment_failure: p_post and lec_to_hpc(x) retrieve well, "
            "but mec_to_hpc(g) does not — frozen grid projection is the bottleneck"
        )

    # Case B: convergence-depth failure.
    if (
        g_post_nn_1 is not None
        and g_post_nn_20 is not None
        and g_post_nn_20 > g_post_nn_1 + 0.15
    ):
        return (
            f"convergence_depth: mec_to_hpc(g_post) improves from "
            f"{g_post_nn_1:.2f} to {g_post_nn_20:.2f} across iterations — "
            f"attractor needs more recurrence depth or higher kappa"
        )

    # Case C: weak sensory correction.
    if (
        g_post_same_obs is not None
        and g_prior_same_obs_5 is not None
        and abs(g_post_same_obs - g_prior_same_obs_5) < 0.05
    ):
        return (
            "weak_sensory_correction: g_post and g_prior queries yield nearly "
            "identical retrieval — MEC posterior correction is not improving the "
            "grid query"
        )

    # Case D: decoder/manifold mismatch.
    if (
        g_post_same_obs is not None
        and g_post_same_obs > 0.5
        and g_post_nn_5 is not None
        and g_post_nn_5 < 0.3
    ):
        return (
            "decoder_manifold_mismatch: retrieval hits the correct observation "
            "basin but misses the exact timestep — decoder may be sensitive to "
            "posterior-specific geometry"
        )

    # Fallback.
    return (
        "mixed or inconclusive: run individual query-type analysis for details"
    )


# =============================================================================
# Public API
# =============================================================================


def produce_tem_query_alignment_probe(  # ----------------------------------------
    model: TEMModelV1,
    input_batch: TEMInputV1,
    *,
    observation_ids: Tensor | None = None,
    position_ids: Tensor | None = None,
    n_iters: tuple[int, ...] = (1, 2, 3, 5, 10, 20),
    mask_kind: str = "full",
) -> QueryAlignmentResult:
    """Run the TEM v1 query-alignment probe for one episode.

    Runs a full forward pass to populate the HPC memory, collects all query
    families, then evaluates AttractorRead retrieval at multiple iteration
    depths for each query type.

    Args:
        model: A TEM v1 model in evaluation mode.
        input_batch: Multi-step ``TEMInputV1``.
        observation_ids: Optional ``(T,)`` observation IDs for NN error
            characterisation.
        position_ids: Optional ``(T, 2)`` spatial (row, col) coordinates.
        n_iters: Attractor iteration depths to test.
        mask_kind: Which attractor masks to use. ``"full"`` or
            ``"hierarchical"``.

    Returns:
        Filled ``QueryAlignmentResult``.

    Raises:
        TypeError: If the HPC backend is not ``HPCAttractor``.
        RuntimeError: If the model is in training mode.
    """
    if model.training:
        raise RuntimeError(
            "TEM query-alignment probe requires evaluation mode. "
            "Call model.evaluation() before probing."
        )

    n_freq = len(model._config.hpc.shape)
    feature_dim = sum(model._config.hpc.shape)
    device = next(model.parameters()).device
    T = int(input_batch.observation_embedding[0].shape[0])

    hpc = model.hpc
    if mask_kind == "full":
        masks = hpc.masks_full
    elif mask_kind == "hierarchical":
        masks = hpc.masks_hierarchical
    else:
        raise ValueError(f"Unknown mask_kind: {mask_kind!r}.")

    # ---- Forward pass: collect all query families ----
    state = model.init_state(1, device=device)

    # Per-timestep storage: each is a list of frequency bundles.
    p_post_store: list[list[Tensor]] = []
    p_recall_store: list[list[Tensor]] = []
    p_path_store: list[list[Tensor]] = []
    g_post_store: list[list[Tensor]] = []
    g_prior_store: list[list[Tensor]] = []

    with torch.no_grad():
        for t in range(T):
            step_obs = [
                freq_t[t : t + 1]
                for freq_t in input_batch.observation_embedding
            ]
            step_input = TEMInputV1(
                observation_embedding=step_obs,
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

            # Posterior place code (T=1 per step).
            p_post_store.append(
                [p.clone().cpu() for p in output.place_codes.post]
            )
            p_path_store.append(
                [p.clone().cpu() for p in output.place_codes.path]
            )
            if output.place_codes.recall is not None:
                p_recall_store.append(
                    [p.clone().cpu() for p in output.place_codes.recall]
                )
            else:
                p_recall_store.append(
                    [p.clone().cpu() for p in output.place_codes.path]
                )

            # Grid codes (multi-scale MEC representations).
            g_post_store.append(
                [g.clone().cpu() for g in output.grid_codes.posterior]
            )
            g_prior_store.append(
                [g.clone().cpu() for g in output.grid_codes.prior]
            )

    # ---- Stack into bundles ----
    def _stack_bundle(bundle: list[list[Tensor]]) -> list[Tensor]:
        return [
            torch.cat([step[f] for step in bundle], dim=0)
            for f in range(n_freq)
        ]

    p_post_bundle = _stack_bundle(p_post_store)
    p_recall_bundle = _stack_bundle(p_recall_store)
    p_path_bundle = _stack_bundle(p_path_store)
    g_post_bundle = _stack_bundle(g_post_store)
    g_prior_bundle = _stack_bundle(g_prior_store)

    # ---- Flatten to (T, S) ----
    p_post_flat = torch.cat(p_post_bundle, dim=-1)
    p_recall_flat = torch.cat(p_recall_bundle, dim=-1)
    p_path_flat = torch.cat(p_path_bundle, dim=-1)

    # ---- Project grid codes into HPC space ----
    g_query_post_bundle = model.mec_to_hpc(g_post_bundle)
    g_query_prior_bundle = model.mec_to_hpc(g_prior_bundle)
    g_query_post_flat = torch.cat(g_query_post_bundle, dim=-1)
    g_query_prior_flat = torch.cat(g_query_prior_bundle, dim=-1)

    # ---- Compute sensory query (lec_to_hpc(x)) via a fresh LEC pass ----
    # Re-run LEC inference on the original observation embeddings so we can
    # extract x_query = lec_to_hpc(x_) without patching the forward pass.
    lec_state = model.lec.init_state(1, device=device)
    x_lec_store: list[list[Tensor]] = []
    with torch.no_grad():
        for t in range(T):
            step_obs = [
                input_batch.observation_embedding[f][t : t + 1]
                for f in range(n_freq)
            ]
            x_inf, lec_state = model.lec.inference(step_obs, lec_state)
            x_lec_store.append([x.clone().detach().cpu() for x in x_inf])

    x_lec_bundle = _stack_bundle(x_lec_store)
    x_query_bundle = model.lec_to_hpc(x_lec_bundle)
    x_query_flat = torch.cat(x_query_bundle, dim=-1)

    # ---- Generate random baseline ----
    random_flat = _generate_random_query(p_post_flat)

    # ---- Assemble queries ----
    queries: dict[str, Tensor] = {
        "p_post": p_post_flat,
        "p_recall": p_recall_flat,
        "p_path": p_path_flat,
        "mec_to_hpc_g_post": g_query_post_flat,
        "mec_to_hpc_g_prior": g_query_prior_flat,
        "lec_to_hpc_x": x_query_flat,
        "random": random_flat,
    }

    # ---- Extract Hebbian memory ----
    memory_entry = state.hpc.memory.g_cued
    if not isinstance(memory_entry, DenseMemoryStore):
        raise TypeError(
            "TEM query-alignment probe requires DenseMemoryStore. "
            f"Got {type(memory_entry).__name__}."
        )
    M = memory_entry.matrix.detach().cpu().to(torch.float64)  # (B, S, S)
    M0 = M[0]  # (S, S)

    # Move reference bank to CPU float64.
    p_post_ref = p_post_flat.to(dtype=M0.dtype)

    # Move optional metadata to CPU.
    obs_ids_cpu = (
        observation_ids.detach().cpu() if observation_ids is not None else None
    )
    pos_ids_cpu = (
        position_ids.detach().cpu() if position_ids is not None else None
    )

    kappa = float(hpc.retrieval_module.config.kappa)

    # ---- Run attractor for each query type and iteration depth ----
    result_by_query: dict[str, QueryTypeMetrics] = {}

    for qkey, qflat in queries.items():
        qflat = qflat.to(dtype=M0.dtype)

        # Initial alignment (before attractor).
        init_align = _initial_alignment(
            qflat,
            p_post_ref,
            observation_ids=obs_ids_cpu,
        )

        per_iter: dict[str, QueryRetrievalMetrics] = {}
        prev_recall: Tensor | None = None

        # Iterate attractor dynamics, recording after each milestone.
        # Follow the exact AttractorRead.forward() pattern:
        #   state = activation(query)
        #   for mask in masks:
        #       field = kappa * state + state @ M
        #       state = (1-mask)*state + mask * activation(field)
        act = hpc.retrieval_module.activation
        state_att = act(qflat)
        n_steps_done = 1  # count initial activation as step 1

        for target_iter in sorted(n_iters):
            steps_needed = target_iter - n_steps_done
            for _ in range(steps_needed):
                # One full attractor pass over all mask stages.
                for stage_mask in masks:
                    mask = stage_mask.to(
                        dtype=state_att.dtype, device=state_att.device
                    ).unsqueeze(0)
                    field = kappa * state_att + (
                        state_att @ M0.to(dtype=state_att.dtype)
                    )
                    state_att = (1 - mask) * state_att + mask * act(field)

            n_steps_done = target_iter

            metrics = _retrieval_metrics(
                query_flat=qflat,
                initial_query_flat=qflat,
                retrieved_flat=state_att,
                previous_recall_flat=prev_recall,
                p_post_flat=p_post_ref,
                observation_ids=obs_ids_cpu,
                position_ids=pos_ids_cpu,
            )
            per_iter[str(target_iter)] = metrics
            prev_recall = state_att.clone()

        result_by_query[qkey] = QueryTypeMetrics(
            label=qkey,
            initial_alignment=init_align,
            per_iter=per_iter,
        )

    # ---- Build result ----
    iter_list = sorted(n_iters)
    result = QueryAlignmentResult(
        model_family="tem-v1",
        episode_steps=T,
        n_freq=n_freq,
        feature_dim=feature_dim,
        n_iters=iter_list,
        query_types=result_by_query,
    )

    # Classify diagnosis.
    result.diagnosis = _classify_diagnosis(result)

    return result


# =============================================================================
def persist_query_alignment_probe(  # --------------------------------------------
    result: QueryAlignmentResult,
    output_dir: Path,
    *,
    filename: str = "query_alignment_probe.json",
) -> Path:
    """Write a query-alignment probe result to a JSON file.

    Args:
        result: The probe result to persist.
        output_dir: Directory for the output file. Created if needed.
        filename: Output filename within *output_dir*.

    Returns:
        Absolute path to the written JSON file.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    out_path.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, allow_nan=True),
        encoding="utf-8",
    )
    return out_path


def load_query_alignment_probe(  # -----------------------------------------------
    path: str | Path,
) -> QueryAlignmentResult:
    """Load a query-alignment probe result from a persisted JSON file.

    Args:
        path: Path to a ``query_alignment_probe.json`` file.

    Returns:
        Deserialised ``QueryAlignmentResult``.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return QueryAlignmentResult.model_validate(raw)


# =============================================================================
def _fmt(v: float | None, decimals: int = 4) -> str:
    """Format a value for table output, handling NaN/None."""
    if v is None:
        return "—"
    if isinstance(v, float) and v != v:  # NaN
        return "—"
    return f"{v:.{decimals}f}"


def format_query_alignment_table(  # ----------------------------------------------
    result: QueryAlignmentResult,
) -> str:
    """Format a query-alignment probe result as a human-readable markdown table.

    Args:
        result: The probe result to format.

    Returns:
        A markdown string with the diagnostic table.
    """
    lines = [
        "## TEM v1 Query-Alignment Probe",
        "",
        f"**Diagnosis:** {result.diagnosis}",
        f"**Episode steps:** {result.episode_steps} | "
        f"**Feature dim:** {result.feature_dim} | "
        f"**Iterations:** {result.n_iters}",
        "",
    ]

    for qkey, qt in result.query_types.items():
        lines.append(f"### {qkey}")
        ia = qt.initial_alignment
        lines.append(
            "| metric | initial | "
            + " | ".join(str(it) for it in result.n_iters)
            + " |"
        )
        lines.append(
            "| --- | --- | " + " | ".join("---" for _ in result.n_iters) + " |"
        )

        # Build rows.
        def _row(label: str, getter):
            vals = [
                _fmt(
                    getattr(ia, label.split("_")[0], None)
                    if label.startswith("init")
                    else None
                )
            ]
            for it in result.n_iters:
                pm = qt.per_iter.get(str(it))
                if pm is None:
                    vals.append("—")
                else:
                    v = getattr(pm, label, None)
                    vals.append(_fmt(v))
            lines.append(f"| {label} | {' | '.join(vals)} |")

        _row("cosine_to_matching", None)
        _row("nn_accuracy", None)
        _row("same_observation_fraction", None)
        _row("retrieval_sharpening", None)
        _row("top5_accuracy", None)
        _row("top10_accuracy", None)
        lines.append("")

        # Also show initial alignment details.
        lines.append(
            f"  *Initial:* cos_match={_fmt(ia.cosine_to_matching)}, "
            f"cos_nearest={_fmt(ia.cosine_to_nearest)}, "
            f"nn_acc={_fmt(ia.nn_accuracy)}, "
            f"same_obs={_fmt(ia.nn_same_observation_fraction)}"
        )
        lines.append("")

    return "\n".join(lines)


# Rebuild forward-reference Pydantic models.
QueryAlignmentResult.model_rebuild()


# =============================================================================
# Attractor vector-field alignment diagnostic
# =============================================================================
#
# Separates the one-step AttractorRead update into three contributions and
# measures whether each pushes the query toward or away from the matching
# posterior place code p_post[t].


class OneStepFieldMetrics(BaseModel, extra="forbid"):
    """Decomposed one-step attractor field alignment against p_post[t] - query[t].

    For each query family, reports the cosine alignment of three update
    components against the target direction d[t] = p_post[t] - query[t]:

    Attributes:
        label: Human-readable label for this query type.
        memory_contribution_cos: cos( (state @ M), d ).
            Pure memory term — does the Hebbian field point toward p_post?
        kappa_contribution_cos: cos( (kappa * state - state), d ).
            Pure decay/self term — does the kappa-driven contraction help?
        full_update_cos: cos( full_field - state, d ).
            Combined effect — does the full one-step field move toward p_post?
        full_update_norm: Mean L2 norm of the full update vector.
        target_direction_norm: Mean L2 norm of the target direction d[t].
        field_magnitude_ratio: mean( ||full_field|| / ||d|| ).
        memory_magnitude: Mean L2 norm of the memory term (state @ M).
        kappa_magnitude: Mean L2 norm of the kappa term (kappa * state - state).
        activation_saturation_fraction: Fraction of units at the activation clamp
            boundary (within 1% of clamp_min or clamp_max).
    """

    label: str = Field(...)
    memory_contribution_cos: float = Field(default=float("nan"))
    kappa_contribution_cos: float = Field(default=float("nan"))
    full_update_cos: float = Field(default=float("nan"))
    full_update_norm: float = Field(default=float("nan"))
    target_direction_norm: float = Field(default=float("nan"))
    field_magnitude_ratio: float = Field(default=float("nan"))
    memory_magnitude: float = Field(default=float("nan"))
    kappa_magnitude: float = Field(default=float("nan"))
    activation_saturation_fraction: float = Field(default=float("nan"))


class AttractorFieldResult(BaseModel, extra="forbid"):
    """Vector-field alignment report for the TEM v1 attractor.

    Attributes:
        model_family: Always ``"tem-v1"``.
        episode_steps: Number of timesteps in the probed episode.
        n_freq: Number of HPC frequency modules.
        feature_dim: Flattened memory-code dimension *S*.
        kappa: The attractor's kappa hyperparameter.
        query_metrics: Per-query-family field alignment metrics.
        summary: One-line interpretation.
    """

    model_family: str = "tem-v1"
    episode_steps: int = 0
    n_freq: int = 0
    feature_dim: int = 0
    kappa: float = float("nan")
    query_metrics: dict[str, OneStepFieldMetrics] = Field(default_factory=dict)
    summary: str = "not evaluated"


# =============================================================================
def produce_tem_attractor_field_probe(  # ----------------------------------------
    model: TEMModelV1,
    input_batch: TEMInputV1,
    *,
    mask_kind: str = "full",
) -> AttractorFieldResult:
    """Run the TEM v1 attractor vector-field alignment diagnostic.

    Runs a forward pass, collects all query families, then computes the
    one-step AttractorRead update direction and measures whether each update
    component moves the query toward or away from its matching ``p_post[t]``.

    Args:
        model: A TEM v1 model in evaluation mode.
        input_batch: Multi-step ``TEMInputV1``.
        mask_kind: Which attractor masks to use. ``"full"`` or
            ``"hierarchical"``.

    Returns:
        Filled ``AttractorFieldResult``.

    Raises:
        TypeError: If the HPC backend is not ``HPCAttractor``.
        RuntimeError: If the model is in training mode.
    """
    if model.training:
        raise RuntimeError(
            "TEM attractor field probe requires evaluation mode. "
            "Call model.evaluation() before probing."
        )

    n_freq = len(model._config.hpc.shape)
    feature_dim = sum(model._config.hpc.shape)
    device = next(model.parameters()).device
    T = int(input_batch.observation_embedding[0].shape[0])

    hpc = model.hpc
    if mask_kind == "full":
        masks = hpc.masks_full
    elif mask_kind == "hierarchical":
        masks = hpc.masks_hierarchical
    else:
        raise ValueError(f"Unknown mask_kind: {mask_kind!r}.")

    # ---- Forward pass: collect all query families ----
    state = model.init_state(1, device=device)

    p_post_store: list[list[Tensor]] = []
    p_recall_store: list[list[Tensor]] = []
    p_path_store: list[list[Tensor]] = []
    g_post_store: list[list[Tensor]] = []
    g_prior_store: list[list[Tensor]] = []

    with torch.no_grad():
        for t in range(T):
            step_obs = [
                freq_t[t : t + 1]
                for freq_t in input_batch.observation_embedding
            ]
            step_input = TEMInputV1(
                observation_embedding=step_obs,
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

            p_post_store.append(
                [p.clone().cpu() for p in output.place_codes.post]
            )
            p_path_store.append(
                [p.clone().cpu() for p in output.place_codes.path]
            )
            if output.place_codes.recall is not None:
                p_recall_store.append(
                    [p.clone().cpu() for p in output.place_codes.recall]
                )
            else:
                p_recall_store.append(
                    [p.clone().cpu() for p in output.place_codes.path]
                )
            g_post_store.append(
                [g.clone().cpu() for g in output.grid_codes.posterior]
            )
            g_prior_store.append(
                [g.clone().cpu() for g in output.grid_codes.prior]
            )

    # ---- Stack into bundles ----
    def _stack_bundle(bundle: list[list[Tensor]]) -> list[Tensor]:
        return [
            torch.cat([step[f] for step in bundle], dim=0)
            for f in range(n_freq)
        ]

    p_post_bundle = _stack_bundle(p_post_store)
    p_recall_bundle = _stack_bundle(p_recall_store)
    p_path_bundle = _stack_bundle(p_path_store)
    g_post_bundle = _stack_bundle(g_post_store)
    g_prior_bundle = _stack_bundle(g_prior_store)

    p_post_flat = torch.cat(p_post_bundle, dim=-1)
    p_recall_flat = torch.cat(p_recall_bundle, dim=-1)
    p_path_flat = torch.cat(p_path_bundle, dim=-1)

    g_query_post_bundle = model.mec_to_hpc(g_post_bundle)
    g_query_prior_bundle = model.mec_to_hpc(g_prior_bundle)
    g_query_post_flat = torch.cat(g_query_post_bundle, dim=-1)
    g_query_prior_flat = torch.cat(g_query_prior_bundle, dim=-1)

    # LEC sensory query.
    lec_state = model.lec.init_state(1, device=device)
    x_lec_store: list[list[Tensor]] = []
    with torch.no_grad():
        for t in range(T):
            step_obs = [
                input_batch.observation_embedding[f][t : t + 1]
                for f in range(n_freq)
            ]
            x_inf, lec_state = model.lec.inference(step_obs, lec_state)
            x_lec_store.append([x.clone().detach().cpu() for x in x_inf])

    x_lec_bundle = _stack_bundle(x_lec_store)
    x_query_bundle = model.lec_to_hpc(x_lec_bundle)
    x_query_flat = torch.cat(x_query_bundle, dim=-1)

    # Random baseline.
    random_flat = _generate_random_query(p_post_flat)

    # ---- Assemble queries ----
    queries: dict[str, Tensor] = {
        "p_post": p_post_flat,
        "p_recall": p_recall_flat,
        "p_path": p_path_flat,
        "mec_to_hpc_g_post": g_query_post_flat,
        "mec_to_hpc_g_prior": g_query_prior_flat,
        "lec_to_hpc_x": x_query_flat,
        "random": random_flat,
    }

    # ---- Extract Hebbian memory ----
    memory_entry = state.hpc.memory.g_cued
    if not isinstance(memory_entry, DenseMemoryStore):
        raise TypeError(
            "TEM attractor field probe requires DenseMemoryStore. "
            f"Got {type(memory_entry).__name__}."
        )
    M = memory_entry.matrix.detach().cpu().to(torch.float64)
    M0 = M[0]
    p_post_ref = p_post_flat.to(dtype=M0.dtype)
    kappa = float(hpc.retrieval_module.config.kappa)
    act = hpc.retrieval_module.activation
    clamp_min = float(hpc.write_module.config.clamp_min)
    clamp_max = float(hpc.write_module.config.clamp_max)
    half_range = (clamp_max - clamp_min) * 0.01

    query_metrics: dict[str, OneStepFieldMetrics] = {}

    for qkey, qflat in queries.items():
        qflat = qflat.to(dtype=M0.dtype)

        # Activated query (this is the actual attractor state after one init activation).
        q_act = act(qflat)

        # ---- Compute one-step field per mask stage ----
        # Start from the initial activated state (same as AttractorRead.forward does).
        state_current = q_act
        # The first mask stage's update.
        stage_mask = (
            masks[0]
            .to(dtype=state_current.dtype, device=qflat.device)
            .unsqueeze(0)
        )

        full_field = kappa * state_current + (
            state_current @ M0.to(dtype=state_current.dtype)
        )
        state_next = (1 - stage_mask) * state_current + stage_mask * act(
            full_field
        )

        # Decompose the update direction for this first stage.
        # state_next = state_current + mask * (act(full_field) - state_current)
        # The "pointing" direction is from the initial activated state toward the next state.
        # But the relevant update vector for the field analysis is the pre-activation field:
        #   field = kappa * state + state @ M
        # And the change relative to the initial activated query is:
        #   update = state_next - q_act
        update_vec = state_next - q_act  # (T, S)

        # Target direction: toward matching p_post.
        target_dir = p_post_ref - qflat  # (T, S)

        # ---- Memory contribution ----
        mem_term = state_current @ M0.to(dtype=state_current.dtype)  # (T, S)
        # Change contributed by memory alone (relative to self-term).
        # field = kappa * state + mem_term
        # The memory-specific push relative to the kappa-only update:
        mem_push = (
            mem_term  # (T, S)  — this is the pure M-contribution to the field
        )

        # ---- Kappa contribution ----
        # kappa * state - state = (kappa - 1) * state = -0.2 * state (for kappa=0.8)
        kappa_push = kappa * state_current - state_current  # (T, S)

        # ---- Compute cosines against target direction ----
        def _cos_with_target(v: Tensor, d: Tensor) -> float:
            c = torch.nn.functional.cosine_similarity(v, d, dim=-1)
            return float(c.mean().item())

        # Normalize to avoid NaN from zero vectors.
        target_norm = target_dir.norm(dim=-1)
        mem_cos = _cos_with_target(mem_push, target_dir)
        kappa_cos = _cos_with_target(kappa_push, target_dir)
        full_cos = _cos_with_target(full_field - state_current, target_dir)

        # ---- Magnitudes ----
        full_update_norm_val = float(update_vec.norm(dim=-1).mean().item())
        target_norm_val = float(target_norm.mean().item())
        field_mag_ratio = float(
            (full_field.norm(dim=-1) / (target_norm + 1e-8)).mean().item()
        )
        mem_mag = float(mem_term.norm(dim=-1).mean().item())
        kappa_mag = float(kappa_push.norm(dim=-1).mean().item())

        # ---- Activation saturation ----
        sat = float(
            (
                (q_act <= clamp_min + half_range)
                | (q_act >= clamp_max - half_range)
            )
            .float()
            .mean()
            .item()
        )

        query_metrics[qkey] = OneStepFieldMetrics(
            label=qkey,
            memory_contribution_cos=mem_cos,
            kappa_contribution_cos=kappa_cos,
            full_update_cos=full_cos,
            full_update_norm=full_update_norm_val,
            target_direction_norm=target_norm_val,
            field_magnitude_ratio=field_mag_ratio,
            memory_magnitude=mem_mag,
            kappa_magnitude=kappa_mag,
            activation_saturation_fraction=sat,
        )

    # ---- Summary ----
    def _fmt_summary(v: float) -> str:
        return f"{v:+.3f}" if not (v != v) else "NaN"

    lines = []
    for qkey, qm in query_metrics.items():
        lines.append(
            f"  {qkey:26s}: mem_cos={_fmt_summary(qm.memory_contribution_cos):>7s} "
            f"kappa_cos={_fmt_summary(qm.kappa_contribution_cos):>7s} "
            f"full_cos={_fmt_summary(qm.full_update_cos):>7s}"
        )
    summary = "\n".join(lines)

    return AttractorFieldResult(
        model_family="tem-v1",
        episode_steps=T,
        n_freq=n_freq,
        feature_dim=feature_dim,
        kappa=kappa,
        query_metrics=query_metrics,
        summary=summary,
    )


# =============================================================================
def persist_attractor_field_probe(  # --------------------------------------------
    result: AttractorFieldResult,
    output_dir: Path,
    *,
    filename: str = "attractor_field_probe.json",
) -> Path:
    """Write an attractor field probe result to a JSON file.

    Args:
        result: The probe result to persist.
        output_dir: Directory for the output file. Created if needed.
        filename: Output filename within *output_dir*.

    Returns:
        Absolute path to the written JSON file.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    out_path.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, allow_nan=True),
        encoding="utf-8",
    )
    return out_path


def load_attractor_field_probe(  # -----------------------------------------------
    path: str | Path,
) -> AttractorFieldResult:
    """Load an attractor field probe result from a persisted JSON file."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return AttractorFieldResult.model_validate(raw)


def format_attractor_field_table(  # ---------------------------------------------
    result: AttractorFieldResult,
) -> str:
    """Format an attractor field probe result as a human-readable table.

    Args:
        result: The probe result to format.

    Returns:
        A markdown string with the diagnostic table.
    """
    lines = [
        "## TEM v1 Attractor Vector-Field Alignment",
        "",
        f"**Kappa:** {result.kappa} | "
        f"**Episode steps:** {result.episode_steps} | "
        f"**Feature dim:** {result.feature_dim}",
        "",
        "| query | mem_cos | kappa_cos | full_cos | mem_mag | kappa_mag | "
        "full_update_norm | target_norm | field/target | sat_frac |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    def _ff(v: float) -> str:
        if v != v:
            return "—"
        return f"{v:+.4f}"

    def _ffn(v: float) -> str:
        if v != v:
            return "—"
        return f"{v:.4f}"

    for qkey, qm in result.query_metrics.items():
        lines.append(
            f"| {qkey} | {_ff(qm.memory_contribution_cos)} | "
            f"{_ff(qm.kappa_contribution_cos)} | "
            f"{_ff(qm.full_update_cos)} | "
            f"{_ffn(qm.memory_magnitude)} | "
            f"{_ffn(qm.kappa_magnitude)} | "
            f"{_ffn(qm.full_update_norm)} | "
            f"{_ffn(qm.target_direction_norm)} | "
            f"{_ffn(qm.field_magnitude_ratio)} | "
            f"{_ffn(qm.activation_saturation_fraction)} |"
        )

    lines.append("")
    lines.append(
        "**cos ⟂ = 0, cos → +1 = update points toward p_post, cos → -1 = away.**"
    )
    lines.append("")
    lines.append(result.summary)

    return "\n".join(lines)


# Rebuild forward-reference Pydantic models.
AttractorFieldResult.model_rebuild()


# =============================================================================
__all__ = [
    "QueryAlignmentResult",
    "QueryTypeMetrics",
    "QueryRetrievalMetrics",
    "InitialQueryAlignment",
    "produce_tem_query_alignment_probe",
    "persist_query_alignment_probe",
    "load_query_alignment_probe",
    "format_query_alignment_table",
    "AttractorFieldResult",
    "OneStepFieldMetrics",
    "produce_tem_attractor_field_probe",
    "persist_attractor_field_probe",
    "load_attractor_field_probe",
    "format_attractor_field_table",
]
