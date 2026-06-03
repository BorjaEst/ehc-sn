"""TEM representation analysis primitives.

Pure functions for computing content/structure selectivity, tuning curves,
pairwise similarity, and RSA-style dissociation from model representations.

These functions are trace-tree independent and do not import from
``ehc_sn.figures``, ``ehc_sn.reporting``, or any other presentation layer.
They can be unit-tested in isolation.

References
----------
Whittington et al. (2020). "The Tolman-Eichenbaum Machine".
    Cell 183(5):1248–1262 e23.
Whittington et al. (2022). "Relating transformers to models and neural
    representations of the hippocampal formation".  arXiv:2112.04035.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# =============================================================================
# Typed results
# =============================================================================


@dataclass(frozen=True)
class StreamEffect:
    """Same/different similarity and effect size for one representational stream.

    The *content_effect* and *location_effect* are defined as:
        effect = mean(same) - mean(different)
    A positive effect indicates clustering by that grouping variable.
    """

    same_observation: float
    different_observation: float
    content_effect: float
    same_location: float
    different_location: float
    location_effect: float


@dataclass(frozen=True)
class ContentStructureRSAResult:
    """Cross-system representational similarity result.

    Attributes
    ----------
    lec, mec, hpc : StreamEffect
        Per-stream effects.
    n_steps : int
        Number of time steps included in the analysis.
    """

    lec: StreamEffect
    mec: StreamEffect
    hpc: StreamEffect
    n_steps: int


# =============================================================================
# Observation tuning
# =============================================================================


def compute_observation_tuning(
    activity: NDArray,
    observation_ids: NDArray,
    n_observations: int | None = None,
) -> NDArray:
    """For each unit, compute mean activation per observation ID.

    Parameters
    ----------
    activity : (T, n_units) float array.
    observation_ids : (T,) int array of observation IDs.
    n_observations : int or None
        Number of distinct observation classes.  Auto-detected when ``None``.

    Returns
    -------
    tuning : (n_units, n_obs) float array.
        ``tuning[u, o]`` = mean activation of unit ``u`` when obs ID == ``o``.
    """
    if n_observations is None:
        n_observations = (
            int(observation_ids.max()) - int(observation_ids.min()) + 1
        )
    n_units = activity.shape[1]
    tuning = np.zeros((n_units, n_observations), dtype=float)
    counts = np.zeros(n_observations, dtype=float)
    for o in range(n_observations):
        mask = observation_ids == o
        n = int(mask.sum())
        counts[o] = n
        if n > 0:
            tuning[:, o] = np.mean(activity[mask], axis=0)
    return tuning


def compute_selectivity_scores(tuning: NDArray) -> NDArray:
    """Compute (max - mean_other) / (max + mean_other) per unit.

    Parameters
    ----------
    tuning : (n_units, n_obs) float array.

    Returns
    -------
    si : (n_units,) float array.
        NaN for units with all-zero tuning.
    """
    max_val = np.max(tuning, axis=1)
    sum_val = np.sum(tuning, axis=1)
    n_obs = tuning.shape[1]
    mean_other = (sum_val - max_val) / np.maximum(n_obs - 1, 1)
    denom = max_val + mean_other
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(denom > 0, (max_val - mean_other) / denom, np.nan)
    return result


def compute_tuning_entropy(tuning: NDArray, base: float = 2.0) -> NDArray:
    """Compute entropy over normalised tuning per unit.

    Parameters
    ----------
    tuning : (n_units, n_obs) float array.
    base : float
        Logarithm base for entropy (default 2 → bits).

    Returns
    -------
    entropy : (n_units,) float array.
    """
    p = tuning / np.maximum(np.sum(tuning, axis=1, keepdims=True), 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.nansum(p * np.log2(p + 1e-12), axis=1) / np.log2(base)


# =============================================================================
# Pairwise similarity
# =============================================================================


def compute_pairwise_cosine_similarity(activity: NDArray) -> NDArray:
    """Pairwise cosine similarity of rows in *activity*.

    Parameters
    ----------
    activity : (T, D) float array.

    Returns
    -------
    sim : (T, T) float array.
    """
    norm = np.linalg.norm(activity, axis=1, keepdims=True)
    activity_normed = activity / np.maximum(norm, 1e-12)
    return activity_normed @ activity_normed.T


# =============================================================================
# Same/different effect
# =============================================================================


def compute_same_different_effect(
    similarity: NDArray,
    labels: NDArray,
) -> tuple[float, float, float]:
    """Compute mean within-group and between-group similarity.

    Parameters
    ----------
    similarity : (T, T) float array, pairwise similarity matrix.
    labels : (T,) int array grouping steps.

    Returns
    -------
    same_mean : float
        Mean similarity for pairs with the same label (excluding diagonal).
    diff_mean : float
        Mean similarity for pairs with different labels.
    effect_size : float
        same_mean - diff_mean (positive = clustering by label).
    """
    T = similarity.shape[0]
    same_vals: list[float] = []
    diff_vals: list[float] = []
    for i in range(T):
        for j in range(i + 1, T):
            val = float(similarity[i, j])
            if np.isfinite(val):
                if labels[i] == labels[j]:
                    same_vals.append(val)
                else:
                    diff_vals.append(val)
    same_mean = float(np.mean(same_vals)) if same_vals else 0.0
    diff_mean = float(np.mean(diff_vals)) if diff_vals else 0.0
    effect_size = same_mean - diff_mean
    return same_mean, diff_mean, effect_size


# =============================================================================
# Full content/structure RSA
# =============================================================================


def compute_content_structure_rsa(
    lec_activity: NDArray,
    mec_activity: NDArray,
    hpc_activity: NDArray,
    observation_ids: NDArray,
    location_ids: NDArray,
) -> ContentStructureRSAResult:
    """Compare LEC, MEC, and HPC representational geometry.

    For each stream, computes pairwise cosine similarity and then the
    same-vs-different effect for observation identity and location identity.

    Parameters
    ----------
    lec_activity : (T, D_lec) float array.
    mec_activity : (T, D_mec) float array.
    hpc_activity : (T, D_hpc) float array.
    observation_ids : (T,) int array.
    location_ids : (T,) int array.

    Returns
    -------
    ContentStructureRSAResult
    """
    lec_sim = compute_pairwise_cosine_similarity(lec_activity)
    mec_sim = compute_pairwise_cosine_similarity(mec_activity)
    hpc_sim = compute_pairwise_cosine_similarity(hpc_activity)

    def _effect(sim, labels) -> StreamEffect:
        so, do, oe = compute_same_different_effect(sim, labels)
        return StreamEffect(
            same_observation=so,
            different_observation=do,
            content_effect=oe,
            same_location=0.0,
            different_location=0.0,
            location_effect=0.0,
        )

    lec_obs = _effect(lec_sim, observation_ids)
    mec_obs = _effect(mec_sim, observation_ids)
    hpc_obs = _effect(hpc_sim, observation_ids)

    lec_loc = compute_same_different_effect(lec_sim, location_ids)
    mec_loc = compute_same_different_effect(mec_sim, location_ids)
    hpc_loc = compute_same_different_effect(hpc_sim, location_ids)

    return ContentStructureRSAResult(
        lec=StreamEffect(
            same_observation=lec_obs.same_observation,
            different_observation=lec_obs.different_observation,
            content_effect=lec_obs.content_effect,
            same_location=lec_loc[0],
            different_location=lec_loc[1],
            location_effect=lec_loc[2],
        ),
        mec=StreamEffect(
            same_observation=mec_obs.same_observation,
            different_observation=mec_obs.different_observation,
            content_effect=mec_obs.content_effect,
            same_location=mec_loc[0],
            different_location=mec_loc[1],
            location_effect=mec_loc[2],
        ),
        hpc=StreamEffect(
            same_observation=hpc_obs.same_observation,
            different_observation=hpc_obs.different_observation,
            content_effect=hpc_obs.content_effect,
            same_location=hpc_loc[0],
            different_location=hpc_loc[1],
            location_effect=hpc_loc[2],
        ),
        n_steps=len(observation_ids),
    )


# =============================================================================
__all__ = [
    "StreamEffect",
    "ContentStructureRSAResult",
    "compute_observation_tuning",
    "compute_selectivity_scores",
    "compute_tuning_entropy",
    "compute_pairwise_cosine_similarity",
    "compute_same_different_effect",
    "compute_content_structure_rsa",
]
