"""Layout-dataset builder for the dagflow source family.

Produces a versioned, immutable interim layout dataset containing DAG
topology, node observations, successor structure, and start/goal flags.
These are reusable structural facts consumed by downstream task CLIs
(build-seqmaze.py).

Layout channels (structural):

- ``node_obs_id``: Per-node observation ID after remapping. Shape ``(N,)`` int32.
- ``node_candidate_index``: Candidate index equals permuted position. Shape ``(N,)`` int32.
- ``node_start_flag``: True for the unique start node. Shape ``(N,)`` bool.
- ``node_goal_flag``: True for the unique goal node. Shape ``(N,)`` bool.
- ``successor_indices``: Successor node indices per slot. Shape ``(N, K)`` int32.
- ``successor_mask``: Valid successor slot mask. Shape ``(N, K)`` bool.
- ``node_mask``: True for actual (non-padded) nodes. Shape ``(N,)`` bool.

Task protocol channels (target_path, path_mask, path_length, edge_label,
edge_mask) belong in the respective task corpus, not here.

Layout dataset path: ``data/interim/dagflow/{preset}/v{version}/``
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ehc_sn.data.lifecycle import (
    extract_version,
    staging_root,
    validate_version_root,
    write_index_at_root,
    write_split,
)
from ehc_sn.data.manifest import write_manifest
from ehc_sn.utils.graph import (
    generate_hamiltonian_dag,
    remap_obs_ids,
)
from ehc_sn.utils.graph import shortest_path as bfs_shortest_path

# =============================================================================
LAYOUT_FAMILY: str = "dagflow"
"""Layout-dataset family name for the dagflow source."""

LAYOUT_CHANNELS: list[str] = [
    "node_obs_id",
    "node_candidate_index",
    "node_start_flag",
    "node_goal_flag",
    "successor_indices",
    "successor_mask",
    "node_mask",
]
"""Canonical layout channels for dagflow worlds."""

_SOURCE_ID: str = "synthetic/dagflow"
_DEFAULT_PRESET: str = "default"

_SPLITS: tuple[str, ...] = ("train", "val", "test")


# =============================================================================
def _generate_graph_sample(
    n_max: int,
    k_max: int,
    rng: np.random.Generator,
    *,
    target_edges: int | None = None,
    fixed_n_actual: bool = False,
    min_extra_edges_per_node: int = 0,
) -> dict[str, np.ndarray]:
    """Generate one random DAG sample with structural channels only.

    Uses :func:`generate_hamiltonian_dag` which guarantees a mandatory
    backbone ``0→1→…→n_actual-1``, full reachability, and no stranded
    nodes (except the terminal).

    Args:
        n_max: Maximum candidate nodes (N). Actual nodes < n_max are padded.
        k_max: Maximum out-degree per node (K).
        rng: Seeded random generator.
        target_edges: Desired total edge count.  ``None`` for probabilistic
            sampling without a fixed target.
        fixed_n_actual: When True, use all ``n_max`` nodes (no randomization).
        min_extra_edges_per_node: Minimum extra edges beyond the mandatory
            backbone edge.  Passed through to :func:`generate_hamiltonian_dag`.

    Returns:
        Sample dict with layout channels (no path or edge labels).
    """
    if fixed_n_actual:
        n_actual = n_max
    else:
        n_actual_low = max(4, min(n_max - 1, 10))
        n_actual = int(rng.integers(n_actual_low, n_max + 1))

    base_seed = int(rng.integers(0, 2**31))
    adjacency = generate_hamiltonian_dag(
        n_actual,
        k_max,
        base_seed,
        target_edges=target_edges,
        min_extra_edges_per_node=min_extra_edges_per_node,
    )

    start_idx = 0
    goal_idx = n_actual - 1

    remap_seed = int(rng.integers(0, 2**31))
    obs_ids = remap_obs_ids(n_actual, remap_seed)

    perm_seed = int(rng.integers(0, 2**31))
    from ehc_sn.utils.graph import permute_candidate_order

    perm, inv = permute_candidate_order(n_actual, perm_seed)

    node_obs_id = np.zeros(n_max, dtype=np.int32)
    node_candidate_index = np.zeros(n_max, dtype=np.int32)
    node_start_flag = np.zeros(n_max, dtype=bool)
    node_goal_flag = np.zeros(n_max, dtype=bool)
    successor_indices = np.zeros((n_max, k_max), dtype=np.int32)
    successor_mask = np.zeros((n_max, k_max), dtype=bool)
    node_mask = np.zeros(n_max, dtype=bool)

    for orig_i in range(n_actual):
        pi = perm[orig_i]
        node_obs_id[pi] = obs_ids[orig_i]
        node_candidate_index[pi] = pi
        node_start_flag[pi] = orig_i == start_idx
        node_goal_flag[pi] = orig_i == goal_idx
        node_mask[pi] = True

        for k, succ_orig in enumerate(adjacency[orig_i]):
            succ_pi = perm[succ_orig]
            successor_indices[pi, k] = succ_pi
            successor_mask[pi, k] = True

    return {
        "node_obs_id": node_obs_id,
        "node_candidate_index": node_candidate_index,
        "node_start_flag": node_start_flag,
        "node_goal_flag": node_goal_flag,
        "successor_indices": successor_indices,
        "successor_mask": successor_mask,
        "node_mask": node_mask,
    }


# =============================================================================
def validate_dagflow_layout_sample(data: dict[str, np.ndarray]) -> None:
    """Validate one dagflow layout sample.

    Args:
        data: Dict of channel arrays for one sample.

    Raises:
        ValueError: On any structural or semantic violation.
    """
    missing = set(LAYOUT_CHANNELS) - data.keys()
    if missing:
        raise ValueError(
            f"dagflow layout sample missing channels: {sorted(missing)}"
        )

    n_max = data["node_obs_id"].shape[0]
    k_max = data["successor_indices"].shape[1]

    if data["node_candidate_index"].shape[0] != n_max:
        raise ValueError("node_candidate_index does not match N.")
    if data["successor_indices"].shape[0] != n_max:
        raise ValueError("successor_indices first dim does not match N.")
    if data["successor_mask"].shape != (n_max, k_max):
        raise ValueError("successor_mask shape mismatch.")
    if data["node_mask"].shape[0] != n_max:
        raise ValueError("node_mask does not match N.")

    n_actual = int(data["node_mask"].sum())
    if n_actual < 2:
        raise ValueError(f"At least 2 actual nodes required, got {n_actual}.")

    n_starts = int(data["node_start_flag"].sum())
    if n_starts != 1:
        raise ValueError(f"Expected exactly one start flag, got {n_starts}.")
    n_goals = int(data["node_goal_flag"].sum())
    if n_goals != 1:
        raise ValueError(f"Expected exactly one goal flag, got {n_goals}.")

    start_idx = int(data["node_start_flag"].argmax())
    goal_idx = int(data["node_goal_flag"].argmax())

    adjacency = [[] for _ in range(n_actual)]
    for i in range(n_actual):
        for k in range(k_max):
            if data["successor_mask"][i, k]:
                succ = int(data["successor_indices"][i, k])
                if succ < n_actual:
                    adjacency[i].append(succ)

    sp = bfs_shortest_path(adjacency, start_idx, goal_idx)
    if not sp:
        raise ValueError(
            "No path exists from start to goal in the graph substrate."
        )


# =============================================================================
def build_dagflow_layouts(
    version_root: Path,
    *,
    preset: str = "default",
    n_max: int = 45,
    t_max: int = 32,
    max_out_degree: int = 4,
    target_edges: int | None = None,
    n_train: int = 4000,
    n_val: int = 500,
    n_test: int = 500,
    seed: int = 42,
    fixed_n_actual: bool = False,
    min_extra_edges_per_node: int = 0,
) -> None:
    """Build the dagflow layout dataset at *version_root*.

    Generates random DAG samples with a mandatory Hamiltonian backbone
    and controlled shortcut edges.  Every non-terminal node has out-degree
    ≥ 1, so there are no stranded nodes.  Only structural channels are
    written — task protocol labels (paths, edges) belong in the task
    corpus builder.

    The version integer is derived from the ``v<N>`` leaf of *version_root*.

    Args:
        version_root: Destination versioned root
            (e.g. ``data/interim/dagflow/default/v1``).  Must not already exist.
        preset: Named source preset (default: "default").
        n_max: Maximum candidate nodes per batch (N).
        t_max: Maximum generated path length (T) — stored in manifest for task
            builders but not a channel of the substrate.
        max_out_degree: Maximum out-degree per node (K).
        target_edges: Desired total edge count per graph.  When set, adds
            approximately this many edges (soft target, capped by out-degree
            limits).  When ``None``, edges are added probabilistically.
        n_train: Number of training samples.
        n_val: Number of validation samples.
        n_test: Number of test samples.
        seed: Deterministic base seed for reproducibility.

    Raises:
        FileExistsError: When *version_root* already exists (immutable root).
        ValueError: When the version leaf name is not ``v<integer>``.
    """
    version = extract_version(version_root)

    stage_params = {
        "preset": preset,
        "n_max": n_max,
        "t_max": t_max,
        "max_out_degree": max_out_degree,
        "target_edges": target_edges,
        "fixed_n_actual": fixed_n_actual,
        "min_extra_edges_per_node": min_extra_edges_per_node,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "seed": seed,
    }
    split_counts = {"train": n_train, "val": n_val, "test": n_test}

    seed_seq = np.random.SeedSequence(seed)
    split_rngs = dict(
        zip(
            _SPLITS,
            [np.random.default_rng(s) for s in seed_seq.spawn(len(_SPLITS))],
        )
    )

    with staging_root(version_root) as tmp:
        all_entries = []
        for split in _SPLITS:
            n = split_counts[split]
            rng = split_rngs[split]

            samples = []
            for _ in range(n):
                sample_seed = int(rng.integers(0, 2**31))
                sample_rng = np.random.default_rng(sample_seed)
                samples.append(
                    _generate_graph_sample(
                        n_max=n_max,
                        k_max=max_out_degree,
                        rng=sample_rng,
                        target_edges=target_edges,
                        fixed_n_actual=fixed_n_actual,
                        min_extra_edges_per_node=min_extra_edges_per_node,
                    )
                )

            entries = write_split(
                output_root=tmp,
                split=split,
                samples=samples,
                source=LAYOUT_FAMILY,
                channels=LAYOUT_CHANNELS,
                topology_kind="dag",
                n_states=n_max,
                extent=[n_max],
                index_kwargs={},
                sample_validator=validate_dagflow_layout_sample,
            )
            all_entries.extend(entries)

        write_index_at_root(all_entries, tmp)

        write_manifest(
            tmp,
            dataset_class="layout_dataset",
            family=LAYOUT_FAMILY,
            version=version,
            channels=LAYOUT_CHANNELS,
            topology_kind="dag",
            n_states=n_max,
            extent=[n_max],
            n_samples=split_counts,
            source_id=_SOURCE_ID,
            builder="ehc_sn.data.substrate.dagflow.build_dagflow_layouts",
            seed=seed,
            stage_params=stage_params,
            preset=preset,
            n_max=n_max,
            t_max=t_max,
            max_out_degree=max_out_degree,
            target_edges=target_edges,
            fixed_n_actual=fixed_n_actual,
            min_extra_edges_per_node=min_extra_edges_per_node,
        )

    n_total = n_train + n_val + n_test
    print(
        f"dagflow layout dataset written to {version_root}  "
        f"({n_total} samples.)"
    )


def validate_dagflow_layout_root(root: Path) -> dict:
    """Validate a dagflow layout dataset root.

    Args:
        root: Resolved versioned dagflow layout dataset root.

    Returns:
        Parsed manifest dict.

    Raises:
        ValueError: On any contract violation.
        FileNotFoundError: When a required file is absent.
    """
    manifest = validate_version_root(root)
    if manifest.get("family") != LAYOUT_FAMILY:
        raise ValueError(
            f"Root family is {manifest.get('family')!r}, "
            f"expected {LAYOUT_FAMILY!r}."
        )
    if manifest.get("dataset_class") != "layout_dataset":
        raise ValueError(
            f"Expected dataset_class 'layout_dataset', "
            f"got {manifest.get('dataset_class')!r}."
        )

    missing_ch = set(LAYOUT_CHANNELS) - set(manifest.get("channels", []))
    if missing_ch:
        raise ValueError(
            f"Manifest missing required dagflow layout channels: "
            f"{sorted(missing_ch)}"
        )

    for split, n in manifest["n_samples"].items():
        split_dir = root / split
        arrays: dict[str, np.ndarray] = {}
        for ch in LAYOUT_CHANNELS:
            ch_file = split_dir / f"{ch}.npy"
            if not ch_file.exists():
                raise FileNotFoundError(
                    f"Missing channel '{ch}' in {split_dir}."
                )
            arrays[ch] = np.load(ch_file, mmap_mode="r")
            if arrays[ch].shape[0] != n:
                raise ValueError(
                    f"Channel '{ch}' in split '{split}' "
                    f"has {arrays[ch].shape[0]} samples, "
                    f"manifest declares {n}."
                )

        for i in range(n):
            sample = {ch: arrays[ch][i] for ch in LAYOUT_CHANNELS}
            validate_dagflow_layout_sample(sample)

    return manifest


__all__ = [
    "LAYOUT_FAMILY",
    "LAYOUT_CHANNELS",
    "build_dagflow_layouts",
    "validate_dagflow_layout_root",
    "validate_dagflow_layout_sample",
]
