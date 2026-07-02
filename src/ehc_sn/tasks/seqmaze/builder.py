"""SeqMaze task corpus materialization.

Owns seqmaze-specific task protocol validation and the builder that produces
the seqmaze task corpus over a parent dagflow shared substrate.

All structural channels (node_obs_id, successor_indices/mask, node_mask)
successor indices/mask, node_mask) are read from the dagflow layout dataset.
The task builder computes shortest paths and encodes task protocol labels
(target_path, path_mask, path_length, edge_label, edge_mask).

Path written: ``data/processed/seqmaze/<corpus>/v<version>/``
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Final

import numpy as np

from ehc_sn.data.lifecycle import (
    extract_version,
    staging_root,
    validate_version_root,
    write_index_at_root,
    write_split,
)
from ehc_sn.data.manifest import read_manifest, write_manifest
from ehc_sn.data.substrate.reader import (
    iter_substrate_entries_and_samples,
)
from ehc_sn.utils.graph import shortest_path as bfs_shortest_path

# =============================================================================
TASK_FAMILY: Final[str] = "seqmaze"
"""Task namespace for the seqmaze task corpus."""

SEQMAZE_TASK_CHANNELS: Final[list[str]] = [
    "node_obs_id",
    "node_rank",
    "rank_to_obs_id",
    "obs_id_to_rank",
    "node_candidate_index",
    "node_start_flag",
    "node_goal_flag",
    "successor_indices",
    "successor_mask",
    "node_mask",
    "target_path",
    "path_mask",
    "path_length",
    "edge_label",
    "edge_mask",
]
"""All channels in the seqmaze task corpus.

Structural channels (node_obs_id, successor_indices/mask, node_mask) are
read from the parent dagflow shared substrate.  ``node_candidate_index``,
``node_start_flag``, ``node_goal_flag`` are derived from rank information
in the substrate.  Source and sink are derived from the graph structure
(in-degree / out-degree) rather than pre-annotated flags.

Edge channels (edge_label, edge_mask) support the Phase 0 probe evaluation.
Path channels (target_path, path_mask, path_length) support the v1
path-prediction task.
"""

_REQUIRED_PARENT_CHANNELS: tuple[str, ...] = (
    "node_obs_id",
    "node_rank",
    "successor_indices",
    "successor_mask",
    "node_mask",
    "obs_id_to_rank",
    "rank_to_obs_id",
)
"""Channels the seqmaze task builder requires in the parent dagflow substrate."""

SEQMAZE_TASK_CHANNEL_DTYPES: dict[str, np.dtype] = {
    "node_obs_id": np.dtype(np.int32),
    "node_rank": np.dtype(np.int32),
    "rank_to_obs_id": np.dtype(np.int32),
    "obs_id_to_rank": np.dtype(np.int32),
    "node_candidate_index": np.dtype(np.int32),
    "node_start_flag": np.dtype(bool),
    "node_goal_flag": np.dtype(bool),
    "successor_indices": np.dtype(np.int32),
    "successor_mask": np.dtype(bool),
    "node_mask": np.dtype(bool),
    "target_path": np.dtype(np.int32),
    "path_mask": np.dtype(bool),
    "path_length": np.dtype(np.int32),
    "edge_label": np.dtype(np.int32),
    "edge_mask": np.dtype(bool),
}
"""Expected numpy dtypes for seqmaze task corpus channels."""

_SPLITS: tuple[str, ...] = ("train", "val", "test")


# =============================================================================
def validate_seqmaze_sample(data: dict[str, np.ndarray]) -> None:
    """Validate a seqmaze task corpus sample against the task channel schema.

    In rank-indexed storage, row ``i`` corresponds to rank ``i``.
    ``successor_indices[i, k]`` contains the **public observation ID** of
    the k-th successor.  The validator maps public IDs back to ranks via
    ``obs_id_to_rank`` to build adjacency in consistent rank space.

    Raises:
        ValueError: On any contract violation.
    """
    missing = set(SEQMAZE_TASK_CHANNELS) - data.keys()
    if missing:
        raise ValueError(f"SeqMaze sample missing channels: {sorted(missing)}")

    for name, arr in data.items():
        if (
            name in SEQMAZE_TASK_CHANNEL_DTYPES
            and arr.dtype != SEQMAZE_TASK_CHANNEL_DTYPES[name]
        ):
            raise ValueError(
                f"Channel '{name}' has dtype {arr.dtype}, expected "
                f"{SEQMAZE_TASK_CHANNEL_DTYPES[name]}."
            )

    # Validate basic shape invariants
    n_max = data["node_obs_id"].shape[0]
    k_max = data["successor_indices"].shape[1]
    t_max = data["target_path"].shape[0]

    if data["successor_indices"].shape[0] != n_max:
        raise ValueError("successor_indices first dim does not match N.")
    if data["successor_mask"].shape != (n_max, k_max):
        raise ValueError("successor_mask shape mismatch.")
    if data["node_mask"].shape[0] != n_max:
        raise ValueError("node_mask does not match N.")
    if data["target_path"].shape[0] != t_max:
        raise ValueError("target_path length mismatch.")
    if data["node_candidate_index"].shape[0] != n_max:
        raise ValueError("node_candidate_index does not match N.")
    if data["node_start_flag"].shape[0] != n_max:
        raise ValueError("node_start_flag does not match N.")
    if data["node_goal_flag"].shape[0] != n_max:
        raise ValueError("node_goal_flag does not match N.")

    # Build rank-space adjacency using obs_id_to_rank.
    # With rank-indexed storage, row i = rank i.
    n_actual = int(data["node_mask"].sum())
    o2r = data["obs_id_to_rank"]
    adjacency = [[] for _ in range(n_actual)]
    for r in range(n_actual):
        for k in range(k_max):
            if data["successor_mask"][r, k]:
                succ_pub = int(data["successor_indices"][r, k])
                if 0 <= succ_pub < n_actual:
                    succ_rank = int(o2r[succ_pub])
                    adjacency[r].append(succ_rank)

    # Derive source (rank 0) and sink (rank n_actual-1) from graph structure.
    in_deg = [0] * n_actual
    out_deg = [0] * n_actual
    for u in range(n_actual):
        for v in adjacency[u]:
            out_deg[u] += 1
            in_deg[v] += 1
    start_idx = next(i for i, d in enumerate(in_deg) if d == 0)
    goal_idx = next(i for i, d in enumerate(out_deg) if d == 0)

    sp = bfs_shortest_path(adjacency, start_idx, goal_idx)

    if not sp:
        raise ValueError("No path exists from start to goal.")
    # Validate stored path is a correct prefix of the BFS shortest path.
    # When the stored path is truncated (path_length == t_max because
    # EOS was forced at position t_max-1), only the stored prefix is
    # checked.  This mirrors the seq2seq truncation contract: the
    # training code uses path_mask to ignore positions beyond the
    # stored length, so only the prefix must match.
    stored_len = int(data["path_length"]) - 1  # actual nodes, excl EOS
    stored_path = list(data["target_path"][:stored_len])
    if stored_path != sp[:stored_len]:
        raise ValueError(
            "Target path prefix does not match BFS shortest path. "
            f"stored={stored_path} expected_prefix={sp[:stored_len]}"
        )


# =============================================================================
def _encode_path_labels(
    n_actual: int,
    n_max: int,
    t_max: int,
    adjacency: list[list[int]],
) -> dict[str, np.ndarray]:
    """Compute path and edge labels from graph structure.

    With rank-indexed storage, row ``i`` = rank ``i``.  Adjacency is in
    rank-space (successor ranks), and the target path is stored as rank
    indices.

    Args:
        n_actual: Number of actual (non-padded) nodes.
        n_max: Maximum padded N.
        t_max: Maximum path length T.
        adjacency: Adjacency list for actual nodes in **rank space**.

    Returns:
        Dict with target_path, path_mask, path_length, edge_label, edge_mask.
    """
    # Source is the unique node with in-degree 0; sink has out-degree 0.
    in_deg = [0] * n_actual
    out_deg = [0] * n_actual
    for u in range(n_actual):
        for v in adjacency[u]:
            out_deg[u] += 1
            in_deg[v] += 1
    start_rank = next(i for i, d in enumerate(in_deg) if d == 0)
    goal_rank = next(i for i, d in enumerate(out_deg) if d == 0)

    sp = bfs_shortest_path(adjacency, start_rank, goal_rank)
    if not sp:
        raise RuntimeError("No path from start to goal.")

    # Path is already in rank space — no obs_to_row conversion needed.
    target_path_raw = sp

    # Edge label matrix: edge_label[r, s] = 1 if s in adj[r] (in rank space)
    edge_label = np.zeros((n_max, n_max), dtype=np.int32)
    edge_mask = np.zeros((n_max, n_max), dtype=bool)

    for r in range(n_actual):
        for sr in adjacency[r]:
            edge_label[r, sr] = 1
        for j in range(n_actual):
            edge_mask[r, j] = True

    # Pad target path to t_max
    target_path = np.full(
        t_max, fill_value=n_max + 1, dtype=np.int32
    )  # PAD = N+1
    path_mask = np.zeros(t_max, dtype=bool)

    path_len = len(target_path_raw)
    for i in range(min(path_len, t_max - 1)):
        target_path[i] = target_path_raw[i]
        path_mask[i] = True
    # EOS token
    target_path[min(path_len, t_max - 1)] = n_max  # EOS = N
    path_mask[min(path_len, t_max - 1)] = True
    path_length = min(path_len + 1, t_max)

    return {
        "target_path": target_path,
        "path_mask": path_mask,
        "path_length": np.array(path_length, dtype=np.int32),
        "edge_label": edge_label,
        "edge_mask": edge_mask,
    }


# =============================================================================
def validate_seqmaze_root(root: Path) -> dict:
    """Validate a seqmaze task corpus root against task-owned semantics.

    Args:
        root: Resolved versioned seqmaze task corpus root.

    Returns:
        Parsed manifest dict.

    Raises:
        ValueError: On any contract violation.
        FileNotFoundError: When a required file is absent.
    """
    manifest = validate_version_root(root)
    if manifest.get("dataset_class") != "task_corpus":
        raise ValueError("Root is not a task_corpus.")
    if manifest.get("task") != TASK_FAMILY:
        raise ValueError(
            f"Root task is {manifest.get('task')!r}, expected {TASK_FAMILY!r}."
        )

    n_max = manifest["n_max"]
    t_max = manifest["t_max"]
    k_max = manifest["max_out_degree"]
    manifest_vocab = manifest["path_vocab_size"]
    expected_vocab = n_max + 2
    if manifest_vocab != expected_vocab:
        raise ValueError(
            f"Manifest path_vocab_size={manifest_vocab}, "
            f"expected {expected_vocab} (= n_max + 2)."
        )

    for split, n in manifest["n_samples"].items():
        split_dir = root / split
        arrays: dict[str, np.ndarray] = {}
        for ch in SEQMAZE_TASK_CHANNELS:
            ch_file = split_dir / f"{ch}.npy"
            if not ch_file.exists():
                raise FileNotFoundError(
                    f"Missing task channel '{ch}' in {split_dir}."
                )
            arrays[ch] = np.load(ch_file, mmap_mode="r")
            if arrays[ch].shape[0] != n:
                raise ValueError(
                    f"Task channel '{ch}' in split '{split}' "
                    f"has {arrays[ch].shape[0]} samples, manifest declares {n}."
                )

            # Validate per-channel dimensions
            if ch == "node_obs_id":
                if arrays[ch].shape[1] != n_max:
                    raise ValueError(
                        f"Channel '{ch}' has N={arrays[ch].shape[1]}, "
                        f"expected {n_max}."
                    )
            if ch == "target_path" or ch == "path_mask":
                if arrays[ch].shape[1] != t_max:
                    raise ValueError(
                        f"Channel '{ch}' has T={arrays[ch].shape[1]}, "
                        f"expected {t_max}."
                    )
            if ch == "successor_indices" or ch == "successor_mask":
                if arrays[ch].shape[2] != k_max:
                    raise ValueError(
                        f"Channel '{ch}' has K={arrays[ch].shape[2]}, "
                        f"expected {k_max}."
                    )

        for i in range(n):
            sample = {ch: arrays[ch][i] for ch in SEQMAZE_TASK_CHANNELS}
            validate_seqmaze_sample(sample)

    return manifest


# =============================================================================
def build_seqmaze_task_corpus(
    version_root: Path,
    *,
    layout_root: Path,
    corpus: str = "default",
    n_max: int = 32,
    t_max: int = 32,
    max_out_degree: int = 4,
    n_train: int = 4000,
    n_val: int = 500,
    n_test: int = 500,
    seed: int = 42,
) -> None:
    """Build the seqmaze task corpus at *version_root* over a dagflow layout.

    All structural channels are read from the dagflow layout dataset.
    The task builder computes shortest paths and encodes task
    protocol labels (target_path, path_mask, path_length, edge_label,
    edge_mask).

    The version integer is derived from the ``v<N>`` leaf of *version_root*.

    Args:
        version_root: Destination versioned root
            (e.g. ``data/processed/seqmaze/default/v1``).  Must not exist.
        layout_root: Path to the dagflow layout dataset
            root.  Must contain a ``manifest.json`` with all required
            structural channels.
        corpus: Corpus label (e.g. ``"default"``).
        n_max: Maximum candidate nodes per batch (N).
        t_max: Maximum generated path length (T).
        max_out_degree: Maximum out-degree per node (K).
        n_train: Number of training samples.
        n_val: Number of validation samples.
        n_test: Number of test samples.
        seed: Deterministic base seed for reproducibility.

    Raises:
        FileExistsError: When *version_root* already exists (immutable root).
        FileNotFoundError: When *layout_root* has no manifest.
        ValueError: When the layout is not a dagflow layout_dataset,
            or the layout lacks required structural channels.
    """
    version = extract_version(version_root)
    layout_manifest = read_manifest(layout_root)

    if layout_manifest.get("family") != "dagflow":
        raise ValueError(
            f"SeqMaze task corpus requires a 'dagflow' layout "
            f"dataset, got family={layout_manifest.get('family')!r}."
        )

    # Validate layout dataset has all required structural channels.
    layout_channels = set(layout_manifest.get("channels", []))
    missing = [
        ch for ch in _REQUIRED_PARENT_CHANNELS if ch not in layout_channels
    ]
    if missing:
        raise ValueError(
            f"Layout dataset lacks required structural channels: "
            f"{', '.join(missing)}. "
            f"Rebuild with:\n"
            f"    python scripts/data-gen/build-dagflow.py build-all"
        )
    # Validate n_max capacity against substrate extent.
    layout_n_max = layout_manifest.get("n_max", 0)
    if n_max < layout_n_max:
        raise ValueError(
            f"n_max ({n_max}) is smaller than substrate n_max "
            f"({layout_n_max}).  Real graph nodes would be truncated. "
            f"Increase --n-max or regenerate the dagflow substrate "
            f"with a smaller --n-max."
        )
    if n_max > layout_n_max:
        warnings.warn(
            f"n_max ({n_max}) exceeds substrate n_max "
            f"({layout_n_max}).  Extra {n_max - layout_n_max} padding "
            f"slots will be added to each sample.",
            UserWarning,
            stacklevel=2,
        )
    split_counts = {"train": n_train, "val": n_val, "test": n_test}
    layout_n = layout_manifest.get("n_samples", {})
    for split, n in split_counts.items():
        avail = layout_n.get(split, 0)
        if n > avail:
            raise ValueError(
                f"Requested {n} {split!r} samples "
                f"but layout dataset only has {avail}."
            )

    stage_params = {
        "corpus": corpus,
        "n_max": n_max,
        "t_max": t_max,
        "max_out_degree": max_out_degree,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "seed": seed,
        "layout_version": layout_manifest["version"],
    }
    canonical_layout = (
        f"data/interim/dagflow/"
        f"{layout_manifest.get('preset', 'balanced-default')}/"
        f"v{layout_manifest['version']}"
    )

    with staging_root(version_root) as tmp:
        all_entries = []
        for split in _SPLITS:
            n = split_counts[split]
            entry_sample_pairs = list(
                iter_substrate_entries_and_samples(
                    layout_root,
                    split,
                    list(_REQUIRED_PARENT_CHANNELS),
                )
            )

            samples = []
            for entry, substrate_sample in entry_sample_pairs[:n]:
                n_actual = int(substrate_sample["node_mask"].sum())
                o2r = substrate_sample["obs_id_to_rank"]  # (n_actual,) int32
                k_max = substrate_sample["successor_indices"].shape[-1]

                # Build rank-space adjacency: map public observation IDs
                # back to ranks using obs_id_to_rank.  With rank-indexed
                # storage, row r = rank r.
                adjacency = [[] for _ in range(n_actual)]
                for r in range(n_actual):
                    for k in range(k_max):
                        if substrate_sample["successor_mask"][r, k]:
                            succ_pub = int(
                                substrate_sample["successor_indices"][r, k]
                            )
                            if 0 <= succ_pub < n_actual:
                                succ_rank = int(o2r[succ_pub])
                                adjacency[r].append(succ_rank)

                path_labels = _encode_path_labels(
                    n_actual=n_actual,
                    n_max=n_max,
                    t_max=t_max,
                    adjacency=adjacency,
                )

                # Derive start/goal flags from rank 0 and rank n_actual-1
                node_start_flag = np.zeros(n_max, dtype=bool)
                node_goal_flag = np.zeros(n_max, dtype=bool)
                node_start_flag[0] = True
                node_goal_flag[n_actual - 1] = True

                # Node candidate index: row index = rank index
                node_candidate_index = np.arange(n_max, dtype=np.int32)

                # Pad rank lookup tables to n_max for uniform shape stacking
                sentinel = n_actual
                padded_o2r = np.full(n_max, sentinel, dtype=np.int32)
                padded_r2o = np.full(n_max, sentinel, dtype=np.int32)
                padded_o2r[:n_actual] = substrate_sample["obs_id_to_rank"]
                padded_r2o[:n_actual] = substrate_sample["rank_to_obs_id"]

                sample = {
                    "node_obs_id": substrate_sample["node_obs_id"],
                    "node_rank": substrate_sample["node_rank"],
                    "rank_to_obs_id": padded_r2o,
                    "obs_id_to_rank": padded_o2r,
                    "successor_indices": substrate_sample["successor_indices"],
                    "successor_mask": substrate_sample["successor_mask"],
                    "node_mask": substrate_sample["node_mask"],
                    **path_labels,
                    "node_candidate_index": node_candidate_index,
                    "node_start_flag": node_start_flag,
                    "node_goal_flag": node_goal_flag,
                }
                samples.append(sample)

            entries = write_split(
                output_root=tmp,
                split=split,
                samples=samples,
                source=f"{TASK_FAMILY}/{corpus}",
                channels=SEQMAZE_TASK_CHANNELS,
                topology_kind="dag",
                n_states=n_max,
                extent=[n_max],
                index_kwargs={
                    "task_metadata": {
                        "task": TASK_FAMILY,
                        "corpus": corpus,
                    },
                },
                sample_validator=validate_seqmaze_sample,
            )
            all_entries.extend(entries)

        write_index_at_root(all_entries, tmp)

        write_manifest(
            tmp,
            dataset_class="task_corpus",
            family=TASK_FAMILY,
            version=version,
            manifest_schema_version=1,
            channels=SEQMAZE_TASK_CHANNELS,
            topology_kind="dag",
            n_states=n_max,
            extent=[n_max],
            n_samples=split_counts,
            source_id="synthetic/dagflow",
            builder="ehp_sn.tasks.seqmaze.builder.build_seqmaze_task_corpus",
            seed=seed,
            stage_params=stage_params,
            task=TASK_FAMILY,
            corpus=corpus,
            task_schema_version=1,
            task_protocol_version=1,
            parents={
                "semantic_graph": {
                    "family": "dagflow",
                    "root": canonical_layout,
                    "version": layout_manifest["version"],
                },
            },
            n_max=n_max,
            t_max=t_max,
            max_out_degree=max_out_degree,
            path_vocab_size=n_max + 2,
            special_tokens={
                "eos": n_max,
                "pad": n_max + 1,
            },
        )
