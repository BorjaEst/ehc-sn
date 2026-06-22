"""Generic layout dataset I/O — source-agnostic reader and writer.

Provides :func:`write_layout_dataset` and :func:`load_layout_dataset` that
operate on the :class:`SpatialLayout` protocol regardless of the layout source
(openfield, dungeongen, etc.).

Storage backends (dispatched via ``manifest.json`` ``storage_format``):

- ``"directory_npz"`` — one NPZ file per layout under ``<root>/layouts/``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ehc_sn.data.layout._protocol import (
    SpatialLayout,
    validate_spatial_layout,
)
from ehc_sn.data.lifecycle._write import create_version_root
from ehc_sn.data.manifest import read_manifest, write_manifest


# =============================================================================
def _derive_extent(layouts: list[SpatialLayout]) -> list[int]:
    """Derive the declared canvas extent from layout ``state_to_row_col`` maxima.

    For homogeneous-size layout datasets, this computes
    ``[max_row + 1, max_col + 1]`` across all layouts.  If any layout has
    a different extent, raises ``ValueError``.

    This is a convenience fallback for generators that do not explicitly
    declare an extent.  Long-term, all generators should pass ``extent``
    explicitly.
    """
    h: int | None = None
    w: int | None = None
    for ly in layouts:
        rc = ly["state_to_row_col"]
        ly_h = int(rc[:, 0].max()) + 1
        ly_w = int(rc[:, 1].max()) + 1
        if h is None:
            h, w = ly_h, ly_w
        elif ly_h != h or ly_w != w:
            raise ValueError(
                f"Layout {ly['layout_id']!r} has extent [{ly_h}, {ly_w}], "
                f"expected [{h}, {w}]. "
                "Heterogeneous extents are not supported by _derive_extent; "
                "pass extent explicitly."
            )
    return [h or 0, w or 0]


# =============================================================================
# Writer
# =============================================================================


def write_layout_dataset(
    layouts: list[SpatialLayout],
    version_root: Path,
    *,
    topology_type: str,
    s_size: int = 45,
    topology_seed: int = 42,
    shard_size: int | None = None,
    version: int = 1,
    layout_family: str = "openfield",
    preset: str = "unknown",
    extent: list[int] | None = None,
) -> None:
    """Write a layout dataset to an interim version root.

    Writes ``manifest.json``, ``index.jsonl``, and one NPZ file per layout
    under ``<root>/layouts/``.

    Args:
        layouts: Validated spatial layout records.  Each record must carry
            a ``split`` key to determine per-split counts and index entries.
        version_root: Destination version root (must not exist).
        topology_type: Topology kind (e.g. ``"square"``, ``"rectangle"``).
        s_size: Sensory vocabulary size.
        topology_seed: Base topology seed (recorded in manifest).
        shard_size: Deprecated; kept for future sharded backend compatibility.
        version: Dataset version integer (written to manifest).
        layout_family: Source family name (e.g. ``"openfield"``, ``"dungeongen"``).
        preset: Named source preset (e.g. ``"tem-square"``, ``"default"``).
        extent: Declared ``[height, width]`` of the spatial canvas.  When
            provided, written to the manifest and used by downstream tasks
            to determine the task grid dimensions.  When ``None``, the
            manifest gets an empty list (legacy behaviour).

    Raises:
        ValueError: When any layout lacks a ``split`` key.
    """
    _ = shard_size  # placeholder for future sharded backend
    create_version_root(version_root)
    layouts_dir = version_root / "layouts"
    layouts_dir.mkdir()

    n_layouts = len(layouts)
    index_lines: list[str] = []
    split_counts: dict[str, int] = {}

    for i, ly in enumerate(layouts):
        split = ly.get("split")
        if split is None:
            raise ValueError(
                f"Layout {ly['layout_id']!r} is missing required 'split' key. "
                "All layouts passed to write_layout_dataset must carry a split assignment."
            )
        split_counts[split] = split_counts.get(split, 0) + 1

        fname = f"{i:04d}.npz"
        path = layouts_dir / fname
        _save_single_layout(path, ly)
        index_lines.append(
            json.dumps(
                {
                    "split": split,
                    "layout_id": ly["layout_id"],
                    "graph_state_count": ly["graph_state_count"],
                    "topology_type": ly["topology_type"],
                    "fname": f"layouts/{fname}",
                }
            )
        )

    (version_root / "index.jsonl").write_text("\n".join(index_lines) + "\n")

    write_manifest(
        version_root,
        dataset_class="layout_dataset",
        family=layout_family,
        version=version,
        channels=[],
        topology_kind="grid2d",
        n_states=-1,
        extent=extent or _derive_extent(layouts),
        n_samples=split_counts,
        source_id=layout_family,
        builder="ehc_sn.data.layout.io.write_layout_dataset",
        seed=topology_seed,
        stage_params={
            "topology_type": topology_type,
            "s_size": s_size,
            "topology_seed": topology_seed,
            "preset": preset,
            "n_layouts": n_layouts,
            "storage_format": "directory_npz",
            "split_counts": split_counts,
        },
        parent_family=layout_family,
        parent_version=version,
        preset=preset,
    )

    print(
        f"Layout dataset written to {version_root}  "
        f"({n_layouts} layouts across {len(split_counts)} splits)."
    )


# =============================================================================
# Reader
# =============================================================================


def load_layout_dataset(version_root: Path) -> list[SpatialLayout]:
    """Load all layout records from an interim layout dataset root.

    Dispatches by ``storage_format`` in ``manifest.json``.
    Currently supports ``"directory_npz"``.

    Args:
        version_root: Path to the interim version root.

    Returns:
        List of :class:`SpatialLayout` records.

    Raises:
        ValueError: When the manifest is missing, the storage format is
            unsupported, or any layout fails validation.
    """
    manifest = read_manifest(version_root)

    # Default to "directory_npz" for backward compatibility with early
    # openfield datasets that lack the storage_format field.
    storage_format = manifest.get("stage_params", {}).get(
        "storage_format", "directory_npz"
    )

    if storage_format == "directory_npz":
        return _load_directory_npz(version_root)
    raise ValueError(f"Unsupported storage_format: {storage_format!r}.")


# =============================================================================
# Internal helpers
# =============================================================================


def _save_single_layout(path: Path, ly: SpatialLayout) -> None:
    """Save one SpatialLayout record to a compressed NPZ file."""
    save_kwargs = dict(
        layout_id=np.array(ly["layout_id"], dtype="U"),
        layout_family=np.array(ly["layout_family"], dtype="U"),
        topology_type=np.array(ly["topology_type"], dtype="U"),
        graph_state_count=np.int32(ly["graph_state_count"]),
        state_to_row_col=ly["state_to_row_col"],
        observation_id=ly["observation_id"],
        next_state=ly["next_state"],
        action_valid=ly["action_valid"],
        action_count=np.int32(ly["action_space"]["action_count"]),
        stay_action=np.int32(ly["action_space"].get("stay_action", -1)),
        action_name=np.array(ly["action_space"]["name"], dtype="U"),
        action_names=np.array(ly["action_space"]["action_names"], dtype="U"),
        action_deltas=np.array(
            ly["action_space"]["action_deltas"], dtype=np.int32
        ).ravel(),
        movement_kind=np.array(
            ly["action_space"].get("movement_kind", "grid4"), dtype="U"
        ),
        topology_seed=np.int32(ly["topology_seed"]),
        observation_seed=np.int32(ly["observation_seed"]),
        observation_vocabulary_size=np.int32(ly["observation_vocabulary_size"]),
    )
    # Persist deprecated valid_state_mask for backward compat (always all-True).
    vsm = ly.get("valid_state_mask")
    if vsm is not None:
        save_kwargs["valid_state_mask"] = vsm
    # Persist split when present (NotRequired field).
    split_val = ly.get("split")
    if split_val is not None:
        save_kwargs["split"] = np.array(split_val, dtype="U")
    # Persist extent when present (NotRequired field).
    extent_val = ly.get("extent")
    if extent_val is not None:
        save_kwargs["extent_h"] = np.int32(extent_val[0])
        save_kwargs["extent_w"] = np.int32(extent_val[1])
    np.savez_compressed(path, **save_kwargs)


def _load_directory_npz(version_root: Path) -> list[SpatialLayout]:
    """Load layouts from a ``directory_npz`` backend.

    Looks for NPZ files under ``<root>/layouts/`` in sorted order.
    Also checks ``<root>/train/`` for backward compatibility with early
    openfield datasets.
    """
    layouts_dir = version_root / "layouts"
    if not layouts_dir.is_dir():
        # Fall back to train/ for backward compatibility.
        layouts_dir = version_root / "train"
    if not layouts_dir.is_dir():
        raise ValueError(
            f"No layouts/ or train/ directory found in {version_root}."
        )

    layouts: list[SpatialLayout] = []
    for fname in sorted(layouts_dir.glob("*.npz")):
        d = np.load(fname)
        layout = _npz_to_spatial_layout(d)
        validate_spatial_layout(layout)
        layouts.append(layout)

    return layouts


def _npz_to_spatial_layout(d: np.lib.npyio.NpzFile) -> SpatialLayout:
    """Convert a loaded NPZ file to a SpatialLayout record.

    Supports both new-format keys (``next_state`` + ``action_valid``,
    ``observation_vocabulary_size``, ``observation_seed``) and
    old-format keys (``movement_adjacency``, ``adjacency``,
    ``sensory_vocab_size``, ``sensory_seed``) with deprecation warnings
    for backward compatibility.
    """
    import warnings

    action_count = int(d["action_count"])
    raw_deltas = d["action_deltas"]
    action_deltas = [
        (int(raw_deltas[i * 2]), int(raw_deltas[i * 2 + 1]))
        for i in range(action_count)
    ]

    # --- Backward-compatible key resolution ---
    if "next_state" in d:
        next_state = d["next_state"].astype(np.int32)
        action_valid = d["action_valid"].astype(bool)
    elif "movement_adjacency" in d:
        # Old-format: construct next_state + action_valid from
        # movement_adjacency + transition_matrix + action_space deltas.
        warnings.warn(
            "NPZ key 'movement_adjacency' is deprecated; "
            "use 'next_state' and 'action_valid'.",
            DeprecationWarning,
            stacklevel=2,
        )
        movement_adj = d["movement_adjacency"].astype(bool)
        N = len(movement_adj)
        A = action_count
        next_state = np.zeros((N, A), dtype=np.int32)
        action_valid = np.zeros((N, A), dtype=bool)
        stay_idx = int(d["stay_action"])
        for s in range(N):
            for a in range(A):
                if a == stay_idx:
                    next_state[s, a] = s
                    action_valid[s, a] = True
                else:
                    dr, dc = action_deltas[a]
                    r, c = divmod(s, int(d.get("extent_w", 1)))
                    # Fallback — movement_adjacency-based reconstruction.
                    neighbors = np.where(movement_adj[s])[0]
                    if len(neighbors) == 0:
                        next_state[s, a] = s
                        action_valid[s, a] = False
                    elif len(neighbors) == 1:
                        next_state[s, a] = neighbors[0]
                        action_valid[s, a] = True
                    else:
                        # Cannot determine which neighbor corresponds to
                        # which action from undirected adjacency.  Use
                        # the first neighbor as a best-effort fallback.
                        next_state[s, a] = neighbors[0]
                        action_valid[s, a] = True
    elif "adjacency" in d:
        warnings.warn(
            "NPZ key 'adjacency' is deprecated; use 'next_state' and "
            "'action_valid'.",
            DeprecationWarning,
            stacklevel=2,
        )
        movement_adj = d["adjacency"].astype(bool)
        N = len(movement_adj)
        A = action_count
        next_state = np.zeros((N, A), dtype=np.int32)
        action_valid = np.zeros((N, A), dtype=bool)
        stay_idx = int(d["stay_action"])
        for s in range(N):
            for a in range(A):
                if a == stay_idx:
                    next_state[s, a] = s
                    action_valid[s, a] = True
                else:
                    neighbors = np.where(movement_adj[s])[0]
                    if len(neighbors) == 0:
                        next_state[s, a] = s
                        action_valid[s, a] = False
                    else:
                        next_state[s, a] = neighbors[0]
                        action_valid[s, a] = True
    else:
        raise KeyError(
            "NPZ file has neither 'next_state' nor 'movement_adjacency' "
            "nor 'adjacency' key."
        )

    if "observation_vocabulary_size" in d:
        vocab_size = int(d["observation_vocabulary_size"])
    elif "sensory_vocab_size" in d:
        warnings.warn(
            "NPZ key 'sensory_vocab_size' is deprecated; "
            "use 'observation_vocabulary_size'.",
            DeprecationWarning,
            stacklevel=2,
        )
        vocab_size = int(d["sensory_vocab_size"])
    else:
        raise KeyError(
            "NPZ file has neither 'observation_vocabulary_size' "
            "nor 'sensory_vocab_size' key."
        )

    if "observation_seed" in d:
        obs_seed = int(d["observation_seed"])
    elif "sensory_seed" in d:
        warnings.warn(
            "NPZ key 'sensory_seed' is deprecated; use 'observation_seed'.",
            DeprecationWarning,
            stacklevel=2,
        )
        obs_seed = int(d["sensory_seed"])
    else:
        obs_seed = -1  # sentinel for missing seed in very old files

    # Action-space name: prefer persisted, fall back to topology_type.
    as_name = str(d.get("action_name", d.get("topology_type", "grid4_dir")))

    # — movement_kind: prefer persisted, fall back to inference —
    if "movement_kind" in d:
        movement_kind = str(d["movement_kind"])
    else:
        movement_kind = "grid4"  # default for old files; hex would be explicit

    result: SpatialLayout = {
        "layout_id": str(d["layout_id"]),
        "layout_family": str(d["layout_family"]),
        "topology_type": str(d["topology_type"]),
        "graph_state_count": int(d["graph_state_count"]),
        "state_to_row_col": d["state_to_row_col"],
        "observation_id": d["observation_id"],
        "next_state": next_state,
        "action_valid": action_valid,
        "action_space": {
            "name": as_name,
            "action_count": action_count,
            "stay_action": int(d["stay_action"]),
            "action_names": [str(n) for n in d["action_names"]],
            "action_deltas": action_deltas,
            "movement_kind": movement_kind,
        },
        "topology_seed": int(d["topology_seed"]),
        "observation_seed": obs_seed,
        "observation_vocabulary_size": vocab_size,
    }
    # Restore deprecated valid_state_mask for backward compatibility.
    if "valid_state_mask" in d:
        result["valid_state_mask"] = d["valid_state_mask"]
    # Restore extent from persisted fields when present.
    if "extent_h" in d and "extent_w" in d:
        result["extent"] = (int(d["extent_h"]), int(d["extent_w"]))
    # Restore persisted split when present.
    if "split" in d:
        result["split"] = str(d["split"])
    return result


__all__ = [
    "load_layout_dataset",
    "write_layout_dataset",
]
