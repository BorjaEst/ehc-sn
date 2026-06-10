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
        topology_kind=topology_type,
        n_states=-1,
        extent=[],
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
        valid_state_mask=ly["valid_state_mask"],
        state_to_row_col=ly["state_to_row_col"],
        observation_id=ly["observation_id"],
        adjacency=ly["adjacency"],
        action_count=np.int32(ly["action_space"]["action_count"]),
        stay_action=np.int32(ly["action_space"].get("stay_action", -1)),
        action_names=np.array(ly["action_space"]["action_names"], dtype="U"),
        action_deltas=np.array(
            ly["action_space"]["action_deltas"], dtype=np.int32
        ).ravel(),
        transition_matrix=ly["transition_matrix"],
        topology_seed=np.int32(ly["topology_seed"]),
        sensory_seed=np.int32(ly["sensory_seed"]),
        sensory_vocab_size=np.int32(ly["sensory_vocab_size"]),
    )
    # Persist split when present (NotRequired field).
    split_val = ly.get("split")
    if split_val is not None:
        save_kwargs["split"] = np.array(split_val, dtype="U")
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
    """Convert a loaded NPZ file to a SpatialLayout record."""
    action_count = int(d["action_count"])
    raw_deltas = d["action_deltas"]
    action_deltas = [
        (int(raw_deltas[i * 2]), int(raw_deltas[i * 2 + 1]))
        for i in range(action_count)
    ]
    result: SpatialLayout = {
        "layout_id": str(d["layout_id"]),
        "layout_family": str(d["layout_family"]),
        "topology_type": str(d["topology_type"]),
        "graph_state_count": int(d["graph_state_count"]),
        "valid_state_mask": d["valid_state_mask"],
        "state_to_row_col": d["state_to_row_col"],
        "observation_id": d["observation_id"],
        "adjacency": d["adjacency"].astype(bool),
        "action_space": {
            "name": str(d.get("topology_type", "grid4_dir")),
            "action_count": action_count,
            "stay_action": int(d["stay_action"]),
            "action_names": [str(n) for n in d["action_names"]],
            "action_deltas": action_deltas,
        },
        "transition_matrix": d["transition_matrix"],
        "topology_seed": int(d["topology_seed"]),
        "sensory_seed": int(d["sensory_seed"]),
        "sensory_vocab_size": int(d["sensory_vocab_size"]),
    }
    # Restore persisted split when present.
    if "split" in d:
        result["split"] = str(d["split"])
    return result


__all__ = [
    "load_layout_dataset",
    "write_layout_dataset",
]
