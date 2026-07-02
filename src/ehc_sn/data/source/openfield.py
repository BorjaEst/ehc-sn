"""Source-spec generation and expansion for openfield layout sources.

Provides the synthetic source stage for openfield: given a preset, split
counts, and seed, writes per-split JSONL topology specs (``source_spec``
artifact).  An expansion step reads those specs and produces topology-only
:class:`SpatialLayout` records with ``observation_id = -1`` sentinel values.

The ``source_spec`` artifact is the openfield equivalent of dungeongen's
raw tar snapshot — a reproducible, versioned source corpus that downstream
stages (``materialize-layouts``) consume.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ehc_sn.data.layout import (
    DEFAULT_GRID_ACTION_SPACE,
    SpatialLayout,
    validate_spatial_layout,
)
from ehc_sn.data.layout.openfield import (
    OPENFIELD_PRESETS,
)
from ehc_sn.data.lifecycle._write import create_version_root
from ehc_sn.data.manifest import write_manifest

_SPLITS: tuple[str, ...] = ("train", "val", "test")
_SPEC_SCHEMA_VERSION: int = 1


# ---------------------------------------------------------------------------
def generate_openfield_source_specs(
    preset: str,
    *,
    n_train: int = 1000,
    n_val: int = 40,
    n_test: int = 40,
    topology_seed: int = 42,
    version: int = 1,
    raw_root: Path = Path("data/raw/openfield"),
    _overridden_widths: list[int] | None = None,
    _overridden_heights: list[int] | None = None,
) -> None:
    """Write per-split topology spec JSONL files for an openfield preset.

    Generates a reproducible ``source_spec`` corpus at::

        {raw_root}/{preset}/v{version}/

    Each split directory contains a ``specs.jsonl`` file where every line is
    a JSON record with ``example_id``, ``topology_type``, ``width``,
    ``height``, and ``seed_offset``.  The manifest includes preset identity,
    split counts, and generation parameters.

    Args:
        preset: Named openfield preset (must be in :data:`OPENFIELD_PRESETS`).
        n_train: Number of source records for the training split.
        n_val: Number of source records for the validation split.
        n_test: Number of source records for the test split.
        topology_seed: Base seed for deterministic topology generation.
        version: Dataset version integer.
        raw_root: Family-level raw root (script appends ``{preset}/v{version}``).

    Raises:
        KeyError: When *preset* is unknown.
        FileExistsError: When the version root already exists.
    """
    cfg = OPENFIELD_PRESETS[preset]
    topology_type: Literal["square", "rectangle", "hex"] = cfg["topology_type"]
    widths: list[int] = (
        _overridden_widths if _overridden_widths is not None else cfg["widths"]
    )
    heights: list[int] | None = (
        _overridden_heights
        if _overridden_heights is not None
        else cfg.get("heights")
    )

    version_root = raw_root / preset / f"v{version}"
    if version_root.exists():
        print(
            f"Source specs already exist at {version_root}, skipping generation."
        )
        return

    n_samples = {"train": n_train, "val": n_val, "test": n_test}

    for split in _SPLITS:
        n = n_samples[split]
        split_dir = version_root / split
        split_dir.mkdir(parents=True)

        spec_lines: list[str] = []
        total_available = len(widths)

        for i in range(n):
            w = widths[i % total_available]
            h = heights[i % total_available] if heights else w
            example_id = f"openfield-{preset}-{split}-{i:06d}"
            record = {
                "example_id": example_id,
                "topology_type": topology_type,
                "width": w,
                "height": h,
                "seed_offset": i,
            }
            spec_lines.append(json.dumps(record))

        (split_dir / "specs.jsonl").write_text("\n".join(spec_lines) + "\n")

    stage_params: dict[str, Any] = {
        "preset": preset,
        "topology_type": topology_type,
        "widths": widths,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "topology_seed": topology_seed,
    }
    if heights is not None:
        stage_params["heights"] = heights

    write_manifest(
        version_root,
        dataset_class="source_spec",
        family="openfield",
        version=version,
        n_samples=n_samples,
        source_id="openfield",
        builder="ehp_sn.data.source.openfield.generate_openfield_source_specs",
        seed=topology_seed,
        stage_params=stage_params,
        preset=preset,
        spec_schema_version=_SPEC_SCHEMA_VERSION,
    )

    total = n_train + n_val + n_test
    print(
        f"Openfield source specs written to {version_root}  "
        f"({total} records, {len(widths)} topology variants)."
    )


# ---------------------------------------------------------------------------
def expand_openfield_source_specs(
    raw_root: Path,
    *,
    preset: str,
    version: int = 1,
    topology_seed: int = 42,
) -> list[SpatialLayout]:
    """Read source specs and produce topology-only SpatialLayout records.

    Each topology spec from the ``source_spec`` artifact is expanded into a
    :class:`SpatialLayout` with ``observation_id = -1`` sentinel and
    ``sensory_seed = -1``.  The returned layouts carry the ``split`` key
    matching their split of origin.

    Args:
        raw_root: Family-level raw root (``{raw_root}/{preset}/v{version}``
            is resolved internally).
        preset: Named openfield preset.
        version: Dataset version integer.
        topology_seed: Base seed (recorded in manifest; used for provenance).

    Returns:
        List of topology-only :class:`SpatialLayout` records.

    Raises:
        FileNotFoundError: When the source spec root does not exist.
    """
    version_root = raw_root / preset / f"v{version}"
    if not version_root.exists():
        raise FileNotFoundError(
            f"Source spec root not found: {version_root}. "
            f"Run generate-topology first."
        )

    layouts: list[SpatialLayout] = []

    for split in _SPLITS:
        spec_file = version_root / split / "specs.jsonl"
        if not spec_file.exists():
            continue

        with spec_file.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                spec = json.loads(line)

                w = spec["width"]
                h = spec["height"]
                n_states = w * h

                # Build action-conditioned transition table for grid4.
                # Action order: STAY=0, UP=1, RIGHT=2, DOWN=3, LEFT=4.
                _GRID4_DELTAS: list[tuple[int, int]] = [
                    (0, 0),  # STAY
                    (-1, 0),  # UP
                    (0, 1),  # RIGHT
                    (1, 0),  # DOWN
                    (0, -1),  # LEFT
                ]
                n_actions = 5
                next_state = np.zeros((n_states, n_actions), dtype=np.int32)
                action_valid = np.zeros((n_states, n_actions), dtype=bool)

                for s in range(n_states):
                    r, c = divmod(s, w)
                    for a, (dr, dc) in enumerate(_GRID4_DELTAS):
                        if a == 0:  # STAY
                            next_state[s, a] = s
                            action_valid[s, a] = True
                        else:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < h and 0 <= nc < w:
                                npos = nr * w + nc
                                next_state[s, a] = npos
                                action_valid[s, a] = True
                            else:
                                next_state[s, a] = s
                                action_valid[s, a] = False

                obs_ids = np.full(n_states, -1, dtype=np.int32)
                example_id = spec["example_id"]

                layout: SpatialLayout = {
                    "layout_id": f"{example_id}-topo-only",
                    "layout_family": "openfield",
                    "topology_type": spec["topology_type"],
                    "graph_state_count": n_states,
                    "state_to_row_col": np.column_stack(
                        (
                            np.arange(n_states, dtype=np.int32) // w,
                            np.arange(n_states, dtype=np.int32) % w,
                        )
                    ),
                    "observation_id": obs_ids,
                    "next_state": next_state,
                    "action_valid": action_valid,
                    "action_space": dict(DEFAULT_GRID_ACTION_SPACE),
                    "topology_seed": topology_seed,
                    "observation_seed": -1,
                    "observation_vocabulary_size": 0,
                    "extent": (h, w),
                    "split": split,
                }
                validate_spatial_layout(layout)
                layouts.append(layout)

    return layouts
    tm = tm / row_sums
    return tm
