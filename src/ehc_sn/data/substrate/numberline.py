"""Shared-substrate builder for the numberline dataset family.

Provides the :func:`build_shared_substrate` builder and
:func:`validate_numberline_shared_root` validator for the NumberLine source
family.

NumberLine is a truthful 1-D substrate: states are integers ``0 .. n_states-1``
on a bounded line.  There are no walls, no observations, and no 2-D geometry.

V1 worlds are reusable world definitions.  All worlds share the same transition
graph (bounded line); there is no world-local variation in V1.

Shared-substrate channels:

- ``state_ids``: Integer state identifiers. Shape ``(n_states,)`` int32.
- ``valid_prev``: True if PREV is legal from that state. Shape ``(n_states,)`` bool.
  Only ``False`` at ``state_id == 0``.
- ``valid_next``: True if NEXT is legal from that state. Shape ``(n_states,)`` bool.
  Only ``False`` at ``state_id == n_states - 1``.

Shared substrate path: ``data/processed/numberline/v<version>/``
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ehc_sn.data.lifecycle import extract_version, staging_root, validate_version_root, write_index_at_root, write_split
from ehc_sn.data.manifest import write_manifest

# =============================================================================
SHARED_FAMILY: str = "numberline"
"""Shared-substrate family name for the NumberLine source."""

TOPOLOGY_KIND: str = "line1d"
"""Canonical topology kind string for 1-D number-line substrates."""

SHARED_CHANNELS: list[str] = ["state_ids", "valid_prev", "valid_next"]
"""Canonical shared-substrate channels for NumberLine worlds."""

_SOURCE_ID: str = "synthetic/numberline"


# =============================================================================
def _make_world(n_states: int) -> dict[str, np.ndarray]:
    """Return one NumberLine world sample dict."""
    state_ids = np.arange(n_states, dtype=np.int32)
    valid_prev = np.ones(n_states, dtype=bool)
    valid_prev[0] = False
    valid_next = np.ones(n_states, dtype=bool)
    valid_next[n_states - 1] = False
    return {"state_ids": state_ids, "valid_prev": valid_prev, "valid_next": valid_next}


# =============================================================================
def validate_numberline_sample(data: dict[str, np.ndarray]) -> None:
    """Validate one NumberLine shared-substrate sample.

    Args:
        data: Dict of channel arrays for one world.

    Raises:
        ValueError: On any structural or semantic violation.
    """
    missing = set(SHARED_CHANNELS) - data.keys()
    if missing:
        raise ValueError(f"NumberLine sample missing channels: {sorted(missing)}")

    state_ids = data["state_ids"]
    valid_prev = data["valid_prev"]
    valid_next = data["valid_next"]

    if state_ids.ndim != 1:
        raise ValueError(f"state_ids must be 1-D, got shape {state_ids.shape}")
    n = len(state_ids)
    if n < 2:
        raise ValueError(f"NumberLine must have at least 2 states, got {n}")
    if state_ids.dtype != np.int32:
        raise ValueError(f"state_ids must be int32, got {state_ids.dtype}")
    if not np.array_equal(state_ids, np.arange(n, dtype=np.int32)):
        raise ValueError("state_ids must be consecutive integers starting from 0")

    if valid_prev.shape != (n,) or valid_prev.dtype != bool:
        raise ValueError(f"valid_prev must be bool shape ({n},), got {valid_prev.shape} {valid_prev.dtype}")
    if valid_next.shape != (n,) or valid_next.dtype != bool:
        raise ValueError(f"valid_next must be bool shape ({n},), got {valid_next.shape} {valid_next.dtype}")

    if valid_prev[0]:
        raise ValueError("valid_prev[0] must be False (PREV is illegal at state 0)")
    if not np.all(valid_prev[1:]):
        raise ValueError("valid_prev must be True for all states except state 0")
    if valid_next[n - 1]:
        raise ValueError(f"valid_next[{n-1}] must be False (NEXT is illegal at the last state)")
    if not np.all(valid_next[: n - 1]):
        raise ValueError("valid_next must be True for all states except the last")


# =============================================================================
def build_shared_substrate(
    version_root: Path,
    *,
    n_states: int = 10,
    n_worlds: int = 100,
    seed: int = 42,
) -> None:
    """Build the NumberLine shared substrate at *version_root*.

    Materialises *n_worlds* world samples, all sharing the same n_states
    bounded line topology.  V1 worlds are structurally identical; the
    world index is the only distinguishing attribute.

    The version integer is derived from the ``v<N>`` leaf of *version_root*.

    Args:
        version_root: Destination versioned root
            (e.g. ``data/processed/numberline/v1``).  Must not already exist.
        n_states: Number of states on the number line (>= 2).
        n_worlds: Total number of world samples to materialise.
        seed: Deterministic base seed (stored in manifest).

    Raises:
        FileExistsError: When *version_root* already exists (immutable root).
        ValueError: When the version leaf or parameters are invalid.
    """
    if n_states < 2:
        raise ValueError(f"n_states must be >= 2, got {n_states}")
    if n_worlds < 1:
        raise ValueError(f"n_worlds must be >= 1, got {n_worlds}")

    version = extract_version(version_root)
    stage_params = {"n_states": n_states, "n_worlds": n_worlds, "seed": seed}

    # Split: 70% train, 15% val, 15% test (round to integers)
    n_val = max(1, n_worlds // 7)
    n_test = max(1, n_worlds // 7)
    n_train = n_worlds - n_val - n_test
    counts = {"train": n_train, "val": n_val, "test": n_test}

    world = _make_world(n_states)

    with staging_root(version_root) as tmp:
        all_entries = []
        for split, n in counts.items():
            samples = [world.copy() for _ in range(n)]
            entries = write_split(
                tmp,
                split,
                samples,
                source=SHARED_FAMILY,
                channels=SHARED_CHANNELS,
                topology_kind=TOPOLOGY_KIND,
                n_states=n_states,
                extent=[n_states],
                index_kwargs={},
                sample_validator=validate_numberline_sample,
            )
            all_entries.extend(entries)

        write_index_at_root(all_entries, tmp)

        write_manifest(
            tmp,
            dataset_class="shared_substrate",
            family=SHARED_FAMILY,
            version=version,
            channels=SHARED_CHANNELS,
            topology_kind=TOPOLOGY_KIND,
            n_states=n_states,
            extent=[n_states],
            n_samples=counts,
            source_id=_SOURCE_ID,
            builder="ehc_sn.data.substrate.numberline.build_shared_substrate",
            seed=seed,
            stage_params=stage_params,
        )

    print(f"numberline shared substrate written to {version_root}  ({n_worlds} worlds).")


# =============================================================================
def validate_numberline_shared_root(root: Path) -> dict:
    """Validate a NumberLine shared substrate root against generic and family rules.

    Calls the generic structural validator then checks family-specific manifest
    fields (family name, topology_kind).

    Args:
        root: Resolved versioned NumberLine shared substrate root.

    Returns:
        Parsed manifest dict.

    Raises:
        ValueError: On any structural or semantic violation.
    """
    manifest = validate_version_root(root)

    if manifest.get("family") != SHARED_FAMILY:
        raise ValueError(f"Expected family '{SHARED_FAMILY}', got '{manifest.get('family')}'.")
    if manifest.get("topology_kind") != TOPOLOGY_KIND:
        raise ValueError(f"Expected topology_kind '{TOPOLOGY_KIND}', got '{manifest.get('topology_kind')}'.")
    extent = manifest.get("extent", [])
    n_states = manifest.get("n_states")
    if not (isinstance(extent, list) and len(extent) == 1 and extent[0] == n_states):
        raise ValueError(f"NumberLine extent must be [n_states] == [{n_states}], got {extent}.")

    return manifest


# =============================================================================
__all__ = [
    "SHARED_FAMILY",
    "TOPOLOGY_KIND",
    "SHARED_CHANNELS",
    "build_shared_substrate",
    "validate_numberline_sample",
    "validate_numberline_shared_root",
]
