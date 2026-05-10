"""Countwalk task corpus materialization.

Produces the Countwalk V1 task corpus over a parent NumberLine shared substrate.

Task corpus channels (per episode):

Trajectory arrays — shape ``(max_steps,)`` or ``(max_steps, width)``:

- ``trajectory_value``: Hidden state value at each step.  int32.
- ``trajectory_previous_action``: Previous action (step 0 = STAY).  int32.
- ``trajectory_anchor_visible``: Cue anchor visibility flag per step.  bool.
- ``trajectory_cue_tokens``: Cue token sequence per step.  int32, shape (T, W).
- ``trajectory_cue_mask``: Cue token validity mask per step.  bool, shape (T, W).
- ``trajectory_episode_start``: True only at step 0.  bool.
- ``trajectory_valid_step``: True for t < trajectory_length.  bool.
- ``query_mask``: True only on the final valid step.  bool.

Per-episode scalars (0-D per sample, shape ``()`` stored, stacked to ``(N,)``):

- ``world_id``: Parent world index.  int32.
- ``cue_surface_id``: Cue surface (CUE_DIGIT=0, CUE_SET=1).  int32.
- ``anchor_regime_id``: Anchor regime (ANCHOR_ONLY=0, SPARSE_REANCHOR=1).  int32.
- ``eval_bucket_id``: Evaluation bucket (BUCKET_ID=0..BUCKET_STRETCH_OOD=4).  int32.
- ``trajectory_length``: Effective trajectory length scalar.  int32.
- ``target_value``: Ground-truth final hidden state value.  int32.

Fixed-slot digit targets — shape ``(DIGIT_WIDTH,)``:

- ``target_digits``: Right-aligned digit class labels (0..9).  int32.
- ``target_digit_mask``: Active digit slot mask.  bool.

Path written: ``data/processed/countwalk/<corpus>/v<version>/``

Primary corpus guarantees:
- Legal-only PREV/NEXT/STAY scripts (PREV never at state 0; NEXT never at last state).
- No ``valid_action_mask`` channel exposed (boundary structure must not leak).
- One cue surface per episode.
- query_mask True only on the terminal valid step.
- trajectory_value is in the corpus for validation/metrics; it is NOT exposed via
  the replay payload (see ``capabilities/replay.py``).
- Evaluation buckets:
    - BUCKET_ID (0): values 0..63, horizon 1..8 — train + val + test.
    - BUCKET_RANGE_OOD (1): values 64..99, horizon 1..8 — test only.
    - BUCKET_HORIZON_OOD (2): values 0..63, horizon 9..16 — test only.
    - BUCKET_JOINT_OOD (3): values 64..99, horizon 9..16 — test only.
    - BUCKET_STRETCH_OOD (4): values 100..199, horizon 1..16 — test only.
- Within each split and each bucket, exact balance across 4 protocol cells.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np

from ehc_sn.data.lifecycle import extract_version, staging_root, validate_version_root, write_index_at_root, write_split
from ehc_sn.data.manifest import write_manifest
from ehc_sn.data.substrate.numberline import SHARED_FAMILY as NUMBERLINE_FAMILY
from ehc_sn.data.substrate.numberline import TOPOLOGY_KIND as NUMBERLINE_TOPOLOGY_KIND
from ehc_sn.data.substrate.reader import load_substrate_manifest
from ehc_sn.tasks.countwalk.contracts import (
    ACTION_NEXT,
    ACTION_PREV,
    ACTION_STAY,
    ANCHOR_ONLY,
    BUCKET_HORIZON_OOD,
    BUCKET_HORIZON_OOD_MAX,
    BUCKET_HORIZON_OOD_MIN,
    BUCKET_ID,
    BUCKET_ID_HORIZON_MAX,
    BUCKET_ID_VALUE_MAX,
    BUCKET_JOINT_OOD,
    BUCKET_RANGE_OOD,
    BUCKET_RANGE_OOD_VALUE_MAX,
    BUCKET_RANGE_OOD_VALUE_MIN,
    BUCKET_STRETCH_OOD,
    BUCKET_STRETCH_OOD_VALUE_MAX,
    BUCKET_STRETCH_OOD_VALUE_MIN,
    CUE_DIGIT,
    CUE_SET,
    DIGIT_WIDTH,
    SPARSE_REANCHOR,
    TASK_FAMILY,
    TOKEN_DIGIT_0,
    TOKEN_ITEM,
    TOKEN_PAD,
)

# =============================================================================
# Channel name constants
# =============================================================================

CHANNEL_WORLD_ID: Final[str] = "world_id"
CHANNEL_CUE_SURFACE_ID: Final[str] = "cue_surface_id"
CHANNEL_ANCHOR_REGIME_ID: Final[str] = "anchor_regime_id"
CHANNEL_EVAL_BUCKET_ID: Final[str] = "eval_bucket_id"
CHANNEL_TRAJECTORY_VALUE: Final[str] = "trajectory_value"
CHANNEL_TRAJECTORY_PREVIOUS_ACTION: Final[str] = "trajectory_previous_action"
CHANNEL_TRAJECTORY_ANCHOR_VISIBLE: Final[str] = "trajectory_anchor_visible"
CHANNEL_TRAJECTORY_CUE_TOKENS: Final[str] = "trajectory_cue_tokens"
CHANNEL_TRAJECTORY_CUE_MASK: Final[str] = "trajectory_cue_mask"
CHANNEL_TRAJECTORY_EPISODE_START: Final[str] = "trajectory_episode_start"
CHANNEL_TRAJECTORY_VALID_STEP: Final[str] = "trajectory_valid_step"
CHANNEL_TRAJECTORY_LENGTH: Final[str] = "trajectory_length"
CHANNEL_QUERY_MASK: Final[str] = "query_mask"
CHANNEL_TARGET_DIGITS: Final[str] = "target_digits"
CHANNEL_TARGET_DIGIT_MASK: Final[str] = "target_digit_mask"
CHANNEL_TARGET_VALUE: Final[str] = "target_value"

COUNTWALK_TASK_CHANNELS: Final[list[str]] = [
    CHANNEL_WORLD_ID,
    CHANNEL_CUE_SURFACE_ID,
    CHANNEL_ANCHOR_REGIME_ID,
    CHANNEL_EVAL_BUCKET_ID,
    CHANNEL_TRAJECTORY_VALUE,
    CHANNEL_TRAJECTORY_PREVIOUS_ACTION,
    CHANNEL_TRAJECTORY_ANCHOR_VISIBLE,
    CHANNEL_TRAJECTORY_CUE_TOKENS,
    CHANNEL_TRAJECTORY_CUE_MASK,
    CHANNEL_TRAJECTORY_EPISODE_START,
    CHANNEL_TRAJECTORY_VALID_STEP,
    CHANNEL_TRAJECTORY_LENGTH,
    CHANNEL_QUERY_MASK,
    CHANNEL_TARGET_DIGITS,
    CHANNEL_TARGET_DIGIT_MASK,
    CHANNEL_TARGET_VALUE,
]
"""Ordered list of all Countwalk task corpus channel names."""

# 4 balanced cells: (cue_surface, anchor_regime)
_CELLS: Final[list[tuple[int, int]]] = [
    (CUE_DIGIT, ANCHOR_ONLY),
    (CUE_DIGIT, SPARSE_REANCHOR),
    (CUE_SET, ANCHOR_ONLY),
    (CUE_SET, SPARSE_REANCHOR),
]

_SPLITS: Final[tuple[str, ...]] = ("train", "val", "test")
_SPLIT_SEED_OFFSET: Final[dict[str, int]] = {"train": 0, "val": 100_000, "test": 200_000}

# Bucket-to-value-range and horizon-range mappings for the test split.
# Each entry: (bucket_id, value_lo, value_hi, horizon_lo, horizon_hi)
# value_lo/hi are the inclusive start/end value range for the clamped random walk.
# horizon_lo/hi are inclusive episode length ranges (sampled uniformly per episode).
_TEST_BUCKET_SPECS: Final[list[tuple[int, int, int, int, int]]] = [
    (BUCKET_ID, 0, BUCKET_ID_VALUE_MAX, 1, BUCKET_ID_HORIZON_MAX),
    (BUCKET_RANGE_OOD, BUCKET_RANGE_OOD_VALUE_MIN, BUCKET_RANGE_OOD_VALUE_MAX, 1, BUCKET_ID_HORIZON_MAX),
    (BUCKET_HORIZON_OOD, 0, BUCKET_ID_VALUE_MAX, BUCKET_HORIZON_OOD_MIN, BUCKET_HORIZON_OOD_MAX),
    (BUCKET_JOINT_OOD, BUCKET_RANGE_OOD_VALUE_MIN, BUCKET_RANGE_OOD_VALUE_MAX, BUCKET_HORIZON_OOD_MIN, BUCKET_HORIZON_OOD_MAX),
    (BUCKET_STRETCH_OOD, BUCKET_STRETCH_OOD_VALUE_MIN, BUCKET_STRETCH_OOD_VALUE_MAX, 1, BUCKET_HORIZON_OOD_MAX),
]


# =============================================================================
# Internal helpers
# =============================================================================


def _encode_digit_cue(value: int, cue_token_width: int) -> tuple[np.ndarray, np.ndarray]:
    """Encode *value* as right-aligned digit tokens in a width-*cue_token_width* slot.

    For example, value=7, DIGIT_WIDTH=3, cue_token_width=9:
    tokens[-3:] = [TOKEN_PAD, TOKEN_PAD, TOKEN_DIGIT_7]
    mask[-3:] = [False, False, True]
    """
    tokens = np.zeros(cue_token_width, dtype=np.int32)
    mask = np.zeros(cue_token_width, dtype=bool)
    # Extract digits right-to-left
    remainder = value
    for d in range(DIGIT_WIDTH - 1, -1, -1):
        digit = remainder % 10
        remainder //= 10
        slot = cue_token_width - DIGIT_WIDTH + d
        tokens[slot] = TOKEN_DIGIT_0 + digit
        mask[slot] = True
    # Clear leading-zero slots where the digit is 0 and the value has fewer digits
    n_sig = max(1, len(str(value)))  # at least 1 significant digit
    for d in range(DIGIT_WIDTH - n_sig):
        slot = cue_token_width - DIGIT_WIDTH + d
        mask[slot] = False
    return tokens, mask


def _encode_set_cue(value: int, cue_token_width: int) -> tuple[np.ndarray, np.ndarray]:
    """Encode *value* as ITEM tokens in a width-*cue_token_width* slot.

    For value=3, cue_token_width=9:
    tokens = [ITEM, ITEM, ITEM, PAD, PAD, PAD, PAD, PAD, PAD]
    mask   = [True, True, True, False, ...]
    """
    tokens = np.full(cue_token_width, TOKEN_PAD, dtype=np.int32)
    mask = np.zeros(cue_token_width, dtype=bool)
    n = min(value, cue_token_width)
    tokens[:n] = TOKEN_ITEM
    mask[:n] = True
    return tokens, mask


def _make_target_digits(value: int) -> tuple[np.ndarray, np.ndarray]:
    """Encode *value* as right-aligned digit class labels (0..9) in DIGIT_WIDTH slots.

    Returns ``(target_digits, target_digit_mask)`` both of shape ``(DIGIT_WIDTH,)``.
    Active mask positions hold raw digit values 0..9; inactive positions hold 0.
    """
    digits = np.zeros(DIGIT_WIDTH, dtype=np.int32)
    mask = np.zeros(DIGIT_WIDTH, dtype=bool)
    remainder = value
    for d in range(DIGIT_WIDTH - 1, -1, -1):
        digit = remainder % 10
        remainder //= 10
        digits[d] = digit
        mask[d] = True
    n_sig = max(1, len(str(value)))
    for d in range(DIGIT_WIDTH - n_sig):
        mask[d] = False
    return digits, mask


def _generate_episode(
    n_states: int,
    world_id: int,
    cue_surface_id: int,
    anchor_regime_id: int,
    eval_bucket_id: int,
    actual_steps: int,
    T_max: int,
    value_lo: int,
    value_hi: int,
    rng: np.random.Generator,
    cue_token_width: int,
) -> dict[str, np.ndarray]:
    """Generate one legal Countwalk episode, padded to *T_max* steps.

    All generated PREV/NEXT actions are legal (never PREV at state 0, never
    NEXT at state n_states-1).  STAY is always legal.

    The random walk starts from a state uniformly sampled in [value_lo, value_hi]
    and all states remain clamped to [0, n_states-1] by the legal-action filter.

    Args:
        n_states: Number of states on the number line.
        world_id: Parent world index.
        cue_surface_id: CUE_DIGIT or CUE_SET.
        anchor_regime_id: ANCHOR_ONLY or SPARSE_REANCHOR.
        eval_bucket_id: BUCKET_* constant for this episode.
        actual_steps: Effective trajectory length (trajectory_length value).
        T_max: Padded array length; must be >= actual_steps.
        value_lo: Inclusive lower bound for the starting state.
        value_hi: Inclusive upper bound for the starting state.
        rng: NumPy random generator.
        cue_token_width: Width of cue token array second dimension.

    Returns:
        Dict of channel arrays for one episode; all sequence channels have
        leading dimension T_max.
    """
    assert T_max >= actual_steps >= 1

    # Clamp value range to valid state bounds
    lo = max(0, min(value_lo, n_states - 1))
    hi = max(lo, min(value_hi, n_states - 1))

    # --- Legal random walk (actual_steps long) ---
    trajectory_value = np.zeros(T_max, dtype=np.int32)
    trajectory_previous_action = np.zeros(T_max, dtype=np.int32)

    start = int(rng.integers(lo, hi + 1))
    trajectory_value[0] = start
    trajectory_previous_action[0] = ACTION_STAY

    current = start
    for t in range(1, actual_steps):
        legal: list[int] = [ACTION_STAY]
        if current > lo:
            legal.append(ACTION_PREV)
        if current < hi:
            legal.append(ACTION_NEXT)
        action = int(rng.choice(legal))
        trajectory_previous_action[t] = action
        if action == ACTION_PREV:
            current -= 1
        elif action == ACTION_NEXT:
            current += 1
        trajectory_value[t] = current

    # --- Anchor placement ---
    anchor_visible = np.zeros(T_max, dtype=bool)
    anchor_visible[0] = True
    if anchor_regime_id == SPARSE_REANCHOR and actual_steps >= 3:
        # One random interior non-terminal step: t in [1, actual_steps-2]
        extra_t = int(rng.integers(1, actual_steps - 1))
        anchor_visible[extra_t] = True

    # --- Cue tokens ---
    cue_tokens = np.zeros((T_max, cue_token_width), dtype=np.int32)
    cue_mask = np.zeros((T_max, cue_token_width), dtype=bool)

    for t in range(actual_steps):
        if anchor_visible[t]:
            v = int(trajectory_value[t])
            if cue_surface_id == CUE_DIGIT:
                tok, msk = _encode_digit_cue(v, cue_token_width)
            else:
                tok, msk = _encode_set_cue(v, cue_token_width)
            cue_tokens[t] = tok
            cue_mask[t] = msk

    # --- Episode metadata ---
    episode_start = np.zeros(T_max, dtype=bool)
    episode_start[0] = True

    valid_step = np.arange(T_max, dtype=np.int32) < actual_steps

    query_mask = np.zeros(T_max, dtype=bool)
    query_mask[actual_steps - 1] = True

    target_value = int(trajectory_value[actual_steps - 1])
    target_digits, target_digit_mask = _make_target_digits(target_value)

    return {
        CHANNEL_WORLD_ID: np.int32(world_id),
        CHANNEL_CUE_SURFACE_ID: np.int32(cue_surface_id),
        CHANNEL_ANCHOR_REGIME_ID: np.int32(anchor_regime_id),
        CHANNEL_EVAL_BUCKET_ID: np.int32(eval_bucket_id),
        CHANNEL_TRAJECTORY_VALUE: trajectory_value,
        CHANNEL_TRAJECTORY_PREVIOUS_ACTION: trajectory_previous_action,
        CHANNEL_TRAJECTORY_ANCHOR_VISIBLE: anchor_visible,
        CHANNEL_TRAJECTORY_CUE_TOKENS: cue_tokens,
        CHANNEL_TRAJECTORY_CUE_MASK: cue_mask,
        CHANNEL_TRAJECTORY_EPISODE_START: episode_start,
        CHANNEL_TRAJECTORY_VALID_STEP: valid_step,
        CHANNEL_TRAJECTORY_LENGTH: np.int32(actual_steps),
        CHANNEL_QUERY_MASK: query_mask,
        CHANNEL_TARGET_DIGITS: target_digits,
        CHANNEL_TARGET_DIGIT_MASK: target_digit_mask,
        CHANNEL_TARGET_VALUE: np.int32(target_value),
    }


# =============================================================================
# Sample validator
# =============================================================================


def validate_countwalk_task_sample(data: dict[str, np.ndarray]) -> None:
    """Validate one Countwalk task corpus sample.

    Args:
        data: Dict of channel arrays for one episode.

    Raises:
        ValueError: On any structural or semantic violation.
    """
    missing = set(COUNTWALK_TASK_CHANNELS) - data.keys()
    if missing:
        raise ValueError(f"Countwalk sample missing channels: {sorted(missing)}")

    traj_len = int(data[CHANNEL_TRAJECTORY_LENGTH])
    T = data[CHANNEL_TRAJECTORY_VALUE].shape[0]
    if T < traj_len:
        raise ValueError(f"trajectory arrays have {T} steps but trajectory_length={traj_len}")

    # trajectory_previous_action[0] must be STAY
    if int(data[CHANNEL_TRAJECTORY_PREVIOUS_ACTION][0]) != ACTION_STAY:
        raise ValueError("trajectory_previous_action[0] must be ACTION_STAY (0)")

    # query_mask True only on final valid step
    qm = data[CHANNEL_QUERY_MASK]
    if not qm[traj_len - 1]:
        raise ValueError("query_mask must be True on the final valid step")
    if qm[: traj_len - 1].any():
        raise ValueError("query_mask must be False on all steps except the final valid step")

    # anchor_visible True at step 0
    if not data[CHANNEL_TRAJECTORY_ANCHOR_VISIBLE][0]:
        raise ValueError("trajectory_anchor_visible[0] must be True")

    # eval_bucket_id must be in valid range
    from ehc_sn.tasks.countwalk.contracts import N_BUCKETS

    bucket_id = int(data[CHANNEL_EVAL_BUCKET_ID])
    if not (0 <= bucket_id < N_BUCKETS):
        raise ValueError(f"eval_bucket_id {bucket_id} is out of range [0, {N_BUCKETS})")

    # No valid_action_mask in the corpus
    if "valid_action_mask" in data:
        raise ValueError("Countwalk primary corpus must not contain 'valid_action_mask'")


# =============================================================================
# Root validator
# =============================================================================


def validate_countwalk_task_root(root: Path) -> dict:
    """Validate a Countwalk task corpus root against generic and family rules.

    Args:
        root: Resolved versioned Countwalk task corpus root.

    Returns:
        Parsed manifest dict.

    Raises:
        ValueError: On any structural or semantic violation.
    """
    manifest = validate_version_root(root)

    if manifest.get("task") != TASK_FAMILY:
        raise ValueError(f"Expected task '{TASK_FAMILY}', got '{manifest.get('task')}'.")
    if manifest.get("parent_family") != NUMBERLINE_FAMILY:
        raise ValueError(f"Expected parent_family '{NUMBERLINE_FAMILY}', got '{manifest.get('parent_family')}'.")
    if manifest.get("topology_kind") != NUMBERLINE_TOPOLOGY_KIND:
        raise ValueError(f"Expected topology_kind '{NUMBERLINE_TOPOLOGY_KIND}', got '{manifest.get('topology_kind')}'.")

    return manifest


# =============================================================================
# Builder
# =============================================================================


def build_countwalk_task_corpus(
    version_root: Path,
    *,
    parent_substrate: Path,
    corpus: str = "default",
    n_episodes_per_world: int = 40,
    id_max_steps: int = 8,
    ood_max_steps: int = 16,
    seed: int = 42,
) -> None:
    """Build the Countwalk task corpus at *version_root*.

    Reads the parent NumberLine shared substrate and generates episodes per
    world per split and bucket, balanced across the 4 protocol cells:

    - (CUE_DIGIT, ANCHOR_ONLY)
    - (CUE_DIGIT, SPARSE_REANCHOR)
    - (CUE_SET, ANCHOR_ONLY)
    - (CUE_SET, SPARSE_REANCHOR)

    Split assignment:
    - train, val: ID bucket only (values 0..63, horizon 1..id_max_steps).
    - test: all 5 buckets (ID + 4 OOD), each balanced across 4 cells.

    All generated trajectories are legal-only (no PREV at state 0, no NEXT at
    last state).  No ``valid_action_mask`` is included in the corpus.

    Args:
        version_root: Destination versioned root
            (e.g. ``data/processed/countwalk/default/v1``).  Must not exist.
        parent_substrate: Path to the parent numberline shared substrate root.
        corpus: Corpus label (e.g. ``"default"``).
        n_episodes_per_world: Episodes generated per world per split per bucket.
            Must be a positive multiple of 4 for exact cell balance.
        id_max_steps: Trajectory length for ID and range-OOD buckets (horizon 1..8).
        ood_max_steps: Trajectory length for horizon-OOD, joint-OOD, and stretch-OOD
            buckets.  Must be >= id_max_steps and >= 3.
        seed: Deterministic base seed.

    Raises:
        FileExistsError: When *version_root* already exists (immutable root).
        ValueError: When the parent is not a numberline shared substrate or
            parameters are invalid.
    """
    if n_episodes_per_world < 4 or n_episodes_per_world % 4 != 0:
        raise ValueError(f"n_episodes_per_world must be a positive multiple of 4, got {n_episodes_per_world}")
    if id_max_steps < 1:
        raise ValueError(f"id_max_steps must be >= 1, got {id_max_steps}")
    if ood_max_steps < 3:
        raise ValueError(f"ood_max_steps must be >= 3 (needed for SPARSE_REANCHOR interior anchor), got {ood_max_steps}")

    version = extract_version(version_root)
    parent_manifest = load_substrate_manifest(parent_substrate)

    if parent_manifest.get("family") != NUMBERLINE_FAMILY:
        raise ValueError(
            f"Countwalk task corpus requires a '{NUMBERLINE_FAMILY}' shared substrate, " f"got family={parent_manifest.get('family')!r}."
        )

    n_states: int = parent_manifest["n_states"]
    parent_n_samples: dict = parent_manifest.get("n_samples", {})

    if n_states <= BUCKET_STRETCH_OOD_VALUE_MAX:
        raise ValueError(
            f"Parent substrate n_states={n_states} is too small to support stretch OOD values "
            f"(need n_states > {BUCKET_STRETCH_OOD_VALUE_MAX})."
        )

    # cue_token_width: wide enough for SET cues (n_states-1) and DIGIT cues (DIGIT_WIDTH)
    cue_token_width = max(DIGIT_WIDTH, n_states - 1)

    stage_params = {
        "corpus": corpus,
        "n_episodes_per_world": n_episodes_per_world,
        "id_max_steps": id_max_steps,
        "ood_max_steps": ood_max_steps,
        "seed": seed,
        "cue_token_width": cue_token_width,
        "parent_version": parent_manifest["version"],
    }
    canonical_parent = f"data/processed/{parent_manifest['family']}/v{parent_manifest['version']}"

    # Per-bucket: (max_steps, horizon_lo, horizon_hi)
    # max_steps is the cap for sampling the actual episode length.
    _BUCKET_HORIZON: dict[int, tuple[int, int, int]] = {
        BUCKET_ID: (id_max_steps, 1, id_max_steps),
        BUCKET_RANGE_OOD: (id_max_steps, 1, id_max_steps),
        BUCKET_HORIZON_OOD: (ood_max_steps, BUCKET_HORIZON_OOD_MIN, BUCKET_HORIZON_OOD_MAX),
        BUCKET_JOINT_OOD: (ood_max_steps, BUCKET_HORIZON_OOD_MIN, BUCKET_HORIZON_OOD_MAX),
        BUCKET_STRETCH_OOD: (ood_max_steps, 1, ood_max_steps),
    }

    # T_max per split: all episodes in a split share the same padded length
    _SPLIT_T_MAX: dict[str, int] = {
        "train": id_max_steps,
        "val": id_max_steps,
        "test": ood_max_steps,
    }

    split_counts: dict[str, int] = {}

    with staging_root(version_root) as tmp:
        all_entries = []

        for split in _SPLITS:
            n_worlds_in_split = parent_n_samples.get(split, 0)
            if n_worlds_in_split == 0:
                split_counts[split] = 0
                continue

            rng_base_offset = _SPLIT_SEED_OFFSET[split]

            # Train/val: ID bucket only; test: all 5 buckets
            active_buckets = _TEST_BUCKET_SPECS if split == "test" else [_TEST_BUCKET_SPECS[0]]
            T_max = _SPLIT_T_MAX[split]

            samples: list[dict[str, np.ndarray]] = []

            for world_idx in range(n_worlds_in_split):
                for bucket_id, value_lo, value_hi, _h_lo, _h_hi in active_buckets:
                    _bucket_max_steps, horizon_lo, horizon_hi = _BUCKET_HORIZON[bucket_id]
                    for episode_idx in range(n_episodes_per_world):
                        cell_idx = episode_idx % 4
                        cue_surface_id, anchor_regime_id = _CELLS[cell_idx]
                        episode_seed = seed + rng_base_offset + bucket_id * 1_000_000 + world_idx * n_episodes_per_world * 10 + episode_idx
                        rng = np.random.default_rng(episode_seed)
                        actual_steps = int(rng.integers(horizon_lo, horizon_hi + 1))
                        ep = _generate_episode(
                            n_states=n_states,
                            world_id=world_idx,
                            cue_surface_id=cue_surface_id,
                            anchor_regime_id=anchor_regime_id,
                            eval_bucket_id=bucket_id,
                            actual_steps=actual_steps,
                            T_max=T_max,
                            value_lo=value_lo,
                            value_hi=value_hi,
                            rng=rng,
                            cue_token_width=cue_token_width,
                        )
                        samples.append(ep)

            n = len(samples)
            split_counts[split] = n

            entries = write_split(
                tmp,
                split,
                samples,
                source=TASK_FAMILY,
                channels=COUNTWALK_TASK_CHANNELS,
                topology_kind=NUMBERLINE_TOPOLOGY_KIND,
                n_states=n_states,
                extent=[n_states],
                index_kwargs={},
                sample_validator=validate_countwalk_task_sample,
            )
            all_entries.extend(entries)

        write_index_at_root(all_entries, tmp)

        write_manifest(
            tmp,
            dataset_class="task_corpus",
            family=TASK_FAMILY,
            version=version,
            channels=COUNTWALK_TASK_CHANNELS,
            topology_kind=NUMBERLINE_TOPOLOGY_KIND,
            n_states=n_states,
            extent=[n_states],
            n_samples=split_counts,
            source_id=NUMBERLINE_FAMILY,
            builder="ehc_sn.tasks.countwalk.builder.build_countwalk_task_corpus",
            seed=seed,
            stage_params=stage_params,
            parent_family=NUMBERLINE_FAMILY,
            parent_version=parent_manifest["version"],
            task_schema_version=1,
            task_protocol_version=1,
            task=TASK_FAMILY,
            corpus=corpus,
            parent_substrate=canonical_parent,
        )

    total = sum(split_counts.values())
    print(f"Countwalk task corpus written to {version_root}  ({total} episodes).")


# =============================================================================
__all__ = [
    "CHANNEL_WORLD_ID",
    "CHANNEL_CUE_SURFACE_ID",
    "CHANNEL_ANCHOR_REGIME_ID",
    "CHANNEL_EVAL_BUCKET_ID",
    "CHANNEL_TRAJECTORY_VALUE",
    "CHANNEL_TRAJECTORY_PREVIOUS_ACTION",
    "CHANNEL_TRAJECTORY_ANCHOR_VISIBLE",
    "CHANNEL_TRAJECTORY_CUE_TOKENS",
    "CHANNEL_TRAJECTORY_CUE_MASK",
    "CHANNEL_TRAJECTORY_EPISODE_START",
    "CHANNEL_TRAJECTORY_VALID_STEP",
    "CHANNEL_TRAJECTORY_LENGTH",
    "CHANNEL_QUERY_MASK",
    "CHANNEL_TARGET_DIGITS",
    "CHANNEL_TARGET_DIGIT_MASK",
    "CHANNEL_TARGET_VALUE",
    "COUNTWALK_TASK_CHANNELS",
    "validate_countwalk_task_sample",
    "validate_countwalk_task_root",
    "build_countwalk_task_corpus",
]
