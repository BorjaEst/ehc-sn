"""Benchmark-owned M0 replay manifest schemas, generation, and I/O helpers."""

from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ehc_sn.data._canonical import canonical_cell_from_mask, first_true_cell, largest_component_mask, shortest_path_distances
from ehc_sn.data.benchmarks.dungeon_b23 import MIN_GOAL_SEPARATION as M0_MIN_TARGET_SEPARATION
from ehc_sn.data.benchmarks.dungeon_b23 import MIN_GOAL_START_DISTANCE as M0_MIN_START_TO_TARGET_DISTANCE
from ehc_sn.data.benchmarks.dungeon_b23 import MIN_PROBE_GOAL_DISTANCE as M0_MIN_PROBE_TO_TARGET_DISTANCE
from ehc_sn.data.index import MazeIndexEntry, read_index
from ehc_sn.data.schema import CHANNEL_MASK_VALID, CHANNEL_OBSERVATIONS, CHANNEL_START, CHANNEL_TOPOLOGY

CueFamily = Literal["observation", "landmark"]

BENCHMARKS_DIRNAME = "benchmarks"
M0_BENCHMARK_DIRNAME = "m0"
BENCHMARK_MANIFEST_FILENAME = "manifest.json"
M0_LEGACY_TRAJECTORIES_FILENAME = "m0-trajectories.jsonl"
M0_TARGET_SLOTS = (5, 6)
M0_PROBE_START_SLOTS = (2, 3)

_NEIGHBORS4: tuple[tuple[int, int], ...] = ((-1, 0), (0, -1), (0, 1), (1, 0))
_ACTION_BY_DELTA: dict[tuple[int, int], int] = {
    (-1, 0): 1,
    (0, 1): 2,
    (1, 0): 3,
    (0, -1): 4,
}


def _stable_id_key(value: str) -> tuple[int, str]:
    """Return a deterministic sort key for string sample or layout ids."""
    return (0, f"{int(value):020d}") if value.isdigit() else (1, value)


class GridPoint(BaseModel):
    """One grid location encoded as row and column coordinates."""

    model_config = ConfigDict(extra="forbid")

    row: int = Field(..., ge=0)
    col: int = Field(..., ge=0)


class CueSpec(BaseModel):
    """Benchmark-owned target cue expressed in TEM-facing cue terms."""

    model_config = ConfigDict(extra="forbid")

    family: CueFamily
    cue_id: int = Field(..., ge=0)


class M0ReplayStep(GridPoint):
    """One benchmark-owned replay step for an M0 episode."""

    model_config = ConfigDict(extra="forbid")

    previous_action: int = Field(..., ge=0, le=4)
    step_index: int = Field(..., ge=0)
    is_target_step: bool = Field(default=False)


class M0EpisodeSpec(BaseModel):
    """One persisted exposure or probe episode for M0."""

    model_config = ConfigDict(extra="forbid")

    phase: Literal["exposure", "probe"]
    start_slot: int = Field(..., ge=1)
    steps: list[M0ReplayStep] = Field(default_factory=list, min_length=1)

    @model_validator(mode="after")
    def validate_episode(self) -> "M0EpisodeSpec":
        if [step.step_index for step in self.steps] != list(range(len(self.steps))):
            raise ValueError("M0 episode steps must be indexed densely from 0 in order.")
        if self.steps[0].previous_action != 0:
            raise ValueError("The first M0 replay step must use the stay/no-op previous action.")
        if any(step.is_target_step for step in self.steps[:-1]):
            raise ValueError("Only the final M0 replay step may be marked as the target step.")
        if not self.steps[-1].is_target_step:
            raise ValueError("The final M0 replay step must be marked as the target step.")
        return self


class M0TargetSpec(BaseModel):
    """One held-out M0 target together with its cue and replay episodes."""

    model_config = ConfigDict(extra="forbid")

    target_slot: int = Field(..., ge=1)
    target_location: GridPoint
    cue: CueSpec
    exposure: M0EpisodeSpec
    probes: list[M0EpisodeSpec] = Field(default_factory=list, min_length=2, max_length=2)

    @model_validator(mode="after")
    def validate_target(self) -> "M0TargetSpec":
        if self.exposure.phase != "exposure" or self.exposure.start_slot != 1:
            raise ValueError("M0 target exposures must use phase='exposure' and start_slot=1.")
        if [probe.phase for probe in self.probes] != ["probe", "probe"]:
            raise ValueError("M0 target probes must all use phase='probe'.")
        if [probe.start_slot for probe in self.probes] != list(M0_PROBE_START_SLOTS):
            raise ValueError("M0 target probes must be ordered by start slots 2 then 3.")
        for episode in (self.exposure, *self.probes):
            final_step = episode.steps[-1]
            if (final_step.row, final_step.col) != (self.target_location.row, self.target_location.col):
                raise ValueError("All M0 episodes must terminate at the persisted target location.")
        return self


class M0LayoutSpec(BaseModel):
    """One per-layout persisted M0 replay contract."""

    model_config = ConfigDict(extra="forbid")

    layout_id: str
    split: str
    dataset_root: str | None = Field(
        default=None,
        description="Legacy per-entry dataset root retained only for backward compatibility.",
    )
    targets: list[M0TargetSpec] = Field(default_factory=list, min_length=2, max_length=2)

    @model_validator(mode="after")
    def validate_layout(self) -> "M0LayoutSpec":
        if [target.target_slot for target in self.targets] != list(M0_TARGET_SLOTS):
            raise ValueError("M0 layout specs must contain held-out target slots 5 then 6.")
        return self


class M0ReplayManifest(BaseModel):
    """In-memory representation of the canonical M0 replay manifest."""

    model_config = ConfigDict(extra="forbid")

    benchmark_id: str = Field(default="m0", description="Canonical benchmark identifier.")
    dataset_root: str | None = Field(default=None, description="Processed dungeon dataset root.")
    entries: list[M0LayoutSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_entries(self) -> "M0ReplayManifest":
        layout_ids = [entry.layout_id for entry in self.entries]
        if len(layout_ids) != len(set(layout_ids)):
            raise ValueError("M0 replay manifest layout ids must be unique.")
        if layout_ids != sorted(layout_ids, key=_stable_id_key):
            raise ValueError("M0 replay manifest entries must be ordered deterministically by layout id.")
        return self


def build_m0_replay_manifest(dataset_root: Path) -> M0ReplayManifest:
    """Build the canonical M0 replay manifest for one processed dungeon dataset root."""
    index_entries = read_index(dataset_root / "index.jsonl")
    split_arrays: dict[str, dict[str, np.ndarray | None]] = {}
    split_offsets: dict[str, int] = defaultdict(int)
    layout_specs: list[M0LayoutSpec] = []

    for entry in index_entries:
        arrays = split_arrays.setdefault(entry.split, _load_split_arrays(dataset_root, entry.split))
        split_index = split_offsets[entry.split]
        split_offsets[entry.split] += 1
        layout_specs.append(
            _build_layout_spec(
                entry=entry,
                observations=arrays[CHANNEL_OBSERVATIONS],
                start_masks=arrays[CHANNEL_START],
                topology=arrays[CHANNEL_TOPOLOGY],
                valid_masks=arrays[CHANNEL_MASK_VALID],
                split_index=split_index,
            )
        )

    return M0ReplayManifest(benchmark_id="m0", dataset_root=str(dataset_root), entries=layout_specs)


def m0_trajectory_manifest_path(dataset_root: Path) -> Path:
    """Return the canonical M0 replay-manifest path for one dataset root."""
    return dataset_root / BENCHMARKS_DIRNAME / M0_BENCHMARK_DIRNAME / BENCHMARK_MANIFEST_FILENAME


def iter_m0_episodes(entry: M0LayoutSpec) -> tuple[M0EpisodeSpec, ...]:
    """Return all persisted M0 episodes in canonical target/phase/start order."""
    episodes: list[M0EpisodeSpec] = []
    for target in entry.targets:
        episodes.append(target.exposure)
        episodes.extend(target.probes)
    return tuple(episodes)


def resolve_m0_layout_cells(
    *,
    valid_mask: np.ndarray,
    observation_grid: np.ndarray,
    start_mask: np.ndarray | None = None,
    entry_id: str | None = None,
    split: str | None = None,
) -> tuple[tuple[int, int], tuple[tuple[int, int], tuple[int, int]], tuple[tuple[int, int], tuple[int, int]]]:
    """Resolve the canonical M0 start cell, held-out targets, and probe starts for one layout."""
    resolved_valid_mask = np.asarray(valid_mask, dtype=bool)
    resolved_observation_grid = np.asarray(observation_grid, dtype=np.int32)
    resolved_start_mask = None if start_mask is None else np.asarray(start_mask, dtype=bool)

    start_cell = None
    if resolved_start_mask is not None:
        start_cell = canonical_cell_from_mask(resolved_start_mask, resolved_valid_mask)
    if start_cell is None:
        start_cell = first_true_cell(resolved_valid_mask)
    if start_cell is None:
        label = "Layout" if entry_id is None else f"Layout {entry_id!r}"
        if split is not None:
            label = f"{label} split {split!r}"
        raise ValueError(f"{label} contains no reachable cells for M0 replay generation.")

    target_cells, probe_starts = _select_layout_cells(resolved_valid_mask, resolved_observation_grid, start_cell)
    return start_cell, target_cells, probe_starts


def validate_m0_layout_feasibility(
    *,
    valid_mask: np.ndarray,
    observation_grid: np.ndarray,
    start_mask: np.ndarray | None = None,
    entry_id: str | None = None,
    split: str | None = None,
) -> None:
    """Raise ``ValueError`` unless one layout admits the canonical M0 replay contract."""
    resolve_m0_layout_cells(
        valid_mask=valid_mask,
        observation_grid=observation_grid,
        start_mask=start_mask,
        entry_id=entry_id,
        split=split,
    )


def _load_split_arrays(dataset_root: Path, split: str) -> dict[str, np.ndarray | None]:
    """Load the split-local arrays needed to build M0 replay contracts."""
    split_dir = dataset_root / split
    valid_path = split_dir / f"{CHANNEL_MASK_VALID}.npy"
    return {
        CHANNEL_OBSERVATIONS: np.load(split_dir / f"{CHANNEL_OBSERVATIONS}.npy", mmap_mode="r"),
        CHANNEL_START: np.load(split_dir / f"{CHANNEL_START}.npy", mmap_mode="r"),
        CHANNEL_TOPOLOGY: np.load(split_dir / f"{CHANNEL_TOPOLOGY}.npy", mmap_mode="r"),
        CHANNEL_MASK_VALID: np.load(valid_path, mmap_mode="r") if valid_path.exists() else None,
    }


def _build_layout_spec(
    *,
    entry: MazeIndexEntry,
    observations: np.ndarray,
    start_masks: np.ndarray,
    topology: np.ndarray,
    valid_masks: np.ndarray | None,
    split_index: int,
) -> M0LayoutSpec:
    """Build one persisted M0 replay contract from one dataset sample."""
    observation_grid = np.asarray(observations[split_index], dtype=np.int32)
    topology_grid = np.asarray(topology[split_index], dtype=bool)
    valid_mask = np.asarray(valid_masks[split_index], dtype=bool) if valid_masks is not None else largest_component_mask(topology_grid)
    start_cell, target_cells, probe_starts = resolve_m0_layout_cells(
        valid_mask=valid_mask,
        observation_grid=observation_grid,
        start_mask=np.asarray(start_masks[split_index], dtype=bool),
        entry_id=entry.id,
        split=entry.split,
    )
    return M0LayoutSpec(
        layout_id=entry.id,
        split=entry.split,
        targets=[
            _build_target_spec(
                target_slot=target_slot,
                target_cell=target_cell,
                observation_grid=observation_grid,
                valid_mask=valid_mask,
                start_cell=start_cell,
                probe_starts=probe_starts,
            )
            for target_slot, target_cell in zip(M0_TARGET_SLOTS, target_cells, strict=True)
        ],
    )


def _select_layout_cells(
    valid_mask: np.ndarray,
    observation_grid: np.ndarray,
    start_cell: tuple[int, int],
) -> tuple[tuple[tuple[int, int], tuple[int, int]], tuple[tuple[int, int], tuple[int, int]]]:
    """Select canonical held-out targets and probe starts for one layout."""
    cells = [tuple(int(coord) for coord in cell) for cell in np.argwhere(valid_mask)]
    distance_cache: dict[tuple[int, int], np.ndarray] = {start_cell: shortest_path_distances(valid_mask, start_cell)}
    path_cache: dict[tuple[tuple[int, int], tuple[int, int]], tuple[tuple[int, int], ...]] = {}

    def distances_from(cell: tuple[int, int]) -> np.ndarray:
        distances = distance_cache.get(cell)
        if distances is None:
            distances = shortest_path_distances(valid_mask, cell)
            distance_cache[cell] = distances
        return distances

    def shortest_path(start: tuple[int, int], goal: tuple[int, int]) -> tuple[tuple[int, int], ...]:
        key = (start, goal)
        cached = path_cache.get(key)
        if cached is not None:
            return cached

        frontier: deque[tuple[int, int]] = deque([start])
        parents: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        while frontier:
            cell = frontier.popleft()
            if cell == goal:
                break
            for d_row, d_col in _NEIGHBORS4:
                nxt = (cell[0] + d_row, cell[1] + d_col)
                if nxt in parents:
                    continue
                if nxt[0] < 0 or nxt[0] >= valid_mask.shape[0] or nxt[1] < 0 or nxt[1] >= valid_mask.shape[1]:
                    continue
                if not valid_mask[nxt]:
                    continue
                parents[nxt] = cell
                frontier.append(nxt)

        if goal not in parents:
            raise ValueError(f"No shortest path exists from {start} to {goal} inside the valid mask.")

        cursor: tuple[int, int] | None = goal
        path: list[tuple[int, int]] = []
        while cursor is not None:
            path.append(cursor)
            cursor = parents[cursor]
        resolved = tuple(reversed(path))
        path_cache[key] = resolved
        return resolved

    start_distances = distances_from(start_cell)
    target_candidates = [
        cell for cell in cells if int(start_distances[cell]) >= M0_MIN_START_TO_TARGET_DISTANCE and int(observation_grid[cell]) >= 0
    ]
    target_candidates.sort(key=lambda cell: (-int(start_distances[cell]), cell))

    for first_target in target_candidates:
        first_distances = distances_from(first_target)
        paired_targets = [
            cell
            for cell in target_candidates
            if cell != first_target
            and int(first_distances[cell]) >= M0_MIN_TARGET_SEPARATION
            and int(observation_grid[cell]) != int(observation_grid[first_target])
        ]
        paired_targets.sort(key=lambda cell: (-min(int(start_distances[cell]), int(first_distances[cell])), cell))

        for second_target in paired_targets:
            probe_candidates = [
                cell
                for cell in cells
                if cell not in (start_cell, first_target, second_target)
                and min(
                    int(distances_from(first_target)[cell]),
                    int(distances_from(second_target)[cell]),
                )
                >= M0_MIN_PROBE_TO_TARGET_DISTANCE
            ]
            probe_candidates.sort(
                key=lambda cell: (
                    -min(
                        int(distances_from(first_target)[cell]),
                        int(distances_from(second_target)[cell]),
                    ),
                    cell,
                )
            )
            if len(probe_candidates) < 2:
                continue

            first_probe, second_probe = probe_candidates[:2]
            target_paths = {
                first_target: (
                    shortest_path(start_cell, first_target),
                    shortest_path(first_probe, first_target),
                    shortest_path(second_probe, first_target),
                ),
                second_target: (
                    shortest_path(start_cell, second_target),
                    shortest_path(first_probe, second_target),
                    shortest_path(second_probe, second_target),
                ),
            }
            if _paths_preserve_target_cues(target_paths, observation_grid):
                return (first_target, second_target), (first_probe, second_probe)

    raise ValueError("Could not derive a deterministic M0 replay contract for the provided layout.")


def _paths_preserve_target_cues(
    target_paths: dict[tuple[int, int], tuple[tuple[tuple[int, int], ...], ...]],
    observation_grid: np.ndarray,
) -> bool:
    """Return whether target cue ids stay off all non-terminal replay steps."""
    for target_cell, episodes in target_paths.items():
        cue_id = int(observation_grid[target_cell])
        for episode in episodes:
            if any(int(observation_grid[cell]) == cue_id for cell in episode[:-1]):
                return False
    return True


def _build_target_spec(
    *,
    target_slot: int,
    target_cell: tuple[int, int],
    observation_grid: np.ndarray,
    valid_mask: np.ndarray,
    start_cell: tuple[int, int],
    probe_starts: tuple[tuple[int, int], tuple[int, int]],
) -> M0TargetSpec:
    """Build one target specification with canonical exposure and probe episodes."""
    return M0TargetSpec(
        target_slot=target_slot,
        target_location=GridPoint(row=target_cell[0], col=target_cell[1]),
        cue=CueSpec(family="observation", cue_id=int(observation_grid[target_cell])),
        exposure=_build_episode(
            phase="exposure",
            start_slot=1,
            cells=_shortest_path_cells(valid_mask, start_cell, target_cell),
        ),
        probes=[
            _build_episode(
                phase="probe",
                start_slot=start_slot,
                cells=_shortest_path_cells(valid_mask, probe_start, target_cell),
            )
            for start_slot, probe_start in zip(M0_PROBE_START_SLOTS, probe_starts, strict=True)
        ],
    )


def _shortest_path_cells(
    valid_mask: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    """Return one deterministic shortest path between two valid cells."""
    frontier: deque[tuple[int, int]] = deque([start])
    parents: dict[tuple[int, int], tuple[int, int] | None] = {start: None}

    while frontier:
        cell = frontier.popleft()
        if cell == goal:
            break
        for d_row, d_col in _NEIGHBORS4:
            nxt = (cell[0] + d_row, cell[1] + d_col)
            if nxt in parents:
                continue
            if nxt[0] < 0 or nxt[0] >= valid_mask.shape[0] or nxt[1] < 0 or nxt[1] >= valid_mask.shape[1]:
                continue
            if not valid_mask[nxt]:
                continue
            parents[nxt] = cell
            frontier.append(nxt)

    if goal not in parents:
        raise ValueError(f"No shortest path exists from {start} to {goal} inside the valid mask.")

    cursor: tuple[int, int] | None = goal
    path: list[tuple[int, int]] = []
    while cursor is not None:
        path.append(cursor)
        cursor = parents[cursor]
    path.reverse()
    return path


def _build_episode(*, phase: Literal["exposure", "probe"], start_slot: int, cells: list[tuple[int, int]]) -> M0EpisodeSpec:
    """Build one M0 episode from a deterministic cell path."""
    steps: list[M0ReplayStep] = []
    previous_cell: tuple[int, int] | None = None

    for step_index, cell in enumerate(cells):
        previous_action = 0
        if previous_cell is not None:
            delta = (cell[0] - previous_cell[0], cell[1] - previous_cell[1])
            previous_action = _ACTION_BY_DELTA[delta]
        steps.append(
            M0ReplayStep(
                row=cell[0],
                col=cell[1],
                previous_action=previous_action,
                step_index=step_index,
                is_target_step=step_index == len(cells) - 1,
            )
        )
        previous_cell = cell

    return M0EpisodeSpec(phase=phase, start_slot=start_slot, steps=steps)


def read_m0_replay_manifest(path: Path) -> M0ReplayManifest:
    """Read one canonical M0 manifest, falling back to the legacy JSONL format."""
    contents = path.read_text(encoding="utf-8")
    stripped = contents.strip()
    if not stripped:
        return M0ReplayManifest(entries=[])

    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return _read_legacy_m0_replay_manifest(contents)

    if isinstance(payload, dict) and "entries" in payload:
        return M0ReplayManifest.model_validate(payload)
    return _read_legacy_m0_replay_manifest(contents)


def write_m0_replay_manifest(manifest: M0ReplayManifest | list[M0LayoutSpec], path: Path) -> None:
    """Write one canonical JSON M0 replay manifest."""
    resolved = _normalize_m0_replay_manifest(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(resolved.model_dump_json(indent=2, exclude_none=True), encoding="utf-8")


def _read_legacy_m0_replay_manifest(contents: str) -> M0ReplayManifest:
    """Read one legacy JSONL M0 replay manifest and lift its metadata to the top level."""
    entries = [M0LayoutSpec.model_validate_json(line) for line in contents.splitlines() if line.strip()]
    dataset_root = _resolve_manifest_dataset_root(entries)
    return M0ReplayManifest(benchmark_id="m0", dataset_root=dataset_root, entries=entries)


def _normalize_m0_replay_manifest(manifest: M0ReplayManifest | list[M0LayoutSpec]) -> M0ReplayManifest:
    """Return the canonical manifest shape used for persisted JSON serialization."""
    resolved = manifest if isinstance(manifest, M0ReplayManifest) else M0ReplayManifest(entries=manifest)
    dataset_root = resolved.dataset_root if resolved.dataset_root is not None else _resolve_manifest_dataset_root(resolved.entries)
    entries = [entry if entry.dataset_root is None else entry.model_copy(update={"dataset_root": None}) for entry in resolved.entries]
    return resolved.model_copy(update={"dataset_root": dataset_root, "entries": entries})


def _resolve_manifest_dataset_root(entries: list[M0LayoutSpec]) -> str | None:
    """Return one manifest-global dataset root inferred from per-entry legacy metadata."""
    dataset_roots = sorted({entry.dataset_root for entry in entries if entry.dataset_root is not None})
    if not dataset_roots:
        return None
    if len(dataset_roots) > 1:
        raise ValueError(f"Legacy M0 replay manifest spans multiple dataset roots: {dataset_roots}.")
    return dataset_roots[0]


__all__ = [
    "BENCHMARKS_DIRNAME",
    "BENCHMARK_MANIFEST_FILENAME",
    "CueSpec",
    "GridPoint",
    "M0_BENCHMARK_DIRNAME",
    "M0_LEGACY_TRAJECTORIES_FILENAME",
    "M0_MIN_PROBE_TO_TARGET_DISTANCE",
    "M0_MIN_START_TO_TARGET_DISTANCE",
    "M0_MIN_TARGET_SEPARATION",
    "M0EpisodeSpec",
    "M0LayoutSpec",
    "M0ReplayManifest",
    "M0ReplayStep",
    "M0TargetSpec",
    "M0_PROBE_START_SLOTS",
    "M0_TARGET_SLOTS",
    "build_m0_replay_manifest",
    "iter_m0_episodes",
    "m0_trajectory_manifest_path",
    "read_m0_replay_manifest",
    "resolve_m0_layout_cells",
    "validate_m0_layout_feasibility",
    "write_m0_replay_manifest",
]
