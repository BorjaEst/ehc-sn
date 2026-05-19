"""MazeHard dataset-level diagnostics and audit helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ehc_sn.data.index import read_index
from ehc_sn.data.manifest import read_manifest
from ehc_sn.tasks.mazehard.runtime import (
    EMPTY_ID,
    GOAL_ID,
    PATH_ID,
    START_ID,
    WALL_ID,
)


def _load_split_channels(root: Path, split: str) -> dict[str, np.ndarray]:
    split_root = root / split
    channels: dict[str, np.ndarray] = {}
    for name in ("topology", "start", "goals", "solution"):
        path = split_root / f"{name}.npy"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing MazeHard channel {name!r} for split {split!r} at {path}."
            )
        channels[name] = np.load(path, mmap_mode="r")
    return channels


def _build_input_ids(channels: dict[str, np.ndarray]) -> np.ndarray:
    topology = channels["topology"]
    grid = np.where(topology, EMPTY_ID, WALL_ID).astype(np.int32, copy=False)
    grid = np.where(channels["start"], START_ID, grid)
    grid = np.where(channels["goals"], GOAL_ID, grid)
    return grid


def _token_counts(values: np.ndarray, *, vocab_size: int) -> np.ndarray:
    flat = values.reshape(-1).astype(np.int64, copy=False)
    return np.bincount(flat, minlength=vocab_size)


def _format_token_counts(counts: np.ndarray) -> dict[str, int]:
    return {str(idx): int(counts[idx]) for idx in range(len(counts))}


def _provenance_coverage(entries: list) -> dict[str, Any]:
    total = len(entries)
    source_present = sum(1 for e in entries if e.source_record_id)
    metadata_present = sum(1 for e in entries if e.task_metadata)

    def _coverage(present: int) -> dict[str, Any]:
        available = present > 0
        return {
            "available": available,
            "n_present": present,
            "n_total": total,
            "coverage": (present / total) if available and total > 0 else None,
        }

    return {
        "source_record_id": _coverage(source_present),
        "task_metadata": _coverage(metadata_present),
    }


def audit_mazehard_task_corpus(
    root: Path,
    *,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Audit a MazeHard task corpus and optionally write a JSON report.

    Args:
        root: Versioned MazeHard task-corpus root.
        output_path: Optional destination for a JSON report.

    Returns:
        Parsed report dictionary.
    """
    manifest = read_manifest(root)
    index_entries = read_index(root / "index.jsonl")
    report: dict[str, Any] = {
        "dataset_path": str(root),
        "task": manifest.get("task"),
        "corpus": manifest.get("corpus"),
        "version": manifest.get("version"),
        "splits": {},
    }

    vocab_size = PATH_ID + 1
    for split in sorted({e.split for e in index_entries}):
        split_entries = [e for e in index_entries if e.split == split]
        channels = _load_split_channels(root, split)
        input_ids = _build_input_ids(channels)
        solution_mask = channels["solution"] > 0
        total_cells = int(solution_mask.size)
        path_count = int(solution_mask.sum())

        input_counts = _token_counts(input_ids, vocab_size=vocab_size)
        solution_input_counts = _token_counts(
            input_ids[solution_mask], vocab_size=vocab_size
        )
        label_counts = input_counts - solution_input_counts
        label_counts[PATH_ID] = path_count

        label_copy_ratio = (
            (total_cells - path_count) / total_cells if total_cells else 0.0
        )
        path_sparsity = (path_count / total_cells) if total_cells else 0.0

        report["splits"][split] = {
            "n_samples": int(input_ids.shape[0]),
            "token_class_distribution": {
                "input_ids": _format_token_counts(input_counts),
                "labels": _format_token_counts(label_counts),
            },
            "label_copy_ratio": float(label_copy_ratio),
            "path_sparsity": float(path_sparsity),
            "provenance": _provenance_coverage(split_entries),
        }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True))

    return report


__all__ = ["audit_mazehard_task_corpus"]
