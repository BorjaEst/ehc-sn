"""Shared orchestration for the B0 MazeHard bridge benchmark."""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel, Field

from ehc_sn.benchmarks._capabilities import BatchPredicts
from ehc_sn.benchmarks._infra import ensure_directory, seed_benchmark_process, write_artifact_json
from ehc_sn.benchmarks.b0.manifest import B0Manifest
from ehc_sn.benchmarks.b0.metrics import build_sample_result, summarize_results
from ehc_sn.data.benchmarks.mazehard_b0 import b0_hard_subset_path, read_mazehard_subset_manifest
from ehc_sn.data.datasets import MazeDataset
from ehc_sn.data.index import filter_index, read_index


# =================================================================================================
class B0BenchmarkConfig(BaseModel, extra="forbid"):
    """Benchmark-owned configuration for B0 MazeHard evaluation."""

    benchmark_id: str = Field(
        default="b0",
        description="Canonical benchmark identifier.",
    )
    dataset_root: Path = Field(
        ...,
        description="Processed MazeHard dataset root.",
    )
    split: str = Field(
        default="test",
        description="Dataset split used for evaluation.",
    )
    hard_subset_manifest_path: Path | None = Field(
        default=None,
        description="Optional override for the persisted B0 hard-subset manifest.",
    )
    compute_budget: int = Field(
        default=16,
        ge=1,
        description="Maximum ACT deliberation steps per sample.",
    )

    seed: int | None = None
    output_dir: Path = Path("outputs/benchmarks/b0")
    write_sample_records: bool = True

    @property
    def resolved_hard_subset_manifest_path(self) -> Path:
        """Return the hard-subset manifest path for this benchmark run."""
        if self.hard_subset_manifest_path is not None:
            return self.hard_subset_manifest_path
        return b0_hard_subset_path(self.dataset_root)

    @classmethod
    def from_config(cls, config_path: Path) -> "B0BenchmarkConfig":
        """Load the B0 benchmark configuration from the given path."""
        loaded = tomllib.load(config_path.open("rb"))
        return cls.model_validate(loaded)


# =================================================================================================
class B0Benchmark:
    """Shared orchestration for the B0 MazeHard bridge benchmark."""

    def __init__(self, config: B0BenchmarkConfig) -> None:
        self.config = config

    def evaluate(self, predictor: BatchPredicts) -> B0Manifest:
        """Run B0 over the configured split and write benchmark artifacts."""
        if self.config.seed is not None:
            seed_benchmark_process(self.config.seed)

        output_root = ensure_directory(self.config.output_dir)
        entries = filter_index(read_index(self.config.dataset_root / "index.jsonl"), split=self.config.split)
        if not entries:
            raise ValueError(f"No MazeHard index entries found for split {self.config.split!r} in {self.config.dataset_root / 'index.jsonl'}.")  # fmt: skip

        hard_subset_manifest = read_mazehard_subset_manifest(self.config.resolved_hard_subset_manifest_path)
        hard_ids = set(hard_subset_manifest.sample_ids)
        known_ids = {entry.id for entry in entries}
        missing_ids = sorted(hard_ids - known_ids)
        if missing_ids:
            raise ValueError(f"B0 hard-subset manifest contains ids not present in split {self.config.split!r}: {missing_ids[:10]}")

        dataset = MazeDataset(entries, self.config.dataset_root / self.config.split)
        sample_results = []
        hard_subset_results = []
        for index, entry in enumerate(entries):
            prediction = predictor.predict_batch(dataset[index])
            result = build_sample_result(
                sample_id=entry.id,
                prediction=prediction,
                is_hard_subset=entry.id in hard_ids,
            )
            sample_results.append(result)
            if result.is_hard_subset:
                hard_subset_results.append(result)

        if not hard_subset_results:
            raise ValueError("B0 hard-subset evaluation selected zero samples; check the manifest and split.")

        full_metrics = summarize_results(sample_results)
        hard_subset_metrics = summarize_results(hard_subset_results)
        artifact_paths = [
            write_artifact_json(output_root / "full_metrics.json", full_metrics),
            write_artifact_json(output_root / "hard_subset_metrics.json", hard_subset_metrics),
        ]
        if self.config.write_sample_records:
            artifact_paths.append(
                write_artifact_json(
                    output_root / "sample_results.json",
                    [result.to_json_dict() for result in sample_results],
                )
            )

        summary_payload = {
            "benchmark_id": self.config.benchmark_id,
            "dataset_root": str(self.config.dataset_root),
            "split": self.config.split,
            "compute_budget": self.config.compute_budget,
            "seed": self.config.seed,
            "hard_subset_manifest_path": str(self.config.resolved_hard_subset_manifest_path),
            "full_metrics": full_metrics,
            "hard_subset_metrics": hard_subset_metrics,
            "artifact_paths": [str(path) for path in artifact_paths],
        }
        artifact_paths.append(write_artifact_json(output_root / "manifest.json", summary_payload))

        return B0Manifest(
            output_root=output_root,
            artifact_paths=tuple(artifact_paths),
            full_metrics=full_metrics,
            hard_subset_metrics=hard_subset_metrics,
        )

    __call__ = evaluate


__all__ = ["B0Benchmark", "B0BenchmarkConfig"]
