"""Shared orchestration for the M0 Episodic Memory Bridge."""

from __future__ import annotations

import tomllib
from pathlib import Path

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field

from ehc_sn.benchmarks._capabilities import EpisodicMemoryAgent
from ehc_sn.benchmarks._capabilities.episodic_memory import EpisodicMemoryQuery, EpisodicMemoryStep
from ehc_sn.benchmarks._infra import ensure_directory, seed_benchmark_process, write_artifact_json
from ehc_sn.benchmarks.m0.manifest import M0Manifest
from ehc_sn.benchmarks.m0.metrics import M0TargetResult, summarize_results
from ehc_sn.data.benchmarks.dungeon_m0 import M0LayoutSpec, m0_trajectory_manifest_path, read_m0_replay_manifest
from ehc_sn.data.datasets import MazeDataset
from ehc_sn.data.index import filter_index, read_index
from ehc_sn.data.schema import CHANNEL_LANDMARKS, CHANNEL_OBSERVATIONS


class M0BenchmarkConfig(BaseModel, extra="forbid"):
    """Benchmark-owned configuration for the M0 Episodic Memory Bridge."""

    benchmark_id: str = Field(default="m0", description="Canonical benchmark identifier.")
    dataset_root: Path = Field(..., description="Processed dungeon dataset root.")
    split: str = Field(default="test", description="Dataset split used for evaluation.")
    trajectory_manifest_path: Path | None = Field(
        default=None,
        description="Optional override for the persisted M0 replay manifest.",
    )

    seed: int | None = None
    output_dir: Path = Path("outputs/benchmarks/m0")
    write_sample_records: bool = True

    @property
    def resolved_trajectory_manifest_path(self) -> Path:
        """Return the M0 replay-manifest path for this benchmark run."""
        if self.trajectory_manifest_path is not None:
            return self.trajectory_manifest_path
        return m0_trajectory_manifest_path(self.dataset_root)

    @classmethod
    def from_config(cls, config_path: Path) -> "M0BenchmarkConfig":
        """Load the M0 benchmark configuration from the given path."""
        loaded = tomllib.load(config_path.open("rb"))
        return cls.model_validate(loaded)


class M0Benchmark:
    """Shared orchestration for the M0 Episodic Memory Bridge."""

    def __init__(self, config: M0BenchmarkConfig) -> None:
        self.config = config

    def evaluate(self, agent: EpisodicMemoryAgent) -> M0Manifest:
        """Run M0 over the configured split and write benchmark artifacts."""
        if self.config.seed is not None:
            seed_benchmark_process(self.config.seed)

        output_root = ensure_directory(self.config.output_dir)
        entries = filter_index(read_index(self.config.dataset_root / "index.jsonl"), split=self.config.split)
        if not entries:
            raise ValueError(
                f"No dungeon index entries found for split {self.config.split!r} in {self.config.dataset_root / 'index.jsonl'}."
            )

        replay_manifest = read_m0_replay_manifest(self.config.resolved_trajectory_manifest_path)
        replay_entries = {entry.layout_id: entry for entry in replay_manifest.entries if entry.split == self.config.split}
        missing_ids = [entry.id for entry in entries if entry.id not in replay_entries]
        if missing_ids:
            raise ValueError(f"M0 replay manifest contains no entries for split {self.config.split!r} layout ids: {missing_ids[:10]}")

        dataset = MazeDataset(entries, self.config.dataset_root / self.config.split)
        results: list[M0TargetResult] = []
        for index, entry in enumerate(entries):
            results.extend(
                self._evaluate_layout(
                    entry=entry,
                    sample=dataset[index],
                    layout_spec=replay_entries[entry.id],
                    agent=agent,
                )
            )

        metrics = summarize_results(results)
        artifact_paths = [write_artifact_json(output_root / "metrics.json", metrics)]
        if self.config.write_sample_records:
            artifact_paths.append(
                write_artifact_json(
                    output_root / "sample_results.json",
                    [result.to_json_dict() for result in results],
                )
            )

        summary_payload = {
            "benchmark_id": self.config.benchmark_id,
            "dataset_root": str(self.config.dataset_root),
            "split": self.config.split,
            "seed": self.config.seed,
            "trajectory_manifest_path": str(self.config.resolved_trajectory_manifest_path),
            "metrics": metrics,
            "artifact_paths": [str(path) for path in artifact_paths],
        }
        artifact_paths.append(write_artifact_json(output_root / "manifest.json", summary_payload))
        return M0Manifest(output_root=output_root, artifact_paths=tuple(artifact_paths), metrics=metrics)

    __call__ = evaluate

    def _evaluate_layout(
        self,
        *,
        entry,
        sample: dict[str, torch.Tensor],
        layout_spec: M0LayoutSpec,
        agent: EpisodicMemoryAgent,
    ) -> list[M0TargetResult]:
        """Evaluate one layout over its two held-out target groups."""
        state = agent.reset_state()
        immediate_results: list[dict[str, float | int | None]] = []
        target_locations: dict[int, int] = {
            target.target_slot: self._flatten_location(target.target_location.row, target.target_location.col, entry.width)
            for target in layout_spec.targets
        }

        for target in layout_spec.targets:
            localization_hits: list[float] = []
            cue_query = EpisodicMemoryQuery(
                kind="cue_location",
                cue_family=target.cue.family,
                cue_id=target.cue.cue_id,
            )

            state, exposure_hits = self._replay_episode(sample=sample, entry=entry, episode=target.exposure, state=state, agent=agent)
            localization_hits.extend(exposure_hits)
            exposure_readout = agent.readout(cue_query, state)
            exposure_location = exposure_readout.location_id
            exposure_correct = float(exposure_location == target_locations[target.target_slot])

            probe_correctness: list[float] = []
            consistency_hits: list[float] = []
            for probe in target.probes:
                state, probe_hits = self._replay_episode(sample=sample, entry=entry, episode=probe, state=state, agent=agent)
                localization_hits.extend(probe_hits)
                probe_readout = agent.readout(cue_query, state)
                probe_location = probe_readout.location_id
                probe_correctness.append(float(probe_location == target_locations[target.target_slot]))
                consistency_hits.append(float(exposure_location is not None and probe_location == exposure_location))

            immediate_results.append(
                {
                    "target_slot": target.target_slot,
                    "cue_family": target.cue.family,
                    "cue_id": target.cue.cue_id,
                    "localization_accuracy": sum(localization_hits) / float(len(localization_hits)),
                    "exposure_recall": exposure_correct,
                    "probe_recall": sum(probe_correctness) / float(len(probe_correctness)),
                    "write_read_consistency": sum(consistency_hits) / float(len(consistency_hits)),
                }
            )

        results: list[M0TargetResult] = []
        last_target_slot = layout_spec.targets[-1].target_slot
        for result in immediate_results:
            target_slot = int(result["target_slot"])
            interference_retention: float | None = None
            if target_slot != last_target_slot:
                retained = agent.readout(
                    EpisodicMemoryQuery(
                        kind="cue_location",
                        cue_family=str(result["cue_family"]),
                        cue_id=int(result["cue_id"]),
                    ),
                    state,
                )
                interference_retention = float(retained.location_id == target_locations[target_slot])

            results.append(
                M0TargetResult(
                    layout_id=layout_spec.layout_id,
                    target_slot=target_slot,
                    localization_accuracy=float(result["localization_accuracy"]),
                    exposure_recall=float(result["exposure_recall"]),
                    probe_recall=float(result["probe_recall"]),
                    write_read_consistency=float(result["write_read_consistency"]),
                    interference_retention=interference_retention,
                )
            )
        return results

    def _replay_episode(
        self, *, sample: dict[str, torch.Tensor], entry, episode, state, agent: EpisodicMemoryAgent
    ) -> tuple[object, list[float]]:
        """Replay one benchmark-owned exposure or probe episode."""
        localization_hits: list[float] = []
        for step in episode.steps:
            payload = self._build_step_payload(sample=sample, entry=entry, step=step)
            state = agent.ingest_step(payload, state)
            localization = agent.readout(EpisodicMemoryQuery(kind="current_location"), state)
            expected_location_id = self._flatten_location(step.row, step.col, entry.width)
            localization_hits.append(float(localization.location_id == expected_location_id))
        return state, localization_hits

    def _build_step_payload(self, *, sample: dict[str, torch.Tensor], entry, step) -> EpisodicMemoryStep:
        """Build one replay-step payload from persisted M0 metadata and dataset arrays."""
        row, col = int(step.row), int(step.col)
        observation_id = int(sample[CHANNEL_OBSERVATIONS][row, col].item())
        observation_dim = int(entry.n_observations)
        if observation_id < 0 or observation_id >= observation_dim:
            raise ValueError(
                f"M0 step for layout {entry.id} references observation_id={observation_id} outside [0, {observation_dim - 1}]."
            )

        inputs = F.one_hot(torch.tensor(observation_id, dtype=torch.int64), num_classes=observation_dim).to(torch.float32)
        location_id = self._flatten_location(row, col, entry.width)
        landmark_id = None
        if CHANNEL_LANDMARKS in sample:
            landmark_id = sample[CHANNEL_LANDMARKS][row, col].to(dtype=torch.int64).view(1, 1)

        return EpisodicMemoryStep(
            inputs=inputs.view(1, -1),
            previous_action=torch.tensor([int(step.previous_action)], dtype=torch.int64),
            episode_start=torch.tensor([step.step_index == 0], dtype=torch.bool),
            observation_target=torch.tensor([observation_id], dtype=torch.int64),
            location_id=torch.tensor([location_id], dtype=torch.int64),
            landmark_id=landmark_id,
        )

    @staticmethod
    def _flatten_location(row: int, col: int, width: int) -> int:
        """Return a flattened location id from row/column coordinates."""
        return int(row) * int(width) + int(col)


__all__ = ["M0Benchmark", "M0BenchmarkConfig"]
