#!/usr/bin/env python3
"""Run the TEM v1 attractor vector-field alignment probe.

Measures the one-step AttractorRead update direction and compares it against
the target direction ``p_post[t] - query[t]`` for each query family.

Usage
-----
    python scripts/diagnostics/run_tem_attractor_field_probe.py \\
        [--checkpoint CHECKPOINT] \\
        [--model-config MODEL_CONFIG] \\
        [--dataset DATASET] \\
        [--output-dir OUTPUT_DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ehc_sn.diagnostics.tem_query_alignment_probe import (
    format_attractor_field_table,
    persist_attractor_field_probe,
    produce_tem_attractor_field_probe,
)
from ehc_sn.models.tem.tem_v1 import ModelSettingsV1, TEMInputV1, TEMModelV1
from ehc_sn.modules.autoencoder import TwoHotEncoder
from ehc_sn.tasks.arena.runtime import ARENA_REPLAY_REQUIRED_KEYS

_LANDMARK_ID = None


# =============================================================================
def _load_checkpoint(model: TEMModelV1, path: Path) -> None:
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    stripped = {}
    for k, v in sd.items():
        key = k
        if key.startswith("adapter.model."):
            key = key[len("adapter.model.") :]
        elif key.startswith("model."):
            key = key[len("model.") :]
        stripped[key] = v
    model.load_state_dict(stripped, strict=False)
    model.eval()


# =============================================================================
def _load_static_batch(dataset_path: Path) -> dict[str, torch.Tensor]:
    split_dir = dataset_path / "val"
    arrays = {}
    for key in ARENA_REPLAY_REQUIRED_KEYS:
        arr_path = split_dir / f"{key}.npy"
        if arr_path.exists():
            import numpy as np

            arrays[key] = torch.from_numpy(np.load(arr_path))
    batch: dict[str, torch.Tensor] = {}
    for key, tensor in arrays.items():
        batch[key] = tensor[:1]
    return batch


# =============================================================================
def _batch_to_tem_input(
    batch: dict[str, torch.Tensor],
    model: TEMModelV1,
    *,
    observation_dim: int = 45,
    feature_dim: int = 10,
) -> TEMInputV1:
    n_freq = len(model._config.hpc.shape)
    T = int(batch["trajectory_observation_id"].shape[1])
    encoder = TwoHotEncoder(observation_dim, feature_dim)
    device = next(model.parameters()).device

    obs_ids = batch["trajectory_observation_id"][0, :T].long().to(device)
    prev_actions = batch["trajectory_previous_action"][0, :T].to(device)
    episode_starts = batch.get(
        "trajectory_episode_start", torch.zeros(T, dtype=torch.bool)
    )[0, :T].to(device)

    one_hot = torch.nn.functional.one_hot(
        obs_ids, num_classes=observation_dim
    ).float()
    code = encoder(one_hot)
    observation_embedding = [code.clone() for _ in range(n_freq)]

    return TEMInputV1(
        observation_embedding=observation_embedding,
        previous_action=prev_actions,
        episode_start=episode_starts,
        landmark_id=_LANDMARK_ID,
    )


# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run TEM v1 attractor vector-field alignment probe"
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/tem-v1/eval-weights-only.pt",
        type=Path,
    )
    parser.add_argument(
        "--model-config",
        default="config/models/tem-v1-base.toml",
        type=Path,
    )
    parser.add_argument(
        "--dataset",
        default="data/processed/arena/default/v1",
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/evaluation/tem_v1/diagnostics",
        type=Path,
    )
    parser.add_argument(
        "--observation-dim",
        type=int,
        default=45,
    )
    parser.add_argument(
        "--feature-dim",
        type=int,
        default=10,
    )
    args = parser.parse_args()

    print(f"Loading model config from {args.model_config}")
    model_settings = ModelSettingsV1.from_config(args.model_config)
    model = TEMModelV1(model_settings)

    print(f"Loading checkpoint from {args.checkpoint}")
    _load_checkpoint(model, args.checkpoint)
    device = next(model.parameters()).device
    print(f"  Model on {device}")

    print(f"Loading dataset from {args.dataset}")
    batch = _load_static_batch(args.dataset)
    T = batch["trajectory_observation_id"].shape[1]
    print(f"  Episode steps: {T}")

    print("Building TEM input...")
    tem_input = _batch_to_tem_input(
        batch,
        model,
        observation_dim=args.observation_dim,
        feature_dim=args.feature_dim,
    )

    print("Running attractor field probe...")
    with torch.no_grad():
        result = produce_tem_attractor_field_probe(model, tem_input)

    print("\n" + format_attractor_field_table(result))

    print(f"\nPersisting to {args.output_dir}")
    out_path = persist_attractor_field_probe(result, args.output_dir)
    print(f"  Written to {out_path}")


# =============================================================================
if __name__ == "__main__":
    main()
