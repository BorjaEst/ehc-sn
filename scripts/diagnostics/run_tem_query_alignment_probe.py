#!/usr/bin/env python3
"""Run the TEM v1 query-alignment probe and persist the result.

Compares Hebbian AttractorRead retrieval from posterior, retrieved, prior,
grid-projected, sensory-projected, and random queries across multiple attractor
iteration depths.

Usage
-----
    python scripts/diagnostics/run_tem_query_alignment_probe.py \\
        [--checkpoint CHECKPOINT] \\
        [--model-config MODEL_CONFIG] \\
        [--dataset DATASET] \\
        [--output-dir OUTPUT_DIR] \\
        [--n-iters N1 N2 N3 ...]

Default paths match the arena_n4 reporting config:
    checkpoint  = checkpoints/tem-v1/eval-weights-only.pt
    model-config = config/models/tem-v1-base.toml
    dataset     = data/processed/arena/default/v1
    output-dir  = artifacts/evaluation/tem_v1/diagnostics
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ehc_sn.diagnostics.tem_query_alignment_probe import (
    format_query_alignment_table,
    persist_query_alignment_probe,
    produce_tem_query_alignment_probe,
)
from ehc_sn.models.tem.tem_v1 import ModelSettingsV1, TEMInputV1, TEMModelV1
from ehc_sn.modules.autoencoder import TwoHotEncoder
from ehc_sn.tasks.arena.runtime import ARENA_REPLAY_REQUIRED_KEYS

# Arena replay does not use landmarks (values are -1/1 sentinels only).
_LANDMARK_ID = None


# =============================================================================
def _load_checkpoint(model: TEMModelV1, path: Path) -> None:
    """Load eval-weights-only checkpoint into the model."""
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
    """Load one Arena val episode as a raw batch dict."""
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
    """Convert an Arena batch into a stacked TEMInputV1 for the whole episode."""
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
        description="Run TEM v1 query-alignment probe"
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
    parser.add_argument(
        "--n-iters",
        type=int,
        nargs="+",
        default=[1, 2, 3, 5, 10, 20],
        help="Attractor iteration depths to test",
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

    # Extract observation IDs and spatial coordinates for NN error char.
    obs_ids = batch["trajectory_observation_id"][0, :].long()
    position_ids = torch.stack(
        [
            batch["trajectory_row"][0, :],
            batch["trajectory_col"][0, :],
        ],
        dim=-1,
    )

    print(f"Running query-alignment probe (n_iters={args.n_iters})...")
    with torch.no_grad():
        result = produce_tem_query_alignment_probe(
            model,
            tem_input,
            observation_ids=obs_ids,
            position_ids=position_ids,
            n_iters=tuple(args.n_iters),
        )

    print("\n" + format_query_alignment_table(result))

    # Print diagnosis.
    print(f"\n=== Diagnosis ===")
    print(f"  {result.diagnosis}")

    # Per-query summary table: best iteration for each query type.
    print(f"\n=== Per-Query Summary (best iteration) ===")
    header = (
        f"{'query':<28s} {'n_iter':>6s} {'match_cos':>10s} "
        f"{'nn_acc':>8s} {'top5':>6s} {'same_obs':>9s} {'sharp':>7s}"
    )
    print(header)
    print("-" * len(header))
    for qkey, qt in result.query_types.items():
        # Find iteration with best nn_accuracy.
        best_it = None
        best_val = -1.0
        for it_str, pm in qt.per_iter.items():
            v = pm.nn_accuracy if pm.nn_accuracy is not None else -1.0
            if v > best_val:
                best_val = v
                best_it = it_str
        if best_it is not None:
            pm = qt.per_iter[best_it]
            print(
                f"  {qkey:<26s} {best_it:>6s} "
                f"{pm.cosine_to_matching:>10.4f} "
                f"{_f(pm.nn_accuracy, '—'):>8s} "
                f"{_f(pm.top5_accuracy, '—'):>6s} "
                f"{_f(pm.same_observation_fraction, '—'):>9s} "
                f"{pm.retrieval_sharpening:>7.4f}"
            )

    print(f"\nPersisting probe to {args.output_dir}")
    out_path = persist_query_alignment_probe(result, args.output_dir)
    print(f"  Written to {out_path}")


def _f(v: float | None, placeholder: str = "—") -> str:
    if v is None:
        return placeholder
    return f"{v:.4f}"


# =============================================================================
if __name__ == "__main__":
    main()
