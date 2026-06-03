#!/usr/bin/env python3
"""Run the TEM v1 Hebbian memory self-retrieval probe and persist the result.

Usage
-----
    python scripts/diagnostics/run_tem_memory_probe.py \\
        [--checkpoint CHECKPOINT] \\
        [--model-config MODEL_CONFIG] \\
        [--dataset DATASET] \\
        [--output-dir OUTPUT_DIR]

Default paths match the arena_n4 reporting config:
    checkpoint  = checkpoints/tem-v1/eval-weights-only.pt
    model-config = config/models/tem-v1-base.toml
    dataset     = data/processed/arena/default/v1
    output-dir  = artifacts/evaluation/tem_v1/diagnostics
"""

from __future__ import annotations

import argparse
from pathlib import Path
from pprint import pprint

import torch

from ehc_sn.diagnostics.tem_memory_probe import (
    produce_tem_memory_probe,
    persist_tem_memory_probe,
    format_memory_probe_table,
)
from ehc_sn.models.tem.tem_v1 import ModelSettingsV1, TEMModelV1, TEMInputV1
from ehc_sn.tasks.arena.runtime import ARENA_REPLAY_REQUIRED_KEYS
from ehc_sn.modules.autoencoder import TwoHotEncoder

# Arena replay does not use landmarks (values are -1/1 sentinels only).
# We pass None to skip OVC correction and avoid shape mismatches in the MEC
# inference path for non-landmark environments.
_LANDMARK_ID = None


# =============================================================================
def _load_checkpoint(model: TEMModelV1, path: Path) -> None:
    """Load eval-weights-only checkpoint into the model."""
    sd = torch.load(path, map_location="cpu", weights_only=False)
    # Handle nested state_dict keys (Lightning wrapping).
    if "state_dict" in sd:
        sd = sd["state_dict"]
    # Strip 'model.' or 'adapter.model.' prefix if present.
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
    """Load one Arena val episode as a raw batch dict.

    Reads pre-materialised numpy arrays directly rather than going through
    the DataLoader pipeline (simpler standalone dependency).
    """
    split_dir = dataset_path / "val"

    # Load all channels.
    arrays = {}
    for key in ARENA_REPLAY_REQUIRED_KEYS:
        arr_path = split_dir / f"{key}.npy"
        if arr_path.exists():
            import numpy as np

            arrays[key] = torch.from_numpy(np.load(arr_path))

    # Pick the first sample.
    batch: dict[str, torch.Tensor] = {}
    for key, tensor in arrays.items():
        batch[key] = tensor[:1]  # (1, T_max) or (1,)

    return batch


# =============================================================================
def _batch_to_tem_input(
    batch: dict[str, torch.Tensor],
    model: TEMModelV1,
    *,
    observation_dim: int = 45,
    feature_dim: int = 10,
) -> TEMInputV1:
    """Convert an Arena batch into a stacked TEMInputV1 for the whole episode.

    The TEM model processes one step at a time, but we build the full
    multi-step input so the probe can iterate over it.
    """
    n_freq = len(model._config.hpc.shape)
    T = int(batch["trajectory_observation_id"].shape[1])

    encoder = TwoHotEncoder(observation_dim, feature_dim)
    device = next(model.parameters()).device

    obs_ids = batch["trajectory_observation_id"][0, :T].long().to(device)  # (T,)
    prev_actions = batch["trajectory_previous_action"][0, :T].to(device)  # (T,)
    episode_starts = (
        batch.get("trajectory_episode_start", torch.zeros(T, dtype=torch.bool))[0, :T]
        .to(device)
    )

    # Encode observations: one-hot → two-hot → replicate per frequency.
    one_hot = torch.nn.functional.one_hot(obs_ids, num_classes=observation_dim).float()
    code = encoder(one_hot)  # (T, feature_dim)

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
        description="Run TEM v1 memory self-retrieval probe"
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/tem-v1/eval-weights-only.pt",
        type=Path,
        help="Path to the TEM v1 eval-weights checkpoint",
    )
    parser.add_argument(
        "--model-config",
        default="config/models/tem-v1-base.toml",
        type=Path,
        help="Path to the TEM v1 model TOML config",
    )
    parser.add_argument(
        "--dataset",
        default="data/processed/arena/default/v1",
        type=Path,
        help="Path to the Arena processed dataset root",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/evaluation/tem_v1/diagnostics",
        type=Path,
        help="Directory for the memory_probe.json artifact",
    )
    parser.add_argument(
        "--observation-dim",
        type=int,
        default=45,
        help="Number of Arena observation IDs",
    )
    parser.add_argument(
        "--feature-dim",
        type=int,
        default=10,
        help="Two-hot encoding feature dimension",
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
    print(
        f"  Episode steps: {batch['trajectory_observation_id'].shape[1]}"
    )

    print("Building TEM input...")
    tem_input = _batch_to_tem_input(
        batch, model,
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

    print("Running memory probe...")
    with torch.no_grad():
        result = produce_tem_memory_probe(
            model, tem_input,
            observation_ids=obs_ids,
            position_ids=position_ids,
        )

    print("\n=== Memory Probe Results ===\n")
    print(format_memory_probe_table(result))

    # Classification
    sr = result.self_retrieval
    nn_acc = sr.nn_accuracy
    print("--- Classification ---")
    if nn_acc is None:
        print("self_retrieval_nn_accuracy: not computed")
    elif nn_acc >= 0.8:
        print(
            f"✅ PASS: self_retrieval_nn_accuracy = {nn_acc:.3f} (>= 0.8)\n"
            "     Memory self-retrieval is reliable."
        )
    elif nn_acc >= 0.5:
        print(
            f"✅ MARGINAL: self_retrieval_nn_accuracy = {nn_acc:.3f} (>= 0.5)\n"
            "     Memory has partial structure but retrieval is not fully reliable."
        )
    else:
        print(
            f"⚠️  LOW: self_retrieval_nn_accuracy = {nn_acc:.3f} (< 0.5)\n"
            "     Memory self-retrieval is unreliable.  Check NN error\n"
            "     characterisation to determine whether the failure is\n"
            "     basin ambiguity (same obs ID / nearby timestep) or\n"
            "     random (arbitrary posterior)."
        )

    if sr.top5_accuracy is not None:
        print(f"\nNN Error Characterisation:")
        print(f"  top5_accuracy:             {sr.top5_accuracy:.3f}")
        print(f"  top10_accuracy:            {sr.top10_accuracy:.3f}")
        print(f"  nn_temporal_dist_mean:     {sr.nn_temporal_distance_mean:.1f}")
        print(f"  nn_temporal_dist_median:   {sr.nn_temporal_distance_median:.1f}")
        if sr.nn_same_observation_fraction is not None:
            print(f"  nn_same_obs_fraction:      {sr.nn_same_observation_fraction:.3f}")
        if sr.nn_same_position_fraction is not None:
            print(f"  nn_same_position_fraction: {sr.nn_same_position_fraction:.3f}")
        if sr.nn_spatial_distance_mean is not None:
            print(f"  nn_spatial_distance_mean:  {sr.nn_spatial_distance_mean:.3f}")

    print(f"\nPersisting memory probe to {args.output_dir}")
    out_path = persist_tem_memory_probe(result, args.output_dir)
    print(f"  Written to {out_path}")

    # ---- Pathway probe ----
    print("\n\nRunning pathway probe...")

    # Build the full adapter, load its state dict, and use it directly.
    # This guarantees parity with the eval path encoder + decoder.
    from ehc_sn.adapters.arena.tem import (
        ArenaTEMAdapterSettings,
        ArenaTEMV1BridgeAdapter,
    )
    from ehc_sn.adapters.arena.tem.core import ArenaEncoderConfig, ArenaDecoderConfig

    adapter_config = ArenaTEMAdapterSettings(
        observation_dim=args.observation_dim,
        action_count=5,
        encoder=ArenaEncoderConfig(kind="two_hot", layout="replicated"),
        decoder=ArenaDecoderConfig(kind="single_scale", prediction_freq=0),
    )
    adapter = ArenaTEMV1BridgeAdapter(model, adapter_config)

    # Load adapter weights from the same checkpoint.
    sd_full = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = sd_full.get("state_dict", sd_full)
    # Strip 'adapter.' prefix from keys that have it.
    adapter_sd = {}
    for k, v in sd.items():
        if k.startswith("adapter."):
            local_key = k[len("adapter.") :]
        else:
            local_key = k
        adapter_sd[local_key] = v
    adapter.load_state_dict(adapter_sd, strict=False)
    adapter.eval()

    # Extract decoder parameters from the loaded adapter.
    decoder_module = adapter._decoder.decoder
    w_x_val = float(adapter._decoder.w_x.item())
    b_x_tensor = adapter._decoder.b_x.detach().cpu()
    print(f"  Decoder loaded: w_x={w_x_val:.4f}, b_x_norm={b_x_tensor.norm():.4f}")

    from ehc_sn.diagnostics.tem_pathway_probe import (
        produce_tem_pathway_probe,
        persist_pathway_probe,
        format_pathway_probe_table,
    )

    # Produce the pathway probe with the correctly-loaded adapter decoder.
    # The forward pass uses the same model(...) call as the memory probe,
    # but the decoder chain now uses the adapter's trained parameters.
    with torch.no_grad():
        pw_result = produce_tem_pathway_probe(
            model, tem_input, obs_ids,
            prediction_freq=0,
            w_x_val=w_x_val,
            b_x=b_x_tensor,
            decoder=decoder_module,
        )

    print("\n=== Pathway Probe Results ===\n")
    print(format_pathway_probe_table(pw_result))

    print(f"\nPersisting pathway probe to {args.output_dir}")
    pw_path = persist_pathway_probe(pw_result, args.output_dir)
    print(f"  Written to {pw_path}")


# =============================================================================
if __name__ == "__main__":
    main()
