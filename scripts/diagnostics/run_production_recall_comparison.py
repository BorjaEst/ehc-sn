#!/usr/bin/env python3
"""Quick comparison: raw grid query vs production p_retrieved vs p_post.

Measures how the production AttractorRead (one pass, hierarchical masks)
changes the grid query relative to the posterior bank.

Usage
-----
    python scripts/diagnostics/run_production_recall_comparison.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ehc_sn.models.tem.tem_v1 import ModelSettingsV1, TEMInputV1, TEMModelV1
from ehc_sn.modules.autoencoder import TwoHotEncoder
from ehc_sn.tasks.arena.runtime import ARENA_REPLAY_REQUIRED_KEYS
from ehc_sn.types import DenseMemoryStore

_LANDMARK_ID = None


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/tem-v1/eval-weights-only.pt",
        type=Path,
    )
    parser.add_argument(
        "--model-config", default="config/models/tem-v1-base.toml", type=Path
    )
    parser.add_argument(
        "--dataset", default="data/processed/arena/default/v1", type=Path
    )
    parser.add_argument("--observation-dim", type=int, default=45)
    parser.add_argument("--feature-dim", type=int, default=10)
    args = parser.parse_args()

    model = TEMModelV1(ModelSettingsV1.from_config(args.model_config))
    _load_checkpoint(model, args.checkpoint)
    device = next(model.parameters()).device

    batch = _load_static_batch(args.dataset)
    tem_input = _batch_to_tem_input(
        batch,
        model,
        observation_dim=args.observation_dim,
        feature_dim=args.feature_dim,
    )
    T = int(tem_input.observation_embedding[0].shape[0])
    print(f"Episode steps: {T}")

    # Metadata
    obs_ids = batch["trajectory_observation_id"][0, :].long()
    pos_ids = torch.stack(
        [batch["trajectory_row"][0, :], batch["trajectory_col"][0, :]], dim=-1
    )

    # Forward pass
    state = model.init_state(1, device=device)

    p_post_flat_list = []
    p_retrieved_flat_list = []
    g_query_post_flat_list = []
    g_query_prior_flat_list = []
    all_obs = []

    with torch.no_grad():
        for t in range(T):
            step_obs = [f[t : t + 1] for f in tem_input.observation_embedding]
            step_input = TEMInputV1(
                observation_embedding=step_obs,
                previous_action=tem_input.previous_action[t : t + 1],
                episode_start=(
                    tem_input.episode_start[t : t + 1]
                    if tem_input.episode_start is not None
                    else None
                ),
                landmark_id=_LANDMARK_ID,
            )
            output, state = model(step_input, state=state)

            # Collect flattened codes
            p_post_flat_list.append(
                torch.cat(
                    [p.clone().cpu() for p in output.place_codes.posterior],
                    dim=-1,
                )
            )
            p_retrieved_flat_list.append(
                torch.cat(
                    [p.clone().cpu() for p in output.place_codes.retrieved],
                    dim=-1,
                )
            )

            # Grid query: project g_post through mec_to_hpc
            g_post_bundle = [
                g.clone().cpu() for g in output.grid_codes.posterior
            ]
            g_query_post = torch.cat(model.mec_to_hpc(g_post_bundle), dim=-1)
            g_query_post_flat_list.append(g_query_post.cpu())

            g_prior_bundle = [g.clone().cpu() for g in output.grid_codes.prior]
            g_query_prior = torch.cat(model.mec_to_hpc(g_prior_bundle), dim=-1)
            g_query_prior_flat_list.append(g_query_prior.cpu())

            all_obs.append(batch["trajectory_observation_id"][0, t].item())

    p_post = torch.cat(p_post_flat_list, dim=0)
    p_retrieved = torch.cat(p_retrieved_flat_list, dim=0)
    g_query_post = torch.cat(g_query_post_flat_list, dim=0)
    g_query_prior = torch.cat(g_query_prior_flat_list, dim=0)

    def _nn_metrics(query: Tensor, ref: Tensor) -> dict:
        """Return nn_acc, same_obs_frac, top5, cos_match, cos_nn."""
        T = query.shape[0]
        # Remove NaN/Inf rows.
        valid = torch.isfinite(query).all(-1) & torch.isfinite(ref).all(-1)
        if not valid.all():
            qv, rv = query[valid], ref[valid]
            T_valid = int(valid.sum().item())
        else:
            qv, rv = query, ref
            T_valid = T
        if T_valid < 2:
            return {
                "cos_match": float("nan"),
                "cos_nn": float("nan"),
                "nn_acc": float("nan"),
                "same_obs": float("nan"),
                "top5": float("nan"),
            }

        # Cosine to matching p_post[t] — safe per-row cosine.
        qn = qv / (qv.norm(dim=-1, keepdim=True) + 1e-8)
        rn = rv / (rv.norm(dim=-1, keepdim=True) + 1e-8)
        cos_match = float((qn * rn).sum(-1).mean().item())

        # NN in ref bank
        sim = qn @ rn.T  # (T_valid, T_valid)
        nn_idx = sim.argmax(-1)
        arange = torch.arange(T_valid)
        nn_acc = float((nn_idx == arange).float().mean().item())
        cos_nn = float(sim[arange, nn_idx].mean().item())

        # Same obs (index into full obs array)
        if valid.all():
            obs_tensor = torch.tensor(all_obs)
            same_obs = float(
                (obs_tensor[nn_idx] == obs_tensor[:T_valid])
                .float()
                .mean()
                .item()
            )
        else:
            same_obs = float("nan")

        # Top5
        topk = sim.topk(k=min(5, T_valid), dim=-1)
        top5 = float(
            (topk.indices == arange.unsqueeze(1)).any(-1).float().mean().item()
        )
        return {
            "cos_match": cos_match,
            "cos_nn": cos_nn,
            "nn_acc": nn_acc,
            "same_obs": same_obs,
            "top5": top5,
        }

    metrics = {
        "p_post (target)": _nn_metrics(p_post, p_post),
        "raw g_query_post  ": _nn_metrics(g_query_post, p_post),
        "raw g_query_prior ": _nn_metrics(g_query_prior, p_post),
        "prod p_retrieved  ": _nn_metrics(p_retrieved, p_post),
    }

    print(
        f"\n{'Code':<22s} {'cos_match':>10s} {'cos_nn':>8s} {'nn_acc':>7s} {'same_obs':>9s} {'top5':>6s}"
    )
    print("-" * 68)
    for label, m in metrics.items():
        print(
            f"  {label:<20s} {m['cos_match']:>10.4f} {m['cos_nn']:>8.4f} {m['nn_acc']:>7.3f} {m['same_obs']:>9.3f} {m['top5']:>6.3f}"
        )

    # Direction: does production improve over raw query?
    g = metrics["raw g_query_post  "]
    p = metrics["prod p_retrieved  "]
    print(f"\n  delta (p_retrieved - g_query_post):")
    print(f"    cos_match: {p['cos_match'] - g['cos_match']:+.4f}")
    print(f"    nn_acc:    {p['nn_acc'] - g['nn_acc']:+.4f}")
    print(f"    same_obs:  {p['same_obs'] - g['same_obs']:+.4f}")
    print(f"    top5:      {p['top5'] - g['top5']:+.4f}")

    # Also check delta from raw to production for g_prior
    g2 = metrics["raw g_query_prior "]
    print(f"\n  delta (p_retrieved - g_query_prior):")
    print(f"    cos_match: {p['cos_match'] - g2['cos_match']:+.4f}")
    print(f"    nn_acc:    {p['nn_acc'] - g2['nn_acc']:+.4f}")
    print(f"    same_obs:  {p['same_obs'] - g2['same_obs']:+.4f}")

    # Final interpretation
    print(f"\n--- Interpretation ---")
    cos_imp = p["cos_match"] - g["cos_match"]
    nn_change = p["nn_acc"] - g["nn_acc"]
    if cos_imp > 0.02 and nn_change >= -0.01:
        print(
            f"✅ Production AttractorRead IMPROVES grid query: cos_match +{cos_imp:.3f}"
        )
    elif cos_imp > 0.02 and nn_change < -0.01:
        print(
            f"⚠️  Production AttractorRead improves cosine (+{cos_imp:.3f}) but harms NN accuracy ({nn_change:.3f})"
        )
        print(f"   → The read changes the manifold in a decoder-relevant way.")
    elif abs(cos_imp) <= 0.02:
        print(f"➖ Production AttractorRead has minimal effect on grid query.")
    else:
        print(
            f"❌ Production AttractorRead DEGRADES grid query: cos_match {cos_imp:.3f}"
        )


if __name__ == "__main__":
    main()
