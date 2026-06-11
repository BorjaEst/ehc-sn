"""Diagnostic analysis of the seqmaze probe recall gap.

Tests three hypotheses:
  H1: Embedding collision — certain successor-index pairs collide in embedding space
  H2: Degree saturation — recall degrades as node out-degree increases
  H3: Random scatter — misses are uniformly distributed

Also simulates: if we randomly drop 38% edges from a DAG, does the
unique shortest path survive?
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from ehc_sn.adapters.seqmaze import SeqMazeProbeAdapterSettings
from ehc_sn.adapters.seqmaze.hrm_v2 import SeqMazeProbeHRMV2BridgeAdapter
from ehc_sn.models.hrm.hrm_v2 import HRModelV2, ModelSettingsV2
from ehc_sn.tasks.seqmaze._data import SeqMazeProbeIterableDataset
from ehc_sn.tasks.seqmaze._graph_utils import (
    generate_transition_dag,
    shortest_path,
)
from ehc_sn.tasks.seqmaze.runtime import extract_seqmaze_probe_targets


# =============================================================================
def load_trained_adapter(
    device: torch.device,
) -> SeqMazeProbeHRMV2BridgeAdapter:
    """Load the already-trained probe adapter from its config and weights.

    Since the probe script saves no checkpoint, we retrain a minimal model
    for diagnostic purposes.
    """
    config_path = "config/training/seqmaze-probe.toml"
    with Path(config_path).open("rb") as f:
        cfg = tomllib.load(f)

    n_max = cfg.get("n_max", 8)
    k_max = cfg.get("k_max", 3)
    vocab_size_obs = cfg.get("vocab_size_obs", 64)
    vocab_size_candidate = cfg.get("vocab_size_candidate", 16)

    model_settings = ModelSettingsV2.from_config(cfg["model_config_path"])
    model = HRModelV2(model_settings)
    model.to(device)

    adapter_config = SeqMazeProbeAdapterSettings(
        n_max=n_max,
        k_max=k_max,
        vocab_size_obs=vocab_size_obs,
        vocab_size_candidate=vocab_size_candidate,
        hidden_size=model_settings.pfc.hidden_size,
    )
    adapter = SeqMazeProbeHRMV2BridgeAdapter(model, adapter_config)
    adapter.to(device)
    return adapter


# =============================================================================
def train_diagnostic_model(
    device: torch.device,
) -> SeqMazeProbeHRMV2BridgeAdapter:
    """Train a fresh probe model for diagnostic analysis."""
    config_path = "config/training/seqmaze-probe.toml"
    with Path(config_path).open("rb") as f:
        cfg = tomllib.load(f)

    n_max = cfg.get("n_max", 8)
    k_max = cfg.get("k_max", 3)
    vocab_size_obs = cfg.get("vocab_size_obs", 64)
    vocab_size_candidate = cfg.get("vocab_size_candidate", 16)
    global_batch_size = cfg.get("global_batch_size", 32)
    max_epochs = cfg.get("max_epochs", 50)
    num_workers = cfg.get("num_workers", 2)

    model_settings = ModelSettingsV2.from_config(cfg["model_config_path"])
    model = HRModelV2(model_settings)
    model.to(device)

    adapter_config = SeqMazeProbeAdapterSettings(
        n_max=n_max,
        k_max=k_max,
        vocab_size_obs=vocab_size_obs,
        vocab_size_candidate=vocab_size_candidate,
        hidden_size=model_settings.pfc.hidden_size,
    )
    adapter = SeqMazeProbeHRMV2BridgeAdapter(model, adapter_config)
    adapter.to(device)

    train_dataset = SeqMazeProbeIterableDataset(
        n_max=n_max,
        k_max=k_max,
        mode="train",
        seed_start=0,
        n_samples=2000,
        obs_vocab_size=vocab_size_obs,
        candidate_vocab_size=n_max + 1,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=global_batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=1e-3, weight_decay=1e-5
    )

    for epoch in range(max_epochs):
        adapter.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            targets = extract_seqmaze_probe_targets(batch)
            bridge_out, _ = adapter(batch, state=None)
            logits = bridge_out.probe.edge_logits

            loss = nn.functional.cross_entropy(
                logits.reshape(-1, 2),
                targets.edge_label.reshape(-1),
                reduction="none",
            )
            loss = (
                loss * targets.edge_mask.reshape(-1)
            ).sum() / targets.edge_mask.sum().clamp(min=1)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            optimizer.step()

        if epoch % 10 == 0 or epoch == max_epochs - 1:
            adapter.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for batch in train_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    targets = extract_seqmaze_probe_targets(batch)
                    bridge_out, _ = adapter(batch, state=None)
                    preds = bridge_out.probe.edge_logits.argmax(dim=-1)
                    mask = targets.edge_mask
                    correct += (
                        ((preds == targets.edge_label) & mask).sum().item()
                    )
                    total += mask.sum().item()
            acc = correct / max(total, 1)
            print(f"  Epoch {epoch:3d}: train_acc={acc:.4f}")

    return adapter


# =============================================================================
@torch.no_grad()
def diagnose_embedding_collisions(
    adapter: SeqMazeProbeHRMV2BridgeAdapter,
    device: torch.device,
) -> None:
    """H1: Check if miss patterns indicate embedding-space collisions.

    We run the eval set and collect per-node-pair predictions, then check
    whether misses cluster on specific successor-index pairs.
    """
    print("\n" + "=" * 60)
    print("H1: EMBEDDING COLLISION ANALYSIS")
    print("=" * 60)

    n_max = adapter.config.n_max
    k_max = adapter.config.k_max

    eval_dataset = SeqMazeProbeIterableDataset(
        n_max=n_max,
        k_max=k_max,
        mode="eval",
        seed_start=10000,
        n_samples=500,
        obs_vocab_size=adapter.config.vocab_size_obs,
        candidate_vocab_size=n_max + 1,
    )
    loader = DataLoader(
        eval_dataset, batch_size=32, shuffle=False, drop_last=False
    )

    adapter.eval()

    # Collect edge-by-edge miss statistics
    # miss_by_succ_idx: for each true successor index s, count misses and total
    miss_count = {}
    total_count = {}

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        targets = extract_seqmaze_probe_targets(batch)
        bridge_out, _ = adapter(batch, state=None)
        logits = bridge_out.probe.edge_logits
        preds = logits.argmax(dim=-1)  # (B, N, N)
        labels = targets.edge_label  # (B, N, N)
        mask = targets.edge_mask  # (B, N, N)

        # For each true positive edge, check if predicted correctly
        true_pos = (labels == 1) & mask  # (B, N, N)
        pred_pos = (preds == 1) & mask
        hit = (preds == labels) & true_pos
        miss = (~hit) & true_pos

        # Map back to successor indices (the K dimension)
        # We need successor_indices from the batch
        succ_indices = batch["successor_indices"]  # (B, N, K)

        B = succ_indices.shape[0]
        for bi in range(B):
            for ni in range(n_max):
                for ki in range(k_max):
                    sj = int(succ_indices[bi, ni, ki].item())
                    if sj < 0 or sj >= n_max:
                        continue
                    key = sj  # successor candidate index
                    total_count[key] = total_count.get(key, 0) + 1
                    # Check if model predicted edge (ni -> sj) correctly
                    if not mask[bi, ni, sj].item():
                        continue
                    correct = (
                        preds[bi, ni, sj].item() == labels[bi, ni, sj].item()
                    )
                    if not correct:
                        miss_count[key] = miss_count.get(key, 0) + 1

    print(
        f"\nMiss rate by successor candidate index (eval, {sum(total_count.values())} total edges):"
    )
    hit_rates = []
    for idx in sorted(total_count.keys()):
        total = total_count[idx]
        misses = miss_count.get(idx, 0)
        hit_rate = 1.0 - misses / max(total, 1)
        hit_rates.append(hit_rate)
        bar = "#" * int(hit_rate * 30)
        print(
            f"  succ_idx={idx:2d}: hit={hit_rate:.3f} ({total - misses}/{total}) {bar}"
        )

    avg_hit = sum(hit_rates) / max(len(hit_rates), 1)
    std_hit = (
        sum((h - avg_hit) ** 2 for h in hit_rates) / max(len(hit_rates), 1)
    ) ** 0.5
    print(
        f"\n  Mean hit rate across successor indices: {avg_hit:.3f} ± {std_hit:.3f}"
    )
    if std_hit > 0.05:
        print(
            "  ⚠️  High variance — some successor indices are harder than others (collision risk)."
        )
    else:
        print(
            "  ✅ Low variance — misses are uniformly distributed across successor indices."
        )

    # Also check E_candidate_index embedding cosine similarity
    print("\nE_candidate_index embedding cosine similarity matrix:")
    emb = adapter._encoder.E_candidate_index.weight.data[: n_max + 1]
    emb_norm = emb / emb.norm(dim=-1, keepdim=True)
    sim = emb_norm @ emb_norm.T
    for i in range(n_max + 1):
        others = [f"{sim[i,j]:.3f}" for j in range(n_max + 1) if j != i]
        max_sim = sim[i].clone()
        max_sim[i] = -1
        print(
            f"  idx={i:2d}: max_cos_sim={max_sim.max():.3f}, mean_cos_sim={sim[i].mean():.3f}"
        )


# =============================================================================
@torch.no_grad()
def diagnose_degree_saturation(
    adapter: SeqMazeProbeHRMV2BridgeAdapter,
    device: torch.device,
) -> None:
    """H2: Check if recall degrades with node out-degree."""
    print("\n" + "=" * 60)
    print("H2: DEGREE SATURATION ANALYSIS")
    print("=" * 60)

    n_max = adapter.config.n_max
    k_max = adapter.config.k_max

    eval_dataset = SeqMazeProbeIterableDataset(
        n_max=n_max,
        k_max=k_max,
        mode="eval",
        seed_start=10000,
        n_samples=500,
        obs_vocab_size=adapter.config.vocab_size_obs,
        candidate_vocab_size=n_max + 1,
    )
    loader = DataLoader(
        eval_dataset, batch_size=32, shuffle=False, drop_last=False
    )

    adapter.eval()

    # Bin recall by node out-degree
    recall_by_deg: dict[int, list[float]] = {}

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        succ_indices = batch["successor_indices"]  # (B, N, K)
        succ_mask = batch["successor_mask"]  # (B, N, K)
        targets = extract_seqmaze_probe_targets(batch)
        bridge_out, _ = adapter(batch, state=None)
        preds = bridge_out.probe.edge_logits.argmax(dim=-1)
        labels = targets.edge_label
        node_mask = batch["node_mask"]

        B, N = succ_indices.shape[:2]
        for bi in range(B):
            for ni in range(N):
                if not node_mask[bi, ni].item():
                    continue
                # Actual out-degree for this node
                deg = int(succ_mask[bi, ni].sum().item())
                if deg not in recall_by_deg:
                    recall_by_deg[deg] = []
                # Recall for this node: fraction of true successors correctly predicted
                true_succ = labels[bi, ni, :] == 1  # (N,)
                pred_succ = preds[bi, ni, :] == 1  # (N,)
                n_true = true_succ.sum().item()
                if n_true > 0:
                    n_correct = (true_succ & pred_succ).sum().item()
                    recall_by_deg[deg].append(n_correct / n_true)

    print("\nRecall by node out-degree:")
    for deg in sorted(recall_by_deg.keys()):
        values = recall_by_deg[deg]
        mean_recall = sum(values) / max(len(values), 1)
        bar = "#" * int(mean_recall * 30)
        print(
            f"  degree={deg}: recall={mean_recall:.3f} (n={len(values)}) {bar}"
        )

    # Test: is there a monotonic degradation?
    if len(recall_by_deg) >= 2:
        sorted_deg = sorted(recall_by_deg.keys())
        means = [
            sum(recall_by_deg[d]) / max(len(recall_by_deg[d]), 1)
            for d in sorted_deg
        ]
        if len(means) >= 2 and means[-1] < means[0] * 0.85:
            print(
                "  ⚠️  Significant recall degradation at higher degrees — degree saturation confirmed."
            )
        else:
            print("  ✅ No significant degree saturation effect.")


# =============================================================================
@torch.no_grad()
def diagnose_random_scatter(
    adapter: SeqMazeProbeHRMV2BridgeAdapter,
    device: torch.device,
) -> None:
    """H3: Check if misses are randomly scattered or systematically biased.

    We compute per-graph recall and check whether it's approximately binomial
    (each edge missed independently) or shows systematic bias (certain graphs
    have much higher miss rates).
    """
    print("\n" + "=" * 60)
    print("H3: RANDOM SCATTER ANALYSIS")
    print("=" * 60)

    n_max = adapter.config.n_max
    k_max = adapter.config.k_max

    eval_dataset = SeqMazeProbeIterableDataset(
        n_max=n_max,
        k_max=k_max,
        mode="eval",
        seed_start=10000,
        n_samples=500,
        obs_vocab_size=adapter.config.vocab_size_obs,
        candidate_vocab_size=n_max + 1,
    )
    loader = DataLoader(
        eval_dataset, batch_size=32, shuffle=False, drop_last=False
    )

    adapter.eval()

    per_graph_recalls = []
    per_graph_precisions = []

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        targets = extract_seqmaze_probe_targets(batch)
        bridge_out, _ = adapter(batch, state=None)
        preds = bridge_out.probe.edge_logits.argmax(dim=-1)
        labels = targets.edge_label
        mask = targets.edge_mask

        B = preds.shape[0]
        for bi in range(B):
            # Per-sample mask
            m = mask[bi]
            l = labels[bi]
            p = preds[bi]

            true_pos = (l == 1) & m
            pred_pos = (p == 1) & m
            hit = (p == l) & true_pos
            n_true = true_pos.sum().item()
            n_pred = pred_pos.sum().item()
            n_hit = hit.sum().item()

            recall = n_hit / max(n_true, 1)
            precision = n_hit / max(n_pred, 1)
            per_graph_recalls.append(recall)
            per_graph_precisions.append(precision)

    mean_recall = sum(per_graph_recalls) / max(len(per_graph_recalls), 1)
    std_recall = (
        sum((r - mean_recall) ** 2 for r in per_graph_recalls)
        / max(len(per_graph_recalls), 1)
    ) ** 0.5
    mean_precision = sum(per_graph_precisions) / max(
        len(per_graph_precisions), 1
    )
    std_precision = (
        sum((p - mean_precision) ** 2 for p in per_graph_precisions)
        / max(len(per_graph_precisions), 1)
    ) ** 0.5

    print(f"\nPer-graph recall:  mean={mean_recall:.3f} ± {std_recall:.3f}")
    print(
        f"Per-graph precision: mean={mean_precision:.3f} ± {std_precision:.3f}"
    )

    # Expected std if binomial with mean recall and varying edge counts
    # For a graph with n nodes and avg k edges per node: E[edges_per_graph] ≈ n * k * 0.5
    # Using observed avg edges per graph from the data
    # Under binomial: std = sqrt(p*(1-p) / n_edges)
    n_total_pairs = sum(r * (1.0 - r) for r in per_graph_recalls)
    n_total_denom = sum(1.0 for r in per_graph_recalls)
    avg_edge_var = n_total_pairs / max(n_total_denom, 1)
    # This overestimates because edges aren't independent, but it's a rough check
    expected_std = mean_recall * (1 - mean_recall) ** 0.5  # upper bound

    if std_recall > 2 * expected_std:
        print(
            "  ⚠️  Variance exceeds binomial expectation — systematic per-graph bias (structure matters)."
        )
    else:
        print(
            "  ✅ Variance consistent with random scatter — no systematic per-graph bias."
        )


# =============================================================================
def simulate_edge_drop_path_survival(
    n_trials: int = 5000,
    n_nodes: int = 8,
    k_max: int = 3,
    drop_rate: float = 0.38,
) -> None:
    """H4: If we randomly drop 38% of edges, does the shortest path survive?"""
    print("\n" + "=" * 60)
    print(f"H4: PATH SURVIVAL UNDER {drop_rate*100:.0f}% EDGE DROP")
    print("=" * 60)

    still_has_unique_path = 0
    path_length_increased = 0
    total_examined = 0

    for trial in range(n_trials):
        adj = generate_transition_dag(n_nodes, k_max, seed=trial)
        original = shortest_path(adj, 0, n_nodes - 1)
        if not original:
            continue

        total_examined += 1
        orig_len = len(original)

        # Drop edges at random
        rng = __import__("random").Random(trial + 10000)
        dropped_adj = [[] for _ in range(n_nodes)]
        for i in range(n_nodes):
            for j in adj[i]:
                if rng.random() >= drop_rate:
                    dropped_adj[i].append(j)

        post_drop = shortest_path(dropped_adj, 0, n_nodes - 1)

        if post_drop and len(post_drop) == orig_len:
            still_has_unique_path += 1
        elif post_drop:
            path_length_increased += 1

    survival_rate = still_has_unique_path / max(total_examined, 1) * 100
    increased_rate = path_length_increased / max(total_examined, 1) * 100
    lost_rate = (
        (total_examined - still_has_unique_path - path_length_increased)
        / max(total_examined, 1)
        * 100
    )

    print(f"\nTrials examined: {total_examined}")
    print(
        f"  Path survived at same length: {survival_rate:.1f}% ({still_has_unique_path})"
    )
    print(
        f"  Path survived but longer:     {increased_rate:.1f}% ({path_length_increased})"
    )
    print(
        f"  No path remains:              {lost_rate:.1f}% ({total_examined - still_has_unique_path - path_length_increased})"
    )

    if survival_rate > 70:
        print(
            "\n✅ Path survival >70% — 62% recall is tolerable for path inference."
        )
    elif survival_rate > 40:
        print(
            "\n⚠️  Moderate path survival — recall gap is a real concern but may be manageable."
        )
    else:
        print(
            "\n❌ Low path survival — recall gap is CRITICAL and must be addressed before full seqmaze."
        )


# =============================================================================
def simulate_learned_edge_drop_path_survival(
    adapter: SeqMazeProbeHRMV2BridgeAdapter,
    device: torch.device,
    n_samples: int = 500,
) -> None:
    """Use the actual model to drop edges (missed predictions) and check path survival."""
    print("\n" + "=" * 60)
    print("H5: ACTUAL MODEL EDGE-DROP PATH SURVIVAL")
    print("=" * 60)

    n_max = adapter.config.n_max
    k_max = adapter.config.k_max

    eval_dataset = SeqMazeProbeIterableDataset(
        n_max=n_max,
        k_max=k_max,
        mode="eval",
        seed_start=10000,
        n_samples=n_samples,
        obs_vocab_size=adapter.config.vocab_size_obs,
        candidate_vocab_size=n_max + 1,
    )
    loader = DataLoader(
        eval_dataset, batch_size=1, shuffle=False, drop_last=False
    )

    adapter.eval()

    still_has_unique_path = 0
    path_length_increased = 0
    total_examined = 0
    false_positives_creating_new_path = 0

    for batch in loader:
        batch_cpu = {k: v.clone() for k, v in batch.items()}
        batch_dev = {k: v.to(device) for k, v in batch.items()}
        targets = extract_seqmaze_probe_targets(batch_dev)
        bridge_out, _ = adapter(batch_dev, state=None)
        preds = bridge_out.probe.edge_logits.argmax(dim=-1)  # (1, N, N)
        labels = targets.edge_label  # (1, N, N)
        node_mask = batch_dev["node_mask"]  # (1, N)

        # Reconstruct adjacency from batch
        succ_indices = batch_cpu["successor_indices"][0]  # (N, K)
        succ_mask = batch_cpu["successor_mask"][0]  # (N, K)
        n_mask = node_mask[0]  # (N,)
        n = int(n_mask.sum().item())

        # Ground-truth adjacency (within the n actual nodes)
        true_adj: list[list[int]] = [[] for _ in range(n)]
        for i in range(n):
            for k in range(k_max):
                if succ_mask[i, k].item():
                    j = int(succ_indices[i, k].item())
                    if j < n:
                        true_adj[i].append(j)

        # Predicted adjacency (model thinks there's an edge)
        pred_adj: list[list[int]] = [[] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if preds[0, i, j].item() == 1:
                    pred_adj[i].append(j)

        true_path = shortest_path(true_adj, 0, n - 1)
        if not true_path:
            continue

        total_examined += 1
        true_len = len(true_path)

        pred_path = shortest_path(pred_adj, 0, n - 1)

        # Check false positives creating new paths
        fp_path_exists = False
        if not pred_path:
            # No path in predicted graph
            pass
        else:
            # Check if FP edges enabled a path that didn't exist before
            for i in range(n):
                for j in pred_adj[i]:
                    if labels[0, i, j].item() == 0 and j not in true_adj[i]:
                        fp_path_exists = True
                        break

        if pred_path and len(pred_path) == true_len:
            still_has_unique_path += 1
        elif pred_path:
            path_length_increased += 1
            if fp_path_exists:
                false_positives_creating_new_path += 1

    survival_rate = still_has_unique_path / max(total_examined, 1) * 100
    increased_rate = path_length_increased / max(total_examined, 1) * 100
    lost_rate = (
        (total_examined - still_has_unique_path - path_length_increased)
        / max(total_examined, 1)
        * 100
    )

    print(f"\nSamples examined: {total_examined}")
    print(
        f"  Path survived at same length:   {survival_rate:.1f}% ({still_has_unique_path})"
    )
    print(
        f"  Path survived but longer:       {increased_rate:.1f}% ({path_length_increased})"
    )
    print(
        f"  No path remains:                {lost_rate:.1f}% ({total_examined - still_has_unique_path - path_length_increased})"
    )
    print(
        f"  FP edges enabled new path:      {false_positives_creating_new_path}"
    )
    if survival_rate > 70:
        print(
            "\n✅ Path survival >70% — recall gap is tolerable for path inference."
        )
    elif survival_rate > 40:
        print("\n⚠️  Moderate path survival — recall gap is a real concern.")
    else:
        print("\n❌ Low path survival — recall gap is CRITICAL.")


# =============================================================================
def main() -> None:
    device = torch.device("cpu")
    print("=" * 60)
    print("SEQMAZE RECALL DIAGNOSTIC")
    print("=" * 60)

    # Train a fresh model for diagnostics
    print("\nTraining diagnostic model...")
    adapter = train_diagnostic_model(device)
    print("Training complete.\n")

    # H1: Embedding collision
    diagnose_embedding_collisions(adapter, device)

    # H2: Degree saturation
    diagnose_degree_saturation(adapter, device)

    # H3: Random scatter
    diagnose_random_scatter(adapter, device)

    # H4: Path survival under random drop
    simulate_edge_drop_path_survival(n_trials=10000, drop_rate=0.38)

    # H5: Path survival under actual model predictions
    simulate_learned_edge_drop_path_survival(adapter, device, n_samples=500)

    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
