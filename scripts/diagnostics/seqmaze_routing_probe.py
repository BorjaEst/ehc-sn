"""SeqMaze routing probe — does the HRM propagate graph info to path-query slots?

Tests whether a probe-trained HRM v2 backbone can route graph-structure
information from graph-region schema slots into path-region schema slots
during deliberation.

Strategy
--------
The probe adapter enforces ``seq_length == n_max`` (no path region).  We train
with N=8 (all slots as graph nodes), then for the routing probe we reinterpret:

    slots [0:4)  → graph region  (4 nodes)
    slots [4:8)  → path region   (4 path queries)

The backbone is frozen after probe training.  Per-position linear classifiers
are trained on path-region output slots.  A control condition zeros the
graph-region *inputs*.

Output table::

    Condition          | Pos-0 acc | Pos-1 acc | Pos-2 acc | Pos-3 acc | Mean acc
    -------------------|-----------|-----------|-----------|-----------|---------
    Experimental (N=4) |   0.XX    |   0.XX    |   0.XX    |   0.XX    |  0.XX
    Control (zeroed)   |   0.XX    |   0.XX    |   0.XX    |   0.XX    |  0.XX
    Random baseline    |   0.167   |   0.167   |   0.167   |   0.167   | 0.167

Interpretation
--------------
- Experimental >> Control and >> Random → HRM routes graph info to path slots.
- Experimental ≈ Control but > Random → path slots carry positional bias only.
- Experimental ≈ Random → no routing; path-query architecture needs redesign.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, IterableDataset

from ehc_sn.adapters.hrm import SeqMazeProbeAdapterSettings
from ehc_sn.adapters.hrm.seqmaze import SeqMazeProbeHRMV2BridgeAdapter
from ehc_sn.models.hrm.hrm_v2 import HRModelV2, ModelSettingsV2
from ehc_sn.utils.graph import (
    generate_transition_dag,
    shortest_path,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# The probe trains on N=8 (all slots as graph nodes).  For the routing probe
# we reinterpret 4 as graph + 4 as path.
PROBE_N_MAX = 8  # graph slots during probe training
ROUTING_N_MAX = 4  # graph slots during routing probe
T_MAX = 4  # path slots during routing probe
S = PROBE_N_MAX  # total schema slots = 8 (= PROBE_N_MAX = ROUTING_N_MAX + T_MAX)
K_MAX = 3

OBS_VOCAB_SIZE = 64
CANDIDATE_VOCAB_SIZE = PROBE_N_MAX + 1  # 0..N_max for successor lookups
HIDDEN_SIZE = 128
VOCAB_PATH = ROUTING_N_MAX + 2  # N_max candidates + EOS + PAD

N_TRAIN_SAMPLES = 2000
N_EVAL_SAMPLES = 500
PROBE_BATCH_SIZE = 64
PROBE_EPOCHS = 30
CLASSIFIER_EPOCHS = 20
CLASSIFIER_LR = 1e-2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------


class SeqMazeRoutingDataset(IterableDataset):
    """Generates DAG shortest-path samples on-the-fly.

    Two modes:
    - ``probe`` (n_max=8): full edge-prediction targets for probe training.
    - ``routing`` (n_max=4): path-prediction targets for routing probe.

    Each sample is a self-contained DAG problem with remapped obs IDs and
    permuted candidate order (anti-memorization contract).
    """

    def __init__(
        self,
        n_max: int,
        k_max: int = K_MAX,
        t_max: int = T_MAX,
        seed_start: int = 0,
        n_samples: int = 1000,
        mode: str = "probe",
    ) -> None:
        super().__init__()
        self.n_max = n_max
        self.k_max = k_max
        self.t_max = t_max
        self.seed_start = seed_start
        self.n_samples = n_samples
        self.mode = mode

    def __iter__(self):
        for i in range(self.n_samples):
            yield self._generate_sample(self.seed_start + i)

    def __len__(self) -> int:
        return self.n_samples

    def _generate_sample(self, seed: int) -> dict[str, Tensor]:
        n = self.n_max
        rng = torch.Generator().manual_seed(seed)

        # 1. Generate DAG
        adj = generate_transition_dag(n, self.k_max, seed)

        # 2. Remap obs IDs
        obs_id = torch.randperm(n, generator=rng)

        # 3. Permute candidate order
        perm = torch.randperm(n, generator=rng)
        inv_perm = torch.zeros(n, dtype=torch.long)
        for orig_idx, perm_idx in enumerate(perm):
            inv_perm[perm_idx] = orig_idx

        adj_remapped: list[list[int]] = [[] for _ in range(n)]
        for i in range(n):
            orig_i = inv_perm[i].item()
            for succ in adj[orig_i]:
                adj_remapped[i].append(int(perm[succ].item()))
            adj_remapped[i].sort()

        start_idx = int(perm[0].item())
        goal_idx = int(perm[n - 1].item())

        # 4. Shortest path
        sp = shortest_path(adj_remapped, start_idx, goal_idx)
        if not sp:
            sp = [start_idx, goal_idx]

        # 5. Successor tensor
        succ_indices = torch.full((n, self.k_max), 0, dtype=torch.long)
        succ_mask = torch.zeros((n, self.k_max), dtype=torch.bool)
        for i in range(n):
            for k, succ in enumerate(adj_remapped[i]):
                if k < self.k_max:
                    succ_indices[i, k] = succ
                    succ_mask[i, k] = True

        # 6. Flags
        start_flag = torch.zeros(n, dtype=torch.bool)
        goal_flag = torch.zeros(n, dtype=torch.bool)
        start_flag[start_idx] = True
        goal_flag[goal_idx] = True

        node_mask = torch.ones(n, dtype=torch.bool)

        result: dict[str, Tensor] = {
            "node_obs_id": obs_id.unsqueeze(0),
            "node_candidate_index": perm.unsqueeze(0),
            "node_start_flag": start_flag.unsqueeze(0),
            "node_goal_flag": goal_flag.unsqueeze(0),
            "successor_indices": succ_indices.unsqueeze(0),
            "successor_mask": succ_mask.unsqueeze(0),
            "node_mask": node_mask.unsqueeze(0),
        }

        # Edge labels (for probe training mode, N=8)
        if self.mode == "probe":
            edge_label = torch.zeros((n, n), dtype=torch.long)
            edge_mask = torch.zeros((n, n), dtype=torch.bool)
            for i in range(n):
                for succ in adj_remapped[i]:
                    edge_label[i, succ] = 1
                edge_mask[i, :] = node_mask
            edge_mask = edge_mask & node_mask.unsqueeze(1)
            result["edge_label"] = edge_label.unsqueeze(0)
            result["edge_mask"] = edge_mask.unsqueeze(0)

        # Path targets (for routing probe mode, N=4)
        if self.mode == "routing":
            t = self.t_max
            path = torch.full((t,), VOCAB_PATH - 1, dtype=torch.long)  # PAD
            for pos, node_idx in enumerate(sp):
                if pos < t:
                    path[pos] = node_idx
            eos_pos = min(len(sp), t - 1)
            path[eos_pos] = VOCAB_PATH - 2  # EOS
            path_mask = path != (VOCAB_PATH - 1)
            result["target_path"] = path.unsqueeze(0)
            result["path_mask"] = path_mask.unsqueeze(0)
            result["path_length"] = torch.tensor([len(sp)], dtype=torch.long)
        else:
            # Emit dummy path keys for the shared batch validation.
            result["target_path"] = torch.zeros(1, 4, dtype=torch.long)
            result["path_mask"] = torch.zeros(1, 4, dtype=torch.bool)
            result["path_length"] = torch.zeros(1, dtype=torch.long)

        return result


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collate(batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Collate a list of (1, ...) samples into batched tensors."""
    out: dict[str, list[Tensor]] = {}
    for key in batch[0]:
        out[key] = [b[key] for b in batch]
    return {k: torch.cat(v, dim=0) for k, v in out.items()}


# ---------------------------------------------------------------------------
# Probe training
# ---------------------------------------------------------------------------


def train_probe_model() -> SeqMazeProbeHRMV2BridgeAdapter:
    """Train a seqmaze edge-lookup probe model (N=8, all-slot graph)."""
    print(
        f"Building probe model (N={PROBE_N_MAX}, D={HIDDEN_SIZE}, "
        f"S={S}, seq_length={PROBE_N_MAX})..."
    )
    model_settings = ModelSettingsV2.from_config(
        "config/models/hrm-v2-seqmaze-n8.toml"
    )
    model = HRModelV2(model_settings)
    model.to(DEVICE)

    adapter_settings = SeqMazeProbeAdapterSettings(
        n_max=PROBE_N_MAX,
        k_max=K_MAX,
        vocab_size_obs=OBS_VOCAB_SIZE,
        vocab_size_candidate=CANDIDATE_VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
    )
    adapter = SeqMazeProbeHRMV2BridgeAdapter(model, adapter_settings)
    adapter.to(DEVICE)

    dataset = SeqMazeRoutingDataset(
        n_max=PROBE_N_MAX,
        k_max=K_MAX,
        seed_start=0,
        n_samples=N_TRAIN_SAMPLES,
        mode="probe",
    )
    loader = DataLoader(
        dataset,
        batch_size=PROBE_BATCH_SIZE,
        collate_fn=_collate,
        shuffle=False,
    )

    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=1e-3, weight_decay=1e-5
    )
    loss_fn = nn.CrossEntropyLoss()

    print(f"Training probe for {PROBE_EPOCHS} epochs...")
    adapter.train()
    for epoch in range(PROBE_EPOCHS):
        total_loss = 0.0
        n_batches = 0
        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            targets = batch["edge_label"]
            target_mask = batch["edge_mask"]

            bridge_out, _ = adapter(batch)
            logits = bridge_out.probe.edge_logits

            loss = loss_fn(logits[target_mask], targets[target_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / max(n_batches, 1)
            print(f"  Epoch {epoch + 1:2d}/{PROBE_EPOCHS}  loss={avg_loss:.4f}")

    return adapter


# ---------------------------------------------------------------------------
# Routing probe — graph+path encoder and per-position classifiers
# ---------------------------------------------------------------------------


class GraphPathEncoder(nn.Module):
    """Encoder that packs graph nodes (0:N) and path queries (N:N+T) into schema tokens.

    Graph-region slots: uses the same embeddings as SeqMazeProbeEncoder.
    Path-region slots: learned query embeddings + position embeddings + region tag.
    """

    def __init__(
        self,
        n_max: int,
        t_max: int,
        hidden_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.n_max = n_max
        self.t_max = t_max

        # Path-query embedding (shared across all path slots)
        self.E_path_query = nn.Embedding(
            1, hidden_size, device=device, dtype=dtype
        )
        # Path position embedding (one per position)
        self.E_path_position = nn.Embedding(
            t_max, hidden_size, device=device, dtype=dtype
        )
        # Region tag for path region (scalar broadcast)
        self.E_region_path = nn.Embedding(
            1, hidden_size, device=device, dtype=dtype
        )

        self.embedding_scale = hidden_size**0.5

    def forward(
        self,
        graph_tokens: Tensor,  # (B, N, D) — from SeqMazeProbeEncoder
        schema_mask: Tensor,  # (B, N) — graph-region mask
    ) -> tuple[Tensor, Tensor]:
        """Encode graph+path tokens.

        Returns:
            schema_tokens: (B, S, D)
            schema_mask_full: (B, S)
        """
        B, N, D = graph_tokens.shape
        T = self.t_max
        S = N + T

        # Path-region tokens
        path_query = self.E_path_query.weight.unsqueeze(0)  # (1, 1, D)
        path_query = path_query.expand(B, T, -1)  # (B, T, D)

        positions = torch.arange(T, device=graph_tokens.device)
        pos_emb = self.E_path_position(positions).unsqueeze(0)  # (1, T, D)

        region_emb = self.E_region_path.weight.unsqueeze(0)  # (1, 1, D)
        region_emb = region_emb.expand(B, T, -1)  # (B, T, D)

        path_tokens = (
            self.embedding_scale * (path_query + pos_emb) + region_emb
        )  # (B, T, D)

        # Concatenate graph + path
        schema_tokens = torch.cat([graph_tokens, path_tokens], dim=1)
        # (B, S, D)

        # Full mask: graph slots use schema_mask, path slots are always valid
        path_mask = torch.ones(
            B, T, dtype=torch.bool, device=graph_tokens.device
        )
        schema_mask_full = torch.cat([schema_mask, path_mask], dim=1)

        return schema_tokens, schema_mask_full


class PathProbeClassifier(nn.Module):
    """Per-position linear classifiers over path-region output slots.

    For each output position t (0..T-1), an independent linear head predicts
    a token from the path vocabulary (N_max + 2).
    """

    def __init__(
        self,
        n_max: int,
        t_max: int,
        hidden_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.n_max = n_max
        self.t_max = t_max
        self.vocab_size = n_max + 2

        self.heads = nn.ModuleList(
            [
                nn.Linear(
                    hidden_size, self.vocab_size, device=device, dtype=dtype
                )
                for _ in range(t_max)
            ]
        )

    def forward(self, path_slots: Tensor) -> Tensor:
        """Classify path-region output slots.

        Args:
            path_slots: (B, T, D) — path-region output slots from HRM.

        Returns:
            logits: (B, T, V) — per-position logits.
        """
        B, T, D = path_slots.shape
        logits = torch.zeros(B, T, self.vocab_size, device=path_slots.device)
        for t in range(T):
            logits[:, t, :] = self.heads[t](path_slots[:, t, :])
        return logits


# ---------------------------------------------------------------------------
# Main routing probe experiment
# ---------------------------------------------------------------------------


def run_routing_probe(
    adapter: SeqMazeProbeHRMV2BridgeAdapter,
    eval_dataset: SeqMazeRoutingDataset,
    *,
    zero_graph: bool = False,
) -> dict[str, float]:
    """Train per-position classifiers and evaluate path-prediction accuracy.

    After probe training (N=8, all-slot graph), we reinterpret 8 slots as
    4 graph + 4 path.  The probe encoder is reused for graph-region encoding.
    Path-region slots receive learned query embeddings.

    The backbone is frozen.  Four per-position linear classifiers are trained
    on the path-region output slots.

    Args:
        adapter: Probe-trained bridge adapter (backbone frozen).
        eval_dataset: Routing-mode evaluation dataset (n_max=4).
        zero_graph: If True, zero out graph-region input slots (control).

    Returns:
        Dict mapping position keys (e.g., 'pos_0', 'pos_1', 'mean') to accuracy.
    """
    loader = DataLoader(
        eval_dataset,
        batch_size=PROBE_BATCH_SIZE,
        collate_fn=_collate,
        shuffle=False,
    )

    # Reuse the probe-trained encoder for graph-region encoding (N=8 trained,
    # but we only feed 4 nodes; the unused slots 4-7 will be overwritten by
    # path-query embeddings below).
    graph_encoder = adapter._encoder  # SeqMazeProbeEncoder (N=8 config)

    # Path-region encoder (places learned query embeddings at slots 4-7)
    path_encoder = GraphPathEncoder(
        ROUTING_N_MAX, T_MAX, HIDDEN_SIZE, device=DEVICE
    )

    # Per-position path classifiers
    classifier = PathProbeClassifier(
        ROUTING_N_MAX, T_MAX, HIDDEN_SIZE, device=DEVICE
    )
    classifier.to(DEVICE)

    optimizer = torch.optim.AdamW(classifier.parameters(), lr=CLASSIFIER_LR)
    loss_fn = nn.CrossEntropyLoss(ignore_index=VOCAB_PATH - 1)  # ignore PAD

    # Freeze backbone
    adapter.eval()
    for param in adapter.parameters():
        param.requires_grad = False

    print(
        f"  Training {T_MAX} per-position classifiers "
        f"({'Experimental' if not zero_graph else 'Control (zeroed)'})..."
    )
    classifier.train()
    for epoch in range(CLASSIFIER_EPOCHS):
        total_loss = 0.0
        n_batches = 0
        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            targets = batch["target_path"]  # (B, T)
            target_mask = batch["path_mask"]  # (B, T)

            # Encode 4 graph nodes using the probe-trained encoder
            with torch.no_grad():
                graph_tokens, graph_mask = graph_encoder(
                    node_obs_id=batch["node_obs_id"],
                    node_candidate_index=batch["node_candidate_index"],
                    node_start_flag=batch["node_start_flag"],
                    node_goal_flag=batch["node_goal_flag"],
                    successor_indices=batch["successor_indices"],
                    successor_mask=batch["successor_mask"],
                    node_mask=batch["node_mask"],
                )
                # graph_tokens: (B, 4, D) — only 4 nodes used

                if zero_graph:
                    graph_tokens = torch.zeros_like(graph_tokens)

                # Pack graph + path into 8-slot schema
                schema_tokens, _ = path_encoder(graph_tokens, graph_mask)
                # schema_tokens: (B, 8, D)

            # Run HRM forward
            from ehc_sn.models.hrm.hrm_v2 import HRMInputV2

            with torch.no_grad():
                hrm_input = HRMInputV2(schema_tokens=schema_tokens)
                hrm_out, _ = adapter.model(hrm_input)

            # Extract path-region output slots (positions 4:8)
            path_slots = hrm_out.schema_slots[
                :, ROUTING_N_MAX : ROUTING_N_MAX + T_MAX, :
            ]

            # Classify
            logits = classifier(path_slots)  # (B, T, V)

            loss = loss_fn(logits[target_mask], targets[target_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / max(n_batches, 1)
            print(
                f"    {'Control' if zero_graph else 'Experimental'} "
                f"Epoch {epoch + 1:2d}/{CLASSIFIER_EPOCHS}  "
                f"loss={avg_loss:.4f}"
            )

    # Evaluate
    classifier.eval()
    correct = torch.zeros(T_MAX, dtype=torch.long, device=DEVICE)
    total = torch.zeros(T_MAX, dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            targets = batch["target_path"]
            target_mask = batch["path_mask"]

            graph_tokens, graph_mask = graph_encoder(
                node_obs_id=batch["node_obs_id"],
                node_candidate_index=batch["node_candidate_index"],
                node_start_flag=batch["node_start_flag"],
                node_goal_flag=batch["node_goal_flag"],
                successor_indices=batch["successor_indices"],
                successor_mask=batch["successor_mask"],
                node_mask=batch["node_mask"],
            )
            if zero_graph:
                graph_tokens = torch.zeros_like(graph_tokens)

            schema_tokens, _ = path_encoder(graph_tokens, graph_mask)
            hrm_input = HRMInputV2(schema_tokens=schema_tokens)
            hrm_out, _ = adapter.model(hrm_input)

            path_slots = hrm_out.schema_slots[
                :, ROUTING_N_MAX : ROUTING_N_MAX + T_MAX, :
            ]
            logits = classifier(path_slots)  # (B, T, V)
            preds = logits.argmax(dim=-1)  # (B, T)

            for t in range(T_MAX):
                pos_correct = (
                    (preds[:, t] == targets[:, t]) & target_mask[:, t]
                ).sum()
                pos_total = target_mask[:, t].sum()
                correct[t] += pos_correct
                total[t] += pos_total

    accuracies = {
        f"pos_{t}": (correct[t].item() / max(total[t].item(), 1))
        for t in range(T_MAX)
    }
    accuracies["mean"] = sum(accuracies.values()) / T_MAX
    return accuracies


def print_table(
    exp_acc: dict[str, float],
    ctrl_acc: dict[str, float],
) -> None:
    """Print the result table."""
    rand_baseline = 1.0 / VOCAB_PATH

    header = (
        f"{'Condition':<22} | {'Pos-0':^9} | {'Pos-1':^9} | {'Pos-2':^9} "
        f"| {'Pos-3':^9} | {'Mean':^9}"
    )
    sep = "-" * len(header)
    print()
    print(header)
    print(sep)
    print(
        f"{'Experimental (N=4)':<22} | {exp_acc['pos_0']:^9.3f} "
        f"| {exp_acc['pos_1']:^9.3f} | {exp_acc['pos_2']:^9.3f} "
        f"| {exp_acc['pos_3']:^9.3f} | {exp_acc['mean']:^9.3f}"
    )
    print(
        f"{'Control (zeroed)':<22} | {ctrl_acc['pos_0']:^9.3f} "
        f"| {ctrl_acc['pos_1']:^9.3f} | {ctrl_acc['pos_2']:^9.3f} "
        f"| {ctrl_acc['pos_3']:^9.3f} | {ctrl_acc['mean']:^9.3f}"
    )
    print(
        f"{'Random baseline':<22} | {rand_baseline:^9.3f} "
        f"| {rand_baseline:^9.3f} | {rand_baseline:^9.3f} "
        f"| {rand_baseline:^9.3f} | {rand_baseline:^9.3f}"
    )
    print()

    # Interpretation
    gap = exp_acc["mean"] - ctrl_acc["mean"]
    if exp_acc["mean"] > rand_baseline + 0.1 and gap > 0.05:
        print(
            "✅  Result: HRM routes graph information to path-region slots. "
            f"Experimental mean ({exp_acc['mean']:.3f}) >> "
            f"control ({ctrl_acc['mean']:.3f}) >> "
            f"random ({rand_baseline:.3f})."
        )
    elif exp_acc["mean"] > rand_baseline + 0.1 and gap <= 0.05:
        print(
            "⚠️  Result: Path slots carry information, but it may come from "
            "positional bias rather than graph routing. "
            f"Experimental ({exp_acc['mean']:.3f}) ≈ "
            f"control ({ctrl_acc['mean']:.3f}) > "
            f"random ({rand_baseline:.3f})."
        )
    elif exp_acc["mean"] > ctrl_acc["mean"] + 0.05:
        print(
            "⚠️  Result: Graph information propagates weakly. "
            f"Experimental ({exp_acc['mean']:.3f}) > "
            f"control ({ctrl_acc['mean']:.3f}) but "
            f"close to random ({rand_baseline:.3f})."
        )
    else:
        print(
            "❌  Result: No evidence of graph-to-path routing. "
            f"Experimental ({exp_acc['mean']:.3f}) ≈ "
            f"control ({ctrl_acc['mean']:.3f}) ≈ "
            f"random ({rand_baseline:.3f}). "
            "Investigate HRM architecture or path-query design."
        )


# =============================================================================
def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-probe", type=str, default=None,
                       help="Save probe-trained model checkpoint to this path")
    args = parser.parse_args()

    print("=" * 64)
    print("SEQMAZE ROUTING PROBE")
    print("=" * 64)
    print(f"Device: {DEVICE}")
    print(
        f"Probe: N={PROBE_N_MAX}, S={S} (all graph)\n"
        f"Routing: N={ROUTING_N_MAX}, T={T_MAX}, D={HIDDEN_SIZE}\n"
    )

    # Step 1: Train probe model
    adapter = train_probe_model()

    # Save probe checkpoint if requested
    if args.save_probe is not None:
        import os
        os.makedirs(os.path.dirname(args.save_probe) or ".", exist_ok=True)
        sd = adapter.model.state_dict()
        torch.save({"state_dict": sd}, args.save_probe)
        print(f"\nSaved probe checkpoint to {args.save_probe!r}")

    # Step 2: Evaluate experimental condition (graph info present)
    print("\n--- Routing probe: experimental condition ---")
    eval_dataset = SeqMazeRoutingDataset(
        n_max=ROUTING_N_MAX,
        k_max=K_MAX,
        t_max=T_MAX,
        seed_start=10000,
        n_samples=N_EVAL_SAMPLES,
        mode="routing",
    )
    exp_acc = run_routing_probe(adapter, eval_dataset, zero_graph=False)

    # Step 3: Evaluate control condition (graph region zeroed)
    print("\n--- Routing probe: control condition (graph zeroed) ---")
    ctrl_dataset = SeqMazeRoutingDataset(
        n_max=ROUTING_N_MAX,
        k_max=K_MAX,
        t_max=T_MAX,
        seed_start=20000,
        n_samples=N_EVAL_SAMPLES,
        mode="routing",
    )
    ctrl_acc = run_routing_probe(adapter, ctrl_dataset, zero_graph=True)

    # Step 4: Print results
    print("\n" + "=" * 64)
    print("RESULTS")
    print("=" * 64)
    print_table(exp_acc, ctrl_acc)


if __name__ == "__main__":
    main()
