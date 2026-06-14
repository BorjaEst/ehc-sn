# TODO

## deferred-post-chunk-hebbian-clamp

**Status:** not-started

**Location:** `src/ehc_sn/lightning/modules/variational_replay.py` lines 501–515
(post-chunk hook) and `src/ehc_sn/modules/hpc/modules.py` line 154
(`finalize_memory`).

**Problem:** HPC Hebbian memory matrices are hard-clamped to `[clamp_min,
clamp_max]` **outside** the computation graph, once per BPTT chunk boundary.
The clamp runs after `.detach()` — gradients never see it, creating a
discontinuity at chunk boundaries and breaking gradient-based optimization
of the Hebbian dynamics.

**Long-term fix:** Move clamping **inside** the computation graph, applied
at every step, so that:

- Gradients flow through the bounded operation.
- No discontinuity when the carry crosses a BPTT chunk boundary.
- The model owns its state invariants — no manual post-chunk state surgery.

**Preferred approach:** Replace hard `torch.clamp` with a differentiable
bound (e.g. `tanh`-scaled or soft-clamp) applied per-step inside
`_update_memory_impl` / `clamp_memory`. Remove `finalize_memory` and the
post-chunk hook entirely once the per-step bound is in place.

**Acceptance criteria:**

- `grep -rn "finalize_memory" src/` returns zero matches.
- `grep -rn "Deferred post-chunk Hebbian clamp" src/` returns zero matches.
- HPC memory values stay bounded `[clamp_min, clamp_max]` across BPTT
  boundaries without manual intervention.
- No training regression on mazehard / arena parity benchmarks.

## hpc-prefix-bias-as-instance-context

**Status:** not-started

**Location:**

- `src/ehc_sn/modules/pfc/__init__.py` lines 340–345 (`prefix_bias` on CLS token, currently unused by adapters)
- `src/ehc_sn/adapters/hrm/_base.py` lines 55–130 (token encoder, no instance-level conditioning)
- `src/ehc_sn/adapters/hrm/mazehard.py` lines 65–70 (`_make_input_v1` always passes `prefix_bias=None`)
- `src/ehc_sn/modules/hpc/query_policy.py` (cue-family retrieval operators)
- Legacy reference: `legacy_hrm/models/hrm/hrm_act_v1.py` lines 140–160 (puzzle embedding prepend)

**Problem:** Legacy HRM uses per-instance puzzle embeddings (a learned lookup
table) to inject instance identity into the reasoning module. The modern PFC
has equivalent plumbing — `prefix_bias` on the CLS token — but adapters never
populate it. Without instance-level context, the model must infer which maze it's
solving from tokens alone, burning capacity on identification rather than
reasoning. A learned lookup table (legacy approach) is not a satisfactory
solution: it requires instance IDs at test time and does not generalize to
unseen instances.

**Research direction (target: EHP implementation):** Use HPC episodic retrieval
as the source of `prefix_bias`:

1. Encode the maze observation into an HPC sensory query cue (`"x"` family).
2. Retrieve the most similar episodic memory via attractor or attention read.
3. Feed the recalled place code as `prefix_bias` into `PFCModel.step()`.

This replaces instance-ID lookup with content-addressable memory — the model
retrieves context based on structural similarity, not a hardcoded ID. It
matches the biological HPC→PFC pathway (hippocampal contextual modulation of
prefrontal working memory) and generalizes to unseen instances at test time.

**Key design decisions to resolve:**

- What does the HPC store during training? Raw observations, structural features, or solution-path embeddings?
- Cue encoding: derive query from observation tokens or from an intermediate PFC representation?
- Read policy: single-shot cue read (`CueRead`) vs targeted read with source/target families (`TargetRead`)?
- Does the HPC memory persist across episodes (cumulative experience) or reset per episode?

**Acceptance criteria:**

- `prefix_bias` is populated by HPC retrieval in at least one HRM adapter (mazehard).
- `grep -rn "prefix_bias=None" src/ehc_sn/adapters/` returns zero matches for the adapted task.
- Mazehard training with HPC context matches or exceeds baseline (no prefix_bias) on sequence-exact accuracy.
- The mechanism uses content-addressable retrieval, not instance-ID lookup (no embedding table keyed by puzzle index).
