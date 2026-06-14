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
