# EHC-SN Benchmark Suite Specification

> Non-required companion spec for canonical benchmark definitions, corpus
> contracts, execution rules, and benchmark-specific reporting constraints.

## 1 Scope

This spec defines the detailed benchmark suite used to support architectural
claims for EHC-SN. Core benchmark ownership rules remain in
`spec/spec-architecture.md` and `spec/spec-requirements.md`.

---

## 2 Canonical Benchmark Family

The canonical benchmark family is divided into two bridge benchmarks and three
primary goal-directed navigation benchmarks.

| Benchmark                           | Purpose                                                                                                                                                | Canonical split / protocol                                                                                                                                              | Current implementation status                                                                        |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| **B0 HRM Deliberation Bridge**      | Bridge benchmark for HRM-style deliberative batch prediction over full-maze token targets.                                                             | Use the processed `maze-30x30-hard-1k` split: `1000/1000/1000` train/val/test 30x30 mazes. Report full test plus a preregistered hard subset derived from `difficulty`. | Partially scaffolded by `envs/mazehard.py`, HRM entry points, and `scripts/benchmarks/b0_bridge.py`. |
| **M0 Episodic Memory Bridge**       | Bridge benchmark for TEM-style structural memory, sensory binding, episodic write/read, and one-shot memory behavior under fixed trajectory contracts. | Reuse the B1 processed dungeon split and OOD corpora together with the B2/B3 six-goal / three-start contract.                                                           | Not yet implemented.                                                                                 |
| **B1 Dungeon Navigation Reasoning** | Main within-episode reasoning benchmark for goal-directed navigation.                                                                                  | Train on the processed dungeon split: `800/100/100` train/val/test layouts, then evaluate in-distribution and on OOD generated corpora.                                 | Requires a goal-reaching reward/binding layer on top of the dungeon processed-data contract.         |
| **B2 One-Shot Goal Relocation**     | Main across-episode one-shot adaptation benchmark.                                                                                                     | Reuse B1 layouts with a six-goal / three-start contract. Train on goals `1-4`, evaluate held-out goals `5-6`.                                                           | Requires explicit frozen-weight one-shot evaluation support.                                         |
| **B3 Interference and Control**     | Main mechanism benchmark for complementary-memory claims.                                                                                              | Reuse B1 layouts with the same six-goal / three-start contract and blocked vs interleaved schedules.                                                                    | Requires retrieval/control diagnostics on top of rollout traces.                                     |

Bridge benchmarks validate inherited predictive or memory capabilities in
isolation. Primary benchmarks evaluate complete goal-directed navigation
agents.

---

## 3 Corpus and Protocol Details

### 3.1 B1 Corpus

- In-distribution corpus: processed dungeon split with `800/100/100`
  train/val/test layouts.
- OOD corpora: `100` layouts each for `medium/classic`, `large/classic`,
  `small/temple`, and `small/cavern`.

### 3.2 B2 and B3 Goal Contract

For each layout, precompute a six-goal / three-start contract from the largest
connected component:

- canonical start `1` plus probe starts `2-3`;
- selected goals must be at least distance `8` from canonical start;
- selected goals must be at least distance `6` from each other;
- probe starts must be reachable and at least distance `6` from every selected
  goal.

### 3.3 B2 One-Shot Protocol

- Training goals: `1-4`.
- Held-out evaluation goals: `5-6`.
- One-shot protocol: one rewarded exposure episode from start `1`, followed by
  immediate probe episodes from starts `2-3`.

### 3.4 M0 Protocol

- M0 reuses the B1 processed dungeon split and OOD corpora together with the
  B2/B3 six-goal / three-start contract.
- Canonical M0 evaluation uses benchmark-owned shortest-path exposure traces
  from start `1` to held-out goals `5-6`, followed by fixed probe traces from
  starts `2-3`.
- M0 reports current-location localization, held-out-goal recall,
  write/read consistency from exposure to probe, and interference under
  sequential held-out-goal exposures.

---

## 4 Execution Constraints

- Claims about preserving **HRM-style deliberative capability** may cite B0.
- Claims about preserving **TEM-style episodic-memory capability** may cite M0.
- Claims about **within-episode navigation reasoning** must be supported by B1.
- Claims about **one-shot adaptation** must be supported by B2.
- Claims about **interference reduction or controlled memory routing** must be
  supported by B3.
- Bridge benchmark results do not substitute for B1-B3 when the claim is about
  navigation.
- Bridge benchmark results on navigation-grounded predictive models do not by
  themselves classify a model family as a navigation agent.
- Any model family evaluated on B1-B3 must expose benchmark-time action
  selection either through a native policy/action head or through an explicit
  attached policy layer satisfying the rollout contract.
- For B1-B3, reports must state explicitly which layer owns action selection.
- M0 is a memory/state benchmark, not a navigation benchmark.
- During M0 evaluation, no benchmark-time action selection, optimizer step,
  gradient update, or learned-weight mutation is allowed. Only declared
  fast-memory state and other recurrent state may change.
- If an M0 adapter or readout is learned, it must be fit only on training
  layouts and goals `1-4`; held-out evaluation on goals `5-6` must remain
  frozen.
- During B2 and B3 evaluation, no optimizer step, gradient update, or weight
  mutation is allowed. Only declared ephemeral rollout state may change.

---

## 5 Reporting Constraints

- All canonical benchmark reports must use `5` independent training seeds.
- Report mean, `95%` confidence interval, and per-seed scatter.
- Adaptive-computation models must include results at internal-compute budgets
  `4`, `8`, and `16` in addition to any unconstrained best result.
- Reports claiming strong ML or RL competence for EHC must include at least one
  HRM-style recurrent baseline, one memory-disabled or `no-HPC-write`
  ablation, one pooled-cue or fused-retrieval ablation, and one generic RL
  baseline when the claim is framed broadly.
- Repository-level reporting rules in `spec/spec-standards.md` still apply.

---

## 6 Ownership Notes

- Benchmark orchestration code intended for reuse across B0, M0, and B1-B3
  belongs in `ehc_sn.benchmarks`.
- Model-aware benchmark bindings, observation-to-tensor conversion, and
  benchmark-time checkpoint hydration belong in `ehc_sn.benchmarks._bindings`.
- Benchmark-semantic packages must not dispatch on `model_kind`, instantiate
  concrete training models, or load checkpoints directly.
