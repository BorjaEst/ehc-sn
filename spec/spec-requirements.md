# EHC-SN Requirements Specification

> Canonical source of truth for repo-level constraints, gating rules, and
> dependency contracts.

## 1 Spec Gate (Mandatory for Agents)

Before generating any design or code artifact, automated agents **MUST**:

1. Verify `spec/spec-manifest.toml` exists.
2. Verify every file listed under `required.files` in that manifest exists and
   is non-placeholder.
3. If any are missing, **STOP** and output exactly:
   `Blocking: missing required specs: <comma-separated list of missing paths>`

The only exception is the **Bootstrap Mode** override defined in the
`bootstrap-specs.prompt.md` prompt, which operates before specs exist.

---

## 2 Namespace Constraint

All new code under `src/ehc_sn/` **MUST** use `ehc_sn` as the sole import
namespace. Forbidden: `torch_tem`, `hrm_sn`. Legacy code is archived
under `temp/` (not on the Python path).

---

## 3 Schema and Contract Constraints

- **Do not invent** new data schemas, type aliases, or configuration contracts
  without first checking whether an equivalent exists in:
  - `ehc_sn/types.py` (type aliases and dataclasses)
  - Existing Pydantic configs in the owning component
- **Extend** existing types rather than creating parallel hierarchies.
- New type aliases in `types.py` must include a docstring with shape conventions
  (following the existing pattern in that file).
- New Pydantic configs must use `extra="forbid"` unless there is an explicit,
  documented reason for leniency.

### 3.1 Policy Contracts

- Reusable policy interfaces and policy configuration contracts belong in
  `ehc_sn/policies/` once they are intended for cross-controller reuse.
- Policy APIs must consume typed rollout-state views or documented tensor
  mappings, not arbitrary controller internals.
- If a canonical model defines an explicit BG or arbitration module, policies
  must consume that module's declared outputs and must not themselves perform
  loop-level bundle-score integration or subsume STR/BG responsibilities.
- Hard action constraints must be represented explicitly in the policy input.
  Environment feasibility, logical compatibility, and other admissibility
  conditions must not be left implicit in downstream losses.
- When action semantics are structured, prefer typed channel families or
  documented admissible-bundle mappings over opaque flat encodings.
- Policy logits used for action selection must remain conceptually distinct
  from reward-prediction or state-value heads unless a canonical spec documents
  a deliberate combined head.
- STR contracts remain narrow by default: reward prediction and optional
  reward-sensitive bias signals. Full bundle arbitration, admissibility-owned
  score integration, and generic sampling semantics belong to an explicit BG /
  arbitration module or to the reusable policy layer according to the canonical
  architecture spec.
- Policy objects own policy-local randomness and sampling semantics; controllers
  may pass mode flags (for example train/eval) but must not duplicate policy
  sampling logic.
- Policy configs must use `extra="forbid"` unless there is an explicit,
  documented compatibility reason.

### 3.2 Figure Contracts

- The canonical public API of `ehc_sn.figures` is limited to registry/context contracts, built-in registration, sinks, and `ehc_sn`-native figure modules.
- A public figure module must define one primary figure class derived from `BaseFigureTemplate` and may expose a thin `plot(...)` convenience wrapper.
- A figure module is public only if it uses the `ehc_sn` namespace exclusively and is exported from the component or registered as a built-in figure.
- Modules under `src/ehc_sn/figures/` that still import `torch_tem` are migration inventory, not public API.

---

## 4 Dependency Constraints

### 4.1 Runtime Dependencies (declared in `pyproject.toml`)

| Dependency           | Role                    | Notes                                   |
| -------------------- | ----------------------- | --------------------------------------- |
| `torch`              | Core tensor computation | Pin-free (user-managed CUDA compat)     |
| `torchrl`            | TorchRL environments    | Controller-owned environment stepping   |
| `lightning`          | Training orchestration  | LightningModule, Trainer, Callbacks     |
| `gymnasium`          | Environment interface   | Maze environments                       |
| `pydantic`           | Configuration schema    | `BaseModel(extra="forbid")` pattern     |
| `pydantic_settings`  | CLI settings            | `BaseSettings(cli_parse_args=True)`     |
| `scipy`              | Scientific computing    | Combinatorics, special functions        |
| `matplotlib`         | Visualization (base)    | Figure rendering                        |
| `SciencePlots`       | Visualization (styles)  | Publication-ready plot styles           |
| `pub-ready-plots`    | Visualization (layout)  | Publication-ready plot utilities        |
| `tensorboard`        | Logging                 | Metric/scalar/image logging             |
| `rich`               | Terminal output         | Progress bars, formatted console output |
| `huggingface_hub`    | Dataset download        | Raw/source dataset retrieval            |
| `maze-nd`            | Maze generation         | Generator dependency used by scripts    |
| `dungeongen`         | Dungeon generation      | Generator dependency used by scripts    |
| `typer`              | Script CLIs             | Data-generation command-line interface  |
| `adam-atan2-pytorch` | Optimizer               | AdamAtan2 for HRM training              |
| `setuptools`         | Build backend           | Package build and version management    |

### 4.2 Audit Candidates

- `networkx`: Declared in `pyproject.toml` but currently has zero imports in `src/`. Audit before next release and remove if it remains unused.
- `maze-nd`: Declared as a runtime dependency but currently used in `scripts/data-gen/` only. Review whether it should move to a script-only optional dependency.
- `dungeongen`: Declared as a runtime dependency but currently used in `scripts/data-gen/` only. Review whether it should move to a script-only optional dependency.
- `huggingface_hub`: Declared as a runtime dependency but currently used in `scripts/data-gen/` only. Review whether it should move to a script-only optional dependency.
- `typer`: Declared as a runtime dependency but currently used in `scripts/data-gen/` only. Review whether it should move to a script-only optional dependency.

### 4.3 Known Dependency Bugs

No known dependency declaration mismatches are currently recorded.

### 4.4 Dev / Script-Only Dependencies

- `pytest`: Testing, used in `tests/`.
- `black`: Formatting, used in dev tooling.
- `flake`: Linting, used in dev tooling.
- `mypy`: Type checking, used in dev tooling.

### 4.5 New Dependency Policy

- New runtime dependencies require justification (why existing deps cannot
  serve the need).
- New dependencies must be added to `pyproject.toml` in the appropriate section.
- Prefer pure-Python or well-maintained packages with compatible licenses.

---

## 5 Data Handling Constraints

- **Pipeline**: `data/raw/` → `data/interim/` → `data/processed/`.
  DataModules consume only `processed` data.
- Raw data is **not** committed to version control.
- Processing scripts live in `scripts/data-gen/`.
- `data/interim/` is optional scratch space; `data/external/` is for
  third-party datasets.
- No hard-coded absolute paths. Use Pydantic settings or CLI args.

---

## 6 Python and Runtime

- **Python**: ≥ 3.12 (as declared in `pyproject.toml`).
- **Build backend**: `setuptools` with `pyproject.toml`-based configuration.
- **Package layout**: `src/` layout (`tool.setuptools.package-dir = {"" = "src"}`).
- **Packages**: `ehc_sn` is the only active first-party package.
- **Version**: Single-source in `src/ehc_sn/VERSION`.

---

## 7 Training Infrastructure Constraints

All code in `training/` is **model-agnostic**: it must not import from
`models/`, `modules/`, or `lightning/`. `training/` owns generic optimization,
step-loop, scheduling, buffer, and loss-support primitives only.

All executable model-specific training orchestration lives in
`lightning/`. This includes Lightning trainers, optimizer assembly, scheduler
assembly, trainer-local checkpoint restore logic, and training/validation
hooks.

All reusable benchmark code lives in `benchmarks/`.

Within `benchmarks/`, `_capabilities/` and `_infra/` remain
**model-agnostic**. They own only benchmark capability contracts and
mechanical benchmark support such as artifact writing, seeding, timing, and
shared result types. These subareas must not import from `models/` or
`lightning/`.

Within `benchmarks/`, `_bindings/` is the only model-aware benchmark subarea.
It may import from `models/` and lower reusable layers to adapt pure models to
benchmark capability protocols, perform observation-to-tensor conversion, and
hydrate resolved checkpoint payloads. `_bindings/` must not own benchmark
semantics, evaluator loops, or benchmark-specific metrics.

Benchmark-semantic packages under `benchmarks/` (for example `b0/`-`b3/`)
own evaluator loops, manifests, metric interpretation, and benchmark-specific
protocols such as B2 frozen-weight exposure/probe rules. These packages remain
model-agnostic and must not import from `models/` or `lightning/`.

All code in `controllers/` and `heads/` is also **model-agnostic**: it must not
import from `models/`. Controllers own rollout carry/state and step orchestration;
heads own step-local loss composition and metric aggregation. Heads may depend on
`controllers/`, `loss/`, `metrics/`, and `training/` primitives, but `training/`
must remain usable without importing from `controllers/` or `heads/`.

All code in `policies/` is **model-agnostic** and **controller-agnostic**:
it must not import from `models/`, `controllers/`, `heads/`, `training/`, or
`modules/`. Policies may depend on external tensor/runtime libraries and on
lightweight shared contracts such as `ehc_sn.types` and `ehc_sn.utils`.

Policies own reusable action-selection behavior over explicit rollout-state
views. Controllers remain responsible for rollout lifecycle and environment
stepping; environments remain responsible for transition dynamics and action
validation.

---

## 8 Evaluation Protocol Constraints

The canonical EHC benchmark contract is defined by the benchmark suite in
`spec/spec-architecture.md`, which is divided into bridge benchmarks (`B0`,
`M0`) and primary navigation benchmarks (`B1`-`B3`).

- Claims about preserving **HRM-style deliberative capability** may cite the
  B0 HRM Deliberation Bridge as supporting evidence.
- Claims about preserving **TEM-style episodic-memory capability** may cite the
  M0 Episodic Memory Bridge as supporting evidence.
- Claims about **within-episode navigation reasoning** for EHC must be
  supported by the B1 Dungeon Navigation Reasoning benchmark. Bridge benchmark
  results do not replace B1 for navigation-centered claims.
- Claims about **one-shot adaptation** must be supported by the B2 One-Shot
  Goal Relocation benchmark.
- Claims that the EHC decomposition reduces interference or improves controlled
  memory routing must be supported by the B3 Interference and Control
  benchmark.
- B1 uses the processed dungeon split as the in-distribution corpus
  (`800/100/100` train/val/test layouts) plus four OOD generated corpora of
  `100` layouts each: `medium/classic`, `large/classic`, `small/temple`, and
  `small/cavern`.
- B2 and B3 must precompute, for each layout, a six-goal / three-start
  contract from the largest connected component: canonical start `1` plus
  probe starts `2-3`. Goal selection must enforce
  shortest-path distance at least `8` from the canonical start and at least `6`
  between selected goals. Probe starts must be reachable and have shortest-path
  distance at least `6` from every selected goal.
- B2 training uses goals `1-4`. Held-out one-shot evaluation uses goals `5-6`.
  The one-shot protocol is: one rewarded exposure episode from start `1`, then
  immediate probe episodes from starts `2-3`.
- M0 reuses the B1 processed dungeon split and OOD corpora together with the
  B2/B3 six-goal / three-start contract.
- Canonical M0 evaluation uses benchmark-owned shortest-path exposure traces
  from start `1` to held-out goals `5-6`, followed by fixed probe traces from
  starts `2-3`.
- M0 reports current-location localization, held-out-goal recall,
  write/read consistency from exposure to probe, and interference under
  sequential held-out-goal exposures.
- Any model family evaluated on B1-B3 must expose benchmark-time action
  selection either through a native policy/action head or through an explicit
  attached policy layer satisfying the rollout contract. Memory or world-model
  backbones that only update latent state do not qualify as standalone
  navigation agents.
- For B1, B2, and B3 specifically, the reported rollout contract may compose a
  recurrent backbone/state updater with a separate policy layer, but
  benchmark-facing docs and reports must state explicitly which layer owns
  action selection.
- M0 is a bridge benchmark over benchmark-owned trajectory or exposure/probe
  contracts. It measures memory/state capability rather than full navigation
  competence and must not be reported as a substitute for B1-B3.
- During M0 evaluation, no benchmark-time action selection, optimizer step,
  gradient update, or learned-weight mutation is allowed. Only online
  fast-memory state and other explicitly declared recurrent state may change.
- If a benchmark adapter or readout for M0 is learned, it must be fit only on
  training layouts and goals `1-4`; held-out evaluation on goals `5-6` must
  remain frozen.
- During B2 and B3 evaluation, **no optimizer step, gradient update, or weight
  mutation is allowed**. Only online fast-memory state and other explicitly
  declared ephemeral rollout state may change.
- All canonical benchmark reports must use `5` independent training seeds and
  report mean, `95%` confidence interval, and per-seed scatter.
- Canonical benchmark reports must include fixed internal-compute budgets of
  `4`, `8`, and `16` micro-steps or recurrent cycles when the model exposes
  adaptive internal computation.
- Any paper or report claiming strong ML / RL performance for EHC must include
  at least the following baseline pack on the relevant benchmark: one HRM-style
  recurrent baseline, one memory-disabled or `no-HPC-write` ablation, one
  pooled-cue or fused-retrieval ablation, and one generic RL baseline when the
  claim is framed as broad RL competence.
- Benchmark orchestration code intended for reuse across B0, M0, and B1-B3 belongs in
  `ehc_sn.benchmarks`.
- Model-aware benchmark bindings, observation-to-tensor conversion, and
  benchmark-time checkpoint hydration belong in `ehc_sn.benchmarks._bindings`.
- Benchmark-semantic packages under `ehc_sn.benchmarks` must not dispatch on
  `model_kind`, instantiate concrete training models, or load checkpoints
  directly.
- Canonical benchmark entrypoints belong under `scripts/benchmarks/` as thin
  wrappers that parse user input, resolve checkpoint paths and binding
  selection, and delegate to evaluators in `ehc_sn.benchmarks`.
- Scripts under experiments/ are exploratory or paper-specific and must not 
  duplicate shared benchmark job planning, runtime resolution, or
  artifact-writing logic.

---

## 9 Cross-Component Change Policy

Code changes that cross component boundaries require a tracked plan in
`.copilot-tracking/plans/` **before** implementation begins. This includes:

- Adding a new top-level package under `ehc_sn/`.
- Moving code between components.
- Changing the public API of a component that other components depend on.
- Adding or removing a runtime dependency.

For new top-level packages, the plan must also update `spec/spec-architecture.md`
so the component taxonomy remains complete and dependency boundaries are explicit.

Single-component changes (bug fixes, internal refactors, new private helpers)
do not require a plan unless they alter the component's public contract.
