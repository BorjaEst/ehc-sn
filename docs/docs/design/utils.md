# Utility Architecture

> Canonical design for `ehp_sn.utils` — a small foundational package for
> **domain-neutral, dependency-light technical primitives**.

`ehp_sn.utils` contains low-level mechanisms that have no legitimate domain
owner. It must not encode scientific or application policy, must import no
other `ehp_sn` domain package, and must have no import-time side effects.

---

## 1. Package contract

### 1.1 Definition

A professional `utils` package satisfies:

| Property                        | Requirement                                                                                                                                     |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| **Domain-neutrality**           | Knows nothing about TEM, HRM, MEC, HPC, LEC, Arena, MazeHard, ACT, rollouts, traces, objectives, metrics, checkpoints, experiments, or task IDs |
| **Dependency direction**        | Depends only on stdlib, NumPy, PyTorch, and narrow infrastructural libraries. Must **never** import any `ehp_sn` domain package                 |
| **Capability-orientation**      | Submodule names communicate concrete technical capabilities (`tensors`, `trees`) — never concealment names (`helpers`, `common`, `misc`)        |
| **No import-time side effects** | Importing `ehp_sn.utils` must not initialize CUDA, configure logging, create directories, register models, or read configuration                |
| **Small root API**              | Root `__init__.py` exposes only submodules, not dozens of flattened symbols                                                                     |
| **Internal by default**         | `ehp_sn.utils` is infrastructure for EHP subsystems, not a user-facing API                                                                      |

### 1.2 Admission rule

A function belongs in `ehp_sn.utils` only when **all** of the following hold:

1. **Domain-neutral** — it does not know about any EHP scientific concept.
2. **No policy** — it implements a mechanism, not a decision.
3. **Multiple independent consumers** — used by at least two top-level
   packages, or clearly an infrastructural primitive.
4. **Narrow dependency cone** — depends only on stdlib / NumPy / PyTorch.
5. **Explicit invariants** — shape, mutation, device, dtype, failure, and
   gradient behaviour are documented and tested.

Submodules are created **only when current code demonstrates a need**. An
empty taxonomy is not architecture. The admission rule applies equally to
new submodules and new symbols within existing submodules.

### 1.3 What does NOT belong in utils

| Excluded category                       | Examples                                                                         | Owner                          |
| --------------------------------------- | -------------------------------------------------------------------------------- | ------------------------------ |
| **Domain contracts**                    | `TaskRuntime`, `LocationBelief`, `RolloutBatch`, `TraceSpec`                     | `contracts/`                   |
| **MEC/HPC logic**                       | `inv_var_trans`, `connections`, `resolve_ovc_slice`, `update_to_masks`           | `modules/mec/`, `modules/hpc/` |
| **Projection construction**             | `create_downsample_matrix`, `create_tiling_matrices`, `create_random_projection` | `modules/projection.py`        |
| **Loss & reduction semantics**          | `squared_error`, `cross_entropy`, `reduce_per_env`                               | `objectives/`                  |
| **Evaluation policy**                   | `snapshot_state_dict`, `FrozenEvalMutationError`                                 | `evaluation/`                  |
| **Directory conventions**               | `make_directories`, `set_directories`, `resolve_envs_path`                       | `training/`, `configuration/`  |
| **Logging setup**                       | `make_logger`                                                                    | `logging/`                     |
| **Task graph generation**               | `generate_hamiltonian_dag`, `generate_transition_dag`                            | `data/substrate/`              |
| **Task permutation & digests**          | `remap_obs_ids`, `canonical_dag_digest`                                          | `data/substrate/`, `tasks/`    |
| **Configuration merging**               | `apply_overrides`                                                                | `configuration/`               |
| **Wrappers over `torch.nn.functional`** | `softmax`, `relu`, `leaky_relu`, `activation_from_str`                           | Remove; use PyTorch directly   |

### 1.4 Dependency direction

```
stdlib, numpy, torch, scipy (narrow)
        ▲
        │
ehp_sn.utils
        ▲
        │
contracts / data / modules / models / controllers /
objectives / rollouts / training / evaluation / traces / figures
```

Enforced invariant — allowed and forbidden:

```python
# Allowed
ehp_sn.models      -> ehp_sn.utils
ehp_sn.training    -> ehp_sn.utils
ehp_sn.traces      -> ehp_sn.utils

# Forbidden
ehp_sn.utils       -> ehp_sn.contracts
ehp_sn.utils       -> ehp_sn.models
ehp_sn.utils       -> ehp_sn.training
ehp_sn.utils       -> ehp_sn.modules
ehp_sn.utils       -> ehp_sn.data
ehp_sn.utils       -> ehp_sn.evaluation
```

The critical defect in the current codebase is: `utils/__init__.py` imports
`from ehc_sn.types import LocationBelief, ...`, reversing the intended
dependency. After migration, `types.py` (or the contract layer) imports from
`utils`, never the reverse.

---

## 2. Package structure

### 2.1 Immediate target

Create only what current code justifies:

```
src/ehp_sn/utils/
├── __init__.py          # Exposes submodules only — no flattened symbols
├── tensors.py           # Shape/rank/device/dtype validation, masked reductions,
│                        #   structural row ops, generic reductions
├── trees.py             # Recursive traversal, detach, clone, device movement,
│                        #   tree_merge_rows; internal pytree facade
└── graph.py             # Generic graph algorithms (BFS, path counting, DAG layout)
```

### 2.2 Conditional modules — create only when demanded

These modules are approved in the taxonomy but must **not** be created
before demonstrated need:

| Module                | Trigger condition                                                                                  |
| --------------------- | -------------------------------------------------------------------------------------------------- |
| `randomness.py`       | Concrete consumers requiring `derive_seed` and `make_torch_generator` across at least two packages |
| `imports.py`          | Multiple packages implementing duplicated optional-dependency detection logic                      |
| `symmetry.py`         | Multiple cross-package consumers (data, figures, evaluation, model tests) of dihedral transforms   |
| `_internal/pytree.py` | Genuine need to isolate PyTorch pytree internals from the public `trees.py` facade                 |

`norms.py` is not retained as a first-class module. The single function
`rms_norm` either lives in `tensors.py` or is removed if PyTorch provides
an equivalent.

### 2.3 Rejected names

These must never exist in `ehp_sn/utils/`:

`common.py`, `helpers.py`, `misc.py`, `general.py`, `math.py`, `config.py`,
`logging.py`, `io.py`, `model_utils.py`, `training_utils.py`, `paths.py`.

---

## 3. Submodule responsibilities

### 3.1 `utils.tensors` — Tensor validation and structural operations

**May know about**: `torch.Tensor`, shapes, ranks, devices, dtypes, masks
(as generic boolean tensors), generic tensor rows, initialization
distributions.

**Must not know about**: MEC frequency bands, HPC memory, rollout carry,
task batches, environment dimensions, trace fields, objective semantics,
model-specific axes.

**Public API**:

```
# Error hierarchy
TensorValidationError        # Base class
  TensorShapeError           # Shape / rank mismatch
  TensorDeviceError          # Device mismatch
  TensorDTypeError           # Dtype mismatch

# Validation (returns tensor; no mutation, no conversion)
require_shape(tensor, expected: Sequence[int|None], *, name: str) -> Tensor
require_rank(tensor, rank: int, *, name: str) -> Tensor
require_same_device(*tensors: Tensor) -> torch.device
require_dtype(tensor, dtype: torch.dtype, *, name: str) -> Tensor
require_finite(tensor, *, name: str) -> Tensor

# Generic reductions
masked_sum(tensor, mask, *, dim) -> Tensor
masked_mean(tensor, mask, *, dim) -> Tensor

# Structural operations  (document rank of flag, semantics of True)
expand_row_mask(flag: Tensor, ref: Tensor) -> Tensor
merge_rows(flag: Tensor, current: Tensor, fresh: Tensor) -> Tensor
split_by_sizes(tensor: Tensor, sizes: Sequence[int]) -> list[Tensor]
```

**Symbols requiring explicit justification before inclusion**:

| Symbol                | Concern                                         | Decision                                                                               |
| --------------------- | ----------------------------------------------- | -------------------------------------------------------------------------------------- |
| `find_multiple(n, k)` | Generic integer arithmetic, not tensor-related. | Keep package-local where used unless multiple non-tensor consumers emerge.             |
| `trunc_normal_init_`  | Initialization may belong in `modules/`.        | Include only if consumed by multiple independent packages (models, adapters, modules). |

**Contract specification for `merge_rows`**:

- `flag`: boolean tensor of shape `(B,)`; `True` selects the `fresh` row,
  `False` selects `current`.
- `current`, `fresh`: same shape `(B, ...)`.
- Returns `torch.where(expand_row_mask(flag, current), fresh, current)`.
- Mixed dtypes are rejected.
- Broadcasting beyond the leading dimension follows PyTorch semantics.

**Boundary examples**:

```python
# Belongs in utils.tensors
require_shape(x, (batch, time, None))

# Does NOT belong in utils.tensors
validate_tem_memory_state(g, x, p)
validate_rollout_batch(batch)
```

### 3.2 `utils.trees` — Nested-structure traversal

**Purpose**: Traverse and transform arbitrary supported nested structures
without understanding their semantic meaning.

**Supported structures**: `dict`, `list`, `tuple`, `namedtuple`, registered
dataclasses, PyTorch pytree-compatible nodes, `torch.Tensor` and scalar
leaves.

**Public API**:

```
# Traversal
tree_map(fn: Callable[[Any], Any], tree: T) -> T
tree_leaves(tree: Any) -> list[Any]

# Tensor-specific structural operations
tree_detach(tree: T) -> T
tree_to(tree: T, *, device, dtype) -> T
tree_to_cpu(tree: T) -> T

# Conditional row merge through nested structures
tree_merge_rows(flag: Tensor, current: T, fresh: T) -> T
```

**Symbols deferred until consumer demand exists**:

| Symbol               | Reason for deferral                                                                                                                                                                                                |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `tree_map_with_path` | Add when path-aware traversal is needed by multiple consumers.                                                                                                                                                     |
| `tree_iter`          | Add when lazy iteration over heterogeneous trees is needed.                                                                                                                                                        |
| `tree_clone`         | Add when `tree_map` with an identity-like clone is a proven bottleneck.                                                                                                                                            |
| `DetachMixin`        | The mixin is inheritance-based convenience. `tree_detach(state)` is usually sufficient. Retain only if it materially improves type safety or eliminates field-by-field boilerplate in a way free functions cannot. |

**Internal layering**: The public facade wraps PyTorch pytree facilities.
Flattening, path normalization, and spec comparison are implementation
details. If isolation from PyTorch's private `_pytree` API becomes necessary,
create `utils/_internal/pytree.py` — but not before.

**PyTree risk**: `torch.utils._pytree` is a private or semi-private API that
may change across PyTorch versions. The facade must remain deliberately
minimal to limit exposure. Do not expose PyTorch's entire pytree model
through EHP wrappers.

**Important distinction**:

```python
# Generic mechanism — belongs in utils.trees
tree_detach(state)

# Semantic operation — belongs in training/ or rollouts/
detach_recurrent_carry_at_tbptt_boundary(carry)
```

### 3.3 `utils.graph` — Generic graph algorithms

**Purpose**: Graph algorithms that operate on abstract adjacency structures
and do not know why the graph exists.

**Public API**:

```
shortest_path(adjacency: Sequence[list[int]], start: int, goal: int) -> list[int]
count_shortest_paths(adjacency: Sequence[list[int]], start: int, goal: int) -> int
compute_layered_dag_positions(*, num_nodes: int, edges: Sequence[tuple[int,int]]) -> NDArray
```

**What does NOT belong here**:

- `generate_hamiltonian_dag` — dataset generation policy. Owner: `data/substrate/`.
- `generate_transition_dag` — dataset generation policy. Owner: `data/substrate/`.
- `remap_obs_ids`, `permute_candidate_order` — task permutation logic. Owner: `data/substrate/`.
- `canonical_dag_digest`, `construction_dag_digest` — graph identity and
  provenance. Owner: `data/substrate/dagflow.py`.

These are rehomed, not removed.

### 3.4 `utils.randomness` — explicit RNG control (conditional)

**Created only when**: at least two packages need `derive_seed` or
`make_torch_generator` with genuinely reusable logic.

**Minimal initial API**:

```
derive_seed(base: int, *components: int | str) -> int
make_torch_generator(seed: int, *, device: str | torch.device = "cpu") -> torch.Generator
```

`derive_seed` hashes a base seed together with a sequence of arbitrary
components (rank, worker, stream, epoch, fold, trial, episode…) — avoiding
fragile arithmetic such as `seed = base + rank * 1000 + worker` without
hardcoding a particular decomposition.

**Deferred until demonstrated need**: `RNGState`, `seed_all`, full RNG
snapshot/restore, `fork_rng`. Full RNG state management introduces global
side effects and a particular distributed-training model into a supposedly
domain-neutral utility. It should be added only when checkpoint or evaluation
code genuinely requires it.

**What the utility does NOT decide**: which seed validation uses, whether
test episodes are fixed, how experiment repetitions are numbered, whether
environment resets are deterministic. Those belong in task, data, or
experiment configuration.

### 3.5 `utils.imports` — optional dependency detection (conditional)

**Created only when**: multiple packages duplicate optional-dependency
detection logic.

**Minimal API**:

```
OptionalDependencyError(ImportError)

module_available(package: str) -> bool
require_optional_dependency(package: str, *, feature: str, extra: str | None) -> None
import_symbol(qualified_name: str) -> object
qualified_name(obj: type | Callable) -> str
```

Must not become a plugin registry, dependency injection container, model
factory, or automatic experiment loader.

### 3.6 `utils.symmetry` — array symmetry groups (conditional)

**Created only when**: dihedral transforms have multiple cross-package
consumers (data augmentation, figures, evaluation, model equivariance tests).

If the only consumer is `data/transforms.py` for spatial data augmentation,
these functions belong in `data/transforms/symmetry.py`, not `utils`.

**Minimal API**:

```
dihedral_transform(arr: NDArray, tid: int) -> NDArray
inverse_dihedral_transform(arr: NDArray, tid: int) -> NDArray
```

Policy remains outside:

```python
# Generic — belongs in utils.symmetry (if created)
dihedral_transform(grid, transform_id)

# Task/data policy — belongs in data transforms
sample_arena_augmentation(...)
```

---

## 4. Root public API

```python
# ehp_sn/utils/__init__.py

"""Low-level, domain-neutral infrastructure for EHP."""

from . import graph
from . import tensors
from . import trees

__all__ = [
    "graph",
    "tensors",
    "trees",
]
```

When conditional modules are created, they are added to `__all__` alongside
the modules above.

**Consumer style**:

```python
# Correct
from ehp_sn.utils.tensors import require_shape, merge_rows
from ehp_sn.utils.trees import tree_detach
from ehp_sn.utils.graph import shortest_path

# Incorrect
from ehp_sn.utils import merge_rows
from ehp_sn.utils import *
```

---

## 5. API levels

| Level                                | Scope                                           | Compatibility                                                                 |
| ------------------------------------ | ----------------------------------------------- | ----------------------------------------------------------------------------- |
| **Level 1** — Package root           | Submodule discovery (`ehp_sn.utils.tensors`)    | Stable namespace                                                              |
| **Level 2** — Documented symbols     | `require_shape`, `tree_detach`, `shortest_path` | Semi-stable internal API; changes require deprecation and repo-wide migration |
| **Level 3** — Private implementation | `utils/_internal/*` (if created)                | No compatibility contract; other packages must not import                     |

`ehp_sn.utils` is an internal project API. Experiment authors interact with
tasks, models, training, evaluation, and configuration — not utility functions.

---

## 6. Governance rules

### 6.1 Admission procedure

Before adding a function, answer these five questions:

1. **Does it know about an EHP concept?** (TEM, HRM, MEC, HPC, LEC, Arena,
   MazeHard, ACT, rollout, trace, objective, checkpoint, experiment) → If
   yes, reject.
2. **Does another package have semantic authority?** (Checkpoint naming →
   `training`, metric reduction → `objectives`, graph generation → `data`)
   → If yes, place it with the owner.
3. **Is it only used within one package?** → If yes, keep it in that
   package's `/_internal/`.
4. **Is it already provided by Python, NumPy, or PyTorch?** → If yes, use
   the existing API directly unless EHP adds a genuine invariant.
5. **Is the behavior precisely specifiable and testable?** → If no, the
   abstraction is too vague for `utils`.

### 6.2 Module naming rules

1. No module named `misc`, `common`, `helpers`, or `general`.
2. No module that duplicates another subsystem's name.
3. Submodule names must communicate concrete technical capability.

### 6.3 Import rules

1. No imports from any `ehp_sn` domain package.
2. No import-time side effects.
3. No ambient global state.

### 6.4 Dependency enforcement

The primary enforcement mechanism is a static import-linter rule, not a
runtime test on `sys.modules`:

```ini
[importlinter:contract:utils-is-foundational]
name = Utils must not depend on EHP domain packages
type = forbidden
source_modules =
    ehp_sn.utils
forbidden_modules =
    ehp_sn.contracts
    ehp_sn.data
    ehp_sn.modules
    ehp_sn.models
    ehp_sn.controllers
    ehp_sn.objectives
    ehp_sn.rollouts
    ehp_sn.training
    ehp_sn.evaluation
    ehp_sn.analysis
    ehp_sn.metrics
    ehp_sn.traces
    ehp_sn.figures
    ehp_sn.reporting
    ehp_sn.tasks
```

A runtime smoke test may supplement this, but it must not be the primary
enforcement mechanism. Runtime `sys.modules` checks are sensitive to test
order, module caching, and transitive import behaviour.

### 6.5 Testing standards

Utilities require disproportionately strong tests because defects propagate
across many subsystems.

**Tensor utilities**: valid/invalid rank, each constrained dimension, empty
tensors, non-contiguous tensors, CPU/CUDA, multiple dtypes, gradients, views
vs. copies, broadcasting, zero-length dimensions, informative error text.

**Tree utilities**: dicts, lists, tuples, named tuples, dataclasses, empty
containers, aliasing, unsupported leaves, nested tensors, container type
preservation, cycles (rejected or explicitly unsupported).

**Graph utilities**: trivial graphs, disconnected graphs, singleton graphs,
cycles (rejected), deterministic DAG layouts, path counting on known examples.

**Randomness** (when created): same seed → same sequence, distinct streams →
distinct sequences, no accidental mutation of unrelated generators.

---

## 7. Migration appendix — resolved destinations

This appendix documents the final destination of every function currently in
`ehc_sn/utils/`. Destinations marked "Remove" should not be migrated.

| Current function                                                                                                         | Final owner                                | Rationale                                            |
| ------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------ | ---------------------------------------------------- |
| `merge_rows`, `expand_row_mask`                                                                                          | `utils/tensors.py`                         | Generic tensor structural ops                        |
| `merge_tree_rows`                                                                                                        | `utils/trees.py` (as `tree_merge_rows`)    | Generic tree structural op                           |
| `to_cpu`                                                                                                                 | `utils/trees.py` (as `tree_to_cpu`)        | Absorb into tree module                              |
| `find_multiple`                                                                                                          | Package-local where used                   | Not a multi-consumer utility                         |
| `trunc_normal_init_`                                                                                                     | `utils/tensors.py` (conditional)           | Only if multiple independent consumers               |
| `split_by_sizes` (was `uncat_to_list`)                                                                                   | `utils/tensors.py` (conditional)           | Only if multiple independent consumers               |
| `tree_detach` (was `detach_any`)                                                                                         | `utils/trees.py`                           | Generic tree detach                                  |
| `DetachMixin`                                                                                                            | `utils/trees.py` (conditional)             | Only if strongly justified over free functions       |
| `shortest_path`, `count_shortest_paths`, `compute_layered_dag_positions`                                                 | `utils/graph.py`                           | Generic graph algorithms                             |
| `generate_hamiltonian_dag`, `generate_transition_dag`                                                                    | `data/substrate/generators.py`             | Task graph generation policy                         |
| `remap_obs_ids`, `permute_candidate_order`                                                                               | `data/substrate/generators.py`             | Task permutation logic                               |
| `canonical_dag_digest`, `construction_dag_digest`                                                                        | `data/substrate/dagflow.py`                | DAG identity and provenance                          |
| `inv_var_weight`                                                                                                         | `types.py` (with `LocationBelief`)         | Domain math owned by the type                        |
| `sample_diag_gaussian`                                                                                                   | `types.py` (with `LocationBelief`)         | Domain sampling owned by the type                    |
| `inv_var_trans`                                                                                                          | `modules/mec/ovc.py`                       | OVC-specific fusion                                  |
| `connections`                                                                                                            | `modules/mec/layout.py`                    | MEC connectivity policy                              |
| `resolve_ovc_slice`                                                                                                      | `modules/mec/config.py`                    | MEC configuration                                    |
| `update_to_masks`, `make_update_full`, `make_update_hierarchical`                                                        | `modules/hpc/_update.py`                   | Attractor update policy                              |
| `make_hebbian_write_mask`                                                                                                | `modules/hpc/_memory.py`                   | Hebbian plasticity rule                              |
| `create_downsample_matrix`, `create_repeat_matrices`, `create_tiling_matrices`, `create_random_projection`, `downsample` | `modules/projection.py`                    | Neural architecture construction                     |
| `create_encoding_table`                                                                                                  | `data/transforms/encoding.py`              | Input encoding construction                          |
| `squared_error`, `cross_entropy`, `reduce_per_env`                                                                       | `objectives/_losses.py`                    | Loss function definitions                            |
| `one_hot_with_zero`                                                                                                      | `tasks/_action.py`                         | Action encoding contract                             |
| `merge_multiscale_rows`, `apply_per_band`, `multiscale_mean_abs`, `multiscale_row_mse`                                   | `modules/_multiscale.py`                   | Multiscale band operations                           |
| `snapshot_state_dict`, `snapshot_optimizer_steps`, `FrozenEvalMutationError`                                             | `evaluation/_frozen.py`                    | Evaluation policy                                    |
| `softmax`, `normalize`, `relu`, `leaky_relu`, `activation_from_str`                                                      | Remove                                     | Use `torch.nn.functional` directly                   |
| `make_directories`, `set_directories`                                                                                    | Remove (legacy `../Summaries/` convention) | Not used by modern `outputs/`/`artifacts/` structure |
| `make_logger`                                                                                                            | `logging/`                                 | Logging infrastructure                               |
| `as_dir_str`, `resolve_envs_path`                                                                                        | Remove (legacy)                            | Replace with pathlib-native operations               |
| `parse_iter_from_stem`                                                                                                   | `training/checkpoints/naming.py`           | Checkpoint naming convention                         |
| `apply_overrides`                                                                                                        | `configuration/`                           | Config merging                                       |
| `require_exists`                                                                                                         | `utils/tensors.py` (as generic path guard) | Only if path guarding is a multi-consumer need       |
| `rms_norm`                                                                                                               | `utils/tensors.py` or Remove               | One function does not justify a module               |

---

## 8. Summary

The governing rule is:

> Keep behaviour with its semantic owner. Move code to `utils` only when no
> domain package has stronger authority over it.

The immediate target is **three submodules** — `tensors`, `trees`, `graph`.
Additional submodules (`randomness`, `imports`, `symmetry`) are approved in
the taxonomy but created only when current code demonstrates a need.

For `ehp_sn`, this means:

- Use direct external libraries for established mechanisms.
- Keep task, model, rollout, training, metric, trace, and artifact semantics
  in their respective packages.
- Expose a small namespaced utility API.
- Make `utils` dependency-light and free of domain imports.
- Treat utility extraction as an architectural decision, not routine code
  cleanup.

A large `utils` package would be a warning sign. A small, stable, well-tested
one is infrastructure.
