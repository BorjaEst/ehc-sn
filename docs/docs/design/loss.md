---
title: Loss and Objective Design
description: Three-layer architecture of loss primitives, objectives, and metric contracts
---

# Loss and Objective Design Contract

> A three-layer architecture: mathematical primitives, task-level optimization
> objectives, and detached evaluation metrics — each with a distinct lifecycle
> and dependency direction.

The loss subsystem is organised into three semantically distinct layers. The
boundary between them is stricter than the module boundary.

```
    ┌──────────────────────────────────────────────────────┐
    │                   TRAINING LOOP                       │
    │  training/, lightning/                                │
    │  • calls objective.evaluate_step() or .compute_step() │
    │  • calls .loss.backward()                            │
    │  • routes results to metrics layer and logging        │
    └──────────────────────┬───────────────────────────────┘
                           │ consumes structured results
    ┌──────────────────────▼───────────────────────────────┐
    │                   OBJECTIVES                          │
    │  objectives/                                          │
    │  • composites/tem.py      → TEMObjective              │
    │  • composites/act.py      → ACTSupervisedScorer       │
    │  • composites/hybrid_rl.py→ HybridRLObjective         │
    │  • supervised/token.py    → TokenPredictionObjective  │
    │  • supervised/field.py    → FieldRegressionObjective  │
    │  • control/halt.py        → HaltClassificationObj     │
    │  • control/q_value.py     → QValueObjective           │
    │  • control/state_value.py → StateValueObjective       │
    │  • Assembles primitives into typed, masked, weighted  │
    │    loss bundles with attached signals and metrics     │
    └──────────────────────┬───────────────────────────────┘
                           │ imports primitives
    ┌──────────────────────▼───────────────────────────────┐
    │              LOSS PRIMITIVES (losses/)                 │
    │  losses/reductions.py   → masked_mean, masked_sum     │
    │  losses/consistency.py  → mse_consistency,            │
    │                           nll_consistency             │
    │  losses/cross_entropy.py→ softmax_cross_entropy,      │
    │                           stablemax_cross_entropy     │
    │  losses/regularization.py→ l1_penalty, l2_penalty     │
    │  losses/divergences.py  → gaussian_kl_divergence      │
    │  • Pure (B,D)→(B,) tensor functions                   │
    │  • No model imports, no state, no logging             │
    └──────────────────────────────────────────────────────┘
```

---

## 1. Vocabulary

| Term               | Definition                                                                                                                  | Example                                           |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------- |
| **Loss primitive** | A single differentiable mathematical function on tensors                                                                    | `mse_consistency(pred, target) → (B,)`            |
| **Objective**      | The complete optimisation policy for one model/task combination — selects terms, applies masks, assigns weights, normalises | `TEMObjective`, `ACTSupervisedScorer`             |
| **Metric**         | Detached evaluation measurement used for selection, interpretation, or reporting — no gradient flow                         | `accuracy_obs_post_revisit`, `field_mae`          |
| **Regulariser**    | A differentiable constraint not tied to a supervised target                                                                 | `l2_penalty(grid_code)`, `l1_penalty(place_code)` |
| **Signal**         | A scalar telemetry value that accompanies an objective result — detached, logged, never back-propagated                     | `grid_post_norm`, `rpe_magnitude`                 |

### Invariant: dependency direction

The dependency graph defines explicit allowed edges. "Upward" and "downward"
are avoided because they depend on diagram orientation.

```
torch
  │
  ▼
losses
  │
  ▼
objectives ──────────────┐
  │                      │
  │ produces             │ uses
  ▼                      ▼
neutral contracts         │
(RatioStat, signal keys,  │
 StepMetrics shape types) │
  │                      │
  ├──────────┐           │
  ▼          ▼           ▼
metrics    diagnostics  training
  │          │           │
  └────┬─────┘           │
       ▼                 ▼
  lightning / training-loop integration
       │
       ├── operational logging
       └── experiment tracking
```

- Primitives (`losses/`) depend only on `torch`.
- Objectives depend on `losses/` and neutral contract types (`RatioStat`,
  `StepMetrics` shape types, signal key constants).
- Neutral contracts define vocabulary shared by objectives, metrics, and
  diagnostics — owned by neither.
- Metrics and diagnostics depend on neutral contracts, never on objectives.
- Training code imports objectives; objectives never import training code.
- Lightning imports objectives, metrics, diagnostics, logging, and tracking.
- Logging and tracking depend on neutral record contracts only.

---

## 2. Layer 1 — Loss Primitives (`src/ehc_sn/losses/`)

### 2.1 Contract

The `losses/` package contains two categories of primitive:

**Elementwise loss primitives** apply a pointwise or per-dimension computation
and return unreduced per-example or per-element tensors. Their contract is:

- **Input**: The leading dimension is the batch dimension. Common shapes are
  `(B, D)` (latent codes), `(B, S, V)` (logits), `(B, N)` (fields), `(B,)`
  (scalar predictions). Each primitive documents its own shape contract.
- **Output**: Per-example `(B,)` or per-element `(B, S)` — **unreduced**.
- **State**: Pure function — no parameters, no buffers, no registered state.
- **Dtype**: Operates on the input dtype; casts internally only when
  numerical stability requires it.
- **Device**: Device-agnostic — no `.cpu()`, `.item()`, or device checks.
- **Logging**: None. No `self.log()`, no MLflow, no console output.
- **Gradient**: Full graph preserved; no `.detach()` in the differentiable path.

**Reduction primitives** explicitly aggregate per-example tensors into scalars
or lower-rank tensors and are exempt from the unreduced-output invariant.
They are pure transformations from values and masks to aggregated results;
the caller chooses whether the result is a loss term or a diagnostic statistic.

### 2.2 Primitive inventory

#### `losses/reductions.py` — Masked aggregation utilities

````python
EmptyReductionPolicy = Literal["zero", "nan", "error"]

def masked_mean(
    values: Tensor,
    mask: Tensor,
    *,
    empty: EmptyReductionPolicy = "zero",
) -> Tensor:
    """Scalar mean over mask-selected elements.

    Args:
        values: Per-example tensor of shape ``(B,)``.
        mask: Boolean eligibility mask of shape ``(B,)``.
        empty: Behaviour when the mask is all-False:
            ``"zero"`` — return 0.0 (batch contributes nothing).
            ``"nan"``  — return NaN (signal undefined metric).
            ``"error"`` — raise ``ValueError`` (flag data/runtime bug).

    Returns:
        Scalar tensor.  The computation is tensor-only — no Python
        branching over tensor values — and is safe under ``torch.compile``
        and CUDA graph capture.
    """
    weights = mask.to(dtype=values.dtype)
    numerator = (values * weights).sum()
    denominator = weights.sum()
    safe_mean = numerator / denominator.clamp_min(1)

    if empty == "zero":
        return torch.where(denominator > 0, safe_mean,
                           values.new_zeros(()))
    if empty == "nan":
        nan = values.new_tensor(float("nan"))
        return torch.where(denominator > 0, safe_mean, nan)
    if empty == "error":
        if denominator == 0:
            raise ValueError(
                "masked_mean received an all-False mask; "
                f"empty policy is {empty!r}."
            )
        return safe_mean
    raise ValueError(f"Unknown empty policy {empty!r}.")


def masked_sum(values: Tensor, mask: Tensor) -> Tensor:
    """Scalar sum over mask-selected elements."""
    return (values * mask.to(dtype=values.dtype)).sum()

| Function      | Input                            | Output | Use                               |
| ------------- | -------------------------------- | ------ | --------------------------------- |
| `masked_mean` | `(B,)` values + `(B,)` bool mask | scalar | TEM revisit-masked loss reduction |
| `masked_sum`  | `(B,)` values + `(B,)` bool mask | scalar | Summation over masked region      |

#### `losses/consistency.py` — Latent-code consistency penalties

```python
def mse_consistency(pred: Tensor, target: Tensor) -> Tensor: ...

def nll_consistency(pred: Tensor, mean: Tensor, std: Tensor, *, min_std=1e-6) -> Tensor: ...
````

| Function          | Input shapes                            | Formula                                              | Use                                                      |
| ----------------- | --------------------------------------- | ---------------------------------------------------- | -------------------------------------------------------- |
| `mse_consistency` | `pred (B,D)`, `target (B,D)`            | $\frac{1}{2}\|p - t\|^2_2$ (sum over D)              | Default TEM latent loss; posterior vs. prior consistency |
| `nll_consistency` | `pred (B,D)`, `mean (B,D)`, `std (B,D)` | $-\log \mathcal{N}(p \mid \mu, \sigma)$ (sum over D) | Gaussian NLL alternative for probabilistic latents       |

**Helper types:**

```python
LatentCode = Tensor | Sequence[Tensor]
# A single or multi-block latent code (e.g. multiple frequency bands).

@dataclass(frozen=True)
class LatentRelation:
    lhs: LatentCode
    rhs: LatentCode
# Semantic-agnostic binary relation. The objective assigns meaning
# through naming and coefficient scheduling — LatentRelation itself
# does not distinguish grid from place, or transition from sensory.

def sum_latent_terms(loss_fn, pred: LatentCode, target: LatentCode) -> Tensor:
    """Apply loss_fn over one or more latent-code blocks and sum per-example."""

def iter_latent_codes(code: LatentCode) -> tuple[Tensor, ...]:
    """Normalise LatentCode to a tuple view."""

def mean_latent_norm(code: LatentCode) -> Tensor:
    """Diagnostic: mean block-wise activation norm (detached)."""
```

#### `losses/cross_entropy.py` — Observation prediction losses

| Function                  | Input shapes                     | Formula                                                                                                                | Use                                      |
| ------------------------- | -------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | ---------------------------------------- |
| `softmax_cross_entropy`   | `logits (B,S,V)`, `labels (B,S)` | Standard `F.cross_entropy(reduction='none')`                                                                           | Default observation loss                 |
| `stablemax_cross_entropy` | `logits (B,S,V)`, `labels (B,S)` | $\text{stablemax}(x)_i = \frac{s(x_i)}{\sum_j s(x_j)},\ s(x) = \begin{cases}1/(1-x) & x<0 \\ x+1 & x \ge 0\end{cases}$ | Numerically stable alternative (float64) |

Stablemax defines a **different transformation** and therefore a different
objective from softmax — it is not a numerically-equivalent drop-in
replacement. It is selected explicitly by configuration
(`observation_loss="stablemax_cross_entropy"`) and treated as an
experimental alternative normalisation/objective rather than a hardening
measure. Its internal `float64` cast affects memory, performance, and
AMP behaviour; these trade-offs are documented in the training configuration
surface, not elided as "stable."

```python
LossType = Literal["stablemax_cross_entropy", "softmax_cross_entropy"]
```

#### `losses/regularization.py` — Activation penalties

| Function           | Input   | Formula                   | Use                 |
| ------------------ | ------- | ------------------------- | ------------------- |
| `l1_penalty(code)` | `(B,D)` | $\sum_d \lvert c_d\rvert$ | Place-cell sparsity |
| `l2_penalty(code)` | `(B,D)` | $\frac12 \sum_d c_d^2$    | Grid-cell energy    |

```python
RegularizationNorm = Literal["none", "l1", "l2"]

def sum_regularization_terms(code: LatentCode, norm: RegularizationNorm) -> Tensor:
    """Apply chosen norm over all blocks of a LatentCode."""
```

#### `losses/divergences.py` — Distribution divergences

```python
def gaussian_kl_divergence(
    posterior_mean: Tensor, posterior_std: Tensor,
    prior_mean: Tensor, prior_std: Tensor,
    *, min_std: float = 1e-6,
) -> Tensor:
    """Per-example KL(q || p) for diagonal Gaussians, shape (B,)."""

def sum_gaussian_kl_divergence(
    posterior_mean: LatentCode, posterior_std: LatentCode,
    prior_mean: LatentCode, prior_std: LatentCode,
    *, min_std: float = 1e-6,
) -> Tensor:
    """Multi-block version, sums over all frequency bands."""
```

### 2.3 Invariants

| Property            | Enforced by                                                                            |
| ------------------- | -------------------------------------------------------------------------------------- |
| No model imports    | Primitives depend only on `torch`, `torch.nn.functional`                               |
| No training state   | Pure functions; `nn.Module` subclasses carry no trainable parameters                   |
| No logging          | No `self.log()`, no MLflow, no console output                                          |
| Unreduced output    | Elementwise primitives return `(B,)` or `(B, ...)`; reduction primitives aggregate     |
| Device-safe         | No `.item()`, `.cpu()`, `.numpy()` in the differentiable path                          |
| Graph-preserving    | No `.detach()` in the loss value path; detached copies are the caller's responsibility |
| Batch-axis explicit | The leading dimension is always the batch dimension unless documented otherwise        |

---

## 3. Layer 2 — Objectives (`src/ehc_sn/objectives/`)

### 3.1 Shared result type

```python
@dataclass(frozen=True)
class ObjectiveResult:
    """Differentiable output from a single objective forward call.

    All fields are immutable.  The ``losses`` and ``terms`` mappings
    are frozen at construction (e.g. via ``MappingProxyType``).
    """
    loss: Tensor                       # Scalar for .backward()
    losses: Mapping[str, Tensor]        # Named reduced loss components (scalar)
    terms: Mapping[str, Tensor]         # Unreduced per-element tensors (optional)
```

`ObjectiveResult` is the return type of all atomic objectives (token, field, halt, Q-value, state-value). Composite objectives (TEM, ACT, Hybrid RL) return richer step-level wrappers that embed an `ObjectiveResult` alongside metrics and signals.

### 3.2 Composite objective results

Composite objectives define domain-specific loss bundles. Each inherits from `DetachMixin` and exposes a `.total` property for back-propagation.

#### TEM Losses

```
TEMLosses.total = loss_obs_nll_sum + loss_latent_sum + loss_reg_sum
```

Where:

```
loss_obs_nll_sum
├── loss_obs_post_sum     ← HPC inference pathway NLL × c_obs
├── loss_obs_recall_sum   ← HPC corrected-grid recall NLL × c_obs
└── loss_obs_path_sum     ← HPC structural prior NLL × c_obs

loss_latent_sum
├── loss_place_consistency_sum
│   ├── loss_place_transition_sum  ← MSE(post, prior) × c_place × temp
│   └── loss_place_sensory_sum     ← MSE(post, sensory) × c_place × temp × p2g_use
└── loss_grid_kl_sum              ← MSE(post, prior) × c_grid × temp

loss_reg_sum
├── loss_grid_reg_sum    ← L2(post_grid) × c_grid_reg × g_cell_reg
└── loss_place_reg_sum   ← L1(post_place) × c_place_reg × p_cell_reg
```

**Config:**

```python
class TEMObjectiveConfig(BaseModel, extra="forbid"):
    observation_loss: LossType = "softmax_cross_entropy"
    c_obs: float = 1.0
    latent_loss: str = "mse_consistency"
    c_grid: float = 1.0
    c_place: float = 1.0
    grid_reg_norm: RegularizationNorm = "l2"
    c_grid_reg: float = 0.01
    place_reg_norm: RegularizationNorm = "l1"
    c_place_reg: float = 0.02
    temp_it: int = 2000          # anneal horizon for temperature
    p2g_use_it: int = 0          # anneal horizon for place→sensory gate
    p2g_scale: float = 200.0
    g_reg_it: int = 40_000_000   # anneal horizon for grid reg
    p_reg_it: int = 4000         # anneal horizon for place reg
```

**Scheduling context** (per-step, from training loop):

| Field         | Source                              | Role                            |
| ------------- | ----------------------------------- | ------------------------------- |
| `temperature` | Linear anneal 0→1 over `temp_it`    | Scales latent MSE terms         |
| `p2g_use`     | Linear anneal 0→1 over `p2g_use_it` | Gates place→sensory consistency |
| `g_cell_reg`  | Linear anneal 0→1 over `g_reg_it`   | Scales grid L2 regularisation   |
| `p_cell_reg`  | Linear anneal 0→1 over `p_reg_it`   | Scales place L1 regularisation  |

#### ACT Losses

```
ACTStepLosses.total = c_task × task_sum + c_halt × halt_sum + c_continue × continue_sum
```

```python
class ACTSupervisedScorerConfig(BaseModel, extra="forbid", frozen=True):
    task_loss_coefficient: float = 1.0
    halt_loss_coefficient: float = 0.5
    continue_loss_coefficient: float = 0.5
```

The task loss is pre-computed by a task-specific evaluator (token, field, etc.) and consumed via `TaskStepEvaluation(task_loss_sum, task_loss_count, ...)`. The ACT scorer only composes — it does not know the task modality.

#### Hybrid RL Losses

```
HybridRLLosses.total = loss_token_sum + c_v × loss_state_value_sum + c_q × loss_q_value_sum
```

```python
class HybridRLLossConfig(BaseModel, extra="forbid"):
    token_loss: LossType = "softmax_cross_entropy"
    gamma: float = 0.99
    c_state_value: float = 0.5
    c_q_value: float = 0.5
```

During warmup (`is_warmup=True`), value losses are zeroed and only the token prediction term contributes.

### 3.3 Masking, normalisation, and gradient-scaling policies

Each objective family has a distinct gradient-scaling contract. The
following table records both the reduction logic and its effect on
gradient magnitude relative to batch size, sequence length, and
distributed world size:

| Objective         | Mask source                                                   | Normalisation                                    | Gradient-scale contract                                        | Empty-mask behaviour                        |
| ----------------- | ------------------------------------------------------------- | ------------------------------------------------ | -------------------------------------------------------------- | ------------------------------------------- |
| TEM               | `protocol_mask` (revisit eligibility)                         | Masked mean over eligible entries                | Independent of eligible count; ~constant per eligible sample   | `clamp_min(1)` → implicit zero contribution |
| ACT               | `TaskStepEvaluation.task_loss_count` for task; `mask` on halt | Task: `sum / count`; Halt: `sum(masked_BCE)`     | Sum over active slots; effective LR scales with active count   | Zero when count is zero                     |
| Hybrid RL / Token | `ignore_index=-100`                                           | Per-sequence mean, then batch sum                | Independent of token count per sequence; grows with batch size | Zero contribution from ignored positions    |
| Field             | `mask (B,N)` bool                                             | Per-sample mean over valid nodes, then batch sum | Independent of valid-node count; grows with batch size         | Zero contribution from masked nodes         |
| Q-Value           | None (all slots active)                                       | Raw MSE sum                                      | Grows linearly with batch size                                 | N/A                                         |
| State-Value       | None (all slots active)                                       | Raw MSE sum                                      | Grows linearly with batch size                                 | N/A                                         |

These choices are **intentional**:

- **TEM** uses a per-eligible-sample mean so that gradient magnitude is
  stable across varying revisit counts.
- **ACT** sums over active slots so that longer deliberation impacts the
  optimisation proportionally to compute invested.
- **Token** and **Field** normalise per sequence to decouple gradient
  magnitude from sequence length, then sum over the batch to preserve
  batch-level scaling.
- **Q-value** and **State-value** use raw sums; the Lightning integration
  applies per-rank normalisation via `sync_dist` where appropriate.

Distributed training (`world_size > 1`) divides the effective global batch.
Objectives do **not** automatically normalise by world size — the Lightning
integration layer owns that policy.

### 3.4 Protocol-based prediction contracts

Objectives consume model outputs through **structural protocols**, not nominal inheritance. This keeps the objectives layer free of adapter imports.

```python
class TEMPrediction(Protocol):
    """Objective-facing prediction contract — satisfied structurally by bridge outputs."""
    grid_transition: LatentRelation
    place_transition: LatentRelation
    place_sensory: LatentRelation | None
    grid_reg_code: LatentCode
    place_reg_code: LatentCode

    @property
    def logits_post(self) -> Tensor: ...
    @property
    def logits_recall(self) -> Tensor: ...
    @property
    def logits_path(self) -> Tensor: ...
```

The adapter bridge outputs (`ArenaTEMBridgeOutput`, `TEMLearningState`) satisfy this protocol by having the right attribute names and types — no explicit `implements` declaration required.

### 3.5 Signals produced by each composite objective

Each composite objective emits a `signals` dict alongside losses and metrics.
Signals **must be detached** — the result container enforces `.detach()` on
every signal value before storing it, preventing accidental graph retention:

```python
@dataclass(frozen=True)
class ScalarSignals:
    """Immutable detached scalar telemetry bundle."""
    _values: Mapping[str, Tensor]

    @classmethod
    def from_tensors(
        cls, raw: Mapping[str, Tensor], *, detach: bool = True
    ) -> "ScalarSignals":
        values = {
            name: (value.detach() if detach else value)
            for name, value in raw.items()
        }
        return cls(_values=MappingProxyType(values))

    def __getitem__(self, key: str) -> Tensor:
        return self._values[key]
```

The caller retains scalar zero-dimensional tensors in the bundle — `.item()`
synchronisation is deferred to the Lightning or tracking boundary.

Signals never contribute to the gradient:

| Composite | Key signals                                                                                                                                               | Count |
| --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ----- |
| TEM       | `loss_total`, `loss_obs_nll`, `loss_latent`, `loss_reg`, `grid_post_norm`, `grid_prior_norm`, `place_post_norm`, `place_prior_norm`, per-pathway obs NLLs | ~15   |
| ACT       | `halt_logit_mean`, `continue_logit_mean`, `greedy_halt_rate`, `target_q_mean`, `target_q_std`, `loss_q_done`                                              | ~6    |
| Hybrid RL | `reward_mean`, `reward_std`, `q_mean`, `q_std`, `rpe_magnitude`, `loss_state_value`, `loss_q_value`                                                       | ~7    |

---

## 4. Layer 3 — Metrics (`src/ehc_sn/metrics/`)

### 4.1 Separation from loss computation

Metrics are **detached evaluation measurements**. They share no gradient path
with the loss. Metrics consume typed predictions, targets, masks, rollout
records, or neutral statistics produced during objective evaluation — they do
not require the objective's differentiable `ObjectiveResult` type.

Metric computation happens in two places:

- inside the objective and bundled as `StepMetrics` (lightweight per-step aggregates); or
- by the evaluation layer (full-dataset aggregation).

### 4.2 Key metric categories

#### Step-level aggregators (`step_metrics.py`)

```python
@dataclass(frozen=True)
class RatioStat:
    numerator_sum: Tensor   # Sum across the step/batch
    denominator_sum: Tensor # Denominator for ratio computation

@dataclass(frozen=True)
class StepMetrics:
    episode: RolloutAgg         # Completed-sequence aggregates
    episode_tokens: TokenAgg    # Token-level over completed sequences
    step: TransitionAgg         # All-evaluated-sequence aggregates
    step_tokens: TokenAgg       # Token-level over all evaluated sequences
    extras: dict[str, RatioStat]  # Algorithm-specific ratio metrics
```

#### Metric keys (`keys.py`)

Stable string constants that serve as lookup keys in `StepMetrics.extras` and in routing tables:

| Prefix          | Count | Examples                                                |
| --------------- | ----- | ------------------------------------------------------- |
| `TEM_ACC_OBS_*` | 6     | `accuracy_obs_post_revisit`, `accuracy_obs_path_all`    |
| `TEM_LOSS_*`    | 7     | `loss_obs_nll`, `loss_grid_kl`, `loss_place_transition` |
| `ACT_LOSS_Q_*`  | 2     | `loss_q_done`, `loss_q_continue`                        |
| `RL_LOSS_*`     | 2     | `loss_state_value`, `loss_q_value`                      |
| `LOSS_TOKEN`    | 1     | `loss_token`                                            |

#### Signal keys (`signals.py`)

Stable scalar telemetry vocabulary organised by paradigm:

| Group                    | Count | Examples                                               |
| ------------------------ | ----- | ------------------------------------------------------ |
| `CROSS_PARADIGM_SIGNALS` | 2     | `steps_mean`, `theta_cls_norm`                         |
| `ACT_SIGNALS`            | 6     | `halt_logit_mean`, `greedy_halt_rate`                  |
| `RL_SIGNALS`             | 7     | `reward_mean`, `q_mean`, `rpe_magnitude`               |
| `VAR_SIGNALS`            | 6     | `loss_total`, `loss_latent`, `latent_post_norm`        |
| `TEM_SIGNALS`            | 9     | `loss_grid_kl`, `loss_place_sensory`, `grid_post_norm` |

### 4.3 Metric routing and reduction

Per-step metrics are accumulated over evaluation episodes and reduced via routing tables in `metrics/routes/`:

```
metrics/routes/
├── tem.py                # TEM-specific metric consolidation
├── act.py                # ACT-specific metric consolidation
├── continuous_field.py   # Field-prediction metric consolidation
└── rl.py                 # RL-specific metric consolidation
```

The `lightning/` layer fetches accumulated metrics via callbacks (`callbacks/evaluation.py`, `callbacks/diagnostics.py`) and logs them through `self.log()`.

---

## 5. Dependency diagram

```
                         ┌───────────────────┐
                         │   torch / nn.functional  │
                         └────────┬──────────┘
                                  │
                         ┌────────▼──────────┐
                         │  losses/           │  pure tensor functions
                         │  • reductions.py   │
                         │  • consistency.py  │
                         │  • cross_entropy.py│
                         │  • regularization  │
                         │  • divergences.py  │
                         └────────┬──────────┘
                                  │ imports
                         ┌────────▼──────────┐
                         │  objectives/       │
                         │  • types.py        │
                         │  • contracts.py    │
                         │  • supervised/     │
                         │  • control/        │
                         │  • composites/     │
                         └────────┬──────────┘
                                  │ uses
                         ┌────────▼──────────┐
                         │  neutral contracts │   shared vocabulary
                         │  • RatioStat       │
                         │  • StepMetrics     │
                         │  • signal keys     │
                         │  • metric keys     │
                         └──┬──────────┬─────┘
                            │          │
                   ┌────────▼──┐  ┌───▼──────────┐
                   │  metrics/  │  │  diagnostics/ │
                   │  • routes/ │  │  • probes/    │
                   │  • build/  │  └───┬──────────┘
                   └─────┬─────┘      │
                         │            │
                   ┌─────▼────────────▼──────┐
                   │  lightning/ / training/  │
                   │  • modules/              │
                   │  • callbacks/            │
                   └──────────┬──────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        logging/         tracking/        figures/
```

### Allowed dependency directions

```
losses        → torch only
objectives    → losses, neutral contract types (RatioStat, StepMetrics shapes)
contracts     → torch (data structures only; no algorithm)
metrics       → neutral contracts
diagnostics   → neutral contracts
lightning     → objectives, metrics, diagnostics, logging, tracking
training      → objectives, metrics, diagnostics
logging       → neutral logging contracts only
tracking      → neutral metric/artifact/run record contracts only
```

### Forbidden dependency directions

```
objectives    → lightning, training, logging, tracking, metrics/, evaluation, figures
metrics       → objectives, lightning, training, logging
diagnostics   → objectives, lightning, training, logging
losses        → any domain or orchestration layer
contracts     → objectives, metrics, losses (data only; no imports from domain)
```

---

## 6. Weighting policy

### 6.1 Static coefficients

Coefficients with no temporal schedule live in immutable config dataclasses:

```python
@dataclass(frozen=True)
class ACTSupervisedScorerConfig:
    task_loss_coefficient: float = 1.0
    halt_loss_coefficient: float = 0.5
    continue_loss_coefficient: float = 0.5
```

### 6.2 Annealed coefficients

Coefficients with temporal schedules are injected per-step via a context object:

```python
@dataclass(frozen=True)
class TEMScoringContext:
    temperature: float    # 0→1 over temp_it steps
    p2g_use: float        # 0→1 over p2g_use_it steps
    g_cell_reg: float     # 0→1 over g_reg_it steps
    p_cell_reg: float     # 0→1 over p_reg_it steps
```

### 6.3 Dynamic weighting (future)

If adaptive multi-task weighting becomes empirically needed, define a narrow callable protocol:

```python
class WeightSchedule(Protocol):
    def __call__(self, *, step: int) -> float: ...
```

Implementations: `ConstantWeight`, `LinearWarmupWeight`, `PiecewiseWeight`.

Do not introduce adaptive weighting before it is empirically justified — adaptive methods alter the optimisation landscape and should be treated as experimental algorithms, not infrastructure.

### 6.4 Weight naming convention

Weights are stored at the config/context level, not as unexplained literals in loss assembly code:

```python
# Good — coefficients are named and configurable
total = c_task * task_loss + c_halt * halt_loss + c_continue * continue_loss

# Bad — unexplained magic constants
total = task_loss + 0.5 * halt_loss + 0.5 * continue_loss
```

---

## 7. Validation policy

### 7.1 Always-cheap checks (every forward call)

```python
if logits.shape[:-1] != labels.shape:
    raise ValueError(
        f"Logits shape {tuple(logits.shape)} does not match "
        f"labels shape {tuple(labels.shape)} (excluding vocab dim)."
    )
```

### 7.2 Diagnostic-only checks (when validation level is FULL)

- `torch.isfinite(loss).all()`
- Probability range checks (`[0, 1]` for completion targets)
- Non-empty mask consistency
- Target index bounds (≤ vocab size)
- Denominator non-zero

```python
class ObjectiveValidation(str, Enum):
    NONE = "none"      # Skip all validation
    SHAPES = "shapes"  # Shape and dtype checks only
    FULL = "full"      # Full numerical validation
```

Validation configuration is repository-wide, not per-objective.

### 7.3 Error message standard

```python
raise ValueError(
    f"[{self.__class__.__name__}] {term_name}: "
    f"shape={tuple(value.shape)}, dtype={value.dtype}, "
    f"device={value.device}, valid_count={valid_count.item()}, "
    f"min={value.min().item():.4f}, max={value.max().item():.4f}"
)
```

Expensive diagnostic summaries (`.item()`, `.min()`, `.max()`) are computed
**only after** a validation predicate has already failed — never eagerly
while constructing the message on every forward pass. Additional precautions:

- `value.min()` fails for empty tensors; check `numel() > 0` first.
- Complex tensors require special handling.
- Integer and boolean tensors may need different formatting.
- Non-finite values make min/max less useful unless filtered:
  use `value[value.isfinite()].min()` when infinities are expected.

### 7.4 What not to do

- Do **not** silently replace NaN or infinity with zero. That conceals optimisation failures.
- Do **not** catch every numerical edge case — let the gradient flow; if it produces NaN, the diagnostic layer should detect it.

---

## 8. Testing requirements

### 8.1 Every loss primitive

| Test            | Description                                                                         |
| --------------- | ----------------------------------------------------------------------------------- |
| Numerical value | Hand-calculated example matches implementation                                      |
| Shape           | Output is `(B,)` for `(B,D)` input                                                  |
| Masking         | Masked positions contribute zero                                                    |
| All-invalid     | All-False mask yields correct `empty`-policy result (reductions)                    |
| Gradient        | `.backward()` produces finite gradients on inputs                                   |
| Finite output   | Extreme but valid inputs produce finite loss                                        |
| Dtype/device    | Works on CPU with `float32` and `float64`. CUDA tests run when device is available. |
| AMP/compile     | Compatible with `autocast` and `torch.compile` where the primitive is hot-path.     |
| Non-contiguous  | Accepts strided/non-contiguous tensors without copy.                                |
| Zero-length     | Zero-length batch dimensions do not error (where semantically valid).               |

### 8.2 Every composite objective

| Test              | Description                                                                  |
| ----------------- | ---------------------------------------------------------------------------- |
| Decomposition     | `losses.total == sum(losses)` for each `TEMLosses`/`HybridRLLosses` property |
| Disabled terms    | A coefficient of `0.0` produces exactly zero contribution                    |
| Signal isolation  | Diagnostic signals do not alter `.total` or gradient graph                   |
| Mask propagation  | Mask affects every applicable term consistently                              |
| Config round-trip | Serialise/deserialise config produces identical objective                    |
| Integration       | `evaluate_step` returns same tensor as direct per-term computation           |

### 8.3 Gradient test example

```python
prediction = torch.randn(4, 8, requires_grad=True)
target = torch.randn(4, 8)

loss = mse_consistency(prediction, target).sum()
loss.backward()

assert prediction.grad is not None
assert torch.isfinite(prediction.grad).all()
```

---

## 9. Objective-to-task mapping

| Objective config            | Objective class               | Task families                                | Adapter bridge output           |
| --------------------------- | ----------------------------- | -------------------------------------------- | ------------------------------- |
| `TEMObjectiveConfig`        | `TEMObjective`                | `arena`                                      | `ArenaTEMBridgeOutput`          |
| `ACTSupervisedScorerConfig` | `ACTSupervisedScorer`         | `arena`, `goaltrace`, `routebind`, `seqmaze` | Family-specific `*BridgeOutput` |
| `HybridRLLossConfig`        | `HybridRLObjective`           | `maze_hard` (HRM v2)                         | `MazeHardHRMBridgeOutput`       |
| (standalone)                | `TokenPredictionObjective`    | Any with token targets                       | Any bridge with `task_logits`   |
| (standalone)                | `FieldRegressionObjective`    | `goaltrace`                                  | `GoaltraceHRMBridgeOutput`      |
| (standalone)                | `HaltClassificationObjective` | Any ACT-capable                              | Via `ACTControlPrediction`      |
| (standalone)                | `QValueObjective`             | RL-capable                                   | Via `SelectedQInput`            |
| (standalone)                | `StateValueObjective`         | RL-capable                                   | Via `ValueRegressionInput`      |

---

## 10. What lives elsewhere

| Concern                                               | Package                             | Reason                                   |
| ----------------------------------------------------- | ----------------------------------- | ---------------------------------------- |
| Shared vocabulary types (RatioStat, StepMetrics)      | Neutral contracts layer             | Owned by neither objectives nor metrics  |
| Signal keys and metric keys                           | Neutral contracts / telemetry       | Shared vocabulary; no domain logic       |
| Evaluation metrics (accuracy, field MAE, ranking AUC) | `metrics/`                          | Detached measurements, no gradient       |
| Metric accumulation over episodes                     | `metrics/` / `lightning/callbacks/` | Stateful aggregation, not differentiable |
| `self.log()` calls                                    | `lightning/` modules                | Lightning lifecycle integration          |
| MLflow parameter/metric/artifact recording            | `tracking/` or `lightning/`         | Experiment persistence, not loss         |
| Gradient histograms and tensor diagnostics            | `diagnostics/`                      | High-volume numerical inspection         |
| Trace schemas and serialisation                       | `traces/`                           | High-volume trajectory persistence       |
| Figure generation                                     | `figures/`                          | Visual rendering, not numerical          |
| Report construction                                   | `reports/`                          | Narrative composition, not evaluation    |

---

## 11. Recommended future moves

| Priority | Change                                                                  | Rationale                                                                                                                                                 |
| -------- | ----------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Now**  | Extract `_masked_mean` → `losses/reductions.py`                         | Currently private in `tem.py`; needed by ACT, TEM, and future objectives. Use the `EmptyReductionPolicy` contract above.                                  |
| **Now**  | Move `gaussian_kl_divergence` → `losses/divergences.py`                 | Pure tensor primitive misplaced in `training/elbo.py`                                                                                                     |
| **Now**  | Relocate `RatioStat` and `StepMetrics` types to neutral contracts       | Acknowledged TODO in `objectives/contracts.py`; removes `objectives → metrics` dependency                                                                 |
| **Now**  | Relocate signal and metric key constants to neutral telemetry contracts | Removes `objectives → metrics/signals` dependency; objectives produce signals using shared constant names from neutral vocabulary                         |
| Soon     | Rename `loss/` → `losses/` (plural)                                     | Follows ecosystem convention (`torch.nn` losses, `monai.losses`, `kornia.losses`)                                                                         |
| Soon     | Add `bfloat16` and `torch.compile` parity tests                         | For hot-path primitives on relevant hardware                                                                                                              |
| Defer    | `LossTerm` wrapper for weighted values                                  | Not needed until dynamic per-step weighting is required                                                                                                   |
| Defer    | Adaptive multi-task weighting                                           | Experimental algorithm, not infrastructure                                                                                                                |
| Never    | Universal reducer that understands all domains                          | Each objective's mask semantics are domain-specific (TEM: revisit; ACT: active-slot; Token: padding); a unified reducer would hide meaningful differences |

---

## 12. Key architectural invariants

1. **Primitive purity**: Loss primitives are stateless, model-agnostic, device-agnostic tensor functions. Each primitive documents its own tensor contract; the leading dimension is always the batch dimension.

2. **Multi-block transparency**: `LatentCode` and `sum_latent_terms`/`sum_regularization_terms` handle the multi-frequency band structure of TEM codes transparently.

3. **Semantic-agnostic relations**: `LatentRelation(lhs, rhs)` is just a pair of code sides. The objective assigns meaning (grid vs. place, transition vs. sensory) through naming and coefficient scheduling — the data structure imposes no interpretation.

4. **Mask-driven reduction**: Every objective has a mask that determines which elements contribute to the gradient. The mask semantics are domain-specific and never unified into a single abstraction.

5. **Separation of loss from metric**: Accuracy, norms, ratios, and signals are computed alongside losses but never flow through `.backward()`. The `StepMetrics` container holds detached values.

6. **Config-driven scheduling**: Dynamic coefficients are injected per-step via typed context objects, not baked into the objective. This keeps objectives stateless with respect to training progress.

7. **Structural protocol, not nominal inheritance**: Objectives read bridge outputs through structural protocols (`TEMPrediction`, etc.). No adapter imports are needed in the objectives layer.

8. **Dependency direction is explicit and enforced**: Neutral contracts define shared vocabulary. Objectives import neutral contracts and primitives, never metrics or logging. Metrics import neutral contracts, never objectives. Training and Lightning import objectives — the dependency arrow never points upward.
