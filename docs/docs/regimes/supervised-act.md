# Supervised ACT (Adaptive Computation Time)

**Status:** conceptual reference — not currently implemented in the
repository. The repository's `act_supervised` regime implements
[supervised Q-halting](supervised-q-halting.md), not classical ACT.

## What it is

Adaptive Computation Time (ACT), introduced by Graves (2016), is a
differentiable mechanism that lets a recurrent model decide how many
computation steps to use per input. Unlike hard halting, ACT produces a
_soft_ halting distribution — every step contributes to the output, with
contribution weights learned through backpropagation.

The regime combines two forces:

1. **Supervised task prediction** — the model learns to produce correct
   outputs from input sequences.
2. **Ponder-cost regularization** — the model is penalized for using more
   computation steps, creating pressure to halt early when possible.

## Mechanism

### Halting as probability mass

At each reasoning step $k$, the model emits a raw halting probability
$p_k \in [0,1]$ from its halting head. The effective output weight
$\alpha_k$ is determined by accumulation:

The number of updates $N$ is the smallest integer satisfying:

$$
N = \min\left\{n : \sum_{k=1}^{n} p_k \geq 1 - \epsilon\right\}
$$

The remainder is:

$$
R = 1 - \sum_{k=1}^{N-1} p_k
$$

The effective weight for each step is:

$$
\alpha_k = \begin{cases}
p_k, & k < N \\
R,   & k = N
\end{cases}
$$

The weights sum to 1 by construction. $\epsilon$ is a small constant
(typically 0.01) that allows halting slightly before the mass reaches 1.

### Soft output aggregation

The final output is a weighted mixture of all intermediate predictions:

$$
\hat{y} = \sum_{k=1}^{N} \alpha_k \hat{y}_k
$$

Every step's prediction influences the returned answer, weighted by
its halting contribution.

### Execution diagram

```text
input x
  │
  ▼
┌───────────────┐
│ reasoning k=1 │
│ state s₁      │
│ prediction ŷ₁ │
│ emitted p₁    │
└───────┬───────┘
        │ α₁ = p₁
        ▼
┌───────────────┐
│ reasoning k=2 │
│ state s₂      │
│ prediction ŷ₂ │
│ emitted p₂    │
└───────┬───────┘
        │ α₂ = R (remainder, since Σp ≥ 1-ε)
        ▼
   ŷ = α₁ŷ₁ + α₂ŷ₂
```

## Objective

### Task loss

The weighted output $\hat{y}$ is compared to the ground-truth target $y$:

$$
\mathcal{L}_{\text{task}} = \ell(\hat{y}, y)
$$

where $\ell$ is typically cross-entropy. Because $\hat{y}$ is a
differentiable function of every $p_k$ and $\hat{y}_k$, gradients flow
through the halting weights into both the prediction head and the halting
head.

### Ponder cost

The model pays a cost proportional to the total computation used:

$$
\mathcal{L}_{\text{ponder}} = \tau \cdot \rho,
\quad \rho = N + R
$$

where $N$ is the number of updates and $R$ is the remainder. The term
$N + R$ is used rather than the discrete $N$ alone because $R$ is a
function of the halting probabilities, providing a differentiable
gradient path through the remainder and weighted output. $\tau$ is a
hyperparameter controlling the trade-off between accuracy and
computation.

Conceptually:

```text
better prediction   versus   more recurrent steps
```

The model receives a direct gradient signal that more computation is
expensive.

### Combined objective

$$
\mathcal{L} = \mathcal{L}_{\text{task}} + \tau(N + R)
$$

Only two terms. There is no separate halting loss — the halting
mechanism is trained entirely through the task-loss gradient and the
ponder penalty.

## Gradient flow

All components are updated by a single backpropagation pass:

```text
task loss + ponder cost
   │
   ├──────────────► prediction head
   │
   ├──────────────► recurrent state
   │
   └──────────────► halting head (p_k)
```

The gradient $\partial\mathcal{L} / \partial p_k$ is well-defined because
the output $\hat{y}$ and the ponder term are functions of the halting
probabilities, providing a differentiable gradient path through the
weighted output and remainder.

## Design properties

**Fully differentiable.** The entire system — prediction, halting,
accumulation — is a single differentiable computation graph. No RL
component, no target networks, no replay buffers, no exploration.

**Soft aggregation by default; optional hard-output approximation at
inference.** During training, the output is always a weighted mixture.
At inference time, the model can either use the same soft accumulation or
hard-stop at the step where $\sum p_k \geq 1-\epsilon$, returning only
$\hat{y}_N$. The latter creates a potential train/test mismatch. Hard
stopping is a deployment choice, not an inherent ACT property.

**No explicit decision margin.** ACT learns a scalar $p_k$ per step.
There is no explicit representation of the _value_ of continuing versus
stopping — the model learns halting tendencies implicitly through the
ponder penalty.

## Comparison with Q-halting

| Property                 | Supervised ACT                             | Supervised Q-halting                                              |
| ------------------------ | ------------------------------------------ | ----------------------------------------------------------------- |
| Halting signal           | Soft $p_k \in [0,1]$                       | Hard halt/continue decision                                       |
| Output                   | Weighted mixture $\sum \alpha_k \hat{y}_k$ | Output from halted step                                           |
| Halting training         | Backprop through soft weights              | Bootstrapped Q-targets (BCE)                                      |
| Computation cost         | Ponder penalty $\tau(N+R)$                 | No explicit cost; bounded by max-step budget                      |
| Exploration              | None                                       | Halt-delay (forced continuation)                                  |
| Gradient to halt choice  | Yes (differentiable)                       | No (discrete action)                                              |
| Target network           | Not needed                                 | Not needed                                                        |
| Train/inference behavior | Soft-to-hard mismatch possible             | Current evaluation ignores halt decisions and executes full depth |

## Practical trade-offs relative to Q-halting

Classical ACT is simpler — one objective, one backpropagation pass. But
it has practical limitations:

- **Diffuse halting mass.** The model can spread probability across many
  steps instead of making a decisive stop.
- **Ponder-cost tuning.** The coefficient $\tau$ must be carefully
  balanced against the task loss. Too small and the model always uses
  maximum depth; too large and it halts before solving.
- **Weighted-output mismatch.** If hard-output approximation is used at
  inference, the training signal (weighted mixture) and deployment
  behavior (single-step output) differ.

Q-halting approaches these differently by making the decision explicit
(halt vs continue action values) and providing a measurable decision
margin. Whether one approach is better depends on the task, the
reliability of intermediate predictions, and the importance of
interpretable stopping decisions.

## References

- Graves, A. (2016). _Adaptive Computation Time for Recurrent Neural
  Networks._ arXiv:1603.08983.
- Wang et al. (2025). _Hierarchical Reasoning Model._ arXiv:2506.21734.
  Describes a mechanism called an "ACT module" that uses Q-learning
  targets for halt and continue — the Q-halting approach documented in
  this repository, not classical ACT.
