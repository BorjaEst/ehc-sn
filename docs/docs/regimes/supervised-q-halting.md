# Supervised sequence learning with Q-learning-based adaptive halting

**Repository identifiers:** legacy `act_supervised` / `ACTController`;
canonical conceptual name _supervised Q-halting_.

## What it is

A training regime for recurrent models that combines two learning
channels:

1. **Supervised sequence prediction** — the model learns to predict
   target tokens, fields, or structured outputs from input sequences.
2. **Q-learning-style adaptive halting** — the controller receives a
   reward derived from the correctness of the current prediction, while
   the continue value bootstraps from the next deliberation step. The
   implementation uses a repository-specific bounded-value formulation
   (BCE over correctness-derived targets), not textbook DQN.

The halting controller is reinforcement-trained in its decision
structure — it learns from outcome-derived value targets rather than
labelled halt actions — but its reward signal is computed using the
supervised ground truth. The controller is not independent of the task
labels; it is independent of _oracle halt/continue labels_, which the
dataset does not provide.

The regime trains an example-dependent halting policy: simple cases can be
assigned earlier halt decisions and hard cases later ones. In the current
evaluation path, these decisions are recorded diagnostically while
execution still runs to the maximum depth to guarantee consistent batch
shapes. Adaptive depth is a trained signal, not a deployed
compute-reduction mechanism.

## What it is not

This regime is **not** classical Adaptive Computation Time (ACT) as
introduced by Graves (2016). The key differences:

| Property           | Classical ACT                              | Supervised Q-halting                                        |
| ------------------ | ------------------------------------------ | ----------------------------------------------------------- |
| Halting signal     | Soft probability $p_k \in [0,1]$           | Hard halt/continue decision                                 |
| Training signal    | Backprop through soft halting weights      | Bootstrapped value targets (BCE)                            |
| Output aggregation | Weighted mixture $\sum \alpha_k \hat{y}_k$ | Output from the halted step                                 |
| Computation cost   | Ponder penalty $\tau(N+R)$ in loss         | No explicit cost; bounded indirectly by maximum-step budget |
| Exploration        | None                                       | Halt-delay exploration (forced continuation)                |

It is also **not** actor-critic halting — there is no stochastic policy
$\pi(a \mid s)$ and no policy-gradient objective. The controller learns
bounded halt and continue value estimates rather than a stochastic policy.

## Mechanism

### Halting as a value-based decision

At each reasoning step $k$, the model produces a state representation
$s_k$ and a prediction $\hat{y}_k$. A controller head maps the state to
action-value logits:

$$
q_k = f_\theta(s_k) \in \mathbb{R}^2
$$

where $q_k[\text{halt}]$ and $q_k[\text{continue}]$ are unnormalized
scores for stopping versus taking another step.

The greedy decision is:

$$
a_k = \begin{cases}
\text{halt} & \text{if } q_k[\text{halt}] > q_k[\text{continue}] \\
\text{continue} & \text{otherwise}
\end{cases}
$$

A step budget $N_{\max}$ forces halting when $k \geq N_{\max}$.

### Execution loop

```text
input x
  │
  ▼
┌───────────────────┐
│ reasoning step 1  │
│ state s₁          │
│ prediction ŷ₁     │
│ q₁[halt], q₁[cont]│
└─────────┬─────────┘
          │ q₁[continue] > q₁[halt]
          ▼
┌───────────────────┐
│ reasoning step 2  │
│ state s₂          │
│ prediction ŷ₂     │
│ q₂[halt], q₂[cont]│
└─────────┬─────────┘
          │ q₂[halt] > q₂[continue]
          ▼
      return ŷ₂
```

The final output is the prediction from the halted step. Earlier
predictions do not contribute directly to the returned answer.

### Halt-delay exploration (forced continuation)

During training, the controller may delay halting to observe outcomes at
deeper steps. With probability $\epsilon$, a slot is forced to continue
to a randomly sampled minimum step:

```python
if rand() < ε:
    min_steps = randint(2, N_max + 1)
    halt = halt & (steps >= min_steps)
```

This ensures the controller observes trajectories where deeper computation
succeeds or fails, preventing premature convergence to always-halt or
always-continue policies. This is not conventional ε-greedy action
selection — the controller always acts greedily; exploration only delays
the halt decision.

## Objective

### Supervised task loss

The prediction $\hat{y}_{k_{\text{halt}}}$ at the halt step is compared
to the ground-truth target $y$:

$$
\mathcal{L}_{\text{task}} = \ell(\hat{y}_{k_{\text{halt}}}, y)
$$

where $\ell$ is typically cross-entropy for token or field prediction
tasks.

### Halting value loss — correctness-derived reward

The halt value receives a reward signal derived from task correctness.
At each step $k$, the correctness indicator $c_k$ is computed
from the prediction $\hat{y}_k$ and the ground-truth target $y$ using
the task-defined correctness contract (e.g., exact token match or
task-specific completion check):

$$
c_k = \mathbf{1}[\hat{y}_k \text{ is correct per task definition}]
$$

The halt value is trained toward this immediate outcome:

$$
\mathcal{L}_{\text{halt}} = \operatorname{BCE}\big(q_k[\text{halt}],\; c_k\big)
$$

Conceptually, $c_k$ answers: _what reward would the controller receive if
it halted now?_ A perfect prediction earns a halt reward of 1; an
incorrect prediction earns 0. The halt target is direct correctness
supervision, not a TD return.

### Continuation bootstrap loss

The continue value is trained against a bootstrapped target from the next
step. Let $q_{k+1}[\text{halt}]$ and $q_{k+1}[\text{continue}]$ be the
scores at step $k+1$. The target for continuing is the best achievable
outcome after one more step:

$$
G_k = \begin{cases}
\sigma(q_{k+1}[\text{halt}]) & \text{if } k+1 = N_{\max} \\
\max\big(\sigma(q_{k+1}[\text{halt}]),\;
\sigma(q_{k+1}[\text{continue}])\big) & \text{otherwise}
\end{cases}
$$

where $\sigma$ is the sigmoid function. The loss is:

$$
\mathcal{L}_{\text{continue}} = \operatorname{BCE}\big(q_k[\text{continue}],\; G_k\big)
$$

When the next step is the final permitted step ($k+1 = N_{\max}$), its
halt value is the bootstrap target because further continuation is
unavailable — the model must stop there. At intermediate steps, the
target is the best of halting or continuing — the controller learns
whether another step is expected to improve the outcome. No discount
factor or explicit step penalty is applied; the finite horizon
$N_{\max}$ is the only bound on computation.

### Combined objective

$$
\mathcal{L} = c_{\text{task}} \cdot \mathcal{L}_{\text{task}}
             + c_{\text{halt}} \cdot \mathcal{L}_{\text{halt}}
             + c_{\text{continue}} \cdot \mathcal{L}_{\text{continue}}
$$

The coefficients are configured by the training recipe; the repository
defaults are $c_{\text{task}} = 1.0$, $c_{\text{halt}} = 0.5$,
$c_{\text{continue}} = 0.5$. No universal weighting is implied.

## What the controller learns

The halt value $q[\text{halt}]$ learns to predict whether the current
prediction is correct. The continue value $q[\text{continue}]$ learns
the expected value of taking one more step.

The difference $q[\text{halt}] - q[\text{continue}]$ is a decision margin:

- **Continue dominates**: additional reasoning has a higher predicted
  probability of eventual success than halting now.
- **Halt dominates**: the current prediction is good enough; further
  computation is unlikely to improve it.

Because there is no explicit step cost or discount, the controller learns
mainly whether eventual success is more likely after continuing, not
whether the expected improvement justifies the computational expense.
The finite horizon $N_{\max}$ is the only compute bound.

## Design properties

**No target network.** The regime uses the same network to compute
next-step targets. With large batch sizes and many parallel slots, the
target distribution is sufficiently stationary without a separate target
network — similar in spirit to Parallel Q-Networks (PQN). However, the
BCE loss over bounded targets differs from conventional DQN's Huber
regression over scalar returns.

**No replay buffer.** Transitions are consumed online within each
training step. The batch provides many independent episodes; the
controller learns from the aggregate without storing past transitions.

**Diagnostic halting at evaluation.** During evaluation, the controller
always executes all $N_{\max}$ steps to guarantee consistent batch shapes
and deterministic outputs. Halt/continue decisions are recorded as
diagnostics but do not truncate execution. Adaptive depth is trained
but not used to reduce evaluation compute.

## Relevant code surfaces

| Component            | Location                                  |
| -------------------- | ----------------------------------------- |
| Controller           | `ehc_sn.controllers.deliberation.act`     |
| Objective            | `ehc_sn.objectives.composites.act`        |
| Lightning module     | `ehc_sn.lightning.modules.act_supervised` |
| Halting control loss | `ehc_sn.objectives.control.halt`          |
| Rollout runner       | `ehc_sn.rollouts.runtime`                 |

## Failure modes

- **Premature halting**: the halt head overestimates current correctness,
  causing the controller to stop before additional deliberation can
  improve the prediction.
- **Always continue**: the continue bootstrap target stays high; the
  model never finds a reason to stop (mitigated by $N_{\max}$).
- **Overestimation**: the max operator in the continue target can inflate
  $q[\text{continue}]$, making the controller reluctant to halt.
