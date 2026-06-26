# Variational replay training

**Repository identifiers:** `VariationalReplayModule`,
`ReplayTrajectoryController`; canonical name _variational replay_.

## What it is

A training regime for models that learn structural representations from
sequential experience. The regime combines three ideas:

1. **Latent representation learning** — the model learns to encode
   sensory observations into structural and conjunctive latent codes
   and reconstruct observations from them.
2. **Learned transition dynamics** — the model learns how actions change
   the structural state, enabling forward simulation.
3. **Trajectory replay** — episodes from the dataset are replayed through
   the model step by step, and the controller walks each trajectory
   deterministically.

There is no halting controller: the regime replays every valid step
supplied by the trajectory batch. The computation depth is fixed by the
episode length.

## What it is not

This regime is **not** a deliberation or reasoning regime. It does not
make online decisions about how many steps to compute. It replays
complete episodes to learn a structured latent transition and observation
model.

It is also **not** a pure supervised regime, and "replay" here does not
mean DQN-style experience replay. Episodes are loaded from the processed
dataset; there is no persistent replay buffer, no value function, and no
policy being learned.

## Mechanism

### Latent variable semantics

The model maintains three kinds of representations:

| Variable | Meaning                                                        |
| -------- | -------------------------------------------------------------- |
| $x_t$    | Encoded sensory observation                                    |
| $g_t$    | Structural/location representation (grid-like code)            |
| $p_t$    | Hippocampal conjunctive representation binding $g_t$ and $x_t$ |

The structural state $g_t$ captures the agent's position in an abstract
state space and evolves under actions. The conjunctive representation
$p_t$ binds structural context to specific sensory content, serving as
the bridge between grid-like and sensory codes.

### State transition

Given structural state $g_{t-1}$ and action $a_{t-1}$, the model predicts
the next structural state:

$$
g_t^- = T(g_{t-1}, a_{t-1})
$$

where $T$ is a learned transition operator — typically a matrix or
MLP-modulated linear map conditioned on the action.

Sensory evidence then refines this prior:

$$
g_t = \text{infer}(g_t^-, x_t)
$$

where $x_t$ is the encoded observation at time $t$.

### Observation encoding and decoding

An encoder maps raw observations to a sensory embedding:

$$
x_t = E(o_t)
$$

A decoder maps sensory content back to observation space:

$$
\hat{o}_t = D(p_t)
$$

where $p_t$ is the grounded representation derived from $g_t$ and the
memory store.

### Memory store (conceptual)

A schematic associative-memory update associates conjunctive
representations with structural states:

$$
M \leftarrow (1 - \eta) M + \eta \cdot g_t \cdot p_t^\top
$$

where $\eta$ controls the learning rate. Retrieval reads:

$$
p = M^\top g
$$

This captures the core idea of hippocampal-style associative memory:
structural states act as addresses, conjunctive representations as
stored content. The repository implementation may support factorized
banks, directional mappings, and learned retrieval beyond this simple
outer-product form.

## Objective

The repository objective (`TEMObjective`) composes several loss terms.
It is not a generic VAE ELBO — the terms are TEM-specific.

### Observation losses

Three prediction pathways are scored against ground-truth observation
labels via cross-entropy:

| Pathway         | Loss key          | Description                                                                      |
| --------------- | ----------------- | -------------------------------------------------------------------------------- |
| Path prediction | `loss_obs_path`   | Decode from the structural state after transition and sensory update             |
| Post-diction    | `loss_obs_post`   | Decode after observing the full sequence (posterior)                             |
| Recall          | `loss_obs_recall` | Decode from a structural state queried from memory without current sensory input |

Summed: $\mathcal{L}_{\text{obs}} = \mathcal{L}_{\text{path}} +
\mathcal{L}_{\text{post}} + \mathcal{L}_{\text{recall}}$

### Place consistency

Two terms encourage agreement between place-code representations:

- **Place transition consistency** (`loss_place_transition`): place code
  predicted from the transition should match the inferred place code.
- **Place sensory consistency** (`loss_place_sensory`): place code
  derived from sensory input should match the memory-recalled place code.

Summed: $\mathcal{L}_{\text{place}} = \mathcal{L}_{\text{place\_trans}} +
\mathcal{L}_{\text{place\_sens}}$

### Grid posterior regularization

The posterior grid code distribution is regularized toward a prior:

- **Grid KL divergence** (`loss_grid_kl`): KL divergence between the
  posterior grid distribution and a learned or fixed prior.

### Regularization

- **Grid regularization** (`loss_grid_reg`): $\ell_2$ penalty on grid
  codes (default coefficient 0.01).
- **Place regularization** (`loss_place_reg`): $\ell_1$ penalty on place
  codes (default coefficient 0.02), encouraging sparsity.

### Combined objective

$$
\mathcal{L} = c_{\text{obs}} \mathcal{L}_{\text{obs}}
             + c_{\text{grid}} \mathcal{L}_{\text{grid\_kl}}
             + c_{\text{place}} \mathcal{L}_{\text{place}}
             + \mathcal{L}_{\text{reg}}
$$

Coefficients and regularization norms are configured via
`TEMObjectiveConfig`. Dynamic parameters (temperature, gating strength,
regularization weight) are scheduled via `TEMScoringContext` and evolve
over the course of training.

## Trajectory replay

Training proceeds by replaying episodes from the processed dataset.
The `ReplayTrajectoryController` walks through each trajectory
step-by-step:

```text
for each trajectory in batch (loaded from dataset):
    g_0 ← initial structural state
    for t = 1 .. T (every valid step):
        g_t⁻ ← T(g_{t-1}, a_{t-1})           # transition
        x_t  ← E(o_t)                         # encode observation
        g_t  ← infer(g_t⁻, x_t)               # posterior update
        p_t  ← retrieve(M, g_t)               # recall from memory
        M    ← update(M, g_t, p_t)            # Hebbian store
        compute losses at step t
```

The controller advances the trajectory cursor deterministically through
the episode. There is no halting decision — every valid step supplied
by the batch is replayed.

## Design properties

**No halting controller.** The regime replays fixed-length trajectories.
Adaptive computation is not part of this training paradigm.

**No RL components.** There is no value function, policy, target
network, or experience-replay buffer. Trajectory replay refers to
replaying episodes from the dataset, not DQN-style experience replay.

**Hebbian memory is online.** Memory updates happen incrementally within
each trajectory. The memory matrix accumulates associations across all
visited states.

**Evaluation replay is deterministic.** During evaluation, the same
trajectory is replayed through the model. The evaluation metrics measure
how well the model reconstructs observations, predicts transitions, and
maintains consistent latent codes.

## Relevant code surfaces

| Component        | Location                                      |
| ---------------- | --------------------------------------------- |
| Controller       | `ehc_sn.controllers.replay.trajectory`        |
| Objective        | `ehc_sn.objectives.composites.tem`            |
| Lightning module | `ehc_sn.lightning.modules.variational_replay` |
| Rollout runner   | `ehc_sn.rollouts.runtime`                     |
| Consistency loss | `ehc_sn.loss.consistency`                     |
| Cross-entropy    | `ehc_sn.loss.cross_entropy`                   |
| Regularization   | `ehc_sn.loss.regularization`                  |

## Relationship to supervised Q-halting

| Property                | Variational replay                     | Supervised Q-halting             |
| ----------------------- | -------------------------------------- | -------------------------------- |
| Objective family        | TEM-specific (obs + consistency + reg) | Supervised + bootstrapped values |
| Halting                 | None (fixed-depth replay)              | Hard halt/continue               |
| Primary learning target | Structural + conjunctive codes         | Task prediction + halt value     |
| Memory                  | Hebbian associative store              | Not applicable                   |
| Current model pairing   | TEM-family                             | HRM-family                       |
