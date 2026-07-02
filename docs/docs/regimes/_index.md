# Training Regimes

A training regime defines how a model is optimized: the objective family,
the controller that drives recurrent computation, and the halting or
replay strategy.

## Taxonomy

```text
Training regimes
├── supervised-act
│   └── conceptual differentiable adaptive computation (not implemented)
├── supervised-q-halting
│   └── implemented hard value-based adaptive computation
└── variational-replay
    └── implemented fixed-depth trajectory learning
```

Supervised ACT and supervised Q-halting are alternative
adaptive-computation mechanisms — they answer "how many steps?".
Variational replay is a different training family — it replays complete
trajectories to learn a world model without adaptive depth.

## Regime map

| Regime                                          | Task signal                            | Computation policy          | Repository status                         |
| ----------------------------------------------- | -------------------------------------- | --------------------------- | ----------------------------------------- |
| [Supervised ACT](supervised-act.md)             | Supervised                             | Differentiable soft halting | Conceptual, not implemented               |
| [Supervised Q-halting](supervised-q-halting.md) | Supervised + bootstrapped values       | Hard halt/continue          | Implemented; legacy name `act_supervised` |
| [Variational replay](variational-replay.md)     | TEM-specific (obs + consistency + reg) | Fixed trajectory replay     | Implemented for TEM family                |

## How regimes relate to model families

- **HRM-family models** (Hierarchical Reasoning Model) use recurrent
  deliberation steps. They pair with a halting regime — currently
  supervised Q-halting. Supervised ACT is documented as the conceptual
  alternative.
- **TEM-family models** (Tolman-Eichenbaum Machine) learn structural
  representations from sequential experience. They use the variational
  replay regime.

## Choosing a regime

1. If the task provides supervised targets, the model produces meaningful
   intermediate predictions, and computation depth should vary across
   examples → supervised Q-halting.
2. If your task requires **structural representation learning** from
   sequential observations without adaptive depth → variational replay.
3. If you want **fully differentiable adaptive depth** with soft halting
   and are willing to tune the ponder coefficient → classical supervised
   ACT (not yet implemented).

## Shared infrastructure

Implemented regimes are coordinated through common lower-layer
infrastructure:

- **Rollout runners** (`ehp_sn.rollouts.runtime`) execute recurrent steps
  and capture per-step records.
- **Objective composites** (`ehp_sn.objectives.composites`) compute loss
  terms from step records without knowing about controller internals.
- **Metrics and reducers** (`ehp_sn.metrics`) aggregate per-step
  statistics into epoch-level summaries.
- **Evaluation executors** (`ehp_sn.evaluation`) run diagnostic and benchmark
  evaluation passes independent of the training path.

## Legacy naming

The codebase uses the legacy prefix `act_` (e.g., `ACTController`,
`act_supervised`) for what is mechanistically a Q-learning-style adaptive
halting controller. This documentation uses the term _supervised
Q-halting_ to describe the mechanism accurately. The two names refer to
the same regime.
