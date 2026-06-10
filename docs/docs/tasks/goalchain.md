# `goalchain` Benchmark Task

## Task identity and overview

Task name: `goalchain`

Benchmark family: goal-conditioned sequence reasoning / memory-conditioned shortest-walk planning

| Symbol   | Description                                                              |
| -------- | ------------------------------------------------------------------------ |
| $obs[t]$ | observation identifier at time t                                         |
| $x'[t]$  | sames as observation identifier at time t (decoded sensory)              |
| $x[t]$   | latent "what" sensory state at time t (LEC encoded sensory)              |
| $g'[t]$  | location identifier at time t (decoded topology)                         |
| $g[t]$   | latent "where" location state at time t (MEC encoded topology)           |
| $M$      | memory state produced by structural learning                             |
| $a[t]$   | action at time t                                                         |
| $p[t]$   | latent sensory-location state of the model at time t (HPC encoded state) |
| $goal'$  | goal cue (decoded goal cue)                                              |
| $goal$   | latent goal cue (PFC encoded goal cue)                                   |

Canonical package path:

```text
src/ehc_sn/tasks/goalchain/
```

The task is built on graph layouts, such as dungeongen layouts, through the public layout API. However, it requires training and the memory ($M$) in the state produced by the model during the structural learning phase. This memory contains the learned environment structure and the observation-location bindings needed to reason about the current layout. The task that trains on the structural learning phase is "arena", see the `arena` benchmark task. Of course, training in the structural learning phase is a hard requirement as the model needs to correctly interpret the memory and understand the structural relations.

The input to `goalchain` includes a layout-matched state produced from the structural learning phase. This state contains the memory ($M$) with the observation-location bindings needed to reason about the current layout and the model believed location ($g$) and possibly other variables ($p$, $x$, etc.). The task is designed to test whether the model can use goal context ($goal$) and episodic memories ($M$) to solve a grounded sequence-reasoning problem.

The core problem is:

```text
given a final goal observation cue (goal[episode] == x[n]),
infer the required prerequisite observation chain (x[0:n]),
retrieve candidate locations for the next required observation (g[0:m]),
and select the best concrete target location under the current spatial structure (g[0:n]).
```

---

## Relationship to other tasks

### Prerequisites and pre knowledge

To successfully engage with the `goalchain` task, a model must have been trained on the `arena` task to learn the environment structure and produce the necessary memory states. The `goalchain` task assumes that the model has already acquired the ability to encode spatial layouts and observation-location bindings through experience in the `arena` task. Therefore, it is essential that the model has been exposed to a variety of layouts and has developed a robust memory representation of those layouts before attempting the `goalchain` task.

```text
context trajectory
    → model is pre-trained to learn structural knowledge (e.g. trained grid cells)
    → produce episode memory state with observation-location bindings ($M$)
    → produce a location belief state ($g$) as a starting point for best-target selection
```

This ensures the model remembers the environment (state memory) and understands the spatial structure (model parameters), which is crucial for solving the `goalchain` task.

### Goal-chain reasoning

Once the model has the correct memories and parameters to understand the layout, the `goalchain` task tests whether it can use that information to solve a goal-conditioned sequence prediction problem. The model receives a final goal observation cue ($goal$), the episodic memory ($M$) and the believed latent location ($g$, where the agent believes it is currently located). The model must then infer the correct prerequisite observation and location sequence needed to solve the shortest-walk navigation problem.

```text
minimal initial state
    → layout-matched memory state
    → location belief state
final goal cue
    → infer the required sequence
    → predict the next required observations / targets for shortest-walk navigation
```

This is `goalchain`. It does not ask the model to learn the graph from scratch. It assumes that the model already has access to the memory state corresponding to the same layout.

The important contract is:

```text
each goalchain episode layout must be paired with the correct memory state for its layout.
```

So if a sample comes from layout `L_i`, the model receives the memory state produced from the structural task on layout `L_i`, not from another layout.

---

## Scientific purpose

`goalchain` tests whether a model can use a learned memory state to solve a goal-conditioned sequence prediction problem.

The relevant EHP computation is no longer:

```text
learn the map
```

It is:

```text
use the learned map-memory to infer the correct sequence toward a goal
```

Interpretation:

| Component              | Role in `goalchain`                                           |
| ---------------------- | ------------------------------------------------------------- |
| HPC / episodic memory  | provides remembered structure and observation locations.      |
| PFC / reasoning module | infer the correct sequence of observations or targets.        |
| MEC-like structure     | provides location identifiers and path integration.           |
| LEC-like content       | provides observation identifiers used to ground the sequence. |

The benchmark therefore isolates whether the reasoning module can use memory to produce the right goal-directed chain.

---

## Input and output

Adapter input (task-data output):

| Variable        | Description                                                          |
| --------------- | -------------------------------------------------------------------- |
| $goal$ (obs_id) | the final goal observation identifier (decoded goal cue)             |
| $M$             | memory state containing the observation-location bindings and layout |
| $g'$            | the model's believed location ("where" state)                        |

The memory state is produced by the previous structural learning task and must correspond to the same layout as the current sample.

Adapter output (evaluation input):

| Variable          | Description                                                                |
| ----------------- | -------------------------------------------------------------------------- |
| $x'[0:n] sequence | the predicted sequence of observation identifiers needed to reach the goal |
| $g'[0:n] sequence | the predicted sequence of location identifiers needed to reach the goal    |

Example:

```text
Initial location state:
    g_7

Goal cue:
    obs_id = 5  # goal observation identifier

Memory state:
    learned structure and observation locations for layout L_i

Target observation sequence:
    x'_0 → x'_1 → x'_2 → x'_3 → x'_4 → x'_5

Target location sequence:
    g'_0 → g'_1 → g'_2 → g'_3 → g'_4 → g'_5
```

The target location sequence should be the one needed to resolve the shortest-walk navigation problem under the current layout.

---

## Benchmark and evaluation

The task is evaluated on multiple axes:

- The capability of the model to discern the correct sequence of observations needed to reach the goal, given the memory state and the goal cue. This is pure reasoning evaluation, it does not require episodic recall of the exact location of the goal observation, but rather whether the model can infer the correct chain of observations that leads to the goal. (supervised evaluation against the decided pattern)

- The capability of the model to select the correct target locations for each required observation in the sequence, given the memory state and the current location belief. This tests whether the model can use the memory to retrieve candidate locations for each required observation and select the best one under the current spatial structure.

- The capability of the model to understand the spatial structure and use it to solve the shortest-walk navigation problem. This is evaluated by the amount of steps required to reach the goal observation from the initial location, following the predicted sequence of locations. The fewer steps, the better.

---

## Open questions

Here are the open questions that need to be resolved to finalize the task design:

- The model needs to know where it is located and indicate what location it wants to go to. To tell the model where start is easy, we can take the final state from structural learning pre-training so the model would correctly believe it is at the end of the trajectory we used to generate that model state. However, the difficulty relies on how to translate the latent location of where the model wants to go to the decoded topology location. It is a encode-decode problem between the latent location and the decoded location. What is the best way to generate this autoencoder? Should we train a separate encoder-decoder on the latent representations ($g_t$ and $g'_t$) states to learn this translation? Or should we use the same model to learn it through experience? This is an open question that needs to be resolved.

- Why HRM model as PFC? Because the reasoning process requires to keep track of the global solution (the sequence of best locations to go) in the high control loop ($z_H$) and the processing of possible best candidates in the low control loop ($z_L$). This is a natural fit for the HRM architecture, where ($z_H$) can maintain the global plan and ($z_L$) can evaluate candidate next steps, see `jolicoeur-martineau_less_2025`.

- The reasoning part of the problem, to predict the observation sequence, is a pure reasoning problem. It does not require episodic recall of the exact location of the goal observation, but rather whether the model can infer the correct chain of observations that leads to the goal. This is a key point because it tests whether the model can use the memory to infer the correct sequence, rather than just recall a specific location. To succeed, there should be a logic in the selection and order of the observations sequence that the model can learn to infer. It probably is wise to create a new task to pre-train the model on this pure reasoning problem, without the spatial structure.

- The token information needed by PFC is not simply the observation identifier, but also the information about the valid transitions between observations. This is needed so the model can infer the correct sequence of observations that leads to a goal, based on the structure of valid transitions (in the future provided by the HPC-EC). This is a key aspect of the reasoning process that we want to test, and it is important to have a task that isolates this reasoning ability without confounding it with spatial memory or other factors. However, the recall of by the "cue" of the HPC might not be enough if it only stores the encoded observation identifier. The latent representation ($p$) should also contain the information about the valid transitions between observations, so the model can use that information to infer the correct sequence. This is an open question that needs to be resolved.

- HRM Mazehard solves the navigation problem because each embedding contains both the "what" (wall, open, start, goal) and the "where" (cell location). This approach is very similar to the role of the HPC which combines both, however, insufficient the the TEM approach where the "what" is a two-hot vector of the observation identifier and the "where" is a simple separate location belief. The question is how to extend TEM design so it provides the same structural knowledge learning but where the tokens contain the information about the valid transitions between observations, so the model can use that information to infer the correct sequence.

---
