# `seqmaze` Benchmark Task

## Task identity and overview

Task name: `seqmaze`

Benchmark family:

| Symbol   | Description                                            |
| -------- | ------------------------------------------------------ |
| $obs[t]$ | observation identifier at time t                       |
| $x'[t]$  | full token information for at time t (decoded sensory) |
| $x[t]$   | embedding for token node at time t (embedding sensory) |

Canonical package path:

```text
src/ehc_sn/tasks/seqmaze/
```

The task is built around the idea of 2D mazes, but the model does not receive any explicit spatial information. Instead, the model receives a set of observation tokens and valid successor transitions between those tokens. The model must infer the correct sequence of observations that leads from a start token to a goal token, given the valid transitions.

The task is designed to test the model's ability to reason about sequences and transitions, rather than relying on spatial memory. The model must learn to read the graph of valid transitions and infer the correct path from start to goal, without any explicit spatial grounding. To do so, the model receives the start (obs[0]) and goal (obs[n]) observation tokens together with all the information needed to infer the correct sequence of observations.

The core problem is:

```text
given the start (obs[0]) and goal (obs[n]) observation tokens
the candidate set of observation tokens (obs[0:n+m])
and valid successor transitions for each token (obs[i] → obs[j])
predict the shortest valid observation sequence from start to goal (obs[0] → obs[1] → ... → obs[n])
```

---

## Relationship to other tasks

### MazeHard analogy

In `mazehard`, each token corresponds to a fixed grid cell, which provides a strong spatial grounding. The model can learn to navigate the maze by learning the spatial layout and the transitions between cells because the embedding adds the "what" (wall, open, start, goal) and the "where" (cell location).

```text
MazeHard:
  cell tokens + wall/open status + start + goal → route
```

In `seqmaze`, each token corresponds to an observation node.

```text
seqmaze:
  observation-node tokens + valid successor information + start + goal → token route
```

The key difference is that `seqmaze` has no fixed 2D position. Therefore, transition structure must be encoded inside the node tokens.

### Anti-memorization contract

The model must not solve the task by memorizing fixed transitions such as:

```text
obs_5 always goes to obs_3
```

So each sample should be generated with sample-local structure:

```text
candidate order is permuted
obs ids may be remapped
successor graph is generated per sample
shortest path is computed offline
```

---

## Scientific purpose

`seqmaze` is designed to test the model's ability to reason about sequences and transitions in a non-spatial, non-memorization-based way. The model must learn to read the graph of valid transitions and infer the correct path from start to goal, without any explicit spatial grounding. This tests the model's ability to perform reasoning over a structured graph of tokens, which is a fundamental aspect of many cognitive tasks.

The relevant HRM computation is not:

```text
read this graph → memorize/recall the path
```

It is:

```text
read this graph → infer the path
```

This is needed so the model can perform the PFC-like reasoning process of inferring the correct sequence of observations that leads to a goal, based on the structure of valid transitions (in the future provided by the HPC-EC). This is a key aspect of the reasoning process that we want to test, and it is important to have a task that isolates this reasoning ability without confounding it with spatial memory or other factors.

## Input and output

Adapter input (task-data output):

| Variable    | Description                                                    |
| ----------- | -------------------------------------------------------------- |
| $x'[0:n+m]$ | candidate set of observation tokens (including start and goal) |

The token info contains all the information needed to describe the valid transitions between tokens, such as successor indices and masks. The model must use this information to infer the correct sequence of observations that leads from the start token to the goal token. The model should not rely on any external spatial information or memory of specific locations, but rather on the reasoning process of inferring the correct path through the graph of tokens and transitions.

Adapter output (evaluation input):

| Variable  | Description                                                         |
| --------- | ------------------------------------------------------------------- |
| $x'[0:T]$ | predicted sequence of observation tokens (including start and goal) |

Use one array of structured node tokens:

```text
[N, F]
```

where `N` is the number of candidate observations.

Each node token should contain:

| Attribute           | Description                                                        |
| ------------------- | ------------------------------------------------------------------ |
| `obs_id`            | unique identifier for the observation token (e.g., obs_5)          |
| `candidate_index`   | index of the token in the candidate set (0 to N-1)                 |
| `start_flag`        | binary flag indicating if this token is the start token            |
| `goal_flag`         | binary flag indicating if this token is the goal token             |
| `successor_indices` | list of indices of valid successor tokens (e.g., [2, 3, PAD, PAD]) |
| `successor_mask`    | binary mask indicating valid successors (e.g., [1, 1, 0, 0])       |

Example:

```text
tokens:
  obs_5, obs_3, obs_4, obs_0, obs_1, obs_2

start:
  obs_0

goal:
  obs_3

valid transitions:
  obs_0 → obs_4
  obs_5 → obs_4
  obs_3 → obs_2
  obs_4 → obs_1
  obs_4 → obs_3

target:
  obs_0 → obs_4 → obs_3, EOS, PAD, PAD, ...
```

## Benchmark and evaluation

The main target is a variable-length token sequence:

```text
[obs_0, obs_4, ..., obs_T, EOS, PAD, PAD, ...]
```

- Like in LLM generation tasks, the model should generate the sequence token by token until it generates an EOS token. Therefore the correct way to evaluate is to compare the generated sequence with the target sequence using sequence-level exact match (sequence_exact) as the primary metric. This means that the generated sequence must exactly match the target sequence, including the order of tokens and the presence of EOS and PAD tokens.

- Like in LLM training, we can also evaluate next-token accuracy (next_token_accuracy) as a secondary metric, which measures the accuracy of predicting the next token in the sequence at each step. This provides a more fine-grained evaluation of the model's performance in generating the correct sequence.

Use teacher-forced sequence cross-entropy with padding masks.

For `seqmaze`, keep the generation constrained so each sample has a **unique shortest path**. Multiple valid solutions can be added later.

The primary metric should be the exact match of the generated sequence with the target sequence:

```text
sequence_exact
```

Secondary metrics can be

```text
next_token_accuracy
valid_transition_rate
reaches_goal
path_length_regret
```

---

## Open questions

Here are the open questions that need to be resolved to finalize the task design:

- How to generate the valid transition graph for each sample? Should it be random, or should it follow some specific structure (e.g., tree, DAG, cyclic graph)?

---
