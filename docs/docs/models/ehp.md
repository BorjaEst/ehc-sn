# EHP Model

EHP (_Entorhinal-Hippocampal–Prefrontal_) is the integrated model family that
combines structured hippocampal memory with prefrontal recurrent reasoning for
goal-directed spatial tasks. It tests whether novel route construction can
emerge from repeated PFC-controlled retrieval over a cognitive map learned
through experience.

## What EHP is

The system first experiences an environment through trajectories (Arena
pretraining). Its entorhinal–hippocampal (EC–HPC) subsystem learns spatial
structure, sensory bindings, and transition relations. Later, the system
receives a start and a goal but is **not** shown the environment topology
again.

```
Previously experienced environment
        +
current start and goal
        ↓
retrieve relevant structural information from memory
        ↓
reason over that information
        ↓
construct a novel route
```

The target route may correspond to a start–goal pair the system never
encountered during training. EHP must compose previously learned local and
structural information into a new solution — it must not simply memorize
complete routes.

## Division of responsibility

EHP establishes a precise division between subsystems inherited from other
model families.

| Subsystem                                                 | Source family | Role                                                                                                       |
| --------------------------------------------------------- | ------------- | ---------------------------------------------------------------------------------------------------------- |
| **EC–HPC** (entorhinal–hippocampal)                       | TEM           | Learns, binds, stores, and retrieves environment structure                                                 |
| **PFC / HRM** (prefrontal / hierarchical recurrent model) | HRM           | Maintains the current task, formulates memory queries, integrates retrieved evidence, constructs the route |

The central unresolved mechanism is:

> How does the evolving PFC state formulate queries to HPC, and how does
> retrieved hippocampal information enter the PFC reasoning process?

The four EHP versions — `ehp_v1` through `ehp_v4` — are four controlled
answers to that question, differing only in the query mechanism.

## Relationship to other model families

**Arena** is the acquisition task. The model experiences trajectories through
a fixed environment. At each step it receives location or observation
information plus an executed action and predicts the next state. Across
sufficiently diverse trajectories, the TEM-like EC–HPC subsystem learns which
locations exist, what sensory content is associated with each, which locations
are connected, which actions produce which transitions, and how local
transitions compose into larger-scale topology. The result is a structured
cognitive map — not a collection of memorized start–goal routes.

**Routebind** provides the downstream reasoning problem. In the ordinary HRM
version, the model sees the full map layout (walls, free cells, observation
identities, start, goals) and predicts a prospective trajectory field. In the
EHP version, the topology is no longer supplied in the current input:

```
HRM Routebind:
Here is the map, start, and goal. Reason over the visible map.

EHP Routebind:
You previously learned the map through experience.
Here are the start and goal. Retrieve what you need and reason.
```

The desired route behavior remains comparable, but the source of structural
information changes from current perception to learned memory.

**HRM** supplies the recurrent PFC-like computation. It maintains a structured
working state across repeated deliberation cycles. The workspace contains
spatial slots (one per grid position), a global CLS token, and — depending on
the EHP variant — additional retrieval tokens. The HRM computation produces
the route through iterative reasoning; it does not receive a complete route
from HPC.

**TEM** supplies the conceptual basis for learning and storing structured
environmental knowledge. The relevant division is MEC-like (structural states
and transitions), LEC-like (sensory or observation content), and HPC (binding
between structural state and arena-specific content). Arena trains this
subsystem through trajectories; EHP later uses it as a memory system.

For detailed theoretical background, see
[EHP Theory Foundations](../ehp-theory-foundations.md) and
[PFC Working Memory Theory](../pfc-working-memory-theory.md).

## Regional modules

EHP comprises four functional modules inherited from the TEM and HRM families.

| Module | Region                    | Primary representation                     | Role                                                                                     |
| ------ | ------------------------- | ------------------------------------------ | ---------------------------------------------------------------------------------------- |
| MEC    | Medial Entorhinal Cortex  | $g_t$ — structural state code              | Encodes the current location or state in a structural coordinate system                  |
| LEC    | Lateral Entorhinal Cortex | $x_t$ — sensory content code               | Encodes observation identity and sensory cues                                            |
| HPC    | Hippocampus               | $\mathcal{M}_t$ — conjunctive memory store | Stores and retrieves bindings between structure and content                              |
| PFC    | Prefrontal Cortex         | $S_t$ — recurrent workspace state          | Maintains the task, formulates memory queries, integrates evidence, constructs the route |

Each module encodes a shared latent referent from a different perspective.
Learnable projections connect the regions without imposing identity between
their codes. The notation used here follows the canonical repository symbol
table defined in [Notation Map](../notation-map.md).

## The PFC workspace

The PFC workspace $S_t$ uses a fixed spatial-slot layout inherited from
Routebind. Each spatial slot corresponds to one physical grid position with a
stable row-major interpretation:

```
slot p ↔ physical position p
```

For the canonical running example of a $30 \times 30$ arena, there are
$P = 900$ spatial slots plus one global CLS token ($\bar z_t$, mapped to
`theta_cls` in HRM and `theta_summary` in EHP). The spatial slots carry fixed
positional identity (row/column embeddings, learned structural location codes),
but their activity content evolves throughout reasoning.

At $t = 0$, the adapter initializes each slot with:

```
position_code(p) + start_role (if p is start) + goal_role (if p is goal)
```

The slots do **not** initially contain wall/free identity, observation
identity, local adjacency, learned transitions, route membership, or path
distance. All topology must come from the learned EC–HPC memory.

The adapter owns the task-representation boundary: it converts start position,
goal-position mask, and spatial schema identity into the initial PFC
workspace. It must not retrieve topology, compute the route, infer semantic
transitions, or decide which HPC memory to use. Those operations belong to the
model and the memory-interaction mechanism.

## The common computational loop

EHP is not a simple feed-forward sequence where TEM retrieves a map and HRM
solves the task. The intended process is recurrent:

```
PFC reasons
→ identifies an information need
→ queries HPC
→ receives memory evidence
→ updates its hypothesis
→ identifies a new information need
→ queries again
→ ...
→ constructs the route
```

Let $\mathcal{Q}_t$ be the query sent to HPC and $\mathcal{M}_t$ be the
memory evidence returned. The recurrent computation is:

$$
\begin{aligned}
\mathcal{Q}_t &= \text{QueryMechanism}(S_t, \bar z_t) \\[2pt]
\mathcal{M}_t &= \text{HPC.retrieve}(\mathcal{Q}_t) \\[2pt]
S_{t+1},\; \bar z_{t+1} &= \text{HRM.update}(S_t, \bar z_t, \mathcal{M}_t)
\end{aligned}
$$

This repeats until the PFC has accumulated and composed enough evidence to
generate a route solution. The HPC response is memory evidence — distributed
and potentially ambiguous — not a symbolic answer. The PFC is responsible for
deciding which returned information is relevant, where it belongs in the
current working hypothesis, and when reasoning is sufficient.

### Two-level temporality

EHP separates two timescales, inherited from the HRM slow-fast recurrence
described in [PFC Working Memory Theory](../pfc-working-memory-theory.md).

**Internal HRM recurrence** ($z_L$, fast). The low-level HRM state performs
local spatial propagation, candidate-route refinement, suppression of
incompatible branches, and trajectory-field construction using the information
currently available in PFC working memory. Multiple fast updates occur per
deliberation cycle.

**PFC–HPC retrieval recurrence** (slow). At selected points, the high-level
HRM state ($z_H$) identifies an unresolved information need, the query
mechanism formulates an HPC cue, memory evidence is retrieved and integrated,
and the refined state feeds back into fast reasoning.

The conceptual schedule is:

```
several fast PFC reasoning updates
        ↓
high-level state identifies an unresolved information need
        ↓
HPC query
        ↓
memory integration
        ↓
further fast PFC reasoning
```

Not every low-level HRM update must trigger retrieval. The mapping between
$z_H$ (slow, integrative) and retrieval control is a design hypothesis, not
an automatic identity. The exact role of $z_H$ in retrieval gating remains
testable.

## The four query-interface variants

The four EHP versions are **proposed experimental variants** of the future EHP
family. They are not descriptions of the existing `EHCModelV1` implementation,
whose name contains "v1" as the first code-level EHP model version. The labels
`ehp_v1`–`ehp_v4` denote proposed query-interface variants and do not imply
equivalence to `EHCModelV1`.

| Version  | Query source                             | Query structure                       | Central hypothesis                                                                                                                                        |
| -------- | ---------------------------------------- | ------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ehp_v1` | CLS token                                | One global query                      | A single global recurrent control state is sufficient for PFC-guided retrieval and compositional route reasoning.                                         |
| `ehp_v2` | Dedicated cue tokens                     | Small fixed bank of queries           | Multiple persistent retrieval channels improve memory-guided reasoning over a single global query.                                                        |
| `ehp_v3` | Selected spatial frontiers               | Queries grounded in current locations | Grounding hippocampal queries in the current spatial reasoning frontier improves compositional route construction.                                        |
| `ehp_v4` | Dynamic projection of complete PFC state | Low-rank distributed queries          | A dynamic low-rank population interface can provide sufficient bandwidth for recurrent PFC-controlled retrieval while preserving compositional reasoning. |

These versions form a meaningful progression:

```
ehp_v1: one global query
ehp_v2: multiple global query channels
ehp_v3: multiple queries explicitly grounded in selected spatial states
ehp_v4: multiple dynamic distributed queries over the full PFC population
```

All four share the same Arena-pretrained memory, the same downstream Routebind
task, the same HRM backbone, the same output heads, the same training and
evaluation data, and the same causal controls. The independent variable is
how the PFC constructs the HPC query. A full comparison with equations,
training stages, controls, and causal evidence requirements is in
[Memory-Mediated Reasoning](../memory-mediated-reasoning.md).

## What HPC should return

The HPC response should be understood as memory evidence, not a symbolic
answer. Depending on the memory representation, a returned vector may encode a
remembered location, bound observation and structural state, successor or
predecessor evidence, action-conditioned transition information, a trajectory
fragment, a local neighborhood, or similarity to previously experienced
states. The return may be distributed and ambiguous.

The safe common contract is:

```
HPC returns evidence in a latent retrieval space.
PFC/HRM integrates that evidence into working memory.
PFC decides what the evidence means for the current route hypothesis.
```

Directly decoding HPC responses into final trajectory outputs would undermine
the intended reasoning claim.

## Route construction

The final route emerges in the PFC workspace $S_t$. The output retains
Routebind-style supervision:

- **trajectory field**: complete physical route through the arena;
- **waypoint field**: selected semantic observations or structurally
  meaningful states;
- **next direction** and **next observation**, where applicable.

The final prediction is decoded from the PFC state after recurrent retrieval
and reasoning:

```
S_final
    ↓
task decoder
    ↓
trajectory field
waypoint field
auxiliary outputs
```

This preserves continuity with the HRM Routebind benchmark while changing the
source of topology from direct input to hippocampal memory.

## What EHP is expected to generalize

The primary generalization target is unseen start–goal composition within a
previously learned environment. The model may have experienced fragments A→B,
B→C, C→D, B→E, E→F during Arena training, but never the complete query
(start = A, goal = F). At inference, the PFC must use retrieved relations to
construct A→B→E→F. The route is novel as a task solution even though its
constituent transitions were experienced.

The initial scientific claim is:

> After learning one environment through Arena experience, EHP can use its
> stored cognitive map to solve novel goal-directed queries in that
> environment.

Transfer to unseen topologies, instant learning of new arenas, and universal
graph reasoning are **not** part of the initial claim.

## Important non-claims

Successful EHP performance does **not** automatically prove:

- exact biological fidelity to hippocampal–prefrontal anatomy;
- explicit symbolic graph storage in HPC;
- human-like planning or episodic recall in the autobiographical sense;
- that every PFC slot corresponds to a literal biological neuron;
- that cue tokens or low-rank projections are biological entities;
- generalization to completely unseen environments.

The 900 slots, CLS token, cue tokens, and projection operators are
computational abstractions. The scientific target is functional: can a
PFC-like recurrent reasoner control retrieval from a TEM-like structured
memory and use the returned evidence to solve novel goal-directed route
problems?

## Implementation status

- **`EHCModelV1`** — the current code-level EHP model (imported as
  `ehp_sn.models.ehp.ehp_v1.EHCModelV1`). Implements an integrated
  PFC/STR/LEC/MEC/HPC step surface with adapter-mediated task binding. This is
  the existing implementation, not one of the four proposed query-interface
  variants.
- **`ehp_v1`–`ehp_v4`** — proposed query-interface taxonomy described in this
  document and detailed in the Memory-Mediated Reasoning theory page. These
  are forward-looking experimental designs and do not yet have dedicated model
  classes. The `ehp_v1` label in this taxonomy should not be confused with the
  existing `EHCModelV1` implementation.

## Entry point

```bash
scripts/training/ehc_v1_pretraining.py
```

## Further reading

- [Memory-Mediated Reasoning](../memory-mediated-reasoning.md) — full
  theoretical framework with equations, training stages, controls, and causal
  evidence requirements for `ehp_v1`–`ehp_v4`.
- [EHP Theory Foundations](../ehp-theory-foundations.md) — the five-ingredient
  theoretical synthesis (TEM, transformer-hippocampus, structured PFC,
  hierarchical reasoning, top-down control).
- [PFC Working Memory Theory](../pfc-working-memory-theory.md) — slot layout,
  $z_H$/$z_L$ dynamics, rotational phase geometry, and the fixed-slot-names
  principle.
- [Notation Map](../notation-map.md) — canonical symbol table mapping
  manuscript notation to code surfaces.
