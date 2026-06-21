# Memory-Mediated Reasoning

This page defines the theoretical framework for memory-mediated route
reasoning in EHP. It owns the PFC→HPC query problem, the four proposed
query-interface variants (`ehp_v1`–`ehp_v4`), the training-stage
decomposition, the required controls, and the causal evidence contract.

It assumes the model-family concepts established in
[EHP Model](../models/ehp.md) and uses the canonical notation defined in
[Notation Map](../notation-map.md). The PFC workspace theory ($z_H$, $z_L$
dynamics, slot layout, fixed-slot-names principle) is documented separately in
[PFC Working Memory Theory](../pfc-working-memory-theory.md). The five
theoretical ingredients (TEM, transformer-hippocampus, structured PFC,
hierarchical reasoning, top-down control) are documented in
[EHP Theory Foundations](../ehc-theory-foundations.md).

### Notation

This page uses the repository symbol table from the Notation Map. Key symbols:

| Symbol               | Meaning                                       | Code surface                             |
| -------------------- | --------------------------------------------- | ---------------------------------------- |
| $S_t$                | PFC workspace (spatial slots + CLS token)     | `state.pfc.memory.z_H`                   |
| $\bar z_t$           | CLS/global summary state                      | `theta_cls` (HRM), `theta_summary` (EHP) |
| $z_{H,t}$, $z_{L,t}$ | High-level and low-level HRM recurrent states | `z_H`, `z_L` in PFC state                |
| $\mathcal{Q}_t$      | PFC→HPC retrieval query set                   | `ReadCues(...)`                          |
| $\mathcal{M}_t$      | Hippocampal memory content                    | `state.hpc.memory`                       |
| $g_t$                | Structural state (MEC code)                   | `grid_prior`, `grid_post`                |
| $x_t$                | Encoded sensory content (LEC code)            | `place_query_from_obs`                   |
| $c_t$                | Cortical cue                                  | `c_prop`                                 |
| $\hat c_t^{retr}$    | Reinstated contextual evidence                | `c_mem`                                  |
| $\xi_t$              | Task context                                  | prose-only in v1                         |

Additional symbols introduced for the query-interface analysis:

| Symbol           | Meaning                                                     |
| ---------------- | ----------------------------------------------------------- |
| $C_t$            | Bank of cue-token states (ehp_v2)                           |
| $R$              | Number of query channels / retrieval rank                   |
| $\alpha_t(r, p)$ | Selection weight of slot $p$ for query channel $r$ (ehp_v3) |
| $A_t$            | Dynamic projection matrix (ehp_v4)                          |

---

## The PFC→HPC query problem

### Why one-shot retrieval is insufficient

Even in one fixed $30 \times 30$ environment, the number of ordered start–goal
pairs approaches $900 \times 899$ before considering walls, multiple
observations, semantic constraints, or changing task context. Arena training
cannot expose every possible pair. Therefore the model must generalize
compositionally.

HPC should not be treated as storing:

```
(start, goal) → complete route
```

Instead, HPC contains reusable structural knowledge such as:

- location $p$ connects to location $q$ under action $a$;
- observation $x$ is bound to structural state $g$;
- trajectory fragment $p \to q \to r$ was experienced;
- region $A$ is connected to region $B$ through bottleneck $c$;
- state $q$ has predecessors $p_1$ and $p_2$.

A novel route is constructed by combining these relations.

The first start–goal cue may retrieve useful information, but that
information is unlikely to be complete, perfectly localized, cleanly
interpretable, or sufficient to determine the route immediately. It may
instead activate memories near the start, memories near the goal, overlapping
trajectory fragments, candidate transitions, coarse structural contexts, and
multiple competing intermediate states. The PFC must reason over this
material, determine what additional information is needed, and issue further
queries.

### The common recurrence

All four EHP versions share the same abstract computational loop.

At deliberation stage $t$, the PFC holds a spatial working state $S_t$ and a
global CLS state $\bar z_t$. The system executes:

$$
\begin{aligned}
\mathcal{Q}_t &= \text{QueryMechanism}(S_t, \bar z_t) \\[2pt]
\mathcal{M}_t &= \text{HPC.retrieve}(\mathcal{Q}_t) \\[2pt]
S_{t+1},\; \bar z_{t+1} &= \text{HRM.update}(S_t, \bar z_t, \mathcal{M}_t)
\end{aligned}
$$

Repeatedly:

```
current PFC hypothesis
        ↓
memory query
        ↓
hippocampal evidence
        ↓
PFC integration and reasoning
        ↓
refined hypothesis
        ↓
new query
```

The four versions differ only in the definition of
$\text{QueryMechanism}$. They otherwise share the same Arena-trained EC–HPC
memory, the same downstream Routebind task, the same HRM backbone, the same
output heads, the same training and evaluation data, and the same evidence and
ablation contract.

---

## Common HPC retrieval contract

### What queries should support

A query should be capable of cueing memory by combinations of:

- location or structural state;
- observation content;
- action or transition relation;
- successor direction;
- predecessor direction;
- arena context;
- current task context;
- partial trajectory fragment;
- high-level goal relevance.

The memory system should not require every query to specify all these
dimensions explicitly. The query is a latent vector or set of vectors learned
through training.

Nevertheless, the system should functionally support queries analogous to:

- _What is associated with this structural state?_
- _What transitions follow from this state?_
- _What states can precede this state?_
- _Where was this observation encountered?_
- _What trajectory fragments overlap this candidate region?_
- _Which stored structural pattern is relevant to this PFC hypothesis?_

### What HPC returns

The HPC response is memory evidence, not a symbolic answer. A returned vector
may encode a remembered location, bound observation and structural state,
successor or predecessor evidence, action-conditioned transition information,
a trajectory fragment, a local neighborhood, a coarse relational direction, an
arena context, or similarity to previously experienced states. The return may
be distributed and ambiguous.

### Response integration contract

The safe common contract is:

```
HPC returns evidence in a latent retrieval space.
PFC/HRM integrates that evidence into working memory.
PFC decides what the evidence means for the current route hypothesis.
```

For each version:

- **ehp_v1**: response enters CLS/global state, then propagates to spatial
  slots through HRM recurrence.
- **ehp_v2**: responses enter the corresponding cue tokens, which interact
  with CLS and spatial slots.
- **ehp_v3**: responses enter frontier-associated retrieval states and then
  influence spatial slots through HRM attention.
- **ehp_v4**: responses enter low-rank integration channels and are
  dynamically expanded into PFC evidence.

Directly decoding HPC responses into final trajectory outputs would undermine
the intended reasoning claim.

---

## `ehp_v1`: CLS-token query

### Central idea

`ehp_v1` uses the CLS token as the sole PFC→HPC query channel. The complete
900-slot working state is internally summarized into the CLS state through HRM
processing. The CLS vector then becomes the hippocampal query.

```
900 spatial slots
        ↓ HRM integration
CLS state  z̄_t
        ↓ projection
HPC query  Q_t
```

Formally:

$$
\mathcal{Q}_t = W_q \, \bar z_t
$$

where $\bar z_t \in \mathbb{R}^D$ and $\mathcal{Q}_t \in \mathbb{R}^{D_q}$.
HPC retrieves one memory response or a small fixed response set:

$$
\mathcal{M}_t = \text{HPC.retrieve}(\mathcal{Q}_t)
$$

The response is fused back into the CLS state:

$$
\bar z_t^{\,retr} = \text{Fuse}(\bar z_t, \mathcal{M}_t)
$$

Then HRM propagates the information from CLS into the 900 spatial slots during
subsequent recurrent computation.

### Query evolution

At $t = 0$, CLS summarizes task identity, start position, goal mask,
positional context, possibly arena identity, and the initial high-level HRM
state. The first query asks approximately:

> Given this arena, start, goal, and current task, retrieve structurally
> relevant memory.

After HRM integrates the first response, the CLS state changes. It may now
summarize a candidate direction, partial route evidence, uncertainty, a
possible intermediate region, or a conflict between candidate branches. The
next CLS query therefore differs from the first:

$$
\mathcal{Q}_{t+1} = W_q \, \bar z_{t+1}
$$

This creates recurrent refinement without adding dedicated query structures.

### Scientific hypothesis

> A single global recurrent control state is sufficient for PFC-guided
> retrieval and compositional route reasoning.

### Strengths

- Simplest and cleanest baseline.
- Low communication cost.
- Minimal architectural complexity.
- Tests whether one global PFC state can control hippocampal retrieval and
  whether repeated global retrieval is enough.

### Limitations

One vector must express all current information needs simultaneously. The PFC
may need to ask several distinct questions (transitions from start, states
leading to goal, candidate location $p$, observation occurrence, branch
validity). A single CLS vector may combine these requests into an ambiguous
superposition. It also lacks explicit separation between global task state,
memory query state, and memory response state.

---

## `ehp_v2`: dedicated cue tokens

### Central idea

`ehp_v2` introduces a small bank of dedicated PFC–HPC cue tokens. The PFC
sequence becomes:

```
[CLS] [CUE_1] [CUE_2] ... [CUE_R] [CELL_0] ... [CELL_899]
```

where $R \ll 900$ (e.g., $R \in \{4, 8, 16\}$). The cue tokens do not
correspond to physical positions. They are persistent retrieval channels.

### Division of roles

| Component                 | Responsibility                                    |
| ------------------------- | ------------------------------------------------- |
| CLS token ($\bar z_t$)    | Global task state and deliberative control        |
| Cue tokens ($C_t$)        | Specific memory queries and returned evidence     |
| 900 spatial slots ($S_t$) | Structured working memory over physical positions |

Each cue token can formulate a different hippocampal request. The model may
learn emergent specializations such as start-related retrieval, goal-related
retrieval, forward structural evidence, backward structural evidence,
candidate branch evidence, observation/content retrieval, route-fragment
retrieval, or uncertainty resolution. These roles need not be hard-coded; the
primary purpose is to provide separate query channels.

### Query generation

After an HRM cycle, the cue token states are projected into HPC query space:

$$
\mathcal{Q}_t = W_q \, C_t
$$

with $C_t \in \mathbb{R}^{R \times D}$ and $\mathcal{Q}_t \in \mathbb{R}^{R
\times D_q}$. HPC returns corresponding evidence:

$$
\mathcal{M}_t = \text{HPC.retrieve}(\mathcal{Q}_t)
$$

with one or several responses per cue. The results are fused back:

$$
C_t^{\,retr} = \text{Fuse}(C_t, \mathcal{M}_t)
$$

Subsequent HRM cycles allow cue tokens to interact with CLS, communicate with
spatial slots, and influence the route hypothesis.

### Scientific hypothesis

> Multiple persistent retrieval channels are necessary because goal-directed
> reasoning requires several simultaneous or sequential memory questions.
> Improvement over `ehp_v1` would support a multi-query interface.

### Strengths

- Greater retrieval bandwidth than `ehp_v1`.
- Explicit separation between global control and memory communication.
- Persistent query identities across recurrent cycles.
- Interpretable token-level diagnostics.
- Computationally efficient (small $R$).

### Limitations

The cue tokens are global latent states. They may still have difficulty
grounding their query in a specific subset of the 900 spatial slots. A cue
token might represent "candidate branch A," but the architecture does not yet
explicitly define which spatial positions constitute that candidate. The token
must learn grounding through ordinary HRM attention and recurrence. Cue-token
specialization may also collapse: several tokens may learn redundant behavior,
one token may dominate, or tokens may fail to maintain distinct roles.

---

## `ehp_v3`: spatial-frontier-derived queries

### Central idea

`ehp_v3` makes memory queries explicitly dependent on the current spatial
reasoning frontier. Rather than relying only on global cue states, the model
selects or pools the PFC spatial slots that are currently relevant and uses
those states to construct HPC queries.

```
current PFC working memory
        ↓ select relevant locations
pooled frontier representations
        ↓
HPC queries
```

### What is a reasoning frontier?

A frontier is a learned set of locations that the current PFC state treats as
relevant to continued reasoning. Possible examples include locations adjacent
to the start, candidate continuation points, locations near the goal, states
where two route hypotheses diverge, an unresolved bottleneck,
observation-bearing cells that may be semantic waypoints, endpoints of
retrieved trajectory fragments, or positions with high uncertainty. The PFC
may maintain multiple frontiers simultaneously.

### Query construction

Assume $R$ query channels. For query channel $r$, the model computes selection
weights over the 900 spatial slots:

$$
\alpha_t(r, p) \quad \text{with} \quad \sum_p \alpha_t(r, p) = 1
$$

It then pools a frontier representation:

$$
u_t[r] = \sum_p \alpha_t(r, p) \, S_t[p]
$$

The query combines the pooled frontier state, a query-role representation, and
the CLS global context:

$$
\mathcal{Q}_t[r] = \text{Project}(u_t[r], \bar z_t, \text{role}_r)
$$

### Forward and backward frontiers

A natural form of `ehp_v3` maintains at least two directional retrieval
processes:

```
forward frontier:
starts from the current start location and asks what follows

backward frontier:
starts from the goal and asks what can lead toward it
```

Conceptually:

```
start → forward relational expansion  ─┐
                                       ├→ route hypothesis
goal  → backward relational expansion ─┘
```

The system need not implement literal graph search. The inductive bias is that
PFC queries memory about its current candidate regions from both ends of the
problem. Additional frontier channels may represent semantic candidates,
alternative branches, unresolved observation bindings, or bottleneck
hypotheses.

### Scientific hypothesis

> Effective retrieval requires explicit grounding in the current spatial
> reasoning frontier. Improvement over `ehp_v2` would indicate that query
> tokens benefit from direct spatial grounding rather than purely global
> latent control.

### Strengths

- Stronger grounding than `ehp_v2`.
- Query explicitly contains information from selected spatial states.
- Naturally supports iterative compositional reasoning over local transitions.
- Easier to relate retrieval behavior to the evolving route hypothesis.

### Limitations

The selection mechanism itself becomes a difficult learned policy. If the PFC
selects the wrong frontier, HPC may return irrelevant memory, and early errors
can propagate. The model must balance concentrated local queries, broad
exploratory queries, multiple competing branches, and bidirectional evidence.
A hard or excessively sparse selector could make learning unstable; a very
soft selector could reduce back to an ambiguous global mixture.

---

## `ehp_v4`: dynamic low-rank projection

### Central idea

`ehp_v4` treats PFC→HPC communication as a dynamic low-rank projection. The
complete PFC working state is compressed into a small number of retrieval
vectors:

$$
S_t \in \mathbb{R}^{901 \times D}
\quad\longrightarrow\quad
\mathcal{Q}_t \in \mathbb{R}^{R \times D_q}
$$

where $R \ll 901$. The projection is not a fixed matrix — it is generated
dynamically from the current PFC state.

### Fixed versus dynamic projection

A fixed projection $\mathcal{Q}_t = A^\top S_t$ would always compress the same
combinations of PFC positions. It could learn broad basis functions but would
not adapt to novel start–goal combinations or changing reasoning needs.

`ehp_v4` instead computes:

$$
A_t = \text{Selector}(S_t, \bar z_t)
$$

with $A_t \in \mathbb{R}^{901 \times R}$. Then:

$$
U_t = A_t^\top S_t
$$

Each column of $A_t$ defines one soft projection over the PFC sequence. The
retrieval queries are constructed from the compressed states:

$$
\mathcal{Q}_t = \text{Project}(U_t, \bar z_t)
$$

### Interpretation

Each rank component can collect information distributed across many spatial
locations. A component might represent the current start-connected region, the
goal-connected region, a candidate route corridor, a distributed semantic
pattern, disagreement between two spatial hypotheses, a larger-scale
topological relation, or a mixture of locations sharing the same observation
identity.

Unlike `ehp_v3`, which emphasizes local or frontier-grounded queries, `ehp_v4`
can form distributed queries over the whole PFC state. It allows a query to
depend simultaneously on start, goal, multiple candidate locations, global
route geometry, distributed uncertainty, and high-level HRM state.

### HPC response and reintegration

HPC returns a small set of retrieved evidence vectors:

$$
\mathcal{M}_t \in \mathbb{R}^{R \times D_h}
$$

The response can be reintegrated through a corresponding dynamic expansion:

$$
\Delta S_t = B_t \, \mathcal{M}_t
$$

with $B_t \in \mathbb{R}^{901 \times R}$. However, direct write-back must be
controlled. The preferred interpretation is that the low-rank response enters a
retrieval or integration state, HRM determines how it should alter the spatial
working memory, and the expansion acts as evidence injection — not an
authoritative map overwrite. A residual update might be:

$$
S_t^{\,evidence} = S_t + \text{Gate}_t \odot (B_t \, \mathcal{M}_t)
$$

followed by further HRM reasoning.

### Scientific hypothesis

> A dynamic population-level projection can compress the complete PFC
> reasoning state into a small retrieval interface without requiring explicit
> slotwise queries. Improvement over the other versions would support
> distributed low-rank communication as an effective interface between
> reasoning and memory.

### Strengths

- Efficient population-level communication.
- Dynamic task-dependent compression.
- Global access to the complete PFC state.
- Natural information bottleneck.
- Biologically plausible convergent/divergent projection interpretation.
- Ability to query distributed relations rather than isolated locations.

### Limitations

Least directly interpretable version. A rank component may mix several
locations, candidate routes, semantic roles, and global control state. It may
be difficult to determine what a particular query means. A low rank may
discard fine-grained information; an excessively high rank may remove the
intended bottleneck. The return projection can accidentally move too much
reasoning into the memory interface.

---

## Version comparison

| Version  | Query source                             | Query structure                       | Inductive bias                      |
| -------- | ---------------------------------------- | ------------------------------------- | ----------------------------------- |
| `ehp_v1` | CLS token                                | One global query                      | Global recurrent control            |
| `ehp_v2` | Dedicated cue tokens                     | Small fixed bank of queries           | Multiple persistent memory channels |
| `ehp_v3` | Selected spatial frontiers               | Queries grounded in current locations | Local relational expansion          |
| `ehp_v4` | Dynamic projection of complete PFC state | Low-rank distributed queries          | Population-level compression        |

The versions form a progression:

```
ehp_v1: one global query
ehp_v2: multiple global query channels
ehp_v3: multiple queries explicitly grounded in selected spatial states
ehp_v4: multiple dynamic distributed queries over the full PFC population
```

They should not initially be merged into one architecture. Their scientific
value comes from isolating different query mechanisms. Later work may combine
successful elements, but the first experiments should preserve clean
distinctions.

---

## HPC memory learning

Arena pretraining must produce not merely a predictive model, but a memory
that the PFC query interface can address. Three distinct learning capabilities
are required.

### Structural acquisition

From Arena trajectories, the EC–HPC system learns transition structure:

```
state_t + action_t → state_t+1
```

This can be supervised through next-observation prediction, next-location
prediction, action-conditioned transition prediction, path integration,
sensory reconstruction, or structural-state consistency.

### Content–structure binding

The system must bind arena-specific content to structural states:

```
structural location ↔ observation/content encountered there
```

Otherwise, it may learn generic grid dynamics without knowing what is present
at each arena location.

### Queryable retrieval

The system must learn to respond to partial cues with useful memory evidence:

```
partial PFC cue → relevant structural or episodic memory
```

This does **not** automatically follow from forward transition prediction. The
retrieval interface must become aligned with the learned memory. Several
possibilities remain open:

- direct pretraining of query–response pairs during Arena;
- masked-state reconstruction from partial trajectory cues;
- bidirectional prediction of predecessors and successors;
- contrastive retrieval of matching structural/content states;
- trajectory-fragment completion;
- joint EHP fine-tuning that teaches PFC queries to address a pretrained HPC;
- a combination of frozen memory and trainable query projections.

The exact loss implementation is a design axis. The theory does not assume
that ordinary next-step prediction guarantees arbitrary retrieval. This is an
open question and should be tested explicitly.

---

## Training stages

The EHP training regime distinguishes at least three stages. The document does
not prescribe exact losses; it defines the conceptual decomposition.

### Stage A: Arena memory acquisition

Train the EC–HPC subsystem to learn the topology, transition dynamics, and
content bindings. The PFC route solver is not yet the primary target. The
canonical running example uses a fixed $30 \times 30$ arena with diverse
trajectories.

### Stage B: EHP query-interface learning

Train the PFC→HPC query mechanism and the HPC→PFC integration pathway so that
start/goal-driven reasoning can retrieve useful information. This is where
`ehp_v1`–`ehp_v4` differ. Each version learns its own query projection and
integration parameters.

### Stage C: Route reasoning

Train or fine-tune the PFC/HRM system to produce Routebind-style targets
(trajectory field, waypoint field) using retrieved memory rather than visible
topology.

### Memory freezing

Whether the EC–HPC memory subsystem is frozen after Arena or jointly
fine-tuned during Routebind training is an open design decision. Both regimes
are scientifically informative:

| Regime                    | Tests                                                                        |
| ------------------------- | ---------------------------------------------------------------------------- |
| Frozen EC–HPC             | Whether the learned map can be reused without rewriting it                   |
| Jointly fine-tuned EC–HPC | Maximum task performance; introduces risk of task-specific memory adaptation |

The cleanest first experiment would likely compare both. The theory should not
silently assume one.

---

## Required controls

High route accuracy alone is insufficient to prove memory-guided reasoning.

### Start–goal pair holdout

Training and evaluation must separate start–goal combinations. The evaluation
set should contain pairs that were not used as downstream route targets during
training. Otherwise the system may memorize pair-specific answers.

### Trajectory-fragment control

The evaluation should distinguish routes that exactly match previously
observed full trajectories from routes that require recombination of fragments
and routes that require novel ordering of familiar local relations. The
strongest tests should require composition.

### HPC ablation

Disable, reset, or disconnect HPC retrieval during EHP evaluation. Expected
result: start + goal alone should be insufficient. Performance should fall
substantially.

### Wrong-memory control

Provide memory from a different arena while keeping the start–goal format
valid. The PFC should either fail or construct a route consistent with the
wrong topology.

### Query-channel ablation

Ablate the query mechanism while preserving HRM capacity. Examples: freeze CLS
query in `ehp_v1`; collapse cue tokens in `ehp_v2`; randomize frontier
selection in `ehp_v3`; replace dynamic projection with fixed projection in
`ehp_v4`.

### Response-shuffling control

Shuffle HPC responses across samples or query channels. If performance remains
unchanged, the model is not using retrieved memory meaningfully.

### Iteration control

Compare one retrieval step with multiple retrieval steps. This tests whether
performance depends on recurrent information gathering rather than one-shot
pattern completion.

### Topology intervention

Modify a transition or observation binding in memory and test whether the
predicted route changes appropriately. This is stronger evidence that the
retrieved topology is causally used.

---

## Evidence levels

The EHP evidence contract distinguishes at least five levels.

1. **Arena representation learning.** The EC–HPC subsystem predicts or
   reconstructs local transitions and observation bindings. Establishes that
   it acquired environment information.
2. **Memory retrieval.** PFC cues retrieve information correlated with the
   correct arena state, transition, or trajectory fragment. Establishes that
   relevant memory is accessible.
3. **Behavioral route competence.** The full model produces valid routes for
   unseen start–goal combinations. Establishes task success.
4. **Iterative retrieval use.** Later queries depend on earlier retrieved
   evidence, and multiple retrieval cycles improve performance. Establishes
   recurrent PFC–HPC interaction.
5. **Causal memory use.** Changing, removing, or substituting HPC memory
   changes the route prediction in the expected way. Supports the strongest
   claim: the PFC constructed its route by using retrieved structured memory.

---

## Cross-version constants

For valid comparison, the following should remain fixed across `ehp_v1`–`ehp_v4`:

- arena topology;
- Arena pretraining trajectories;
- EC–HPC memory capacity;
- memory training objective;
- Routebind downstream corpus;
- start–goal split;
- HRM hidden dimension;
- number of deliberation cycles;
- number of retrieval opportunities;
- output decoder;
- supervision losses;
- evaluation metrics;
- parameter-count budget, where practical.

The major independent variable is how the PFC constructs the HPC query.
Differences in parameter count and retrieval bandwidth must be recorded and
controlled. Comparing one CLS vector against 32 cue tokens without accounting
for capacity would confound mechanism with bandwidth.

---

## Goal-location shortcut

A fully memory-dependent task would provide a semantic goal identity (e.g.,
`goal = obs_10`) and require the system to retrieve the physical positions
where that observation occurs. This is itself a difficult memory query.

For the initial EHP experiments, the adapter is allowed to know the physical
goal positions and mark them with a goal flag. Thus semantic goal identity
localization is provided externally, but the topology needed to connect start
to goal is not. This is a deliberate decomposition of the problem — it
isolates PFC–HPC topology retrieval before adding semantic localization.

This is a **benchmark profile**, not a model version. The correct
decomposition is:

```
EHP query-interface version:
    ehp_v1, ehp_v2, ehp_v3, or ehp_v4

Task input profile:
    goal_position_known
    goal_identity_only
```

The initial profile uses `goal_position_known`. The later full profile uses
`goal_identity_only` and may be evaluated with any of the four query-interface
versions.

---

## Open questions

The following design axes are not yet settled and should be treated as
testable hypotheses rather than assumed answers.

1. **Exact retrieval loss.** Which training objective best aligns the
   PFC→HPC query interface with the Arena-trained memory?
2. **Memory freezing regime.** Is the EC–HPC memory frozen, partially frozen,
   or jointly fine-tuned during Routebind training?
3. **Retrieval cadence.** At what points during HRM recurrence should the
   system issue HPC queries? Fixed schedule? Learned gating?
4. **Retrieval budget.** What is the minimum number of query channels $R$
   needed for each version? How does performance scale with $R$?
5. **Query interpretability.** Can the emergent query behavior in each version
   be related to interpretable spatial or relational questions?
6. **Version combination.** Can successful elements of `ehp_v2` (multi-query),
   `ehp_v3` (spatial grounding), and `ehp_v4` (population compression) be
   combined without losing scientific isolation?

---

## Central hypothesis

The central EHP hypothesis, consistent across all four versions and both
benchmark profiles, is:

> Goal-directed reasoning can be implemented as recurrent, controlled
> retrieval over a learned cognitive map, where hippocampal memory supplies
> compositional structural evidence and PFC constructs the novel solution.

- The freezing contract is unchanged: EC–HPC substrate frozen during goalchain; PFC↔HPC projections trainable.
