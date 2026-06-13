## EHP Theory Foundations

This note explains the main theoretical lines that motivate the
entorhinal-hippocampal-prefrontal synthesis used in EHP-style models.

It focuses on five ingredients:

- the Tolman-Eichenbaum Machine as a theory of structural and grounded memory;
- the transformer-to-hippocampus correspondence as a theory of memory readout;
- structured prefrontal working memory as a theory of controllable slots;
- hierarchical reasoning as a theory of slow-fast recurrent computation;
- goal-directed prefrontal control over episodic memory as a theory of
  contextual retrieval.

The goal is not to claim that these theories are identical. The goal is to
state what each one contributes, where each one is strongest, and how they can
be composed without collapsing their differences.

### Source Map

| Source                                                                   | Core claim                                                                           | Main contribution to EHP-style thinking                                                |
| ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------- |
| [Whittington et al., 2020](https://doi.org/10.1016/j.cell.2020.10.024)   | Hippocampal memory binds sensory content to a structural state space.                | Gives the basic entorhinal-hippocampal loop over structural state and grounded memory. |
| [Whittington et al., 2022](https://doi.org/10.48550/arXiv.2112.04035)    | Transformer-style attention can be read as hippocampal-style associative retrieval.  | Justifies attention-like episodic memory as a hippocampal read mechanism.              |
| [Whittington et al., 2025](https://doi.org/10.1016/j.neuron.2024.10.017) | Prefrontal working memory can be modeled as structured, controllable activity slots. | Gives a theory for a maintained cortical workspace rather than a passive sequence.     |
| [Wang et al., 2025](https://doi.org/10.48550/arXiv.2506.21734)           | Reasoning can be organized by coupled slow and fast recurrent modules.               | Gives the slow-fast cortical update schedule.                                          |
| [Zheng et al., 2025](https://doi.org/10.48550/arXiv.2503.02303)          | Prefrontal cortex can control episodic memory retrieval according to task demands.   | Gives the top-down cueing and contextual-retrieval story.                              |

### The Common Problem

All five theories address a different part of the same broad question:

How can an agent combine structure, experience, control, and deliberation in a
single system?

At a high level, the problem can be written as follows. Let

- $o_t$ be the current percept;
- $a_{t-1}$ be the previous action;
- $g_t$ be a structural or abstract state;
- $p_t$ be a grounded episodic or place-like state;
- $S_t$ be a maintained cortical workspace;
- $c_t$ be a cortical retrieval cue.

Then the overall computation is a composition of:

$$
g_t \leftarrow \text{structure update}(g_{t-1}, a_{t-1}, o_t),
$$

$$
p_t \leftarrow \text{episodic retrieval}(g_t, o_t, c_t),
$$

$$
S_t \leftarrow \text{cortical maintenance and reasoning}(S_{t-1}, p_t, o_t),
$$

$$
c_t \leftarrow \text{control policy over memory and workspace}(S_t, \xi_t).
$$

Each theory below clarifies one of these terms.

### Tolman-Eichenbaum Machine

The Tolman-Eichenbaum Machine explains hippocampal-entorhinal computation as a
division between structural state and grounded memory.

The central distinction is:

- the entorhinal pathway carries a structural code $g_t$;
- the hippocampal pathway carries a grounded code $p_t$ tied to specific
  observations or experiences.

The structural state evolves under action and transition structure,
approximately as

$$
g_t^{-} = T(g_{t-1}, a_{t-1}),
$$

where $g_t^{-}$ is a prior or path-integrated state.

Sensory evidence then constrains grounded retrieval,

$$
x_t = E(o_t),
$$

$$
p_t = M(x_t, g_t),
$$

where $x_t$ is encoded sensory content and $M$ is hippocampal associative
memory.

The main theoretical claim is that memory is not stored as isolated episodes in
an unstructured list. Instead, specific sensory content is bound to a reusable
structural scaffold. That is why a learned map can generalize across new
layouts: the structure is reused while the grounded bindings are updated.

The strongest contribution of this theory is the idea that entorhinal and
hippocampal pathways should not be collapsed into one homogeneous latent state.
One pathway carries structure; the other carries specific grounded memories.

### Relating Transformers To Hippocampal Memory

The transformer-to-hippocampus account argues that attention is not merely a
generic sequence operator. It can be interpreted as a concrete associative
memory read.

For a query matrix $Q$, key matrix $K$, and value matrix $V$, the canonical
attention read is

$$
R(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V.
$$

This equation can be read as a content-addressable memory lookup:

- keys determine which memory items are relevant;
- values return the recalled content;
- the query specifies the current retrieval demand.

The theory matters because it gives a modern computational interpretation of
hippocampal retrieval. A memory read can be selective, differentiable,
high-capacity, and compatible with learned representations without abandoning
the logic of associative recall.

The strongest contribution here is not that transformers replace hippocampal
theory. It is that a standard attention mechanism can be reinterpreted as a
memory system whose operations look strikingly similar to hippocampal recall.

In an EHP-style synthesis, this theory mainly justifies using attention-like
episodic reads rather than requiring only dense attractor dynamics.

### Structured Prefrontal Working Memory

The structured-slot account of prefrontal working memory treats working memory
as a maintained set of controllable activity subspaces rather than as a bag of
indistinguishable recurrent activations.

Let

$$
S_t \in \mathbb{R}^{N \times d}
$$

denote a cortical workspace with $N$ slots of width $d$.

The main claim is that these slots are not just positions in a sequence. They
are controllable activity coordinates that support selective maintenance,
binding, and readout. A compact summary variable can then be derived from the
workspace,

$$
\bar z_t = \rho(S_t),
$$

where $\rho$ is a structured readout rather than a claim that the summary is
the whole workspace.

This theory contributes two important ideas.

First, cortex should maintain a persistent workspace rather than recomputing
everything from scratch at each step. Second, not every slot needs a fixed,
human-readable interpretation. Some slots may have clear roles, while others
remain internal degrees of freedom that are useful but only partially
interpretable.

That is why a mixed layout is reasonable:

- a few slots may have stable roles such as state, replay, or cue;
- the remaining slots can be indexed internal content slots.

The theory does not require every slot to have a hand-written semantic label.
It only requires that the workspace be structured, controllable, and stable
enough to support selective reasoning.

### Hierarchical Reasoning Model

The hierarchical reasoning account explains how a cortical system can perform
deep computation through recurrence at multiple timescales rather than through
one very deep feedforward stack.

Let $z^L_{t,k}$ denote a fast state and $z^H_t$ denote a slow state. A minimal
two-timescale scheme is

$$
z^L_{t,k+1} = F_L(z^L_{t,k}, z^H_t, x_t),
$$

$$
z^H_{t+1} = F_H(z^H_t, z^L_{t,K}),
$$

where $K$ fast updates happen before one slow update.

The conceptual point is that different computations can live at different
timescales:

- the fast process handles local, transient, detail-sensitive updates;
- the slow process maintains broader, more stable, more abstract state.

This theory is strongest when the task requires iterative refinement rather than
one-shot pattern matching. It explains how a model can deliberate, revise, and
integrate evidence over an internal rollout without needing an explicit chain of
symbolic steps.

In a broader synthesis, this theory belongs on the cortical side rather than on
the hippocampal side. It explains how the workspace evolves, not how episodic
memory itself is stored.

### Flexible Prefrontal Control Over Episodic Memory

The top-down control account focuses on the fact that memory retrieval should be
conditional on current goals or task context, not only on bottom-up sensory or
structural similarity.

Let $\xi_t$ denote task context and let $S_t$ denote the maintained cortical
workspace. Then a cortical cue can be generated as

$$
c_t = C(S_t, \xi_t).
$$

That cue can then be used to bias encoding or retrieval,

$$
\hat m_t = \operatorname{Read}(c_t, M_t),
$$

or, in a more selective form,

$$
\hat m_t = \operatorname{Read}(g_t, c_t, M_t).
$$

The core claim is that prefrontal cortex should not only consume retrieved
memory. It should shape which memory becomes relevant in the first place.

This matters for generalization because two situations may be structurally
similar while demanding different retrieved associations. A top-down cue allows
retrieval to depend on the active goal, rule, or context rather than only on
the current observation.

The strongest contribution of this theory is therefore selective control over
episodic access. It is not just a memory-capacity story. It is a memory-routing
story.

### How The Theories Fit Together

These theories are complementary if they are assigned distinct roles.

The Tolman-Eichenbaum Machine contributes the entorhinal-hippocampal structure:

$$
(o_t, a_{t-1}) \rightarrow (x_t, g_t, p_t).
$$

The transformer-memory account contributes the read mechanism:

$$
p_t = \operatorname{MemoryRead}(q_t, K_t, V_t).
$$

The structured-slot account contributes the cortical substrate:

$$
S_t \in \mathbb{R}^{N \times d}.
$$

The hierarchical reasoning account contributes the cortical update schedule:

$$
S_t \leftarrow \text{slow-fast recurrent reasoning over } S_{t-1}.
$$

The top-down control account contributes the contextual routing signal:

$$
c_t = C(S_t, \xi_t),
$$

which then biases replay, retrieval, or memory selection.

The synthesis is coherent only if these roles remain distinct. Structure is not
the same thing as working memory. Working memory is not the same thing as
episodic memory. A control cue is not the same thing as a summary token. And a
memory read mechanism is not the same thing as a theory of why one memory
should be selected instead of another.

### Main Tensions

There are still real theoretical tensions between these accounts.

- The Tolman-Eichenbaum Machine emphasizes structural generalization through a
  reusable state space, while the top-down control account emphasizes retrieval
  selection according to current demands.
- Attention-style memory reads are elegant and scalable, but they do not by
  themselves explain which cues should be generated.
- Structured-slot workspace theories explain maintained control state, but they
  do not by themselves explain hippocampal binding.
- Slow-fast recurrent reasoning explains deliberation, but it does not by
  itself explain memory indexing.

These tensions are productive. They show why a full EHP-style theory needs
more than one ingredient.

### Bottom Line

The five theories answer different questions.

- The Tolman-Eichenbaum Machine explains how structural and grounded memory can
  coexist.
- The transformer-memory account explains how memory can be read through
  attention-like associative retrieval.
- Structured-slot working-memory theory explains why cortex should maintain a
  controllable workspace.
- Hierarchical reasoning explains how that workspace can compute over multiple
  timescales.
- Top-down prefrontal control explains how retrieval can be routed by current
  goals and context.

Taken together, they motivate a model family in which hippocampal memory,
cortical workspace, hierarchical reasoning, and contextual cueing are distinct
but interacting parts of a single cognitive system.
