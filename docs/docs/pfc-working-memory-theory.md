## PFC Workspace And Working-Memory Dynamics

This note explains how the current PFC workspace layer relates to the
working-memory theory discussed around HRM, Tale, and cognitive maps.

The core claim is simple:

- fixed workspace names define an address space for slots;
- recurrent dynamics move activity within that address space;
- any phase-like or rotational structure lives in latent activity, not in the
  slot names themselves.

### Source Map

| Source                                                                                                                                        | Claim used here                                                                                           | Why relevant                                                                                                                                         |
| --------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Whittington et al., 2025](https://doi.org/10.1016/j.neuron.2024.10.017)                                                                      | Prefrontal working memory can be understood as structured, controllable activity slots.                   | This is the closest theoretical source for treating workspace slots as active working-memory subspaces rather than passive token positions.          |
| [Wang et al., 2025](https://doi.org/10.48550/arXiv.2506.21734)                                                                                | Reasoning can be organized as coupled slow and fast recurrent modules.                                    | This is the closest algorithmic source for the $z_H$ and $z_L$ split used by HRM-style reasoning in this repository.                                 |
| [Whittington et al., 2022](https://doi.org/10.1038/s41593-022-01153-y)                                                                        | Cognitive maps are best understood as structured representational systems rather than single code motifs. | This provides the broader map-level framing used here when treating working memory as a dynamic map rather than a plain tape.                        |
| [Perich et al., 2025](https://doi.org/10.1038/s41593-025-02031-z)                                                                             | Neural activity should often be analyzed as trajectories on low-dimensional manifolds.                    | This provides the right language for talking about latent rotations, phase structure, and local subspaces without confusing them with slot renaming. |
| Repository implementation: `src/ehc_sn/modules/pfc/workspace.py`, `src/ehc_sn/modules/pfc/reasoning.py`, `src/ehc_sn/modules/pfc/__init__.py` | Concrete implementation of slot naming, recurrent state, and public PFC state.                            | The theory here is grounded in the current code, not in a generic RNN or transformer abstraction.                                                    |

### Repository Reading Rule

The repository separates three things that are easy to conflate.

| Layer                    | What it owns                                | Current surface                      |
| ------------------------ | ------------------------------------------- | ------------------------------------ |
| Slot layout              | Concrete slot names and order               | `WorkspaceSpec`                      |
| Named slot bank          | One instantiated bank of slot vectors       | `NamedWorkspace`                     |
| Recurrent working memory | Fast and slow latent dynamics               | `WorkingMemory` over $z_H$ and $z_L$ |
| Public semantic state    | Named workspace, summary, and scratch carry | `PFCState`                           |

The most important implication is that `workspace.py` does not define the
working-memory dynamics. It defines the addressing scheme over the slot bank.

### What The Workspace Layer Really Means

`WorkspaceSpec` is a flat ordered list of concrete slot names. It is best read
as a compiled layout, not as a complete ontology.

For example, a conceptual family such as $\mathrm{schema}[n]$ is currently compiled into
concrete names like `schema_0`, `schema_1`, and `schema_2`. That is practical,
but it means slot families are represented by naming convention rather than a
first-class type.

This is why the current abstraction can feel slightly weak:

- every slot must have a concrete unique name;
- groups are implicit in prefixes such as `schema_`;
- exchangeability is not represented directly in the type system.

That does not make the module wrong. It means the current workspace layer is a
layout contract, not a full theory of slot families.

`NamedWorkspace` then pairs that layout with a tensor of shape `(B, S, D)`.
The tensor holds slot values; the spec tells the rest of the system how to read
those values by role.

### Prefix Slots Versus Body Slots

The current implementation also separates the controller-like prefix slot from
the body slots.

The public named workspace corresponds to the body of the slot bank, while the
full public working memory is reconstructed later by prepending the summary or
controller slot inside `PFCState.working_memory(...)`.

This matters for interpretation:

- the named workspace is not the entire recurrent state;
- it is the role-addressable body of that state;
- the summary or controller token is handled separately.

`workspace_from_prefixed_tokens(...)` should be read with that in mind. It does
not infer rich semantics. It slices a prefix-plus-body sequence, discards the
prefix block, and attaches the declared slot names to the remaining body.

### Mathematical View Of The Current PFC

Let:

- $X_t$ be the input slot bank for one outer step, with shape $(S, D)$;
- $H_t$ be the high-level state $z_H$, with shape $(S + 1, D)$;
- $L_{t,k}$ be the low-level state $z_L$ after inner step $k$, also with shape
  $(S + 1, D)$.

The repository update schedule in `reasoning.py` is a fast-slow recurrent
system. In compact notation:

$$
\begin{aligned}
L_{t,k+1} &= F_L(L_{t,k}, H_t + X_t) \\
H_{t+1} &= F_H(H_t, L_{t,K})
\end{aligned}
$$

where $K$ is the number of low-level inner updates executed before one
high-level update.

This is the key structural reason to expect different representational roles.

- $L$ is updated repeatedly within one reasoning episode;
- $H$ changes more slowly;
- $H$ conditions the fast updates of $L$;
- $L$ can therefore carry transient, phase-sensitive structure while $H$
  carries slower integrative or control-like structure.

### Local Linearization

To make the phase claim precise, flatten the slot bank into a vector and
linearize the fast dynamics around one operating point. Then the low-level
update is approximately:

$$
\delta l_{t,k+1} \approx A_L \, \delta l_{t,k} + B_L \, \delta h_t + C_L \, \delta x_t
$$

and the slow update is approximately:

$$
\delta h_{t+1} \approx A_H \, \delta h_t + B_H \, \delta l_{t,K}
$$

The matrix $A_L$ controls the local geometry of the fast dynamics. If $A_L$
has a complex conjugate eigenpair

$$
\lambda_{+} = r e^{+i\omega}, \qquad \lambda_{-} = r e^{-i\omega}
$$

then, in the corresponding two-dimensional real subspace, the fast dynamics are
locally equivalent to a rotation with decay or gain:

$$
u_{k+1} = r R(\omega) u_k + b(h, x)
$$

$$
R(\omega) =
\begin{bmatrix}
\cos(\omega) & -\sin(\omega) \\
\sin(\omega) & \cos(\omega)
\end{bmatrix}
$$

This is the correct mathematical reading of a Tale-style rotational or
phase-like code.

It does not mean that slot $i$ becomes slot $i + 1$.
It means the activity vector inside a fixed slot layout moves through a latent
subspace with phase.

### Why Lag Cells Follow From Phase Geometry

Suppose a readout cell has weight vector $w_i$ over the fast latent state. Its
response after an observation-dependent perturbation is:

$$
y_i^{(o, h)}(k) = w_i^\top u_k
$$

Under the local oscillatory approximation above, this becomes:

$$
y_i^{(o, h)}(k) \approx \alpha_i^{(o, h)} r^k \cos\bigl(\omega k + \phi_i^{(o, h)}\bigr)
$$

The preferred lag of that readout is determined by the phase offset
$\phi_i^{(o, h)}$. This gives the intended interpretation of statements such as
"a cell fires one or two steps after a particular observation": the cell is
not coding only observation identity, but observation-conditioned phase or lag.

This is also why phase relationships can shift across tasks. If the task or
context changes $h$, then either:

- the initial perturbation lands in a different direction;
- the local linearization itself changes;
- or both.

In that case, the same readout can keep a within-task lag interpretation while
changing its relative phase structure across tasks.

### Why $z_L$ Is The More Plausible Carrier

Given the current implementation, $z_L$ is the more plausible place for
phase-like or rotational structure to emerge.

Reasons:

- $z_L$ is updated multiple times per outer reasoning episode;
- $z_L$ directly receives the current observation-conditioned drive;
- $z_H$ changes only after a block of low-level updates;
- $z_H$ is therefore structurally closer to a slow control or integration
  state.

In manifold language, $z_H$ is a natural candidate for selecting or stabilizing
the active regime, while $z_L$ is a natural candidate for tracing the fast
trajectory within that regime.

This is a structural argument, not a theorem. The current architecture permits
this division of labor, but it does not force it.

### What The Current Public State Exposes

The current public PFC state is rebuilt from $z_H$ rather than from $z_L$.
That choice is architecturally sensible because the public state is meant to be
stable and architecture-native.

It also has one important consequence:

- if fast phase geometry lives mainly in $z_L$, then the public workspace view
  exposes only a slower projection of that computation;
- the fast geometry is still present in the scratch carry, but it is not the
  primary public surface.

This is the main reason the repository can be compatible with a Tale-like story
without explicitly exposing a rotating public workspace.

The names stay fixed. The fast latent activity is what may rotate.

### What This Means For `workspace.py`

From this perspective, the workspace module is doing a narrower job than it may
appear to be doing.

It is not implementing the dynamical theory of working memory. It is doing
three simpler things:

1. fixing the concrete slot layout;
2. attaching role labels to the body slot bank;
3. giving higher layers a stable way to select slots by role.

That is still useful, because a structured activity-slot theory still needs a
stable address space. But it also explains the current smell around indexed
families: the abstraction stops at concrete names.

If the repository later needs first-class slot families, that should likely be
added above `WorkspaceSpec`, not forced into its current flat naming contract.

### Testable Predictions

If the Tale-style interpretation is correct for this implementation, the
following predictions should hold empirically.

1. Local Jacobians of the fast update should more often exhibit stable complex
   modes than the slow update.
2. Observation-conditioned trajectories in $z_L$ should show clearer curved or
   phase-like low-dimensional structure than trajectories in $z_H$.
3. Probes trained to decode short lag or phase should work better on $z_L$
   during inner updates than on the public $z_H$ workspace surface.
4. Probes trained to decode coarse task state or control regime should work at
   least as well, and likely better, on $z_H$ than on $z_L$.

These are empirical claims. They should be checked with Jacobian spectra,
trajectory visualizations, and targeted probes rather than assumed from the
architecture alone.

### Bottom Line

The current repository is best read as a compatible hybrid of two ideas.

- From Tale, it takes the view that working memory can be treated as structured
  activity slots with stable role-addressable surfaces.
- From HRM, it takes the view that reasoning benefits from fast and slow
  recurrent interaction.

The most coherent synthesis is therefore:

- workspace names fix the slot address space;
- $z_H$ provides the slow integrative and control-facing surface;
- $z_L$ is the most plausible carrier of fast lag or phase geometry;
- any rotational structure should be understood as latent subspace dynamics,
  not slot renaming.

### Related Internal Docs

- [Notation Map](notation-map.md)

### References

- Whittington, J. C. R., Dorrell, W., Behrens, T. E. J., Ganguli, S., and
  El-Gaby, M. (2025). A tale of two algorithms: Structured slots explain
  prefrontal sequence memory and are unified with hippocampal cognitive maps.
  _Neuron_, 113(2), 321-333.e6.
  DOI: [10.1016/j.neuron.2024.10.017](https://doi.org/10.1016/j.neuron.2024.10.017)
- Wang, G., Li, J., Sun, Y., Chen, X., Liu, C., Wu, Y., Lu, M., Song, S., and
  Yadkori, Y. A. (2025). Hierarchical Reasoning Model.
  DOI: [10.48550/arXiv.2506.21734](https://doi.org/10.48550/arXiv.2506.21734)
- Whittington, J. C. R., McCaffary, D., Bakermans, J. J. W., and Behrens,
  T. E. J. (2022). How to build a cognitive map. _Nature Neuroscience_, 25,
  1257-1272.
  DOI: [10.1038/s41593-022-01153-y](https://doi.org/10.1038/s41593-022-01153-y)
- Perich, M. G., Narain, D., and Gallego, J. A. (2025). A neural manifold view
  of the brain. _Nature Neuroscience_, 28, 1582-1597.
  DOI: [10.1038/s41593-025-02031-z](https://doi.org/10.1038/s41593-025-02031-z)
