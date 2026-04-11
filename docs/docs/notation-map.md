## Notation Map

This page freezes the theory-to-code naming used across the manuscript,
environments, models, and the EHC controller.

Use it as the canonical semantic reference for public batch keys, cortical
cue names, and manuscript symbols.

### EHC V1 Decision

- In EHC v1, $\xi_t$ stays prose-only.
- Do not introduce `context_summary` or `context_slots` as public API objects
  in v1.
- The nearest current implementation mapping is the exogenous conditioning
  pair `external_context` and `input_ids -> sequence_summary`, which together
  condition the cue query in EHC.
- `context_slots` is wrong for v1 because slots already denote the cortical
  workspace $S_t$.
- `context_summary` is also wrong for v1 because the current runtime still
  exposes two distinct conditioning channels rather than one stable public
  context object.
- If a later version promotes $\xi_t$ into a typed runtime contract, reserve
  `TaskContext` as the public type name.

### Manuscript To Code

| Concept | Manuscript | Current code surface | Reading rule |
|---|---|---|---|
| Current percept payload | $o_t$ | `observation` | Step-local raw percept only. |
| Encoded sensory content | $x_t$ | `place_query_from_obs`, `sensory.recall`, `place_sensory` | Encoded sensory cue, not the raw observation. |
| Structural state | $g_t$ | `grid_prior`, `grid_post` | Code exposes prior and posterior structural states separately. |
| Hippocampal memory | $M_t$, $\mathcal{M}_t$ | `state.hpc.memory` | Bank `c` exists only in EHC. |
| Cortical workspace | $S_t$ | `state.pfc.memory.z_H`, `state.pfc.memory.z_L` | Implementation state backing the workspace. |
| Task context | $\xi_t$ | prose-only in v1; nearest code inputs are `external_context` and `input_ids -> sequence_summary` | Not a standalone runtime object in v1. |
| Cortical cue | $c_t$ | `c_prop` | PFC-derived cue proposal read from workspace. |
| Reinstated contextual evidence | $\hat c_t^{retr}$ | `c_mem` | Grounded target-bank read from HPC bank `c`. |
| Routed cue | $c_{t,m}$ | `c_use` | Transient routed cue used for replay bias and control. |
| Retrieval query set | $\mathcal{Q}_t$ | `ReadCues({"x", "g"})` in TEM, `ReadCues({"x", "g", "c"})` in EHC | Family-indexed retrieval cue set. |
| Control summary | $\bar z_{t,m}$ | `theta_cls` in HRM, `theta_summary` in EHC | Summary bottleneck, not the canonical cortical cue. |
| Internal control logits | $\ell_{t,m}^{int}$ | `q_logits` in HRM, `internal_control_logits` in EHC | Distinct from motor action logits. |
| Reward prediction | $\hat r_{t,m}$ | `r_logits` in HRM v2, `reward_logits` in EHC | Narrow STR-side reward output. |
| Motor action logits | overt action factors | `motor_logits` | Overt action scores only; currently EHC-specific. |

### Forward Surfaces

| Surface | Required step-local keys | Optional or static keys | Recurrent state | Public output | Notes |
|---|---|---|---|---|---|
| `DungeonWalk` | `observation`, `observation_id`, `previous_action`, `location_id`, `region_id`, `valid_action_mask`, `step_count` | `landmark_id`; maze tensors are provided on reset | Environment TensorDict | next TensorDict | Spatial step payload used by TEM and EHC rollouts. |
| `MazeHardEnv` | `input_ids`, `labels`, `prev_accuracy`, `step_count` | none | Environment TensorDict | next TensorDict | Static token-tape environment used by HRM-style rollouts. |
| `TEMModelV1` | `observation`, `previous_action` | `episode_start`, `landmark_id` | `TEMState(lec, mec, hpc)` | `(state, obs_logits, None, grid_codes, place_codes)` | Pure memory model; no direct control head. |
| `TEMModelV2` | `observation`, `previous_action` | `episode_start`, `landmark_id` | `TEMState(lec, mec, hpc)` | `(state, obs_logits, None, grid_codes, place_codes)` | Same public step contract as TEM v1. |
| `HRModelV1` | `input_ids` | none | `HRMState(pfc)` | `(new_state, (logits, q_logits), theta_cls)` | Token-only reasoning surface. |
| `HRModelV2` | `input_ids` | none | `HRMState(pfc, str)` | `(new_state, (logits, q_logits, r_logits), theta_cls)` | Adds STR reward output. |
| `EHCModelV1` | `observation`, `previous_action` | `episode_start`, `landmark_id`, `external_context`, `input_ids` | `EHCState(pfc, str, lec, mec, hpc)` | `EHCOutput(state, obs_logits, memory, control)` | Structured model step only; no environment stepping. |
| `EHCController` current payload | `observation`, `observation_id`, `previous_action`, `location_id`, `region_id`, `valid_action_mask`, `step_count` | `landmark_id` plus static `input_ids` and `external_context` | `EHCRolloutState(model_state, data, env_td, conditioning_data)` | `EHCControllerOutput(model_output, internal_action, commit_mask, motor_action, reward, cycle_count)` | `input_ids` and `external_context` live in `conditioning_data`, not in `env_td`. |

### Naming Freeze

| Name | Freeze to | Status |
|---|---|---|
| `observation` | current step-local percept payload | use |
| `observation_id` | current-step categorical percept id | use |
| `input_ids` | serialized discrete tape | use |
| `external_context` | exogenous continuous conditioning input | use |
| `sequence_summary` | encoded summary of `input_ids` inside EHC | use locally; do not equate with $\xi_t$ by itself |
| `c_prop` | cortical cue proposal derived from workspace | use |
| `c_mem` | reinstated contextual evidence read from bank `c` | use |
| `c_use` | transient routed cue used to bias replay and control | use |
| `theta_cls` | HRM control-summary token | use |
| `theta_summary` | EHC control-summary token | use |
| `internal_control_logits` | internal policy scores | use |
| `motor_logits` | overt action scores | use |
| `inputs` | split into `observation` or `input_ids` | avoid |
| `context` | split into `external_context` or manuscript $\xi_t$ | avoid |
| `context_summary` | implies a first-class $\xi_t$ object that v1 does not expose | avoid in v1 |
| `context_slots` | conflates $\xi_t$ with workspace slots $S_t$ | avoid in v1 |
| `TaskContext` | reserved future public type name for a typed $\xi_t$ contract | reserve |

### Canonical Reading Rule

- `observation` is the current percept payload.
- `input_ids` is the serialized discrete tape, not the current percept.
- `external_context` is exogenous conditioning presented to cortex.
- $S_t$ is the maintained cortical workspace represented by PFC state.
- In EHC v1, $\xi_t$ is theory-level task context. Read it as the current
  conditioning pair `external_context` plus `sequence_summary`, not as a
  standalone runtime object.
- `c_prop` is the PFC-derived cue proposal.
- `c_mem` is reinstated contextual evidence from HPC bank `c`.
- `c_use` is the routed transient cue used for replay bias and control.
- `theta_cls` and `theta_summary` are control summaries, not canonical
  cortical cues.