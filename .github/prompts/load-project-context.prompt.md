---
agent: "Mentor mode"
description: "Ground the current task in required specs, papers, and legacy code before any design, coding, or advice."
argument-hint: "Current task or focus, optional"
---

# Load Project Context

## Mission

Ground the current task in the required specs, listed bibliography entries, and listed legacy code before giving any design, implementation, or advice.
This prompt is retrieval-only. Stop after producing the grounding brief.

## Input

Current Focus
${input:CurrentFocus:optional}

## Scope And Preconditions

1. Pass the spec gate and read the required files from [spec/spec-manifest.toml](../../spec/spec-manifest.toml).
2. Read every listed paper and code anchor below before expanding further.
3. Keep the reading set tight. Expand only one hop if Current Focus requires it.

## Core Sources

### TEM

- Papers in [article/references.bib](../../article/references.bib): `whittington_relating_2022`, `whittington_tolman-eichenbaum_2020`
- Code: [legacy_tem/README.md](../../legacy_tem/README.md), [legacy_tem/tem_tf1/tem_model.py](../../legacy_tem/tem_tf1/tem_model.py), [legacy_tem/tem_tf1/run_tem.py](../../legacy_tem/tem_tf1/run_tem.py), [legacy_tem/tem_tf2/tem_model.py](../../legacy_tem/tem_tf2/tem_model.py), [legacy_tem/tem_tf2/run_tem.py](../../legacy_tem/tem_tf2/run_tem.py)
- Working map: TEM = LEC, MEC, HPC

### HRM

- Papers in [article/references.bib](../../article/references.bib): `whittington_how_2022`, `wang_hierarchical_2025`
- Code: [legacy_hrm/README.md](../../legacy_hrm/README.md), [legacy_hrm/pretrain.py](../../legacy_hrm/pretrain.py), [legacy_hrm/models/hrm/hrm_act_v1.py](../../legacy_hrm/models/hrm/hrm_act_v1.py)
- Working map: HRM = PFC

### HPC-PFC Interaction

- Paper in [article/references.bib](../../article/references.bib): `zheng_flexible_2025`
- Use this as the main interaction anchor unless Current Focus requires one closer neighboring code source.

## Workflow

1. Read the required spec files first.
2. Read each listed paper entry and each listed code file above.
3. Resolve terminology from the cited bibliography entries and the legacy code. Do not infer from titles alone.
4. Extract evidence instead of writing a freeform summary:
   - from each paper: one task-relevant claim, one mechanism, and one limitation or uncertainty
   - from each code source: one owning symbol or module, and the behavior it controls
5. Build cross-source mappings from paper concept to legacy code surface to Current Focus.
6. Stop after the Grounding Check. Do not design, code, critique, or recommend next steps in this prompt.

## Grounding Check

Before any design, coding, or advice, return:

### Papers Read

- citation key
- task-relevant claim
- mechanism
- limitation or uncertainty

### Code Read

- file
- owning symbol or module
- behavior it controls

### Mappings

- for each relevant family, map paper concept -> legacy code surface -> relevance to Current Focus

### Gaps

- anything not yet grounded, ambiguous, contradictory, or missing from the sources above

Only make claims that can be tied to a bibliography key and a concrete code surface read in this run.

## Stop Conditions

Stop and report the gap instead of giving advice or implementation guidance if:

- any listed paper was not read
- any listed code source was not read
- any claim cannot be tied to a bibliography key
- any code statement cannot be tied to a concrete file and owning symbol or module
- any required mapping cannot be made from the sources above

## Output Expectations

Return a short, human-readable grounding brief only. Do not include design proposals, implementation steps, or recommendations.
