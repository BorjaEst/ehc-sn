## Model Family Selection

This page is a practical guide for choosing TEM, HRM, or EHC surfaces.

## TEM Family

Use TEM when you need navigation-grounded predictive memory with explicit
entorhinal-hippocampal dynamics.

Characteristics:

- Predictive memory focus
- Replay-style trajectory objectives
- Strong compatibility with spatial analysis and cell-level figures

Entrypoints:

- scripts/training/tem_v1_baseline.py
- scripts/training/tem_v2_softmax.py

## HRM Family

Use HRM when you need hierarchical recurrent reasoning with ACT-style halting
over tokenized task inputs.

Characteristics:

- PFC-centric reasoning dynamics
- v1: ACT-supervised control surface
- v2: deliberation actor-critic extensions

Entrypoints:

- scripts/training/hrm_v1_baseline.py
- scripts/training/hrm_v2_rl-striatum.py

## EHC Family

Use EHC when you need integrated PFC/STR/LEC/MEC/HPC interactions under one
training surface.

Characteristics:

- Multi-subsystem integration
- Supports spatial_pretrain and reason_pretrain modes
- Adapter-mediated bridge to task semantics

Entrypoint:

- scripts/training/ehc_v1_pretraining.py

## Rule of Thumb

- Start with TEM for spatial predictive baselines.
- Use HRM when task framing is token reasoning and halting control.
- Move to EHC for integrated cross-region experiments and richer control routes.
