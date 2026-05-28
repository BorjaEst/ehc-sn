## Model Families

Model implementations live under src/ehc_sn/models.

## Core Families

- TEM family: src/ehc_sn/models/tem
- HRM family: src/ehc_sn/models/hrm
- EHC family: src/ehc_sn/models/ehc

## Contract Summary

- Models remain task-agnostic.
- Models do not own benchmark semantics.
- Task binding and runtime protocol adaptation are handled by adapters.

## Related Surfaces

- Adapters: src/ehc_sn/adapters
- Tasks: src/ehc_sn/tasks
- Objectives: src/ehc_sn/objectives
