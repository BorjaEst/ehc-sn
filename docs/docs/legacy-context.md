## Legacy Context

This repository includes legacy reference codebases for historical context and
migration parity checks.

## Included Legacy Trees

- legacy_tem
- legacy_hrm

These directories are references and should not be treated as canonical runtime
surfaces for new development.

## Why They Exist

- Preserve implementation context from prior TEM and HRM codebases.
- Enable parity checks during migration and adapter/runtime evolution.
- Support interpretation of manuscript-level terminology against prior code.

## Current Canonical Namespace

New work belongs under src/ehc_sn and imports should use ehc_sn.

## Reference Files

- `legacy_tem/README.md`
- `legacy_hrm/README.md`

## Usage Guidance

- Use legacy code as a historical reference, not as a source for new imports.
- Confirm current behavior against `src/ehc_sn` before making design or API decisions.
- If you need to compare runtime semantics, prefer side-by-side tests rather than
  copying legacy implementation directly.

## Migration Notes

- New development should remain in `src/ehc_sn` and use the canonical package
  namespace `ehc_sn`.
- When updating models or tasks, preserve the intent of legacy proofs-of-concept
  while replacing legacy wiring with current adapters and controllers.
- Keep legacy artifacts isolated from production code and use `legacy_*`
  directories only for parity validation and historical understanding.
