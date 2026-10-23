# Equinox Forge v0.23 release status

Milestone: Typed Slot Constraints / Semantic Admission.

Verification performed in the build environment:

- v0.1 through v0.17: green in the combined regression runner before the 120s execution window expired.
- v0.18 through v0.23: each run separately and green.
- v0.22 backward-compatibility frame test: green.
- v0.23 typed-slot test: green.
- Runtime cognition, base-pack and mechanism-attribution critical regressions: green.
- CLI `induce-frames` typed-slot smoke test: produced valid v0.23 IR.

Controlled v0.23 fixture:

- 1 typed frame
- 35 runtime nodes
- 16 runtime relations
- 3,509-byte `.eqbase`
- semantic admission applicability recall: 100%
- semantic admission applicability precision: 100%
- open-slot baseline false-positive: `Stein ist ein -> Tier`
- typed-slot result: `Stein ist ein -> abstain`

These are deterministic pipeline/fixture results, not claims of broad natural-language semantic understanding.
