# Equinox Forge v0.27 — Hierarchical Specificity / Override Resolution

## Added
- categorical frame-scope precedence in `RuntimeCognition`;
- `enable_frame_specificity_override` runtime switch;
- suppression diagnostics and `frame_specificity_suppressed` work metric;
- `specificity_resolution.py` pack-level A/B benchmark;
- `benchmark-specificity` CLI;
- deterministic conflict regression where a lower-scope rule overrides a higher-scoring inherited rule.

## Precedence
`exact context > direct/lower prototype frame > hierarchical transferred frame > untyped open frame`.

This is an override rule. Suppressed broader frames contribute no prediction evidence. Equal-scope frames remain eligible to ensemble.

## Regression
The new v0.27 conflict test passes together with v0.22–v0.26 frame/prototype/hierarchy tests and core runtime pack/cognition tests.
