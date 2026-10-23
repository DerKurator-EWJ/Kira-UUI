# v0.40 Exact Context Authority / Evidence Ordering

v0.40 attacks the scale-dependent runtime error isolated by v0.39. It reuses the exact same neural-teacher-generated `.eqbase`; the treatment arm performs no additional teacher probing and no recompilation.

A strong direct PREDICTION from an exact or longest-known suffix context forms a higher evidence tier than recall-only alternatives once its direct score clears the configured floor. This is not unconditional exact precedence: if no direct candidate clears the floor, the historical aggregate-score ordering remains active.

Frozen 32-entity result:

```text
v0.39 replay:       92.1875% -> v0.40 100%
v0.39 paraphrase:   92.1875% -> v0.40 100%
heldout facts:      14.0625% -> unchanged
opaque relation:    100% abstain -> unchanged
outside domain:     100% abstain -> unchanged
```

The `löwe lebt im` failure is corrected while preserving the original numeric scores: `Land` still has the larger aggregate score, but `Savanne` has strong direct-context support and therefore wins by evidence class.

Run with the neural benchmark dependency available:

```bash
eqforge-exact-authority --out out/v0_40_exact_context_authority
```
