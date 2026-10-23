# Equinox Forge v0.55 — Unified Numeric Execution Kernel

v0.55 flattens the numeric runtime stack introduced across v0.48-v0.54 into one DB-free runtime artifact and one execution engine. Compilation may still use the historical packs for backward-compatible validation, but the deployed v0.55 kernel needs none of those runtime layers.

## Runtime model

```text
flat facts
+ surface bindings
+ validated concept-macro shortcuts
+ numeric relation routes
        ↓
UnifiedNumericExecutionKernel
        ↓
resolve(subject_id, relation_id)
```

The runtime directly implements the frozen semantics of specific-to-general inheritance, validated concept-macro shortcuts, scalar relation execution, and relation-set / relation-macro execution. It does not instantiate `SpecificToGeneralNumericRuntime`, `MacroAwareNumericRuntime`, `FacetedMacroNumericRuntime`, `RelationMacroNumericRuntime`, or `UnifiedNumericRelationRuntime`.

## Frozen parity audit

The reference fixture compares every compiled relation route across every subject present in the v0.54 runtime graph:

- 240 subject/relation query pairs compared,
- 97 resolved query pairs,
- 0 semantic parity failures,
- 24 validated concept shortcuts retained,
- 15 numeric relation routes retained.

The flat JSON artifact is 7,559 bytes versus 11,243 bytes for the nested v0.54 runtime artifact on the same fixture, or about 67.23% of the previous serialized size.

A fresh process can load the v0.55 pack and execute a query without importing any of the historical v0.48-v0.54 runtime modules.

## Scope

v0.55 is a runtime consolidation release, not a new inference mechanism. It is intentionally required to preserve all v0.54 answers, scores, confidence, specificity, inheritance provenance, and validated macro routing semantics.
