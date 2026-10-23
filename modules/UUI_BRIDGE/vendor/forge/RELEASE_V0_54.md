# Equinox Forge v0.54 — Unified Numeric Relation Routing

v0.54 removes the need for runtime-side semantic family dispatch from the new numeric path. The compiler turns relation-family IDs into opaque numeric aliases to exact recursive relation macros, while ordinary scalar predicates, relation facets, and relation-macro IDs use the same public runtime API.

## Core rule

```text
query relation ID
       ↓
compiled numeric route table
       ↓
scalar predicate OR relation-macro ID
       ↓
existing concept-side numeric resolver
```

The runtime does not inspect names such as `PROPERTY` or `ACTION`. Family membership is used only at compile time to discover an exact macro leaf-set match. The exported route table contains numeric IDs and route kinds only.

## Frozen reference result

The reference graph contains 8 relation facets, 2 exact family-level relation macros, and 1 higher relation macro.

- 2 family IDs are compiled as numeric aliases to their exact relation macros.
- 3 relation-macro IDs remain directly executable.
- 10 scalar predicate/facet IDs use scalar execution.
- PROPERTY through the unified API returns the same 3 values as the legacy v0.51 family view.
- ACTION through the unified API returns the same 3 values as the legacy v0.51 family view.
- The higher relation macro still returns the 6-value union.
- An ordinary scalar `IS_A` query is executed through the same `resolve(subject_id, relation_id)` method.
- The regression test sabotages the legacy `resolve_family()` method; unified family queries continue to work unchanged.

## Architectural significance

v0.53 proved that recursive relation macros can execute grouped predicate queries. v0.54 makes those macros the compiled execution target for relation-family IDs as well. The new runtime interface therefore no longer needs to know that a numeric ID corresponds to a human category such as PROPERTY or ACTION.

Legacy family APIs remain in the package for backward compatibility and regression comparison, but the v0.54 runtime path does not depend on them.

## Scope boundary

v0.54 is a routing unification, not a new inference mechanism. It does not change concept-side truth authority, scores, inheritance, or macro semantics, and it does not claim a speedup over the specialized v0.51 hash index. Its purpose is to replace domain-specific runtime dispatch with compiled numeric routing.
