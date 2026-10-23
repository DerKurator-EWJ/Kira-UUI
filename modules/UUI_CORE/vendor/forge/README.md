# Equinox Forge v0.56 — Direct Teacher → Flat Kernel Compiler

v0.56 removes the historical nested runtime-pack chain from the deployment compiler path. Forge may still materialize research/audit state in SQLite, but the deployable output is compiled directly from Forge build runs into the flat v0.55 numeric execution format:

```text
Teacher
  ↓
Top-down structure extraction
  ↓
Canonical IDs + relation facets
  ↓
Structural closure
  ↓
Concept + relation macro collapse
  ↓
DIRECT FLAT COMPILE
  ↓
knowledge.eqkernel.json
  ↓
UnifiedNumericExecutionKernel
```

No v0.48-v0.54 runtime pack is persisted or required by the direct compiler. Those historical pack builders remain only as regression oracles.

Frozen v0.56 reference results:

- 16 Teacher queries,
- 2 concept-macro levels / 3 concept macros,
- 2 relation-macro levels / 3 relation macros,
- 24 flat canonical facts,
- 24 validated concept shortcuts,
- 15 numeric relation routes,
- 100% reference runtime-query accuracy,
- direct flat core exactly equal to the legacy nested→flattened oracle,
- exactly one deployable runtime artifact: `knowledge.eqkernel.json`,
- Teacher and Forge DB removed before runtime evaluation.

The flat execution format remains **v0.55** because v0.56 changes compilation, not runtime semantics.

Run:

```bash
eqforge-direct-teacher-flat --out out/v0_56_direct_teacher_flat
```

See `RELEASE_V0_56.md` and `benchmarks/v0_56/` for the frozen contract. Historical release notes and regression tests remain included.
