# Equinox Forge v0.56 — Direct Teacher → Flat Kernel Compiler

v0.56 turns the v0.55 flat kernel from a *flattened historical runtime stack* into a direct compiler target.

## Compiler model

```text
Teacher evidence
      ↓
Forge canonical / closure / macro build state
      ↓
compile_flat_kernel_direct(...)
      ↓
ONE knowledge.eqkernel.json
      ↓
UnifiedNumericExecutionKernel
```

The direct compiler reads canonical facts, surfaces, concept-macro state, relation-facet state, and relation-macro state directly from Forge build runs. It does **not** build or persist v0.48 numeric, v0.49 macro-numeric, v0.51 faceted, v0.53 relation-macro-runtime, or v0.54 unified-relation runtime packs.

Those older pack builders remain available only as independent regression oracles.

## Frozen reference

The controlled neural-teacher trial produces:

- 16 Teacher queries,
- 2 concept-macro levels / 3 concept macros,
- 2 relation-macro levels / 3 relation macros,
- 24 canonical flat facts,
- 17 surface bindings,
- 24 validated concept shortcuts,
- 15 relation routes,
- 2 compiled family→relation-macro aliases,
- 100% runtime accuracy on the frozen scalar + multivalue reference queries.

Exactly one deployable runtime artifact is left in the trial directory: `knowledge.eqkernel.json`. The Teacher state and Forge SQLite database are removed before runtime evaluation.

## Independent parity oracle

For validation only, v0.56 builds the historical nested v0.48→v0.54 stack inside a temporary directory and flattens it through v0.55. That temporary directory is destroyed immediately.

The direct pack and legacy-oracle pack must have the same:

- inheritance predicates,
- canonical facts,
- surface bindings,
- validated concept shortcuts,
- numeric relation routes,
- runtime policy.

The frozen reference reports exact core parity.

## Runtime format

The runtime artifact format remains `equinox-unified-numeric-execution-kernel-v1`, version **0.55**. v0.56 changes only the compiler path; `UnifiedNumericExecutionKernel` semantics are unchanged.

## Scope

The release reference still uses the controlled local neural Teacher because the execution environment has no external pretrained Hugging Face model installed/cached. The compiler API is Teacher-backend agnostic; an external pretrained HF run remains a separate environment validation, not a claim of this release.
