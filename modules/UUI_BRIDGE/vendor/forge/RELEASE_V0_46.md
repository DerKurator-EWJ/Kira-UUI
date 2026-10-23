# Equinox Forge v0.46 — Recursive Macro Collapse

v0.46 adds the first recursive collapse layer on top of the canonical numeric ID space and v0.45 structural closure.

Stable common relation structures across several concept IDs can now be assigned their own deterministic numeric macro ID. That macro ID is itself a first-class numeric identity and can become a member of a higher macro on the next collapse level.

```text
Concept IDs
   ↓ common structural signature
Macro ID (level 1)
   ↓ common structure between macro IDs
Macro ID (level 2)
   ↓
...
```

The canonical semantic graph remains the audit source and is not destructively rewritten.

## Numeric macro identity

Macro identity is derived from the stable common `(predicate_id, object_id)` signature, not from insertion order or a SQLite row number.

A macro therefore has the same deterministic 63-bit handle when the same structural signature is rediscovered in a rebuild.

The numeric macro layer stores:

```text
macro_id
members: [numeric IDs]
local residual relations: [(predicate_id, object_id), ...]
```

No language surface is required for collapse.

## Recursive residual compaction

The central v0.46 behavior is not merely grouping.

When two level-1 macros share facts and collapse into a level-2 macro, those parent-owned facts are removed from the level-1 macros' **local** relation rows.

```text
Before level-2 collapse:

Bird-like macro:
  IS_A animal
  alive
  move
  feathers
  fly

Mammal-like macro:
  IS_A animal
  alive
  move
  fur
  walk

After level-2 collapse:

Higher macro:
  IS_A animal
  alive
  move

Bird-like macro local residual:
  feathers
  fly
  (+ its specific class edge)

Mammal-like macro local residual:
  fur
  walk
  (+ its specific class edge)
```

`effective_macro_facts()` recursively reconstructs the full child signature from local residuals plus higher macro relations.

This is the first actual recursive structural compression in the new numeric ID space.

## Persistent structures

v0.46 adds:

- `recursive_macro_collapse_runs`
- `recursive_macros`
- `recursive_macro_members`
- `recursive_macro_relations`

`recursive_macro_relations` stores local/residual macro facts. Higher shared facts are accessed through recursive macro membership instead of being duplicated at every macro level.

## Reference proof

The frozen demo starts with four concrete roots:

```text
owl    ┐
falcon ┘ → specific level-1 macro

dog  ┐
wolf ┘ → specific level-1 macro
```

The two level-1 macros share a more general animal structure and collapse again:

```text
level-1 macro A ┐
                ├→ level-2 macro
level-1 macro B ┘
```

Frozen v0.46 result:

```text
canonical semantic triples:     12 before / 12 after
macro levels built:             2
macro count:                    3
membership rows:                6
local macro relation rows:      9
effective macro relation rows: 15
potential reference savings:    9
```

The two level-1 macros each retain six effective facts but only three local facts after level-2 residual compaction.

## Boundaries

v0.46 does **not** yet replace RuntimeCognition with macro-ID routing.

It also does not delete the canonical source graph. Collapse is a derived numeric layer so provenance, rebuilds, and future alternative collapse policies remain possible.

The next logical step is to make the runtime consume this recursive macro layer as its primary routing structure rather than treating it only as a compiled overlay.

Run:

```bash
eqforge-recursive-macro --out out/v0_46_recursive_macro_collapse
```

See `benchmarks/v0_46/` and `tests/test_recursive_macro_collapse_v046.py`.
