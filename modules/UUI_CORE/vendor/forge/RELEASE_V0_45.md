# Equinox Forge v0.45 — Structural Closure

v0.45 builds the first non-duplicating structural view on top of the v0.44 canonical numeric ID space.

A concept handle can now address its own relations and the relations reachable through its configured hierarchy without copying those ancestor facts into the child concept.

```text
ID_Eule
├─ local fact -------------------- origin: Eule, depth 0
├─ ID_Vogel relation ------------ origin: Vogel, depth 1
└─ ID_Tier relation ------------- origin: Tier,  depth 2
```

The semantic base still stores each canonical relation exactly once.

## Core representation

The canonical graph remains:

```text
(subject_id, predicate_id, object_id)
```

A structural-closure entry is only a routing reference:

```text
root_id
predicate_id
object_id
origin_subject_id
inheritance_depth
inheritance_path
```

Scores, confidence, source count, and evidence count are **not copied into the closure table**. Loading a closure joins the reference back to `canonical_relation_triples`.

This means two children can both address:

```text
Vogel --HAS--> Federn
```

without Forge materializing redundant semantic facts such as:

```text
Eule  --HAS--> Federn
Falke --HAS--> Federn
```

unless those facts independently exist in the canonical source graph.

## Persistent structures

v0.45 adds:

- `structural_closure_runs`
- `structural_closure_entries`

`structural_closure_entries` is explicitly an index of references into one canonical-ID build, not a second fact store.

## Deterministic routing

Hierarchy traversal is breadth-first and numeric. For the same effective `(predicate_id, object_id)` reachable from several hierarchy levels, the closure view selects the nearest origin. Equal-depth ties are resolved deterministically by evidence quality and then by numeric path.

The canonical graph itself remains lossless; farther duplicate evidence is not deleted.

Cycles in malformed hierarchy input are guarded by visited concept IDs and cannot recursively expand forever.

## Frozen reference controls

The owl/falcon reference graph verifies:

```text
canonical semantic triples before closure: 7
canonical semantic triples after closure:  7
closure roots:                             2
closure entries:                           11
  direct:                                  3
  inherited/addressed:                     8
```

Both owl and falcon address the one stored bird→feathers relation with `origin_subject_id = bird` and `inheritance_depth = 1`. Owl reaches animal→alive at depth 2 and retains its own owl→nocturnal relation at depth 0.

Reversing insertion order leaves the closure digest unchanged.

## Important boundary

Structural closure is **addressability**, not yet universal semantic inheritance.

v0.45 deliberately does not claim that every relation found on an ancestor should automatically be asserted for a child. It preserves relation origin and specificity depth so a later runtime/routing layer can decide which relation types inherit, which require overrides, and which are only useful as structural context.

v0.45 also does not yet collapse a stable closure into a new macro ID. That is the intended next layer.

Run:

```bash
eqforge-structural-closure --out out/v0_45_structural_closure
```

See `benchmarks/v0_45/` and `tests/test_structural_closure_v045.py`.
