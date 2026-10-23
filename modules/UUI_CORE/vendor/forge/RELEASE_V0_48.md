# Equinox Forge v0.48 — Specific-to-General Numeric Runtime

v0.48 is the first runtime release that directly executes the canonical numeric ID substrate introduced in v0.44–v0.47.

## Core rule

For a query `(subject_id, predicate_id)` the runtime searches:

```text
specific concept ID
  ↓ local fact?
parent ID
  ↓ fact?
next parent ID
  ↓ ...
```

The nearest hierarchy depth with qualifying direct evidence owns the answer. A valid local fact is therefore not outvoted by a stronger general fact. A local fact below the configured quality floor is ignored and the runtime may fall back to a parent.

## What is new

- DB-independent numeric runtime pack (`knowledge.eqnum.json`)
- numeric `(subject,predicate,object)` runtime core
- separate surface→concept bindings for I/O only
- cycle-safe parent traversal
- local/specific override semantics
- quality-gated parent fallback
- same-depth conflict abstention
- explanation fields: origin ID, inheritance depth and numeric path

## Reference proof

```text
Eule → covering      = Federn     (depth 1, origin Vogel)
Eule → alive         = true       (depth 2, origin Tier)
Eule → can_fly       = true       (depth 1, origin Vogel)
Pinguin → can_fly    = false      (depth 0 local override)
WeakBird → can_fly   = true       (weak local false rejected; parent fallback)
```

The Pinguin local override is intentionally lower-quality than the inherited Vogel fact. It still wins because specificity is categorical once the local evidence clears the minimum quality gate.

## Non-goals

v0.48 does not yet route through recursive macro IDs and does not replace the historical `RuntimeCognition` implementation. It establishes the execution semantics of the new numeric substrate first.

Run:

```bash
eqforge-numeric-runtime --out out/v0_48_numeric_runtime
```

See `benchmarks/v0_48/` for the frozen contract and reference.
