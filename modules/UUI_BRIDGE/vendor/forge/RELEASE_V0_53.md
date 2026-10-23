# Equinox Forge v0.53 — Relation-Macro Runtime Routing

v0.53 connects the recursive predicate hierarchy from v0.52 to a DB-free numeric runtime. Relation-macro IDs become executable query handles over descendant facet relations while concrete facts remain resolved by the existing v0.51/v0.49 concept-side runtime.

## Core rule

A relation macro is routing, not evidence:

```text
relation Macro-ID
  ↓ select descendant facet predicates
facet relation IDs
  ↓ existing scalar resolver
objects / inherited values
```

No relation macro may invent a value. It only narrows or groups the predicate set that is evaluated.

## Frozen reference result

The reference relation space contains 8 facets: 4 PROPERTY-derived and 4 ACTION-derived facets. v0.52 produces 2 level-1 relation macros and 1 level-2 macro.

- PROPERTY macro candidate facets: 4 / 8.
- ACTION macro candidate facets: 4 / 8.
- Candidate reduction versus an unindexed 8-facet scan: 50%.
- PROPERTY macro output exactly matches the existing PROPERTY family-index output.
- ACTION macro output exactly matches the existing ACTION family-index output.
- The level-2 root relation macro returns the union of both child macro result sets.
- The relation macro does **not** beat the already-specialized family hash index: both inspect 4 candidate facets for PROPERTY. v0.53 intentionally records this as a non-win rather than manufacturing a speedup claim.

## Architectural significance

Before v0.53, PROPERTY/ACTION multi-value queries depended on a dedicated family→facet runtime index. With v0.53, the same kind of grouped predicate execution can be addressed by an ordinary recursive numeric relation-macro ID. This is the first runtime step toward a uniform relation space where higher-order relation IDs can route to lower-order relation IDs without domain-specific code for every family.

## Scope boundary

v0.53 does not claim that relation macros outperform a perfect specialized hash index on this tiny fixture, and it does not change concept-side semantics. Performance gains should be evaluated later on larger relation spaces where generic macro routing can replace broader scans or multiple specialized indexes.
