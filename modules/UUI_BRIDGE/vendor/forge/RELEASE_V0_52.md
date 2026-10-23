# Equinox Forge v0.52 — Recursive Relation-ID Collapse

v0.52 makes the predicate axis recursively collapsible. The first-class relation facets introduced in v0.51 are no longer terminal IDs: Forge derives a structural meta-signature for each facet relation, finds repeated signatures, assigns deterministic relation-macro IDs, residualizes lower relation macros when a higher relation macro absorbs shared structure, and permits the resulting IDs to serve as ordinary canonical predicates.

```text
(subject ID, relation ID, object ID)
                  ↓
          relation structure
                  ↓
        relation Macro-ID
                  ↓
   relations between relation Macros
                  ↓
      higher relation Macro-ID
```

## Frozen v0.52 result

The controlled teacher/facet fixture contains 8 canonical relation facets: 4 PROPERTY-derived and 4 ACTION-derived facets.

- 8 facet relation IDs collapse to 2 level-1 relation macros.
- Those 2 relation macros collapse to 1 level-2 relation macro.
- Total recursive relation macros: 3.
- Recursive levels: 2.
- Membership references: 10 (8 facet→L1 + 2 L1→L2).
- Local residual meta-facts after collapse: 3.
- Effective recursively reconstructable meta-facts across macros: 5.
- All 3 macro IDs are registered in namespace `relation` as `recursive_relation_macro` identities.
- A level-2 relation macro is validated as a legal predicate ID in a canonical `(subject, predicate, object)` triple.
- Reversing source insertion order produces the same relation-macro IDs, memberships, effective signatures, and digest.

## Structural rule

Each v0.51 facet relation initially has a numeric meta-signature equivalent to:

```text
facet relation ID
  ├─ relation_is_a       → coarse family ID
  ├─ relation_kind       → relation-facet kind ID
  └─ relation_bound_obj  → bound object ID
```

Facets in the same family share two structural facts and collapse first. The resulting PROPERTY and ACTION relation macros still share the generic relation-facet kind and therefore collapse again at the next level. When the higher macro owns that shared fact, child macros retain only their family-specific residual fact and recover the shared fact recursively.

## Scope boundary

v0.52 does not yet route runtime queries through relation macros and does not claim semantic discovery of human relation labels. It proves that relation identities can undergo the same deterministic, residual, recursive collapse already demonstrated for concept/macros, while remaining legal first-class predicate IDs. A later runtime layer can exploit this predicate hierarchy for routing/compression.
