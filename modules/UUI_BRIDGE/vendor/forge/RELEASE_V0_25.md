# Release v0.25 — Prototype Hierarchy / Recursive Concept Induction

v0.25 adds the first recursive semantic abstraction layer above v0.24 slot prototypes.

## Added

- `equinox_forge.prototype_hierarchy`
- `PrototypeHierarchyPolicy`
- `induce_prototype_hierarchy(...)`
- deterministic `recursive_latent_concept` nodes
- child-concept `IS_A` parent evidence
- complete-link cross-concept compatibility over underlying leaf fillers
- depth-aware thresholds with a hard floor
- full leaf provenance and merge metrics
- `RuntimeIndex.traverse_relation(...)`
- CLI: `induce-prototype-hierarchy`
- regression: `test_recursive_prototype_hierarchy.py`
- demo: `examples/build_recursive_prototype_hierarchy_demo.py`

## Controlled regression result

```text
base prototypes:          4
recursive concepts:       3
memberships:              6
levels generated:         2
highest hierarchy depth:  2
all base concepts reach root: yes
```

No human ontology label is asserted automatically. All recursive concepts remain provisional and evidence-derived.
