# Equinox Forge v0.51 — Canonical Relation IDs / Multi-valued Structural Closure

v0.51 removes the scalar collision exposed by v0.50. The original teacher evidence remains unchanged, but Forge derives a first-class numeric relation identity for every coarse `(relation family, object)` pair. A relation ID is therefore a normal canonical identity and is itself linked numerically to its coarse relation family.

```text
subject ID
   ↓
facet relation ID
   ↓
object ID

facet relation ID
   ↓ relation_is_a
coarse relation-family ID
```

No domain label such as `covering` or `locomotion` is required to create the facet. The relation identity is a deterministic handle for structure, not a word-level ontology inserted by Forge.

## Frozen v0.51 results

- 8 coarse teacher attribute relations become 8 canonical facet relation IDs.
- All 8 relation IDs are linked numerically to PROPERTY or ACTION.
- The four leaf concepts each recover all three inherited PROPERTY values and all three inherited ACTION values: 24/24 values total.
- Multi-valued family-query accuracy: 100%.
- Recursive macro collapse remains two levels / three macros.
- Safe macro shortcuts rise from 0 in v0.50 to 24 in the frozen v0.51 fixture.
- Macro routing is used for all 24 resolved family values.
- Hierarchy-node work falls from 112 to 88 (21.43%) for the frozen multi-value query set.
- Teacher state and Forge DB are offline before runtime evaluation.

## Scope boundary

v0.51 does **not** claim that semantic human relation labels are autonomously discovered. It proves a domain-agnostic structural relation identity: `(family ID, object ID) -> relation ID`. Later work may collapse or name relation IDs using teacher evidence, exactly as concept IDs are already recursively collapsed.
