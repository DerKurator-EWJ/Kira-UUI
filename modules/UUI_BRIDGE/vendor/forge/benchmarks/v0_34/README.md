# v0.34 — Evidence-Backed Relation Composition

This benchmark attacks the frozen L5 case without allowing opaque zero-shot relation guessing.

`Hund wohnt in → Haus` is absent from predictive training. It becomes executable only when the BASE contains both:

1. a stable compiled composition schema for `[SLOT_0] wohnt in`; and
2. the complete non-predictive semantic path `Hund --is_a--> Haustier --property--> Haus`.

Graph-only, schema-only, runtime-ablation, incomplete-path and unmatched-pattern controls must abstain.

Run:

```bash
eqforge-relation-compose --out out/v0_34_relation_composition
```
