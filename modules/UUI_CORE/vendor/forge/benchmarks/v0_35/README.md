# v0.35 Automatic Relation Composition Induction Benchmark

This benchmark freezes the first automatically induced relation-composition program in Equinox Forge.

The induction set contains three direct observations with different slot fillers. The held-out runtime query is `Hund wohnt in`, whose answer must be obtained from the induced program plus the semantic graph rather than from a compiled direct observation.

Run:

```bash
eqforge-relation-induce --out out/v0_35_relation_composition_induction
```

`relation_composition_induction_contract.json` freezes the core examples and controls. `reference.json` records the expected deterministic mechanism-level result for v0.35.
