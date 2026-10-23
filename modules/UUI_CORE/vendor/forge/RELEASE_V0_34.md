# Equinox Forge v0.34 — Evidence-Backed Relation Composition

## Purpose

v0.33 recovered novel-entity transfer while deliberately leaving L5 at zero. v0.34 asks a narrower and scientifically defensible version of the L5 question:

> Can a relation family that is absent from direct predictive training be executed when its semantics are already compiled as an explicit composition over other relations?

The answer in the controlled fixture is yes.

## Core rule

v0.34 does **not** turn arbitrary graph proximity into prediction authority. Runtime relation composition requires all of the following:

1. the live token sequence matches a compiled `relation_composition_schema`;
2. the schema is in an allowed lifecycle status (`stable` by default);
3. the schema contains exactly one entity slot in this first implementation;
4. every path hop names an allowed non-predictive relation type and direction;
5. the complete path exists from the live entity in the compiled `.eqbase`;
6. the multiplicative path quality clears `relation_composition_min_path_quality`.

Only then may the terminal node become a prediction candidate.

## Deterministic L5 fixture

The frozen v0.31 predictive training set still excludes the full prompt:

```text
Hund wohnt in
```

No evaluation teacher prediction is inserted into the BASE and no direct `PREDICTION` edge for that prompt is added.

Auxiliary non-predictive support provides:

```text
stable schema:
[SLOT_0] wohnt in
    => path [is_a(out), property(out)]

semantic graph:
Hund --is_a--> Haustier --property--> Haus
Stein --is_a--> Objekt
```

Result:

```text
Hund wohnt in  -> Haus
Stein wohnt in -> ABSTAIN
```

## Necessary-condition controls

The benchmark compiles five tracks over the same frozen predictive training data:

| Track | L5 Top-1 |
|---|---:|
| opaque control: no schema, no path | 0.0 |
| graph only | 0.0 |
| schema only | 0.0 |
| schema + complete path | **1.0** |
| same BASE, runtime composition disabled | 0.0 |

L0–L4 remain at `1.0` in the full track.

Additional negative controls:

- `Stein wohnt in` matches the schema but lacks the complete path and abstains;
- `Hund wohnt aus` has the semantic graph but does not match the schema and abstains.

## Runtime diagnostics

`RuntimePredictionReport` now exposes `relation_composition` diagnostics and work counters including:

```text
relation_composition_schema_checks
relation_composition_path_relation_checks
relation_composition_matches
relation_composition_admissions
relation_composition_rejections
```

Prediction candidates also expose a separate `composition_score`.

## Policy switches

Key controls:

```text
enable_relation_composition_prediction
relation_composition_weight
relation_composition_min_path_quality
relation_composition_max_hops
relation_composition_max_matches
relation_composition_allowed_premise_types
relation_composition_allowed_statuses
```

Direct `PREDICTION`, `TRANSITION`, and nested `COMPOSITION` relations are not allowed as premise hops by default.

## CLI

```bash
eqforge-relation-compose --out out/v0_34_relation_composition
```

Historical benchmarks remain available:

```bash
eqforge-novel-entity --out out/v0_33_novel_entity
eqforge-support-gate --out out/v0_32_support_gate
eqforge-gauntlet --out out/v0_31_baseline_gauntlet
```

## Non-claims

- v0.34 does not infer a genuinely unknown relation from nothing.
- The composition schema in this release is fixture-authored, not automatically induced.
- A stable schema's provenance is represented only by compiled metadata in this first runtime mechanism; source-lineage validation of schema authorship is not yet enforced at runtime.
- The benchmark is synthetic and validates mechanism behavior, not broad natural-language reasoning.
- L5 in this experiment means held out from **direct predictive training**. Its compositional semantics are intentionally present through a different representation.
