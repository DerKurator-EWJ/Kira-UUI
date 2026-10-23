# Equinox Forge v0.26 — Higher-Concept Utility / Hierarchical Frame Transfer

v0.26 is the first milestone that tests whether a recursively induced higher concept is **useful for behavior**, not merely representable.

## Core result

A frame learned only from one lower subtype can be promoted to an already-grounded higher latent concept:

```text
Hund / Katze  → Concept_A ─┐
                           ├─→ Concept_ROOT
Adler / Falke → Concept_B ─┘

original frame:
[SLOT:Concept_A] atmet → Luft

promoted frame:
[SLOT:Concept_ROOT] atmet → Luft
```

`Adler atmet` was never compiled as a predictive context. With ordinary v0.24 direct-prototype admission it remains unresolved. With v0.26 hierarchical admission the runtime follows:

```text
Adler --is_a→ Concept_B --is_a→ Concept_ROOT
```

and may therefore reuse the promoted `… atmet → Luft` frame.

A known lexical counterexample (`Stein`) that has no compiled path into the higher concept is rejected.

## What is new

- `hierarchical_frame_transfer.py`;
- non-destructive promotion of prototype-typed frames to reachable `recursive_latent_concept` ancestors;
- no new prediction targets are introduced during promotion;
- transferred slots persist their base prototype, selected higher concept, compile-time path and path quality;
- runtime multi-hop `IS_A` admission for frame slots;
- bounded hop count and minimum path-quality threshold;
- auditable runtime admission trace (`hierarchical_prototype_membership`);
- CLI: `promote-hierarchical-frames`;
- isolated v0.26 regression proving that the higher concept enables a transfer that the lower prototype alone cannot make.

## Deliberate limitation

A higher latent concept is still not automatically a human ontology label. v0.26 only shows that an evidence-grounded higher class can function as a reusable admission boundary for a frame. Whether that boundary is semantically useful at scale must be measured on broader held-out data.

## Controlled fixture

The v0.26 demo compiles only these predictive contexts:

```text
Hund atmet  → Luft
Katze atmet → Luft
Adler fliegt → Himmel
Falke fliegt → Himmel
```

Independent semantic evidence induces two lower prototypes and one shared higher concept. The `… atmet` frame is then promoted from the Hund/Katze prototype to the higher concept without adding any `Adler atmet` or `Falke atmet` prediction evidence.

Isolated runtime result (all exact/suffix/lexical/parent/relation recall paths disabled):

```text
old direct-prototype admission:
Adler atmet → unresolved
Falke atmet → unresolved

v0.26 hierarchical admission:
Adler atmet → Luft
Falke atmet → Luft
Stein atmet → abstain
```

For both successful transfers the runtime admission trace contains a two-hop path:

```text
surface → lower prototype → higher recursive concept
```

Demo pack:

```text
runtime nodes:      50
relations:          28
pack bytes:     16,068
promoted frames:     2
```

These values are fixture diagnostics, not a general performance or semantic-accuracy claim.

## CLI

```bash
eqforge promote-hierarchical-frames master.sqlite hierarchical-frames.eqir.json \
  --min-path-edge-quality 0.50 \
  --admission-min-path-score 0.70 \
  --persist
```
