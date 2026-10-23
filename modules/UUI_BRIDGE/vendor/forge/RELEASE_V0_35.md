# Equinox Forge v0.35 — Automatic Relation Composition Induction

## Purpose

v0.34 proved that an explicitly compiled relation-composition schema can execute a predictive relation family that was absent from direct runtime training. Its main limitation was equally explicit: the schema itself was authored by the fixture.

v0.35 attacks that limitation.

> Can Forge infer a reusable relation-composition program from repeated direct observations and an independently represented semantic graph, then compile only the induced program into a runtime that transfers to a held-out entity?

In the controlled fixture, yes.

## What is now induced

Forge receives three offline observations:

```text
Katze wohnt in     -> Haus
Kaninchen wohnt in -> Bau
Fuchs wohnt in     -> Bau
```

and a semantic graph containing, among other edges:

```text
Katze     --is_a--> Haustier --property--> Haus
Kaninchen --is_a--> Wildtier --property--> Bau
Fuchs     --is_a--> Wildtier --property--> Bau
```

The inducer is not supplied with either the relation phrase or a path schema. It:

1. tokenizes each observation prompt;
2. replaces each token position in turn with a candidate slot;
3. groups examples that share the same remaining literal structure;
4. requires repeated distinct fillers;
5. searches allowed semantic paths from each filler to that observation's target;
6. aggregates path signatures across examples and independent sources;
7. promotes only a unique sufficiently supported signature to `stable`.

The resulting stable program is:

```text
pattern: [SLOT_0] wohnt in
path:    is_a(out) -> property(out)
```

## Held-out transfer

`Hund wohnt in` is not present in the induction observations. The evaluated runtime contains the induced schema and the non-predictive semantic path:

```text
Hund --is_a--> Haustier --property--> Haus
```

Result:

```text
Hund wohnt in  -> Haus
Stein wohnt in -> ABSTAIN
Hund wohnt aus -> ABSTAIN
```

The three direct induction observations are offline evidence only and are not compiled as `PREDICTION` edges into the evaluated runtime BASE.

## Default induction contract

A stable schema requires by default:

- at least 3 direct examples;
- at least 3 distinct slot fillers;
- at least 3 supporting examples for the winning path;
- at least 2 distinct `source_id@source_version` sources;
- at least 75% observation coverage;
- mean multiplicative path quality >= 0.50;
- a support margin of at least 1 over the runner-up path;
- path length <= 3;
- premise relation types restricted to `is_a`, `property`, and `action`.

These are research defaults, not claimed universal constants.

## Adversarial controls

### Same-source pseudo-consensus

All three observations are relabeled as one source. Support remains numerically high, but `source_count=1`, so no stable schema is emitted.

### Incomplete coverage

One training filler loses the semantic path to its target. The best program covers only 2/3 observations (~0.667), below the 0.75 threshold, so it remains non-stable.

### Competing programs

A second graph program `action -> property` is added with exactly the same three-example support as `is_a -> property`. Forge does not break the tie by deterministic ordering. Both top candidates become `contested`; no stable runtime schema is available.

## Frozen ladder result

With the positively induced schema:

| Level | Top-1 |
|---|---:|
| L0 exact replay | 1.0 |
| L1 suffix transfer | 1.0 |
| L2 recombination | 1.0 |
| L3 novel order | 1.0 |
| L4 evidence-backed novel entity | 1.0 |
| L5 automatically induced composition | 1.0 |

## CLI

```bash
eqforge-relation-induce --out out/v0_35_relation_composition_induction
```

Historical explicit-schema benchmark remains available:

```bash
eqforge-relation-compose --out out/v0_34_relation_composition
```

## Non-claims

- v0.35 does not discover arbitrary natural-language semantics from unrestricted text.
- The current inducer handles one varying prompt slot and outgoing semantic path programs only.
- Independence currently counts distinct `source_id@source_version`; source lineage/family correlation is not yet modeled.
- Absence of a path is treated as missing support/coverage, not a logically proven contradiction.
- The benchmark is synthetic and validates mechanism behavior, not broad language understanding.
- Path and promotion thresholds are heuristic research parameters and require empirical calibration at scale.
