# Equinox Forge v0.28 — Automatic Exception / Override Induction

v0.28 learns **specific subtype overrides** from systematic residual behaviour against a broader hierarchical rule.

The key constraint is that the subtype must already exist independently in the compiled Equinox concept hierarchy. The exception target is not allowed to define its own class.

Example:

```text
existing broad rule:
Concept_ROOT / [SLOT] atmet → Luft

existing subtype from independent semantic evidence:
Adler ─┐
       ├→ Concept_Bird → Concept_ROOT
Falke ─┘

new concrete observations:
Adler atmet → Sauerstoff
Falke atmet → Sauerstoff

v0.28:
Concept_Bird / [SLOT] atmet → Sauerstoff   # provisional override
```

The general frame is retained unchanged. Runtime precedence is still handled by v0.27 specificity resolution, so the direct subtype override suppresses the broader inherited frame only for fillers admitted to that subtype.

## Default acceptance contract

An override requires:

- at least 2 concrete matching contexts;
- at least 2 unique fillers;
- an already compiled latent subtype prototype;
- an `IS_A` path from that subtype to the broad frame's higher concept;
- consistent target support across the subtype contexts;
- a configurable score margin over the broad rule's target inside the subtype;
- a different subtype Top-1 target by default.

A single anomaly therefore cannot create an exception under the default policy.

## Runtime ablation

`RuntimeCognitionPolicy.enable_exception_override_frames` can disable automatic exception frames on the same immutable `.eqbase`. Disabled exception frames are excluded from direct matching **and** parent/relation recall, so the baseline cannot leak through the same override node.

## CLI

```bash
eqforge induce-exceptions master.sqlite overrides.eqir.json \
  --min-contexts 2 \
  --min-unique-fillers 2 \
  --min-exception-target-score 0.50 \
  --min-disagreement-margin 0.20 \
  --persist
```

Benchmark the same pack with automatic exception frames disabled versus enabled:

```bash
eqforge benchmark-exceptions knowledge.eqbase cases.jsonl
```

## Deterministic fixture

The v0.28 fixture first compiles:

```text
Hund atmet  → Luft
Katze atmet → Luft

Adler/Falke form a separate latent subtype under the same higher concept.
```

Only after that hierarchy exists, two new observations are added:

```text
Adler atmet → Sauerstoff
Falke atmet → Sauerstoff
```

Forge induces one bird-subtype override. `Eule` is semantically known and already classified into the bird prototype, but **`Eule atmet` is never observed during induction**. With all alternate runtime routes disabled:

```text
without automatic exception frames:
Eule atmet → Luft
Hund atmet → Luft
accuracy = 50 %

with v0.28 exception frames:
Eule atmet → Sauerstoff
Hund atmet → Luft
accuracy = 100 %
```

The override itself is learned only from `Adler atmet` and `Falke atmet`. No held-out `Eule atmet` answer is used to define the subtype or the override, so this fixture tests transfer of the learned exception to a new member of the already known subtype.

## Non-claims

- v0.28 does not prove human-like exception learning.
- The induced override is provisional derived knowledge, not an asserted universal truth.
- This first implementation supports one-slot text frames and already compiled latent subtype prototypes.
- It does not yet model temporal exception decay, contradictory exception sources, or confidence hysteresis across repeated rebuilds.
