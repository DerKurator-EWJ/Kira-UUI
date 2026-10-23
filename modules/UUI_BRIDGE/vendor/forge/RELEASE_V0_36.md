# Equinox Forge v0.36 — Composition Hypothesis Lifecycle

## Purpose

v0.35 can induce a reusable relation-composition program from repeated observations and an independently represented semantic graph. Its promotion decision, however, exists only inside one rebuild: a program can be `stable` within that run without any persistent history of whether the same explanation survives later evidence.

v0.36 separates these two questions:

1. **Induction strength:** is the program well supported in the current rebuild?
2. **Runtime authority:** has the program survived enough independent evidence epochs over time to be executable?

The persistent lifecycle is deliberately more conservative than the single-run inducer.

## Lifecycle contract

The benchmark tracks the same induced program:

```text
[SLOT_0] wohnt in
    => is_a(out) -> property(out)
```

through six evidence epochs:

```text
support_1   -> candidate
support_2   -> stable
conflict_1  -> contested
conflict_2  -> quarantined
recovery_1  -> candidate
recovery_2  -> stable
```

The source sets differ between successive support/conflict epochs. Replaying an identical source set does not count as a new epistemic epoch.

## Runtime authority

Only lifecycle `stable` schemas retain the ordinary runtime metadata status `stable`. All other lifecycle states are compiled as non-authoritative schemas and therefore ignored by the default relation-composition runtime policy.

For the held-out prompt:

```text
Hund wohnt in
```

the benchmark result is:

| Epoch | Lifecycle | Runtime |
|---|---|---|
| support_1 | candidate | ABSTAIN |
| support_2 | stable | Haus |
| conflict_1 | contested | ABSTAIN |
| conflict_2 | quarantined | ABSTAIN |
| recovery_1 | candidate | ABSTAIN |
| recovery_2 | stable | Haus |

A single strong rebuild therefore cannot activate a newly discovered composition program, and a single source-novel conflict withdraws runtime authority immediately.

## Source-novel epoch control

v0.36 distinguishes repeated execution from new evidence.

Two identical clean rebuilds remain:

```text
candidate -> candidate
```

rather than manufacturing `stable` by rerunning the compiler.

After a program has become stable, replaying the exact same conflict set twice remains:

```text
contested -> contested
```

rather than manufacturing `quarantined` from duplicate evidence.

## Persistent storage

v0.36 adds:

- `composition_hypothesis_runs`
- `composition_hypothesis_states`

The persistent state records hypothesis/family identity, status, first/last seen runs, support/conflict streaks, missed rebuilds, and the previous support/conflict source sets. `CanonicalDB` initializes and reports these tables.

Hypothesis identity is derived from the invariant prompt pattern, slot position, and semantic relation path. This keeps the lifecycle stable across rebuilds even when runtime node metadata changes.

## Default lifecycle policy

Research defaults:

- 2 source-novel clean support epochs for `stable`;
- at least 2 supporting sources in a clean epoch;
- at least 1 new source after the first consecutive support epoch;
- 2 source-novel conflict epochs for `quarantined`;
- at least 1 new source after the first consecutive conflict epoch;
- 2 missed rebuilds for quarantine of a disappeared hypothesis.

These are explicit heuristic defaults, not universal constants.

## CLI

```bash
eqforge-composition-lifecycle --out out/v0_36_composition_lifecycle
```

The historical v0.35 induction benchmark remains available:

```bash
eqforge-relation-induce --out out/v0_35_relation_composition_induction
```

## Non-claims

- v0.36 does not yet infer source lineage or detect correlated teachers that share a common ancestor.
- A source-novel epoch means a different `source_id@source_version` set, not proven causal independence.
- The current conflict signal relies on the v0.35 inducer exposing an equally supported competing program as `contested`; semantic contradiction models can be richer later.
- Lifecycle thresholds are synthetic research defaults and require calibration on larger real teacher/corpus workloads.
- The benchmark validates persistent rule governance, not broad natural-language competence.
