# Equinox Forge v0.33 — Evidence-Backed Novel Entity Admission

## Purpose

v0.32 fixed unsupported lexical guessing by separating recall from prediction authority. The cost was deliberate: generic L4 transfer (`Wolf ist ein`) dropped to zero because an unseen `Wolf` and an unseen `Stein` were structurally indistinguishable.

v0.33 restores that transfer only when the new entity is supported by **independent compiled semantic evidence**.

## New distinction

v0.33 separates two meanings of "known":

```text
known somewhere in BASE
!=
seen in predictive training
```

An entity may therefore be present through an embedding/semantic source without ever occurring in a predictive teacher prompt.

## Admission rule

For recalled predictive contexts, a token that is new to the predictive-context vocabulary may be substituted only when:

1. the live and recalled contexts are otherwise aligned;
2. there is exactly one novel aligned substitution in this first implementation;
3. both surfaces exist in the compiled BASE;
4. an allowed compiled semantic relation supports the new entity against the trained entity;
5. `score * confidence >= novel_entity_min_semantic_quality` (default `0.65`).

Allowed relation channels default to:

- `semantic_similarity`
- `contextual_similarity`

The mechanism is controlled by `enable_evidence_backed_novel_entity_admission`.

## Benchmark result

The benchmark uses the unchanged v0.31 predictive training ladder plus an auxiliary **non-predictive** semantic document.

- `Wolf` has strong independent semantic similarity to trained fillers `Hund` / `Katze`.
- `Stein` is present in the BASE but only weakly related.
- Neither `Wolf` nor `Stein` occurs in predictive training prompts.
- Evaluation teacher predictions are never inserted into the BASE.

Result:

| Track | v0.32 | v0.33 |
|---|---:|---:|
| L0 exact replay | 1.0 | 1.0 |
| L1 suffix transfer | 1.0 | 1.0 |
| L2 recombination | 1.0 | 1.0 |
| L3 novel order | 1.0 | 1.0 |
| L4 novel entity | 0.0 | **1.0** |
| L5 held-out relation family | 0.0 | 0.0 |
| `Stein ist ein` | abstain | **abstain** |
| surface-only `Wolf ist ein` | n/a | **abstain** |

The surface-only control is important: compiling the token `Wolf` without semantic evidence does **not** restore the old v0.31 guess.

## Architectural consequence

```text
semantic knowledge about entity
        +
known predictive relational context
        ↓
evidence-backed substitution
        ↓
prediction authority
```

This is the first generic runtime bridge between independent semantic evidence and support-gated predictive transfer. It does not require the entity to have been trained in the predictive task itself.

## Generalization ladder support documents

`compile_and_benchmark_generalization_ladder(...)` now accepts optional `support_documents`. These documents are compiled into the training BASE but are kept distinct from evaluation teacher references. This allows controlled experiments where semantic knowledge comes from a source independent of predictive behavior.

## CLI

```bash
eqforge-novel-entity --out out/v0_33_novel_entity
```

Historical benchmarks remain available:

```bash
eqforge-support-gate --out out/v0_32_support_gate
eqforge-gauntlet --out out/v0_31_baseline_gauntlet
```

## Non-claims

- v0.33 handles one conservative aligned substitution; it is not a general analogy engine.
- The semantic threshold is a research parameter, not a universal constant.
- Runtime v0.33 checks compiled semantic evidence but does not yet enforce source-lineage independence itself; the benchmark guarantees source separation at compile time.
- L5 remains unsolved: v0.33 does not infer entirely unseen relation families.
- The benchmark is synthetic and demonstrates mechanism validity, not broad language competence.
