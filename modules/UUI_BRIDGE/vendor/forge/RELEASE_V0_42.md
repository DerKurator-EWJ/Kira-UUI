# Equinox Forge v0.42 — Iterative Active Learning / Frozen Shadow Holdout Audit

v0.42 tests the next obvious acquisition idea: recompile after small query batches, recompute the knowledge frontier, and choose the next teacher questions from the updated Equinox state.

The result is scientifically useful but intentionally not presented as a success where there is none.

## Stronger evaluation design

v0.41 compared accuracy on each strategy's own "unqueried" remainder. That remainder is strategy-dependent: an active selector may query difficult facts and leave easier facts unqueried, so a higher unqueried score does not by itself prove better transfer.

v0.42 therefore freezes 32 core facts (one per entity) as a **shadow holdout**. These prompts are never eligible for teacher queries by active, fixed, or random strategies. Transfer is evaluated on exactly the same facts for every strategy and budget.

The remaining 96 core facts form the selectable pool. Every strategy starts with the same 32-query seed and may spend up to 64 unique teacher queries total.

## Iterative policy

The active strategy now runs in four 8-query rounds:

1. compile current selected prompts;
2. evaluate the remaining selectable frontier using Equinox only;
3. rank by abstention / uncertainty plus coverage tie-breakers;
4. ask eight new teacher questions;
5. rebuild and repeat.

A v0.41-style one-shot control scores the frontier only once after the seed and fixes all 32 additional selections immediately.

Teacher answers and benchmark labels remain hidden until a prompt has actually been selected. Query caching ensures recompilation does not inflate unique teacher-query counts.

## Frozen full-domain curve

| Queries | Iterative full | One-shot full | Fixed full | Iterative shadow | Fixed shadow |
|---:|---:|---:|---:|---:|---:|
| 32 | 36.71875% | 36.71875% | 36.71875% | 15.625% | 15.625% |
| 40 | 42.96875% | 42.96875% | 39.84375% | 15.625% | 15.625% |
| 48 | 49.21875% | 49.21875% | 45.3125% | 15.625% | 15.625% |
| 56 | 55.46875% | 55.46875% | 51.5625% | 15.625% | 15.625% |
| 64 | **61.71875%** | **61.71875%** | 57.03125% | **15.625%** | **15.625%** |

Seeded random (5-run mean) at 64 queries reaches 57.34375% full-core accuracy and the same 15.625% shadow-holdout accuracy.

## Main finding

Iterative rescoring chooses the **same final query schedule** as the v0.41 one-shot policy and produces the same full-domain curve. More importantly, none of the acquisition strategies improves the identical frozen shadow holdout: it remains at 15.625%.

Therefore the current acquisition proxy has a genuine **query-allocation advantage** — it spends queries on harder selectable facts and raises total known-domain accuracy — but v0.42 finds **no evidence that it increases structural transfer to untouched facts**.

This corrects the interpretation of v0.41's 23.4375% strategy-specific unqueried score. That number remains a valid property of the selected remainder, but it is not by itself a transfer metric.

## Consequence

The next acquisition mechanism should optimize **marginal structural gain**, not merely local uncertainty. A future selector needs to estimate which query is likely to change predictions for other prompts, rather than simply asking where the current BASE is most uncertain.

RuntimeCognition is unchanged and remains v0.40.

Run:

```bash
eqforge-iterative-active --out out/v0_42_iterative_active_learning --random-trials 5
```

See `benchmarks/v0_42/` for the frozen contract and reference.
