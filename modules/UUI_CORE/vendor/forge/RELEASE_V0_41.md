# Equinox Forge v0.41 — Active Probe Selection / Information-Gain Proxy

v0.41 moves the bottleneck from runtime evidence ordering to **which teacher questions Forge chooses to ask**.
No RuntimeCognition mechanism changes in this release; runtime cognition remains v0.40.

## Frozen experiment

The existing locally trained neural teacher is reused on the controlled 32-entity animal domain.
The candidate pool contains 128 core facts (32 entities × 4 known relation templates). Answers are hidden from the selector.
Every strategy receives exactly **64 unique teacher queries**.

Active selection uses a two-stage acquisition policy:

- 32 deterministic seed queries: one rotating relation per entity, covering every entity and relation family;
- compile the seed BASE;
- score the remaining 96 prompts using only Equinox runtime abstention / prediction uncertainty;
- choose 32 additional prompts using uncertainty plus relation/entity coverage tie-breaks.

A caching teacher wrapper guarantees that recompiling the 32 seed prompts for the final pack creates cache hits rather than additional neural-teacher queries.

## Frozen results

| Strategy | Full core accuracy | Accuracy on 64 unqueried facts |
|---|---:|---:|
| Balanced fixed (v0.39-style) | 57.03125% | 14.0625% |
| Seeded random, 5-run mean | 57.34375% | 14.6875% |
| **Active v0.41** | **61.71875%** | **23.4375%** |

Active improvement:

- vs fixed: **+4.6875 percentage points** full-domain accuracy;
- vs fixed on unqueried facts: **+9.375 points**;
- vs 5-run random mean: **+4.375 points** full-domain accuracy;
- vs random mean on unqueried facts: **+8.75 points**.

The active run still abstains on 100% of the completely unprobed `food` relation family and on 100% of outside-domain object controls.

## What the selector actually discovered

After the broad seed, Equinox could already produce predictions for many `class` and `habitat` frontier prompts, while the remaining `ability` and `trait` prompts were predominantly abstentions. The active second batch therefore spent its 32 additional queries on those unresolved regions instead of querying prompts the current BASE could already often answer.

This is a pool-based information-gain **proxy**, not autonomous question generation and not a proof of optimal active learning. The prompt universe and relation templates are known in advance; only the answers are hidden.

Run:

```bash
eqforge-active-probe --out out/v0_41_active_probe_selection --random-trials 5
```

See `benchmarks/v0_41/` for the frozen contract and reference.
