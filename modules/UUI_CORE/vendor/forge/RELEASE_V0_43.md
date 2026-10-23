# Equinox Forge v0.43 — Marginal Structural Information-Gain Feasibility Audit

v0.43 tests the hypothesis proposed after v0.42: perhaps a better acquisition score could improve transfer by preferring teacher questions with larger structural reach.

The audit deliberately changes **no runtime cognition**. Instead it asks a harder prerequisite question: under the current representation, does query selection have any observed ability to change top-1 decisions on the frozen, never-queryable shadow holdout at all?

## Frozen audit

The same 32-entity / 128-core-fact neural-teacher domain is used. One fact per entity (32 total) remains a frozen shadow holdout and is never teacher-queryable.

| Schedule | Teacher-queryable facts compiled | Full-core accuracy | Shadow accuracy | Shadow abstention |
|---|---:|---:|---:|---:|
| Seed | 32 | 36.71875% | **15.625%** | 50% |
| v0.41-style active | 64 | 61.71875% | **15.625%** | 50% |
| All selectable saturation | 96 | 78.90625% | **15.625%** | 50% |

Four relation-extreme 64-query schedules also remain at 15.625% shadow accuracy. Twenty-four deterministic random 64-query schedules produce the same 15.625% shadow accuracy in every trial.

## What changes under saturation

The absence of accuracy movement does not mean the added knowledge has no effect. For the 16 shadow prompts on which the runtime already emits a prediction, the mean top-1 score rises from approximately:

```text
0.749414 -> 0.966288
```

Sixteen shadow score vectors change, but **zero shadow top-1 outputs change**. The other 16 ability/trait shadow prompts remain abstentions.

The runtime is therefore absorbing more relation-family evidence, but it is not routing that evidence through entity-specific cross-fact structure. More class/habitat observations mostly amplify dominant priors such as `säugetier` or `land` instead of using an entity's known facts to identify a more appropriate peer/prototype.

## Interpretation

v0.43 does not claim a mathematical impossibility result. It establishes an empirical ceiling for the tested architecture and domain:

- v0.41/v0.42 uncertainty selection does not move the frozen shadow;
- relation-biased schedules do not move it;
- 24 random schedules do not move its accuracy;
- even compiling **every legally selectable non-shadow fact** does not move its top-1 decisions.

Therefore another selector heuristic is not the next justified intervention. The current bottleneck is representation/transfer.

The next experiment should introduce an explicit cross-fact entity-signature or prototype-routing mechanism and test whether facts known about one entity can route a missing relation through structurally similar entities without reverting to global majority guessing.

RuntimeCognition remains v0.40 and is byte-identical to v0.42.

Run:

```bash
eqforge-structural-gain-audit --out out/v0_43_structural_gain_audit --random-trials 24
```

See `benchmarks/v0_43/` for the frozen contract and reference.
