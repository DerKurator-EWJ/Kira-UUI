# Equinox Forge v0.39 — Real Teacher Domain Trial

v0.39 is a measurement release, not a new cognition release. It freezes the first larger end-to-end neural-teacher trial and deliberately leaves the v0.38 runtime mechanisms unchanged.

## Trial boundary

A real PyTorch neural next-token teacher is trained deterministically on a controlled 32-entity animal domain plus eight outside-domain object controls. The teacher contains five animal relation families:

- class
- habitat
- ability
- trait
- food

Forge is allowed to probe only two of the four core relation facts per animal. The food relation family is never probed into Equinox. The neural teacher is trained on base and article-prefixed prompt variants, while Forge compiles only the base variants.

The evaluation then removes the teacher from the Equinox answer path and measures:

1. **replay** — compiled prompts;
2. **paraphrase** — uncompiled surface variants of compiled facts;
3. **heldout_fact** — teacher-known animal facts never directly probed into Equinox;
4. **opaque_relation** — a whole teacher-known relation family never probed into Equinox;
5. **outside_domain_control** — teacher-known object facts outside the compiled animal domain.

## Frozen v0.39 observation

The teacher reaches 100% training/evaluation accuracy on the controlled domain. Direct compiler/pack replay preserves the teacher Top-1 at 100%, proving that accepted probe evidence survives IR -> DB -> EQBASE serialization.

With the normal associative runtime enabled, the 32-entity reference run exposes a new scale-dependent error: full-runtime replay/paraphrase retention is 92.1875%. Minority exact-context targets such as `löwe lebt im -> savanne` can be outranked by accumulated recall support for frequent alternatives such as `land`.

Teacher-known but unprobed core facts reach only 14.0625% Top-1 retention, with 50% runtime abstention. The entirely unprobed food relation family is rejected at 100% abstention. Outside-domain object facts are also rejected at 100% abstention.

A concrete failure trace is `löwe lebt im`. The exact compiled `savanne` candidate receives about 0.880 direct-context support and finishes near 0.919 after combination, but accumulated recalled support from common `... lebt im -> land` contexts reaches about 0.939 and wins Top-1. This isolates evidence ordering/aggregation rather than serialization loss.

The 32-entity reference uses 64 teacher compilation probes, i.e. 40% of the 160 animal fact slots represented in the neural teacher. The resulting EQBASE is 12,738 bytes versus a 93,151-byte serialized teacher state (ratio ~0.1367). This is a storage diagnostic only: the BASE contains only extracted probe evidence and must not be interpreted as equivalent teacher compression.

This isolates two different limits:

- **runtime evidence ordering:** broad associative recall can overpower an exact compiled context;
- **extraction coverage:** teacher knowledge that was never probed and has no independent semantic representation is generally unavailable to Equinox.

These are deliberately left unfixed in v0.39 so that subsequent releases can be measured against this frozen reference.

## Scope exclusions

v0.39 does not claim:

- distillation of a pretrained general-world LLM;
- broad natural-language generalization;
- mobile latency or energy efficiency;
- recovery of information that Forge never extracted.

## Run

```bash
eqforge-domain-trial --out out/v0_39_real_teacher_domain_trial
```

See `benchmarks/v0_39/domain_trial_contract.json` and `benchmarks/v0_39/reference.json`.
