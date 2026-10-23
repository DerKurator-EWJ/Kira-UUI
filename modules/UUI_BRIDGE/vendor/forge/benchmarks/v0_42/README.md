# v0.42 Iterative Active Learning / Frozen Shadow Holdout

v0.42 audits the v0.41 acquisition claim under a stricter evaluation design.

The 128 core facts are split into:

- 96 selectable prompts;
- 32 frozen shadow-holdout facts (one per entity), which no strategy may ever query.

All strategies start from the same 32-query seed. Iterative active learning then adds four batches of eight queries, recompiling and rescoring the frontier after every batch. A v0.41-style one-shot active control ranks all 32 additional prompts from the seed snapshot only. Fixed and seeded-random controls use the same final 64-query budget.

The shadow holdout fixes an important v0.41 evaluation confound: strategy-specific "unqueried facts" are different sets, so higher accuracy on that remainder can reflect querying the hard cases rather than transferring knowledge to new cases.

Frozen full-domain result:

- iterative and one-shot active end at the same 61.71875% full-core accuracy;
- fixed ends at 57.03125%;
- seeded random (5-run mean) ends at 57.34375%;
- all strategies remain at 15.625% on the identical frozen 32-fact shadow holdout at the final budget;
- iterative active chooses the same final query schedule as one-shot active.

Therefore v0.42 finds a real query-allocation advantage but **no measured fixed-holdout transfer improvement** from the current uncertainty proxy, and no benefit from iterative rescoring under this domain.
