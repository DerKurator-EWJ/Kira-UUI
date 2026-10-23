# v0.43 Marginal Structural Information-Gain Feasibility Audit

v0.43 asks whether **prompt selection alone** can improve the fixed never-queryable shadow holdout under the unchanged v0.40 RuntimeCognition representation.

The audit compares:

- the 32-query seed;
- the v0.41-style 64-query active selector;
- all 96 legally selectable facts (a diagnostic saturation ceiling, not a fair acquisition budget);
- four 64-query relation-extreme schedules;
- 24 deterministic random 64-query schedules.

Frozen full-domain finding: the shadow remains at **15.625% accuracy and 50% abstention** in seed, active, and full-selectable saturation. Saturation raises the mean score of the 16 non-abstained shadow predictions from ~0.7494 to ~0.9663 and changes 16 score vectors, but changes **zero top-1 outputs**. All 24 random schedules also remain at 15.625% shadow accuracy.

Interpretation: the current acquisition policy has no observed leverage over fixed-holdout transfer because the runtime lacks an entity-specific cross-fact transfer path. More teacher facts mostly strengthen relation-level priors. The next useful mechanism should target representation/routing rather than another selector score.
