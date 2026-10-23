# Equinox Forge v0.24

Milestone: **Slot Prototype / Latent Class Induction**.

v0.24 promotes a coherent frame-slot filler cohort into a reusable provisional concept node. Candidate membership is compiled offline from independent semantic evidence as `IS_A` relations and can be consumed by runtime frame admission through a direct indexed lookup. Identical filler cohorts are deduplicated and reused across multiple frames.

The release intentionally does not infer classes for completely unknown surfaces and does not assign human semantic labels to latent prototypes without separate evidence.

Primary regression: `tests/test_slot_prototype_induction.py`.
