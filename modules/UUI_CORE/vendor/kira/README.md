# KIRA NUMERIC CORE 0.2 — indexed numeric learning experiment

This builds directly on 0.1 and retains the prior Efficient Core / RRM components for compatibility.

## New
- `BlockIndex`: sorted bounded blocks instead of one global Python list. Insertions shift elements inside one block; block split when needed. No fundamental topology, edges, nodes or spatial semantics.
- `NumericPatternLearner`: learns **exact rational differences** between consecutive observations, conditioned on the preceding difference. Learned patterns persist in SQLite and can be reused on a second named sequence without copying training observations. All arithmetic uses `Fraction`.
- Incomplete evidence returns `None`; multiple observed outcomes remain distinct and support counts are retained.

## Limits
- The new learner is **not yet wired into the Efficient Core dispatcher**; it is an executable numeric candidate alongside the existing compatible numeric store. No claim of a fully migrated Kira.
- Block index is in RAM and its block-maximum list can itself require O(number_of_blocks) shifts; not a guaranteed O(log n) insertion. A disk-backed B-tree with an order-preserving rational key or balanced tree is the next production step.
- This learner models 1D numeric change only; it does not discover arbitrary macros, objects, causality, semantic equivalence or universal one-shot learning. Numeric proximity is not automatically meaning proximity.
- Source legacy graph processing remains unchanged; no silent conversion of old graph evidence.

## Run
`PYTHONPATH=.:RRM_UNIFIED_0_1 python -m unittest -v test_numeric_core test_integration test_numeric_patterns`
