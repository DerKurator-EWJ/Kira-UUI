# v0.33 — Evidence-Backed Novel Entity Admission

This benchmark restores L4 novel-entity transfer without returning to the lexical guessing exposed by v0.31.

The entity surface may exist in the compiled BASE through an independent semantic source while remaining absent from all predictive training prompts. Surface presence alone is a negative control. Only compiled semantic evidence above the v0.33 threshold may authorize a single aligned substitution in a recalled predictive context.

Run:

```bash
eqforge-novel-entity --out out/v0_33_novel_entity
```
