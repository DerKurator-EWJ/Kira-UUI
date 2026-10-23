# v0.45 Structural Closure

This benchmark freezes the first ID-rooted structural view on top of the v0.44 canonical numeric graph. It verifies that child concept IDs can address local and ancestor-rooted information while the canonical fact graph remains non-duplicated.

Run:

```bash
eqforge-structural-closure --out out/v0_45_structural_closure
```

The closure is a routing/addressability layer. An ancestor relation appearing in a closure retains its original subject, depth, and inheritance path; v0.45 does not yet declare every such relation to be a valid semantic inheritance rule.
