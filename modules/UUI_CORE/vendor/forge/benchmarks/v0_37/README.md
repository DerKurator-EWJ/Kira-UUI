# Equinox Forge v0.37 — Source Lineage / Correlated Evidence

This benchmark attacks pseudo-consensus caused by correlated source identities.

Three different source IDs that all descend from the same registered ancestor must count as one independent evidence line. Two disjoint ancestry roots may count as two. A multi-parent derivative conservatively bridges all of its roots into one correlated component.

The lifecycle benchmark additionally verifies that renamed descendants of already-seen roots cannot manufacture a new support or conflict epoch.

Run:

```bash
eqforge-source-lineage --out out/v0_37_source_lineage
```
