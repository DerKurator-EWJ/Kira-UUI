# Equinox Forge v0.38 — Global Lineage-Aware Consensus

v0.38 generalizes the source-lineage semantics introduced in v0.37 beyond composition induction and composition lifecycle. Correlated source descendants are now treated as one epistemic evidence line in three general subsystems:

1. compiled relation aggregation,
2. provisional-sense consensus,
3. exception consensus.

## Core rule

A distinct `source_id` is not automatically an independent source of evidence.

If several sources share transitive ancestry, they form one lineage component for consensus and source-count thresholds. Unregistered sources remain backward-compatible and act as independent roots of themselves.

## Compiled relation aggregation

Before v0.38, three correlated aliases with confidence 0.60 could produce:

```text
source_count = 3
confidence = 1 - (1 - .60)^3 = .936
```

With registered common ancestry, v0.38 produces:

```text
raw evidence rows = 3
effective source_count = 1
confidence = .60
```

Adding a genuinely independent lineage produces:

```text
effective source_count = 2
confidence = 1 - (1 - .60)^2 = .84
```

Relation score aggregation is also alias-count invariant inside one lineage: correlated aliases cannot move a score merely by multiplying identities. Source-local scores are first aggregated, then a conservative component estimate is derived from the component score range.

## Sense consensus

Per-source votes are collapsed by lineage component before stability thresholds are evaluated.

- unanimous support inside one family counts once;
- unanimous conflict inside one family counts once;
- support and conflict inside the same family become lineage-level `abstain`;
- component weight is capped by the strongest participating source weight instead of summed across aliases.

Therefore three siblings from one foundation model cannot satisfy `stable_min_sources=2`.

## Exception consensus

The same lineage collapse now applies to exception override votes. Repeated builds can still provide temporal confirmation, but the within-run independence requirement is based on lineage components rather than raw source ids.

## Backward compatibility

For databases where none of the participating sources has registered ancestry, relation aggregation keeps the pre-v0.38 implementation and consensus votes remain one-source/one-vote. This protects historical fixtures and existing databases while allowing lineage-aware behavior when lineage metadata is supplied.

## Frozen benchmark

Run:

```bash
eqforge-global-lineage --out out/v0_38_global_lineage_consensus
```

The deterministic benchmark verifies:

- three correlated relation sources collapse to one effective source;
- correlated confidence is not noisy-OR multiplied;
- correlated aliases cannot satisfy `min_sources=2`;
- an independent root restores two-source eligibility;
- correlated sense support remains `candidate`;
- independent sense support can become `stable`;
- correlated exception support remains `candidate`;
- independent exception support can become `stable`;
- internal support/conflict inside one lineage is treated conservatively.

v0.38 does not claim that lineage metadata can be inferred automatically. It assumes the ancestry graph is known or registered. Automatic provenance discovery remains a separate research problem.
