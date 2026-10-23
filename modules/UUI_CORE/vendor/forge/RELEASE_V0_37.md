# Equinox Forge v0.37 — Source Lineage / Correlated Evidence

## Purpose

v0.36 introduced source-novel lifecycle epochs, but "novel" still meant a different `source_id@source_version`. That allows pseudo-consensus when several teachers, fine-tunes, snapshots, or distilled models descend from the same underlying source family.

v0.37 adds explicit source-lineage ancestry and replaces raw source identity with conservative **independent lineage components** for automatic relation-composition induction and lifecycle novelty.

## Core rule

```text
three source IDs
    !=
three independent evidence lines
```

A registered source with no parent is a lineage root. A derived source may point to one or more parents. Root ancestry is resolved transitively.

Two observed sources are treated as correlated whenever their transitive root sets overlap. This rule is deliberately conservative.

Example:

```text
foundation-X
   /   |   \
  A    B    C
```

`A`, `B`, and `C` are three raw sources but only one independent lineage component.

## Multi-parent / distilled source control

A source derived from multiple roots bridges those root families:

```text
root-X ---- blend-XY ---- root-Y
   |                       |
 x-only                  y-only
```

The set `{blend-XY, x-only, y-only}` is counted as one correlated component, not three independent opinions. This prevents a multi-parent derivative from being counted as independent of its own ancestors.

## Induction changes

The v0.35 composition inducer still records raw supporting sources, but `source_count` now means effective independent lineage components when a `SourceLineageGraph` is provided.

The candidate also records:

- `raw_source_count`
- `supporting_sources`
- `supporting_lineage_components`

The frozen v0.37 induction controls are:

| Fixture | Raw sources | Effective lineages | Stable? |
|---|---:|---:|---|
| three siblings from one root | 3 | 1 | no |
| two disjoint ancestry roots | 3 | 2 | yes |
| multi-parent bridge | 3 | 1 | no |

## Lifecycle changes

v0.36 lifecycle promotion/quarantine now consumes lineage roots from the induced candidate.

A later support epoch counts as novel only when it contains an independent lineage component whose roots are disjoint from all roots already accepted in the current support streak.

Therefore simply renaming or fine-tuning descendants of already-seen roots cannot manufacture promotion.

The same logic applies independently to conflict streaks.

Frozen lifecycle sequence:

```text
support_1                    -> candidate
support_2_same_lineages      -> candidate
support_3_new_lineage        -> stable
conflict_1                   -> contested
conflict_2_same_lineages     -> contested
conflict_3_new_lineage       -> quarantined
```

Runtime authority remains tied to lifecycle status: only `stable` composition schemas execute by default.

## Persistence

v0.37 adds the canonical table:

- `source_lineage_edges`

Each edge stores child source, parent source, relation label, and metadata. The schema supports multiple parents and transitive ancestry.

Cycles are rejected during ancestry resolution rather than silently flattened.

## Backward compatibility

An unregistered source is treated as its own independent root. Existing v0.35/v0.36 fixtures therefore preserve their historical behavior unless lineage metadata is explicitly supplied.

## CLI

```bash
eqforge-source-lineage --out out/v0_37_source_lineage
```

## Non-claims

- v0.37 does not infer lineage automatically from model weights or model names; ancestry is explicit metadata.
- Shared ancestry is treated conservatively as correlation, but the current system does not estimate a continuous correlation coefficient.
- Lineage-aware independence is wired into relation-composition induction/lifecycle first; older generic consensus subsystems are not yet universally migrated to the same policy.
- Distinct lineage roots are evidence-independence candidates, not proof of causal or dataset independence.
- Thresholds remain research defaults requiring larger real-model calibration.
