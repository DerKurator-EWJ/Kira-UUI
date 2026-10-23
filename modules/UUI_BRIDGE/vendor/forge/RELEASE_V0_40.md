# Equinox Forge v0.40 — Exact Context Authority / Evidence Ordering

v0.40 is the first cognition release driven directly by the larger neural-teacher domain trial introduced in v0.39. It fixes the observed case where many weaker recalled contexts could beat a strong direct prediction from the live exact context merely through accumulation.

## The v0.39 failure

The frozen v0.39 pack contained the correct direct relation for:

```text
löwe lebt im -> savanne
```

but broad recall activated many other habitat contexts predicting `land`. Numerically:

```text
Savanne direct support   ~= 0.880
Savanne combined score  ~= 0.919
Land recalled score     ~= 0.939
```

Historical aggregate sorting therefore returned `land` even though the live context itself explicitly supported `savanne`.

## v0.40 rule

v0.40 introduces an evidence-ordering tier for exact and longest-known suffix contexts.

A candidate receives context authority only when:

1. the live query resolves to an exact or permitted suffix predictive context;
2. the candidate has direct `PREDICTION` support from that context; and
3. its direct score meets `exact_context_authority_min_direct_score` (default `0.50`).

When at least one candidate qualifies, qualifying direct candidates rank ahead of recall-only alternatives. Candidates inside the authority tier are ordered by direct score first and aggregate score second.

This is deliberately **not** an unconditional `exact always wins` rule. If no direct candidate clears the quality floor, the historical aggregate-score ordering remains active. Frame-schema and relation-composition outputs do not enter this authority tier.

## Frozen v0.40 neural-domain result

The exact same v0.39 compiled `.eqbase` is evaluated twice. The v0.40 arm performs zero additional teacher queries and no recompilation.

```text
Category                         v0.39      v0.40
-------------------------------------------------
Replay Top-1                    92.1875%   100%
Paraphrase Top-1                92.1875%   100%
Held-out fact Top-1             14.0625%   14.0625%
Held-out abstention             50%        50%
Opaque relation abstention      100%       100%
Outside-domain abstention       100%       100%
```

For `löwe lebt im`, `Land` still retains the numerically larger aggregate score. v0.40 returns `Savanne` because its strong direct live-context evidence belongs to the stronger evidence class. No candidate score is secretly inflated.

## Diagnostics

Runtime reports now expose:

```text
exact_context_authority_active
exact_context_authority_candidates
context_resolution_kind:exact
context_resolution_kind:suffix
context_resolution_kind:frame
```

The runtime cognition format version advances to `0.40`.

## Anti-cheating controls

The v0.40 regression fixture verifies that:

- with authority disabled, accumulated recalled `Land` beats direct `Savanne`;
- with authority enabled at the normal floor, `Savanne` wins even while `Land.score > Savanne.score`;
- if the direct-score floor is raised above the direct evidence, authority does not activate and historical aggregate ordering returns;
- no extra teacher evidence or recompilation is used in the neural-domain treatment arm.

## Scope

v0.40 fixes runtime evidence ordering for already-extracted direct knowledge. It does **not** improve the separate extraction-coverage limit measured by v0.39. Teacher-known facts that Forge never extracted remain unavailable unless another compiled mechanism supports them.

## Run

```bash
eqforge-exact-authority --out out/v0_40_exact_context_authority
```

See `benchmarks/v0_40/` for the frozen contract and reference result.
