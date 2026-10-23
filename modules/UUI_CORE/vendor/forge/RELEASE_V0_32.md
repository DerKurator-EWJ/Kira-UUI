# Equinox Forge v0.32 — Support-Gated Prediction

## Purpose

v0.32 addresses the main failure exposed by the frozen v0.31 Baseline Gauntlet: **associative recall was being treated as prediction authority**.

The old runtime could correctly abstain when a prompt had no known lexical foothold, yet it could emit unrelated predictions when familiar tokens appeared in an unsupported combination.

Examples from the frozen gauntlet:

```text
Hund Baum  -> Tier / bellen / heute
Haus kann  -> bellen / miauen / groß
Maus kann  -> bellen / miauen / heute
Stein ist ein -> Tier / groß / Haustier
```

v0.32 keeps recall broad but gates whether a *recalled* predictive context may execute its `PREDICTION` edge.

## New runtime rule

A recalled predictive context is admitted only when:

1. the live context and recalled predictive context share at least two lexical tokens;
2. their overlap fraction is at least `0.66` by default;
3. a live context containing an unseen token is **not** generalized over through lexical recall alone, unless an explicit compiled context already resolved it.

These defaults are exposed in `RuntimeCognitionPolicy`:

```text
enable_support_gated_recalled_prediction = True
recalled_support_min_shared_tokens = 2
recalled_support_min_overlap_fraction = 0.66
recalled_support_allow_unknown_tokens_without_context = False
```

The historical behavior remains available for ablation by setting:

```text
enable_support_gated_recalled_prediction = False
```

## Frozen-gauntlet result

The exact v0.31 case set and dataset hash are reused unchanged.

| Track | v0.31 | v0.32 |
|---|---:|---:|
| L0 exact replay | 1.0 | 1.0 |
| L1 known suffix | 1.0 | 1.0 |
| L2 known-parts recombination | 1.0 | 1.0 |
| L3 novel order / known lexicon | 1.0 | 1.0 |
| L4 lexical-only novel entity | 1.0 | 0.0 |
| L5 held-out relation family | 0.0 | 0.0 |
| unsupported-query rejection | 0.25 | 1.0 |
| known/mixed unsupported rejection | 0.0 | 1.0 |
| typed-frame precision | 1.0 | 1.0 |
| typed-frame recall | 1.0 | 1.0 |

No composite score is used.

## Why L4 intentionally drops

The generic ladder contains both:

```text
Wolf ist ein
Stein ist ein
```

Neither `Wolf` nor `Stein` exists in that compiled ladder BASE. Both prompts therefore expose the same runtime support pattern:

```text
[unknown entity] + known tokens "ist ein"
```

The v0.31 runtime guessed `Tier` for both. The benchmark happened to count the Wolf guess as a success and the Stein guess as a false positive, but the runtime had no independent evidence that could justify that distinction.

v0.32 treats this as an epistemic error in the old benchmark interpretation rather than something to preserve at all costs. Both lexical-only cases abstain.

Novel-entity transfer is instead routed through mechanisms that possess independent admission evidence: typed frames, latent slot prototypes and hierarchy paths. The typed-frame counterexample remains `1.0` precision and `1.0` recall: `Wolf` and `Fuchs` are admitted through semantic support, while `Stein` is rejected.

## Architectural consequence

v0.32 establishes a stronger separation:

```text
Recall = "this may be relevant"
Prediction support = "this context is justified to emit an answer"
```

A memory being reachable is no longer itself evidence that its output relation applies to the live context.

## Diagnostics

Prediction work reports now include:

- `support_gate_checks`
- `support_gate_rejections`
- `support_gate_admissions`
- per-reason counters such as `support_gate_reason:insufficient_shared_tokens`
- `support_gate_reason:unknown_live_token_without_compiled_context`

Runtime cognition format advances to `equinox-runtime-cognition-v2`, version `0.32`.

## CLI

```bash
eqforge-support-gate --out out/v0_32_support_gate
```

For the immutable historical v0.31 semantics:

```bash
eqforge-gauntlet --out out/v0_31_baseline_gauntlet
```

## Non-claims

- v0.32 does not solve held-out relation-family acquisition.
- Two-token overlap is a conservative first support criterion, not a learned universal law.
- The deterministic gauntlet remains synthetic.
- Semantic admission for genuinely unseen entities still requires independent evidence; lexical form alone is intentionally insufficient.
