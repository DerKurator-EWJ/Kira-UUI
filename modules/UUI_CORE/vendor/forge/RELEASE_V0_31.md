# Equinox Forge v0.31 — Baseline Gauntlet

## Purpose

v0.31 deliberately adds **no new cognitive mechanism**. It attacks the frozen v0.30 baseline with a deterministic multi-track benchmark and records both strengths and failures without collapsing them into a single score.

## New benchmark tracks

1. **Generalization ladder L0–L5** — exact replay through held-out relation-family transfer.
2. **Unsupported-query rejection** — fully unknown prompts and adversarial recombinations of known lexical material.
3. **Typed-frame counterexample** — demonstrates the difference between open-slot overgeneralization and semantic slot admission.
4. **Ambiguity separation** — contextual clustering of the two scripted senses of `Bank`, with gold labels used only for evaluation purity.
5. **Exception consensus lifecycle** — candidate → stable → contested → quarantined → candidate → stable, with runtime lifecycle gating.
6. **Runtime routing** — repeated-structure macro routing and indexed-work reduction.

Run it with:

```bash
eqforge-gauntlet --out out/v0_31_baseline_gauntlet
```

or:

```bash
python -m equinox_forge.baseline_gauntlet --out out/v0_31_baseline_gauntlet
```

## Frozen baseline result

The deterministic v0.31 fixture reports:

- L0 exact replay: `1.0`
- L1 known-suffix transfer: `1.0`
- L2 known-parts recombination: `1.0`
- L3 novel order with known lexicon: `1.0`
- L4 novel entity in known frame: `1.0`
- L5 held-out relation family: `0.0`
- overall unsupported-query rejection: `0.25`
- unsupported known/mixed lexical recombination rejection: `0.0`
- typed-frame applicability precision / recall: `1.0 / 1.0`
- ambiguity cluster purity: `1.0`
- exception lifecycle contract: pass
- held-out stable exception transfer: `Eule atmet → Sauerstoff`
- macro-route indexed work reduction in the fixed routing fixture: `0.746875`

See `benchmarks/v0_31/baseline_reference.json`.

## Main failure exposed

The strongest failure is not total unknown input. The runtime abstains correctly when it has no lexical foothold at all. The problem appears when a prompt contains known tokens but no compiled predictive context: lexical recall can combine unrelated evidence and still emit predictions.

Examples in the frozen gauntlet include:

```text
Hund Baum
Haus kann
Maus kann
Stein ist ein
```

These prompts are intentionally outside compiled predictive support. Under the v0.30 runtime semantics they nevertheless produce predictions. This makes unsupported-combination rejection a concrete target for the next cognitive change.

## Scientific rule

v0.31 prohibits a composite overall score for the gauntlet. A release may improve one track and regress another; both must remain visible.

The case set and deterministic non-timing baseline are frozen under `benchmarks/v0_31/`. Later releases may append new cases but must not rewrite these rows or metric meanings.

## Non-claims

- Success on L0–L4 in this scripted fixture is not evidence of general language understanding.
- The benchmark is deterministic and synthetic; it is a diagnostic instrument, not a natural-language benchmark.
- v0.31 does not solve the false-positive problem it exposes.
- Timing values are machine-dependent and are not frozen as strict cross-machine assertions.
