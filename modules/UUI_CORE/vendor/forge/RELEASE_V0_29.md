# Equinox Forge v0.29 — Exception Consensus, Stability & Reversal

v0.29 separates **discovering an exception hypothesis** from **authorizing it as a runtime rule**.
A v0.28 automatic override may now move through a persistent lifecycle:

```text
candidate → stable → contested → quarantined
                       ↘
                        recovery → candidate → stable
```

## Cross-source votes

Each independent `source_id@source_version` is evaluated only on the concrete support contexts that
originally induced the override. For every source Forge compares its confidence-weighted prediction
support for:

- the exception target, and
- the inherited/general target.

A source votes `support`, `conflict`, or `abstain`. `source_policies.trust_weight` affects vote weight,
but never creates extra independent sources.

Default stability contract:

- at least 2 independent support sources,
- support weight ≥ 1.5,
- consensus ≥ 0.75,
- 2 consecutive clean confirmation runs.

One conflicting source makes a hypothesis `contested`; two independent weighted conflicts can
`quarantine` it. Recovery resets the clean-support streak, so a formerly quarantined exception does
not become stable again after a single clean rebuild.

## Runtime lifecycle gate

`RuntimeCognitionPolicy.enable_exception_lifecycle_gating=True` is on by default. Once an exception
frame carries lifecycle metadata, the runtime executes it only when its status is in
`allowed_exception_lifecycle_statuses` (default: `("stable",)`). Candidate, contested, and quarantined
frames remain in the BASE for auditability but do not affect predictions.

v0.28 packs have no lifecycle metadata and remain executable for backward compatibility.

## CLI

```bash
eqforge track-exception-consensus master.sqlite overrides.eqir.json \
  --output overrides.lifecycle.eqir.json \
  --report exception-consensus.json

eqforge exception-states master.sqlite
```

## Deterministic fixture

Two independent exception teachers support:

```text
Adler atmet → Sauerstoff
Falke atmet → Sauerstoff
```

while the inherited broad rule remains `→ Luft`.

```text
run 1: support A+B               → candidate → runtime uses Luft
run 2: support A+B               → stable    → runtime uses Sauerstoff
run 3: + conflict C              → contested → runtime falls back to Luft
run 4: + conflict D              → quarantined → runtime uses Luft
run 5: conflicts disabled        → candidate → still Luft
run 6: clean support repeated    → stable    → Sauerstoff again
```

The held-out `Eule atmet` context is still not used to induce the exception. It is used only to verify
runtime transfer and fallback behavior.

## Non-claims

- Lifecycle status is evidence management, not a truth label.
- The first implementation evaluates text-frame exception predictions only.
- Consensus quality depends on source independence; correlated teachers can still create misleading agreement.
- Temporal decay by wall-clock age is not modeled yet; stability is rebuild/event based.
