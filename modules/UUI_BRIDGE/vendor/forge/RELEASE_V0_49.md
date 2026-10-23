# Equinox Forge v0.49 — Macro-Aware Numeric Routing

v0.49 connects the recursive macro layer from v0.46 to the numeric runtime introduced in v0.48.

## Core rule

The runtime still checks the most specific concept first. If no qualifying local fact exists, it may
jump into a compile-time validated macro ID whose effective relation reproduces the frozen v0.48
specific-to-general answer. If no validated macro shortcut exists, resolution falls back to v0.48.

```text
Concept ID
  ↓ local fact?
validated Macro ID
  ↓ effective macro fact?
answer

otherwise:
  ↓
v0.48 parent traversal
```

## Safety boundary

Macros do not receive independent semantic authority in v0.49. Every exported `(concept,predicate)`
shortcut is checked during pack compilation against the v0.48 resolver. A shortcut is emitted only
when both paths resolve to the same object and the original answer was inherited rather than local.

## Frozen reference

The reference graph builds two level-1 macros and one level-2 macro. Across 12 queries:

- answer parity with v0.48: 12/12
- origin/depth/path/score/confidence/source/evidence metadata parity: 12/12
- v0.48 hierarchy nodes touched: 38
- macro-aware nodes touched: 23
- reduction: 39.47%
- macro-routed queries: 11/12
- local leaf fact remains local: yes
- canonical semantic triples rewritten: no

Run:

```bash
eqforge-macro-numeric-runtime --out out/v0_49_macro_numeric_runtime
```

See `benchmarks/v0_49/` for the frozen contract and reference.
