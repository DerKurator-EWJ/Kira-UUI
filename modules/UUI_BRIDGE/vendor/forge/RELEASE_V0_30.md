# Equinox Forge v0.30 — Engineering Freeze & Benchmark Foundation

## Purpose

v0.30 freezes the v0.29 cognitive behaviour before large-scale teacher compilation work. No recall, prediction, hierarchy, specificity, exception or consensus semantics are intentionally changed.

## Engineering changes

- package version raised to `0.30.0`;
- explicit PEP 517 setuptools build configuration;
- package discovery restricted to `equinox_forge*`;
- regression runner now discovers all `test_*.py` files automatically;
- per-test duration, timeout, aggregate pass/fail summary and optional JSON output;
- frozen benchmark contract added under `benchmarks/v0_30/`;
- release consistency test added.

## Scientific rule introduced by v0.30

Future releases must be compared against the same metric definitions. New benchmark cases may be appended, but existing v0.30 metric semantics must not be retroactively changed.

The core metric groups are:

- behavioural retention;
- held-out transfer/generalization;
- false-positive / unknown rejection;
- specificity and exception correctness;
- consensus/conflict correctness;
- runtime work per query;
- binary pack size;
- compile cost;
- retained function per byte.

## Non-claims

- v0.30 does not improve Equinox intelligence by itself.
- v0.30 does not establish large-scale teacher compression.
- v0.30 does not claim that the current heuristic weights are optimal.
- v0.30 exists to make subsequent claims harder to fake accidentally.
