# UUI Bridge 0.2 — KIRA Unified Core 1.4 × Equinox Forge 0.56

Python integration adapter; **not** a UUI programming language, VM, or native runtime.

## Changes from 0.1
- Replaced vendored Kira Python modules with KIRA_UNIFIED_CORE_1_4(1).zip.
- Refreshed `equinox_forge` Python package from Forge 0.56 upload.
- Integrated Kira 1.4 `ExperimentPlanner` with `UUIBridge`: `ask_about`, `register_capability`, `propose_experiments`, `approve_experiment`, `record_verified_experiment`.
- Experiment proposals do **not** execute anything; completion requires a caller-provided verified result. No invented actions or motor learning.
- Kept Forge query as read-only; results enter evidence as external claims, never direct observations.
- Namespaced identifiers remain distinct from rational numeric values.

## Tests
From this folder: `python -m unittest -v test_bridge.py`.
To run Kira vendor tests: `cd vendor/kira && python -m unittest discover -p 'test_*.py'` (some may require additional paths/dependencies).

## Limits
No real compiled Forge knowledge pack shipped or exercised by bridge tests; no E2E Forge compilation, Android runtime, UUI parser/bytecode, unified transaction or concurrent-writer guarantee. This package contains Python source modules from the supplied archives, not their full demos or build assets.
