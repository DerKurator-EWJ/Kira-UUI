# UUI Bridge 0.1 — minimal integration of KIRA 1.3 and Equinox Forge 0.56

This is a **working Python adapter**, NOT yet the UUI language, bytecode or runtime. The original modules are retained under `vendor/` (without large demos/binaries). No direct semantic merge of numeric values and opaque Forge IDs is attempted.

## Implemented
- Namespaced identities prevent Forge subject/relation IDs from colliding with Kira numeric IDs.
- Exact rational observations via original Kira NumericCore.
- Forge flat-kernel query through original `UnifiedNumericExecutionKernel`, provided a compatible compiled pack.
- Optional import of resolved Forge query results into Kira evidence as **external_claim**, not as sensor observations.
- Kira observation -> evidence review -> inquiry synchronization.

## Run
`python -m unittest -v test_bridge.py` from this directory.

## Deliberately not implemented
- Forge pack generation / conversion, general program execution, UUI syntax, compiler, common transaction coordinator, automated resolution of conflicting claims, automatic ID alignment or cross-domain inference.
- Forge query is not tested against a real compiled pack in this package; only Kira-side bridge tests are included.

## Source versions
KIRA_UNIFIED_CORE_1_3(1).zip and Equinox_Forge_v0_56_DirectTeacherFlatCompiler(4).zip. Kira 1.4 was not provided in this turn; do not treat this as a 1.4 integration.
