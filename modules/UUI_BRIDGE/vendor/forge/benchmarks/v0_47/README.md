# Equinox Forge v0.47 — Top-Down Teacher Structure Extraction

This benchmark verifies that Forge can extract a generic parent hierarchy from teacher evidence without embedding the concrete ontology in the extractor, materialize the discovered concepts root-to-leaf into the canonical numeric ID space, and pass those IDs directly into v0.45 Structural Closure and v0.46 Recursive Macro Collapse.

The controlled concrete knowledge is isolated in `top_down_teacher_fixture.json`. The extractor itself only knows generic `IS_A`, `PROPERTY`, and `ACTION` ordering operations.

Run (PyTorch required for the controlled neural-teacher fixture):

```bash
eqforge-top-down-teacher --out out/v0_47_top_down_teacher_structure
```

The generic `extract_top_down_teacher_structure()` function itself is backend-agnostic and accepts any `CausalTeacherBackend`, including the local Hugging Face backend when that optional runtime and model files are available.
