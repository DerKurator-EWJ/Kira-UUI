# Equinox Forge v0.50 — Teacher → Numeric Equinox Compiler Trial

v0.50 freezes the first end-to-end validation of the new v0.44–v0.49 architecture:

```text
computed neural teacher
  → top-down structure extraction
  → canonical numeric ID space
  → structural closure
  → recursive macro collapse
  → DB-free macro-aware numeric runtime pack
  → teacher + Forge DB offline
  → runtime evaluation
```

The default offline fixture is intentionally a locally trained neural teacher rather than a scripted backend. It is **not** presented as an external pretrained Hugging Face result because the release environment has no cached pretrained HF model and no `transformers` runtime. The CLI accepts an optional local HF model directory for static HF manifest/tensor inspection through the existing adapter boundary.

## Frozen v0.50 claims

- Every structural fact extracted from the controlled teacher survives in the numeric pack.
- Runtime evaluation happens after the Forge DB is closed and the teacher state is moved out of its original path.
- Specific→general scalar queries retain the expected nearest facts.
- Recursive macros survive the full compiler path. In this fixture, v0.49 correctly emits **no** macro shortcut because the coarse `PROPERTY`/`ACTION` predicates contain multiple values and would be ambiguous.
- The logical runtime graph is numeric; surfaces remain I/O bindings.

## Explicitly exposed limitation

`PROPERTY` and `ACTION` are still generic scalar predicates in the v0.48 numeric resolver. The pack retains all extracted relations, but one scalar query returns the nearest value rather than enumerating the complete inherited multi-valued closure. This is an API/representation issue for the next release, not hidden loss in the compiled graph. It also explains why the macro-aware runtime cannot safely accelerate the v0.50 teacher fixture yet: the compiled macro has several distinct objects under the same generic predicate.

Run:

```bash
eqforge-teacher-numeric-compiler --out out/v0_50_teacher_numeric_compiler
```
