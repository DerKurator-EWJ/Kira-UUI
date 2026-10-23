# Equinox Forge v0.47 — Top-Down Teacher Structure Extraction

v0.47 connects the new numeric knowledge substrate to a teacher-driven ordering pass. Forge no longer needs a domain ontology such as `Tier → Vogel → Eule` in its extraction code. It knows only generic structural operations and obtains the concrete hierarchy from teacher evidence.

## Core contract

The extractor receives:

- a `CausalTeacherBackend`,
- a seed surface inventory,
- a generic parent prompt template,
- optional generic property/action probe templates,
- extraction thresholds and stop boundaries.

It does **not** receive a hard-coded domain tree.

The controlled v0.47 fixture keeps the concrete animal/category knowledge in the teacher training data under `benchmarks/v0_47/top_down_teacher_fixture.json`. The extraction algorithm remains domain-agnostic.

## Two phases

### 1. Teacher boundary discovery

For each seed, Forge follows teacher-supplied parent evidence until a configured boundary is reached or the teacher abstains.

```text
leaf surface
   -- generic parent probe --> parent surface
                                  ↓
                              parent probe
                                  ↓
                                 ...
```

Cycles are rejected. Low-confidence teacher outputs are ignored.

### 2. Root-to-leaf materialization

After the parent forest is validated, Forge orders the discovered concepts from the most general roots to the most specific leaves and only then materializes the canonical concept IDs and `IS_A` evidence.

```text
root
 ↓
more specific concept
 ↓
more specific concept
 ↓
leaf
```

This is the top-down learning/materialization direction used by the new ID substrate. The initial parent observations are an extraction mechanism; they are not the stored runtime organization.

## Attributes at abstraction levels

By default v0.47 probes `PROPERTY`/`ACTION` only on established **internal hierarchy nodes** rather than blindly probing every leaf. This reflects the intended model:

```text
shared/general information → highest stable abstraction level
specific differences        → lower concept level
```

The controlled teacher therefore yields shared facts on the discovered intermediate concepts. Structural Closure then makes those facts addressable from the leaves without copying them.

## End-to-end controlled proof

The reference teacher is a small trained PyTorch neural network. Forge receives only its next-token behaviour through the same `CausalTeacherBackend` boundary used by the local Hugging Face runtime.

The frozen fixture produces:

```text
Teacher structure concepts:       9
Teacher parent edges:             8
Teacher attribute edges:          8
Teacher queries:                 16
Hierarchy roots:                  1
Max hierarchy depth:              4
Structural-closure roots:         4
Recursive macro levels:           2
Recursive macro count:            3
```

The leaf chains are reconstructed from teacher evidence, then converted into the v0.44 numeric ID graph. v0.45 Structural Closure consumes those IDs directly, and v0.46 Recursive Macro Collapse again discovers two specific level-1 macros and one higher level-2 macro.

## Provenance

The teacher manifest and a stable manifest hash are persisted in `source_models`. Parent and attribute observations are persisted separately from the canonical numeric graph:

- `top_down_teacher_runs`
- `top_down_teacher_parent_edges`
- `top_down_teacher_attribute_edges`

The final canonical graph still uses the ordinary Forge evidence/provenance pipeline.

## Determinism and safety controls

v0.47 freezes controls for:

- seed-order invariance,
- root-first deterministic materialization,
- hierarchy-cycle rejection,
- numeric canonical output,
- source-manifest persistence,
- no concrete domain ontology in the extractor defaults,
- v0.45 closure reuse,
- v0.46 recursive macro reuse.

## Boundary

The reference release does **not** claim autonomous world-ontology extraction from a large pretrained Hugging Face model. No such model is bundled with this repository.

The generic extractor is backend-agnostic and can accept the existing local Hugging Face `CausalTeacherBackend` when `transformers` and local model files are available. The controlled release proof uses the local trained neural teacher so the extraction contract remains fully reproducible offline.

v0.47 also does not yet make RuntimeCognition execute the new macro hierarchy. RuntimeCognition remains unchanged from v0.40.

Run:

```bash
eqforge-top-down-teacher --out out/v0_47_top_down_teacher_structure
```

See `benchmarks/v0_47/` and `tests/test_top_down_teacher_structure_v047.py`.
