# Equinox Forge v0.44 — Canonical ID Space

v0.44 changes the center of gravity of Forge without changing RuntimeCognition. Knowledge can now be materialized into a deterministic, language-separated numeric identity graph:

```text
canonical node identity      -> stable positive 63-bit ID
canonical relation identity  -> stable positive 63-bit ID
compiled knowledge           -> (subject_id, predicate_id, object_id)
surface labels               -> separate bindings to concept IDs
```

The numeric value itself carries no semantic meaning. Meaning exists only in the relation graph rooted at that handle. The old symbolic node IDs remain available as provenance/debug identities, so existing Forge databases can be migrated without losing auditability.

## Why this exists

The long-term Equinox representation is intended to separate teacher/model language from runtime knowledge. `Eule`, `owl`, an image-derived owl concept, or another modality may eventually resolve to the same canonical concept handle while the runtime graph remains numeric.

v0.44 deliberately does **not** infer that two surfaces are the same concept. They share an ID only after an explicit Forge binding. This avoids turning string similarity into semantic identity.

## New persistent structures

- `canonical_identities`
- `canonical_surface_bindings`
- `canonical_id_build_runs`
- `canonical_relation_triples`

IDs are deterministic functions of canonical symbolic identity and namespace, independent of insertion order. A detected numeric collision is a hard error, never an implicit merge.

## Frozen controls

The reference demo materializes an owl/bird/animal graph and verifies:

```text
Eule@de -> same concept ID as owl@en
knowledge triples contain numeric subject/predicate/object IDs
surface strings are absent from the triple payload
numeric closure traverses owl -> bird -> animal
reversing DB insertion order leaves IDs and graph digest unchanged
```

This release does not yet perform recursive macro collapse, inheritance, or top-down teacher extraction. Those are intentionally left for later releases on top of the ID layer.

Run:

```bash
eqforge-canonical-id-space --out out/v0_44_canonical_id_space
```

See `benchmarks/v0_44/` and `tests/test_canonical_id_space_v044.py`.
