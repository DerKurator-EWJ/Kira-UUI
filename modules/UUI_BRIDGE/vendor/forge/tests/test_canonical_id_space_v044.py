from __future__ import annotations

import json
from pathlib import Path
import tempfile
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from equinox_forge.canonical_id_space import (
    ID_SPACE_VERSION,
    bind_surface,
    canonical_node_id,
    canonical_relation_id,
    identity_record,
    load_numeric_triples,
    materialize_canonical_id_space,
    numeric_closure,
    resolve_surface,
    run_canonical_id_demo,
)
from equinox_forge.database import CanonicalDB
from equinox_forge.ir import IRNode, Modality, NodeKind, RelationEvidence, RelationType, SourceRef

assert ID_SPACE_VERSION == "0.44"


def node(node_id: str, level: int = 3) -> IRNode:
    return IRNode(node_id, Modality.CROSSMODAL.value, NodeKind.CONCEPT.value, level, surface=None)


def build(path: Path, reverse: bool = False):
    nodes = [node("opaque:owl"), node("opaque:bird"), node("opaque:animal"), node("opaque:fly", 2)]
    src = SourceRef("id-test", "1", "unit")
    evs = [
        RelationEvidence("opaque:owl", "opaque:bird", RelationType.IS_A.value, 1.0, 0.9, src),
        RelationEvidence("opaque:bird", "opaque:animal", RelationType.IS_A.value, 1.0, 0.9, src),
        RelationEvidence("opaque:owl", "opaque:fly", RelationType.ACTION.value, 1.0, 0.8, src),
    ]
    if reverse:
        nodes = list(reversed(nodes))
        evs = list(reversed(evs))
    with CanonicalDB(path) as db:
        for n in nodes:
            db.upsert_node(n)
        for e in evs:
            db.add_evidence(e)
        db.db.commit()
        owl = canonical_node_id(db.db, "opaque:owl")
        bird = canonical_node_id(db.db, "opaque:bird")
        animal = canonical_node_id(db.db, "opaque:animal")
        is_a = canonical_relation_id(db.db, RelationType.IS_A.value)
        bind_surface(db.db, surface="Eule", concept_node_id="opaque:owl", locale="de")
        bind_surface(db.db, surface="owl", concept_node_id="opaque:owl", locale="en")
        b = materialize_canonical_id_space(db.db)
        triples = load_numeric_triples(db.db, b.run_id)
        closure = numeric_closure(db.db, owl, run_id=b.run_id, max_depth=3)
        return {
            "owl": owl, "bird": bird, "animal": animal, "is_a": is_a,
            "digest": b.digest,
            "triples": [(t.subject_id,t.predicate_id,t.object_id) for t in triples],
            "closure": [(t.subject_id,t.predicate_id,t.object_id) for t in closure],
            "de": resolve_surface(db.db, "Eule", locale="de"),
            "en": resolve_surface(db.db, "owl", locale="en"),
            "owl_record": identity_record(db.db, owl),
        }


with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    a = build(td / "a.sqlite", reverse=False)
    b = build(td / "b.sqlite", reverse=True)

    # The number is a stable handle, not an insertion-order-generated primary key.
    assert a["owl"] == b["owl"]
    assert a["bird"] == b["bird"]
    assert a["animal"] == b["animal"]
    assert a["is_a"] == b["is_a"]
    assert a["digest"] == b["digest"]
    assert a["triples"] == b["triples"]

    # Surface language is secondary; both labels resolve to one concept handle.
    assert a["de"] == [a["owl"]]
    assert a["en"] == [a["owl"]]
    assert a["de"] == a["en"]

    # Numeric closure can traverse owl -> bird -> animal without using labels.
    assert (a["owl"], a["is_a"], a["bird"]) in a["closure"]
    assert (a["bird"], a["is_a"], a["animal"]) in a["closure"]
    assert a["owl_record"].symbolic_id == "opaque:owl"

    report = run_canonical_id_demo(td / "demo")
    assert all(report["controls"].values())
    assert report["surface_resolution"]["Eule@de"] == report["surface_resolution"]["owl@en"]
    assert report["build"]["triple_count"] == 4

print("FORGE_V044_CANONICAL_ID_SPACE_OK")
