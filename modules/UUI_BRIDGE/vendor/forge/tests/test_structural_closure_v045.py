from __future__ import annotations

from pathlib import Path
import tempfile
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from equinox_forge.structural_closure import (
    STRUCTURAL_CLOSURE_VERSION,
    StructuralClosurePolicy,
    compute_structural_closure,
    load_materialized_structural_closure,
    materialize_structural_closures,
    run_structural_closure_demo,
)
from equinox_forge.canonical_id_space import (
    canonical_node_id,
    canonical_relation_id,
    load_numeric_triples,
    materialize_canonical_id_space,
)
from equinox_forge.database import CanonicalDB
from equinox_forge.ir import IRNode, Modality, NodeKind, RelationEvidence, RelationType, SourceRef

assert STRUCTURAL_CLOSURE_VERSION == "0.45"


def node(name: str, level: int = 3) -> IRNode:
    return IRNode(f"opaque:{name}", Modality.CROSSMODAL.value, NodeKind.CONCEPT.value, level, surface=None)


def build(path: Path, reverse: bool = False):
    names = ["owl", "falcon", "bird", "animal", "fly", "feathers", "alive", "nocturnal"]
    nodes = {name: node(name, 2 if name in {"fly", "feathers", "alive", "nocturnal"} else 3) for name in names}
    src = SourceRef("closure-test", "1", "unit")
    evidence = [
        RelationEvidence(nodes["owl"].node_id, nodes["bird"].node_id, RelationType.IS_A.value, 1.0, 0.99, src),
        RelationEvidence(nodes["falcon"].node_id, nodes["bird"].node_id, RelationType.IS_A.value, 1.0, 0.99, src),
        RelationEvidence(nodes["bird"].node_id, nodes["animal"].node_id, RelationType.IS_A.value, 1.0, 0.99, src),
        RelationEvidence(nodes["bird"].node_id, nodes["fly"].node_id, RelationType.ACTION.value, 0.95, 0.93, src),
        RelationEvidence(nodes["bird"].node_id, nodes["feathers"].node_id, RelationType.PROPERTY.value, 0.98, 0.97, src),
        RelationEvidence(nodes["animal"].node_id, nodes["alive"].node_id, RelationType.PROPERTY.value, 0.99, 0.99, src),
        RelationEvidence(nodes["owl"].node_id, nodes["nocturnal"].node_id, RelationType.PROPERTY.value, 0.93, 0.91, src),
        # The same effective fact exists at two levels. The closure should route
        # to the nearest origin instead of exposing duplicate semantic rows.
        RelationEvidence(nodes["animal"].node_id, nodes["feathers"].node_id, RelationType.PROPERTY.value, 0.70, 0.70, src),
    ]
    node_values = list(nodes.values())
    if reverse:
        node_values.reverse()
        evidence.reverse()
    with CanonicalDB(path) as cdb:
        for n in node_values:
            cdb.upsert_node(n)
        for ev in evidence:
            cdb.add_evidence(ev)
        cdb.db.commit()
        canonical = materialize_canonical_id_space(cdb.db)
        ids = {name: canonical_node_id(cdb.db, n.node_id) for name,n in nodes.items()}
        rel = {
            "is_a": canonical_relation_id(cdb.db, RelationType.IS_A.value),
            "action": canonical_relation_id(cdb.db, RelationType.ACTION.value),
            "property": canonical_relation_id(cdb.db, RelationType.PROPERTY.value),
        }
        base_count_before = len(load_numeric_triples(cdb.db, canonical.run_id))
        closure_build = materialize_structural_closures(
            cdb.db,
            canonical_run_id=canonical.run_id,
            root_ids=[ids["owl"], ids["falcon"]],
            policy=StructuralClosurePolicy(max_parent_depth=4),
        )
        base_count_after = len(load_numeric_triples(cdb.db, canonical.run_id))
        owl = load_materialized_structural_closure(cdb.db, ids["owl"], structural_run_id=closure_build.run_id)
        falcon = load_materialized_structural_closure(cdb.db, ids["falcon"], structural_run_id=closure_build.run_id)
        return {
            "ids": ids,
            "rel": rel,
            "canonical_digest": canonical.digest,
            "closure_digest": closure_build.digest,
            "base_before": base_count_before,
            "base_after": base_count_after,
            "owl": [
                (x.predicate_id, x.object_id, x.origin_subject_id, x.inheritance_depth, x.inheritance_path)
                for x in owl
            ],
            "falcon": [
                (x.predicate_id, x.object_id, x.origin_subject_id, x.inheritance_depth, x.inheritance_path)
                for x in falcon
            ],
        }


with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    a = build(td / "a.sqlite", reverse=False)
    b = build(td / "b.sqlite", reverse=True)

    # Canonical and closure routing are deterministic under insertion reordering.
    assert a["canonical_digest"] == b["canonical_digest"]
    assert a["closure_digest"] == b["closure_digest"]
    assert a["owl"] == b["owl"]
    assert a["falcon"] == b["falcon"]

    # Building the closure index never adds synthetic semantic triples.
    assert a["base_before"] == a["base_after"] == 8

    ids = a["ids"]
    rel = a["rel"]
    owl = {(p,o): (origin,depth,path) for p,o,origin,depth,path in a["owl"]}
    falcon = {(p,o): (origin,depth,path) for p,o,origin,depth,path in a["falcon"]}

    assert owl[(rel["property"], ids["nocturnal"])][1] == 0
    assert owl[(rel["property"], ids["feathers"])][0] == ids["bird"]
    assert owl[(rel["property"], ids["feathers"])][1] == 1
    assert owl[(rel["property"], ids["alive"])][0] == ids["animal"]
    assert owl[(rel["property"], ids["alive"])][1] == 2
    assert owl[(rel["is_a"], ids["animal"])][1] == 1
    assert falcon[(rel["property"], ids["feathers"])][0] == ids["bird"]

    # The nearest identical fact wins in the closure view; the base graph still
    # keeps the farther duplicate for audit/rebuild purposes.
    assert owl[(rel["property"], ids["feathers"])][0] != ids["animal"]

    # Cycle safety: a malformed hierarchy cannot recurse forever.
    with CanonicalDB(td / "cycle.sqlite") as cdb:
        x, y = node("x"), node("y")
        cdb.upsert_node(x); cdb.upsert_node(y)
        src = SourceRef("cycle", "1", "unit")
        cdb.add_evidence(RelationEvidence(x.node_id, y.node_id, RelationType.IS_A.value, 1.0, 1.0, src))
        cdb.add_evidence(RelationEvidence(y.node_id, x.node_id, RelationType.IS_A.value, 1.0, 1.0, src))
        cdb.db.commit()
        cb = materialize_canonical_id_space(cdb.db)
        xid = canonical_node_id(cdb.db, x.node_id)
        facts = compute_structural_closure(cdb.db, xid, canonical_run_id=cb.run_id, policy=StructuralClosurePolicy(max_parent_depth=50))
        assert len(facts) <= 2

    report = run_structural_closure_demo(td / "demo")
    assert all(report["controls"].values())
    assert report["closure_build"]["inherited_entry_count"] > 0
    assert report["closure_build"]["direct_entry_count"] > 0

print("FORGE_V045_STRUCTURAL_CLOSURE_OK")
