from __future__ import annotations

from pathlib import Path
import tempfile
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from equinox_forge.recursive_macro_collapse import (
    RECURSIVE_MACRO_VERSION,
    RecursiveMacroPolicy,
    effective_macro_facts,
    macro_members,
    materialize_recursive_macro_collapse,
    run_recursive_macro_demo,
)
from equinox_forge.canonical_id_space import canonical_node_id, load_numeric_triples, materialize_canonical_id_space
from equinox_forge.structural_closure import StructuralClosurePolicy, materialize_structural_closures
from equinox_forge.database import CanonicalDB
from equinox_forge.ir import IRNode, Modality, NodeKind, RelationEvidence, RelationType, SourceRef

assert RECURSIVE_MACRO_VERSION == "0.46"


def node(name: str) -> IRNode:
    return IRNode(f"opaque:{name}", Modality.CROSSMODAL.value, NodeKind.CONCEPT.value, 3, surface=None)


def build(path: Path, reverse: bool = False):
    names = [
        "owl", "falcon", "dog", "wolf", "bird", "mammal", "animal",
        "feathers", "fly", "fur", "walk", "alive", "move",
    ]
    nodes = {x: node(x) for x in names}
    src = SourceRef("v046-test", "1", "unit")
    evidence = [
        RelationEvidence(nodes["owl"].node_id, nodes["bird"].node_id, RelationType.IS_A.value, 1, 1, src),
        RelationEvidence(nodes["falcon"].node_id, nodes["bird"].node_id, RelationType.IS_A.value, 1, 1, src),
        RelationEvidence(nodes["bird"].node_id, nodes["animal"].node_id, RelationType.IS_A.value, 1, 1, src),
        RelationEvidence(nodes["bird"].node_id, nodes["feathers"].node_id, RelationType.PROPERTY.value, .99, .98, src),
        RelationEvidence(nodes["bird"].node_id, nodes["fly"].node_id, RelationType.ACTION.value, .98, .97, src),
        RelationEvidence(nodes["dog"].node_id, nodes["mammal"].node_id, RelationType.IS_A.value, 1, 1, src),
        RelationEvidence(nodes["wolf"].node_id, nodes["mammal"].node_id, RelationType.IS_A.value, 1, 1, src),
        RelationEvidence(nodes["mammal"].node_id, nodes["animal"].node_id, RelationType.IS_A.value, 1, 1, src),
        RelationEvidence(nodes["mammal"].node_id, nodes["fur"].node_id, RelationType.PROPERTY.value, .99, .98, src),
        RelationEvidence(nodes["mammal"].node_id, nodes["walk"].node_id, RelationType.ACTION.value, .98, .97, src),
        RelationEvidence(nodes["animal"].node_id, nodes["alive"].node_id, RelationType.PROPERTY.value, .99, .99, src),
        RelationEvidence(nodes["animal"].node_id, nodes["move"].node_id, RelationType.ACTION.value, .99, .98, src),
    ]
    node_values = list(nodes.values())
    if reverse:
        node_values.reverse(); evidence.reverse()
    with CanonicalDB(path) as cdb:
        for n in node_values:
            cdb.upsert_node(n)
        for ev in evidence:
            cdb.add_evidence(ev)
        cdb.db.commit()
        canonical = materialize_canonical_id_space(cdb.db)
        ids = {name: canonical_node_id(cdb.db, n.node_id) for name,n in nodes.items()}
        roots = [ids[x] for x in ("owl", "falcon", "dog", "wolf")]
        before = len(load_numeric_triples(cdb.db, canonical.run_id))
        closure = materialize_structural_closures(
            cdb.db, canonical_run_id=canonical.run_id, root_ids=roots,
            policy=StructuralClosurePolicy(max_parent_depth=4),
        )
        build = materialize_recursive_macro_collapse(
            cdb.db, canonical_run_id=canonical.run_id, structural_closure_run_id=closure.run_id,
            root_ids=roots, policy=RecursiveMacroPolicy(min_members=2, min_common_facts=2, max_levels=4),
        )
        after = len(load_numeric_triples(cdb.db, canonical.run_id))
        rows = cdb.db.execute(
            """SELECT macro_id,level,effective_fact_count,local_fact_count
               FROM recursive_macros WHERE recursive_macro_run_id=? ORDER BY level,macro_id""",
            (build.run_id,),
        ).fetchall()
        macros = [
            (
                int(mid), int(level), int(ef), int(lf),
                macro_members(cdb.db, int(mid), run_id=build.run_id),
                tuple(sorted(effective_macro_facts(cdb.db, int(mid), recursive_macro_run_id=build.run_id))),
            )
            for mid,level,ef,lf in rows
        ]
        return {
            "canonical_digest": canonical.digest,
            "closure_digest": closure.digest,
            "macro_digest": build.digest,
            "before": before,
            "after": after,
            "build": build,
            "macros": macros,
        }


with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    a = build(td / "a.sqlite", False)
    b = build(td / "b.sqlite", True)

    # Full numeric collapse is deterministic under insertion reordering.
    assert a["canonical_digest"] == b["canonical_digest"]
    assert a["closure_digest"] == b["closure_digest"]
    assert a["macro_digest"] == b["macro_digest"]
    assert a["macros"] == b["macros"]

    # Canonical semantic facts are never destructively rewritten by collapse.
    assert a["before"] == a["after"] == 12
    assert a["build"].levels_built == 2
    assert a["build"].macro_count == 3
    assert a["build"].potential_reference_savings > 0

    level1 = [m for m in a["macros"] if m[1] == 1]
    level2 = [m for m in a["macros"] if m[1] == 2]
    assert len(level1) == 2
    assert len(level2) == 1

    # Higher-order recursion is real: the level-2 macro's members are macro IDs.
    assert all(kind == "macro" for _,kind in level2[0][4])
    assert {member for member,_ in level2[0][4]} == {m[0] for m in level1}

    # Parent-owned shared facts are removed from child-local storage but remain
    # visible in each child's effective recursive signature.
    assert all(local < effective for _,_,effective,local,_,_ in level1)
    assert all(len(sig) == effective for _,_,effective,_,_,sig in level1)

    report = run_recursive_macro_demo(td / "demo")
    assert all(report["controls"].values()), report["controls"]

print("FORGE_V046_RECURSIVE_MACRO_COLLAPSE_OK")
