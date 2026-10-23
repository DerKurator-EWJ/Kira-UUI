from __future__ import annotations

import sqlite3

from equinox_forge.canonical_id_space import CanonicalTriple, canonical_relation_id, register_canonical_identity
from equinox_forge.canonical_relation_facets import _insert_derived_canonical_run, materialize_relation_faceted_id_space
from equinox_forge.recursive_relation_collapse import (
    RelationMacroPolicy,
    effective_relation_macro_facts,
    materialize_recursive_relation_macro_collapse,
    relation_macro_members,
)


def _build(order: list[str]) -> tuple[str, dict[str, int], list[tuple[int, int, str]], dict[int, set[tuple[int, int]]]]:
    db = sqlite3.connect(":memory:")
    prop = canonical_relation_id(db, "property")
    action = canonical_relation_id(db, "action")
    subject = register_canonical_identity(db, namespace="node", symbolic_id="entity", identity_kind="concept")
    names = ["feathers", "body", "cells", "fur", "fly", "move", "live", "run"]
    objects = {name: register_canonical_identity(db, namespace="node", symbolic_id=name, identity_kind="concept") for name in names}
    rows = {
        "p1": CanonicalTriple(subject, prop, objects["feathers"], 1, 1, 1, 1),
        "p2": CanonicalTriple(subject, prop, objects["body"], 1, 1, 1, 1),
        "p3": CanonicalTriple(subject, prop, objects["cells"], 1, 1, 1, 1),
        "p4": CanonicalTriple(subject, prop, objects["fur"], 1, 1, 1, 1),
        "a1": CanonicalTriple(subject, action, objects["fly"], 1, 1, 1, 1),
        "a2": CanonicalTriple(subject, action, objects["move"], 1, 1, 1, 1),
        "a3": CanonicalTriple(subject, action, objects["live"], 1, 1, 1, 1),
        "a4": CanonicalTriple(subject, action, objects["run"], 1, 1, 1, 1),
    }
    source_run, _ = _insert_derived_canonical_run(db, source_run_id=0, triples=[rows[x] for x in order], metadata={"fixture":"v052"})
    facets = materialize_relation_faceted_id_space(db, source_canonical_run_id=source_run)
    build = materialize_recursive_relation_macro_collapse(
        db,
        relation_facet_run_id=facets.run_id,
        policy=RelationMacroPolicy(min_members=2, min_common_meta_facts=1, max_levels=4),
    )
    assert build.levels_built == 2, build
    assert build.macro_count == 3, build
    assert build.predicate_ready_macro_count == 3
    macros = db.execute(
        "SELECT macro_id,level FROM recursive_relation_macros WHERE relation_macro_run_id=? ORDER BY level,macro_id",
        (build.run_id,),
    ).fetchall()
    memberships: list[tuple[int, int, str]] = []
    effective: dict[int, set[tuple[int,int]]] = {}
    for mid, level in macros:
        rec = db.execute("SELECT namespace,identity_kind FROM canonical_identities WHERE numeric_id=?", (mid,)).fetchone()
        assert rec == ("relation", "recursive_relation_macro")
        for child, kind in relation_macro_members(db, relation_macro_run_id=build.run_id, macro_id=mid):
            memberships.append((int(mid), int(child), kind))
        effective[int(mid)] = set(effective_relation_macro_facts(db, int(mid), relation_macro_run_id=build.run_id))
    levels = {int(mid): int(level) for mid, level in macros}
    l1 = [mid for mid, level in levels.items() if level == 1]
    l2 = [mid for mid, level in levels.items() if level == 2]
    assert len(l1) == 2 and len(l2) == 1
    assert all(len(effective[mid]) == 2 for mid in l1)  # family + inherited facet-kind
    assert len(effective[l2[0]]) == 1
    parent_members = relation_macro_members(db, relation_macro_run_id=build.run_id, macro_id=l2[0])
    assert {m for m, kind in parent_members if kind == "relation_macro"} == set(l1)
    # A collapsed relation-macro is itself a legal predicate ID in the canonical space.
    demo_run, _ = _insert_derived_canonical_run(
        db, source_run_id=source_run,
        triples=[CanonicalTriple(subject, l2[0], objects["body"], 1, 1, 1, 1)],
        metadata={"predicate_ready_demo": True},
    )
    demo = db.execute(
        "SELECT predicate_id FROM canonical_relation_triples WHERE canonical_id_run_id=?", (demo_run,)
    ).fetchone()
    assert int(demo[0]) == l2[0]
    digest = build.digest
    ids = {"prop": prop, "action": action, **objects}
    db.close()
    return digest, ids, sorted(memberships), effective


def main() -> None:
    a = _build(["p1","p2","p3","p4","a1","a2","a3","a4"])
    b = _build(["a4","a3","a2","a1","p4","p3","p2","p1"])
    assert a[0] == b[0], (a[0], b[0])
    assert a[1] == b[1]
    assert a[2] == b[2]
    assert a[3] == b[3]
    print("v0.52 recursive relation-ID collapse: PASS")


if __name__ == "__main__":
    main()
