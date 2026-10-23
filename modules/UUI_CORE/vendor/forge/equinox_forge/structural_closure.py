from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable
import argparse
import hashlib
import json
import sqlite3

from .canonical_id_space import (
    CanonicalTriple,
    canonical_node_id,
    canonical_relation_id,
    ensure_canonical_id_schema,
    latest_run_id as latest_canonical_run_id,
    load_numeric_triples,
    materialize_canonical_id_space,
)
from .ir import IRNode, Modality, NodeKind, RelationEvidence, RelationType, SourceRef

STRUCTURAL_CLOSURE_FORMAT = "equinox-structural-closure-v1"
STRUCTURAL_CLOSURE_VERSION = "0.45"


@dataclass(frozen=True)
class StructuralClosurePolicy:
    """Policy for a non-duplicating, inheritance-addressable view of a numeric graph.

    The closure does not assert that every ancestor relation is semantically true of
    the child.  It creates an *addressable structural view* and preserves the origin
    and inheritance depth for every exposed fact. Later runtime policy can decide
    which relation types are inheritable, overridable, or only useful for routing.
    """

    inheritance_relation_types: tuple[str, ...] = (RelationType.IS_A.value,)
    accessible_relation_types: tuple[str, ...] = ()  # empty => all relation types
    max_parent_depth: int = 8
    min_score: float = 0.0
    min_confidence: float = 0.0
    include_hierarchy_relations: bool = True

    def validate(self) -> None:
        if self.max_parent_depth < 0:
            raise ValueError("max_parent_depth must be >= 0")
        if not 0.0 <= self.min_score <= 1.0:
            raise ValueError("min_score must be in [0,1]")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0,1]")
        if not self.inheritance_relation_types:
            raise ValueError("inheritance_relation_types must not be empty")


@dataclass(frozen=True)
class StructuralFact:
    root_id: int
    predicate_id: int
    object_id: int
    origin_subject_id: int
    inheritance_depth: int
    inheritance_path: tuple[int, ...]
    score: float
    confidence: float
    source_count: int
    evidence_count: int

    @property
    def inherited(self) -> bool:
        return self.inheritance_depth > 0

    @property
    def quality(self) -> float:
        return max(0.0, min(1.0, float(self.score) * float(self.confidence)))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["inheritance_path"] = list(self.inheritance_path)
        d["inherited"] = self.inherited
        d["quality"] = self.quality
        return d


@dataclass
class StructuralClosureBuild:
    format: str
    version: str
    run_id: int
    canonical_id_run_id: int
    root_count: int
    entry_count: int
    inherited_entry_count: int
    direct_entry_count: int
    digest: str
    policy: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def ensure_structural_closure_schema(db: sqlite3.Connection) -> None:
    ensure_canonical_id_schema(db)
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS structural_closure_runs(
          structural_closure_run_id INTEGER PRIMARY KEY AUTOINCREMENT,
          canonical_id_run_id INTEGER NOT NULL,
          format TEXT NOT NULL,
          version TEXT NOT NULL,
          policy_json TEXT NOT NULL,
          root_count INTEGER NOT NULL,
          entry_count INTEGER NOT NULL,
          inherited_entry_count INTEGER NOT NULL,
          direct_entry_count INTEGER NOT NULL,
          digest TEXT NOT NULL,
          metadata_json TEXT NOT NULL DEFAULT '{}',
          created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
          FOREIGN KEY(canonical_id_run_id) REFERENCES canonical_id_build_runs(canonical_id_run_id)
        );

        -- This table is an index of references into canonical_relation_triples.
        -- It deliberately stores no duplicate score/confidence/evidence payload.
        CREATE TABLE IF NOT EXISTS structural_closure_entries(
          structural_closure_run_id INTEGER NOT NULL,
          root_id INTEGER NOT NULL,
          predicate_id INTEGER NOT NULL,
          object_id INTEGER NOT NULL,
          origin_subject_id INTEGER NOT NULL,
          inheritance_depth INTEGER NOT NULL,
          inheritance_path_json TEXT NOT NULL,
          PRIMARY KEY(structural_closure_run_id,root_id,predicate_id,object_id),
          FOREIGN KEY(structural_closure_run_id) REFERENCES structural_closure_runs(structural_closure_run_id)
        );
        CREATE INDEX IF NOT EXISTS idx_structural_closure_root
          ON structural_closure_entries(structural_closure_run_id,root_id,inheritance_depth,predicate_id,object_id);
        CREATE INDEX IF NOT EXISTS idx_structural_closure_origin
          ON structural_closure_entries(structural_closure_run_id,origin_subject_id,predicate_id,object_id);
        """
    )


def _policy_dict(policy: StructuralClosurePolicy) -> dict[str, Any]:
    return asdict(policy)


def _relation_id_set(db: sqlite3.Connection, relation_types: Iterable[str]) -> set[int]:
    return {canonical_relation_id(db, str(x)) for x in relation_types}


def _triple_better(
    candidate: tuple[CanonicalTriple, int, tuple[int, ...]],
    incumbent: tuple[CanonicalTriple, int, tuple[int, ...]],
) -> bool:
    """Deterministic nearest-origin selection for one effective (predicate, object).

    The canonical base graph remains lossless.  The structural closure is a routing
    view and therefore keeps only the nearest occurrence of an identical effective
    fact.  Equal-depth duplicates prefer higher quality, then a stable numeric path.
    """
    c_t, c_depth, c_path = candidate
    i_t, i_depth, i_path = incumbent
    if c_depth != i_depth:
        return c_depth < i_depth
    c_q = float(c_t.score) * float(c_t.confidence)
    i_q = float(i_t.score) * float(i_t.confidence)
    if abs(c_q - i_q) > 1e-15:
        return c_q > i_q
    if abs(float(c_t.confidence) - float(i_t.confidence)) > 1e-15:
        return float(c_t.confidence) > float(i_t.confidence)
    return tuple(c_path) < tuple(i_path)


def compute_structural_closure(
    db: sqlite3.Connection,
    root_id: int,
    *,
    canonical_run_id: int | None = None,
    policy: StructuralClosurePolicy | None = None,
) -> list[StructuralFact]:
    """Compute the numeric structural closure for one concept ID.

    Traversal follows only configured inheritance relations. Facts remain stored once
    in the canonical graph; returned rows carry their origin subject and hierarchy
    path so callers can distinguish local from inherited/accessed information.
    """
    policy = policy or StructuralClosurePolicy()
    policy.validate()
    rid = int(canonical_run_id or latest_canonical_run_id(db))
    triples = load_numeric_triples(db, rid)
    by_subject: dict[int, list[CanonicalTriple]] = {}
    for t in triples:
        if float(t.score) + 1e-15 < policy.min_score:
            continue
        if float(t.confidence) + 1e-15 < policy.min_confidence:
            continue
        by_subject.setdefault(int(t.subject_id), []).append(t)
    for values in by_subject.values():
        values.sort(key=lambda t: (t.predicate_id, t.object_id, -t.score, -t.confidence))

    inheritance_ids = _relation_id_set(db, policy.inheritance_relation_types)
    accessible_ids = (
        _relation_id_set(db, policy.accessible_relation_types)
        if policy.accessible_relation_types else None
    )

    # BFS yields the shortest deterministic hierarchy path to each ancestor.
    paths: dict[int, tuple[int, ...]] = {int(root_id): (int(root_id),)}
    depths: dict[int, int] = {int(root_id): 0}
    frontier: list[int] = [int(root_id)]
    while frontier:
        subject = frontier.pop(0)
        depth = depths[subject]
        if depth >= policy.max_parent_depth:
            continue
        parent_ids = sorted({
            int(t.object_id)
            for t in by_subject.get(subject, [])
            if int(t.predicate_id) in inheritance_ids
        })
        for parent_id in parent_ids:
            if parent_id in paths:
                continue  # also blocks inheritance cycles deterministically
            paths[parent_id] = paths[subject] + (parent_id,)
            depths[parent_id] = depth + 1
            frontier.append(parent_id)

    selected: dict[tuple[int, int], tuple[CanonicalTriple, int, tuple[int, ...]]] = {}
    for subject, path in sorted(paths.items(), key=lambda x: (len(x[1]), x[1])):
        depth = depths[subject]
        for t in by_subject.get(subject, []):
            pred = int(t.predicate_id)
            if pred in inheritance_ids:
                if not policy.include_hierarchy_relations:
                    continue
            elif accessible_ids is not None and pred not in accessible_ids:
                continue
            key = (pred, int(t.object_id))
            candidate = (t, depth, path)
            incumbent = selected.get(key)
            if incumbent is None or _triple_better(candidate, incumbent):
                selected[key] = candidate

    out: list[StructuralFact] = []
    for (pred, obj), (t, depth, path) in sorted(selected.items()):
        out.append(StructuralFact(
            root_id=int(root_id),
            predicate_id=pred,
            object_id=obj,
            origin_subject_id=int(t.subject_id),
            inheritance_depth=int(depth),
            inheritance_path=tuple(int(x) for x in path),
            score=float(t.score),
            confidence=float(t.confidence),
            source_count=int(t.source_count),
            evidence_count=int(t.evidence_count),
        ))
    return out


def materialize_structural_closures(
    db: sqlite3.Connection,
    *,
    canonical_run_id: int | None = None,
    root_ids: Iterable[int] | None = None,
    policy: StructuralClosurePolicy | None = None,
    metadata: dict[str, Any] | None = None,
) -> StructuralClosureBuild:
    """Materialize closure *references* for fast ID-rooted access.

    No inherited semantic triple is copied into canonical_relation_triples. The index
    only points to an origin subject/predicate/object plus the hierarchy path.
    """
    ensure_structural_closure_schema(db)
    policy = policy or StructuralClosurePolicy()
    policy.validate()
    canonical_rid = int(canonical_run_id or latest_canonical_run_id(db))

    if root_ids is None:
        roots = [int(r[0]) for r in db.execute(
            "SELECT DISTINCT subject_id FROM canonical_relation_triples WHERE canonical_id_run_id=? ORDER BY subject_id",
            (canonical_rid,),
        ).fetchall()]
    else:
        roots = sorted({int(x) for x in root_ids})

    policy_json = json.dumps(_policy_dict(policy), sort_keys=True, separators=(",", ":"))
    db.execute(
        """INSERT INTO structural_closure_runs(
             canonical_id_run_id,format,version,policy_json,root_count,entry_count,
             inherited_entry_count,direct_entry_count,digest,metadata_json
           ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
        (
            canonical_rid, STRUCTURAL_CLOSURE_FORMAT, STRUCTURAL_CLOSURE_VERSION,
            policy_json, len(roots), 0, 0, 0, "pending",
            json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True),
        ),
    )
    run_id = int(db.execute("SELECT last_insert_rowid()").fetchone()[0])

    signature: list[tuple[Any, ...]] = []
    inherited = 0
    direct = 0
    entry_count = 0
    for root_id in roots:
        facts = compute_structural_closure(
            db, root_id, canonical_run_id=canonical_rid, policy=policy
        )
        for f in facts:
            db.execute(
                """INSERT INTO structural_closure_entries(
                     structural_closure_run_id,root_id,predicate_id,object_id,
                     origin_subject_id,inheritance_depth,inheritance_path_json
                   ) VALUES(?,?,?,?,?,?,?)""",
                (
                    run_id, f.root_id, f.predicate_id, f.object_id,
                    f.origin_subject_id, f.inheritance_depth,
                    json.dumps(list(f.inheritance_path), separators=(",", ":")),
                ),
            )
            entry_count += 1
            if f.inherited:
                inherited += 1
            else:
                direct += 1
            signature.append((
                f.root_id, f.predicate_id, f.object_id, f.origin_subject_id,
                f.inheritance_depth, list(f.inheritance_path),
            ))

    digest = hashlib.sha256(
        json.dumps(sorted(signature), separators=(",", ":"), sort_keys=False).encode("utf-8")
    ).hexdigest()
    db.execute(
        """UPDATE structural_closure_runs SET
             entry_count=?,inherited_entry_count=?,direct_entry_count=?,digest=?
           WHERE structural_closure_run_id=?""",
        (entry_count, inherited, direct, digest, run_id),
    )
    db.commit()
    return StructuralClosureBuild(
        STRUCTURAL_CLOSURE_FORMAT, STRUCTURAL_CLOSURE_VERSION, run_id,
        canonical_rid, len(roots), entry_count, inherited, direct, digest,
        _policy_dict(policy),
    )


def latest_structural_closure_run_id(db: sqlite3.Connection) -> int:
    ensure_structural_closure_schema(db)
    row = db.execute("SELECT MAX(structural_closure_run_id) FROM structural_closure_runs").fetchone()
    if not row or row[0] is None:
        raise ValueError("no structural closure build exists")
    return int(row[0])


def load_materialized_structural_closure(
    db: sqlite3.Connection,
    root_id: int,
    *,
    structural_run_id: int | None = None,
) -> list[StructuralFact]:
    """Load a materialized closure by joining reference rows back to base triples."""
    run_id = int(structural_run_id or latest_structural_closure_run_id(db))
    row = db.execute(
        "SELECT canonical_id_run_id FROM structural_closure_runs WHERE structural_closure_run_id=?",
        (run_id,),
    ).fetchone()
    if not row:
        raise KeyError(f"unknown structural closure run: {run_id}")
    canonical_rid = int(row[0])
    rows = db.execute(
        """SELECT e.root_id,e.predicate_id,e.object_id,e.origin_subject_id,
                  e.inheritance_depth,e.inheritance_path_json,
                  t.score,t.confidence,t.source_count,t.evidence_count
           FROM structural_closure_entries e
           JOIN canonical_relation_triples t
             ON t.canonical_id_run_id=?
            AND t.subject_id=e.origin_subject_id
            AND t.predicate_id=e.predicate_id
            AND t.object_id=e.object_id
           WHERE e.structural_closure_run_id=? AND e.root_id=?
           ORDER BY e.predicate_id,e.object_id""",
        (canonical_rid, run_id, int(root_id)),
    ).fetchall()
    return [
        StructuralFact(
            int(root), int(pred), int(obj), int(origin), int(depth),
            tuple(int(x) for x in json.loads(path_json)),
            float(score), float(conf), int(sc), int(ec),
        )
        for root,pred,obj,origin,depth,path_json,score,conf,sc,ec in rows
    ]


def _demo_node(node_id: str, level: int = 3) -> IRNode:
    return IRNode(node_id, Modality.CROSSMODAL.value, NodeKind.CONCEPT.value, level, surface=None)


def run_structural_closure_demo(out_dir: str | Path) -> dict[str, Any]:
    """Reference proof: one fact stored once, addressable through multiple child IDs."""
    from .database import CanonicalDB

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    db_path = out / "structural-closure-demo.sqlite"
    if db_path.exists():
        db_path.unlink()

    names = ["owl", "falcon", "bird", "animal", "fly", "feathers", "alive", "nocturnal"]
    nodes = {name: _demo_node(f"concept:{name}", 2 if name in {"fly", "feathers", "alive", "nocturnal"} else 3)
             for name in names}
    src = SourceRef("v045-demo", "1", "structural-closure")
    evidence = [
        RelationEvidence(nodes["owl"].node_id, nodes["bird"].node_id, RelationType.IS_A.value, 1.0, 1.0, src),
        RelationEvidence(nodes["falcon"].node_id, nodes["bird"].node_id, RelationType.IS_A.value, 1.0, 1.0, src),
        RelationEvidence(nodes["bird"].node_id, nodes["animal"].node_id, RelationType.IS_A.value, 1.0, 1.0, src),
        RelationEvidence(nodes["bird"].node_id, nodes["fly"].node_id, RelationType.ACTION.value, 0.97, 0.95, src),
        RelationEvidence(nodes["bird"].node_id, nodes["feathers"].node_id, RelationType.PROPERTY.value, 0.99, 0.98, src),
        RelationEvidence(nodes["animal"].node_id, nodes["alive"].node_id, RelationType.PROPERTY.value, 0.99, 0.99, src),
        RelationEvidence(nodes["owl"].node_id, nodes["nocturnal"].node_id, RelationType.PROPERTY.value, 0.95, 0.94, src),
    ]

    with CanonicalDB(db_path) as cdb:
        for node in nodes.values():
            cdb.upsert_node(node)
        for ev in evidence:
            cdb.add_evidence(ev)
        cdb.db.commit()
        canonical = materialize_canonical_id_space(cdb.db)
        ids = {name: canonical_node_id(cdb.db, node.node_id) for name,node in nodes.items()}
        relation_ids = {
            "is_a": canonical_relation_id(cdb.db, RelationType.IS_A.value),
            "action": canonical_relation_id(cdb.db, RelationType.ACTION.value),
            "property": canonical_relation_id(cdb.db, RelationType.PROPERTY.value),
        }
        base_before = load_numeric_triples(cdb.db, canonical.run_id)
        build = materialize_structural_closures(
            cdb.db,
            canonical_run_id=canonical.run_id,
            root_ids=[ids["owl"], ids["falcon"]],
            policy=StructuralClosurePolicy(max_parent_depth=4),
        )
        base_after = load_numeric_triples(cdb.db, canonical.run_id)
        owl = load_materialized_structural_closure(cdb.db, ids["owl"], structural_run_id=build.run_id)
        falcon = load_materialized_structural_closure(cdb.db, ids["falcon"], structural_run_id=build.run_id)

        owl_by_pair = {(x.predicate_id, x.object_id): x for x in owl}
        falcon_by_pair = {(x.predicate_id, x.object_id): x for x in falcon}
        feathers_pair = (relation_ids["property"], ids["feathers"])
        alive_pair = (relation_ids["property"], ids["alive"])
        nocturnal_pair = (relation_ids["property"], ids["nocturnal"])
        fly_pair = (relation_ids["action"], ids["fly"])
        animal_pair = (relation_ids["is_a"], ids["animal"])
        synthetic_owl_feathers = (
            ids["owl"], relation_ids["property"], ids["feathers"]
        )
        base_keys = {(x.subject_id, x.predicate_id, x.object_id) for x in base_after}

        report = {
            "format": "equinox-forge-structural-closure-demo-v1",
            "version": STRUCTURAL_CLOSURE_VERSION,
            "canonical_build": canonical.to_dict(),
            "closure_build": build.to_dict(),
            "numeric_ids": ids,
            "relation_ids": relation_ids,
            "controls": {
                "canonical_triple_count_unchanged": len(base_before) == len(base_after),
                "inherited_fact_not_copied_to_owl": synthetic_owl_feathers not in base_keys,
                "owl_sees_bird_feathers": feathers_pair in owl_by_pair,
                "owl_sees_animal_alive": alive_pair in owl_by_pair,
                "owl_sees_transitive_animal_class": animal_pair in owl_by_pair,
                "owl_keeps_local_nocturnal": nocturnal_pair in owl_by_pair and owl_by_pair[nocturnal_pair].inheritance_depth == 0,
                "falcon_sees_bird_fly": fly_pair in falcon_by_pair,
                "shared_fact_keeps_same_origin": (
                    feathers_pair in owl_by_pair and feathers_pair in falcon_by_pair
                    and owl_by_pair[feathers_pair].origin_subject_id == ids["bird"]
                    and falcon_by_pair[feathers_pair].origin_subject_id == ids["bird"]
                ),
                "bird_fact_depth_is_one_for_children": (
                    owl_by_pair[feathers_pair].inheritance_depth == 1
                    and falcon_by_pair[feathers_pair].inheritance_depth == 1
                ),
                "animal_fact_depth_is_two_for_owl": owl_by_pair[alive_pair].inheritance_depth == 2,
                "closure_is_numeric_only": all(
                    isinstance(v, int)
                    for fact in owl + falcon
                    for v in (fact.root_id, fact.predicate_id, fact.object_id, fact.origin_subject_id, *fact.inheritance_path)
                ),
            },
            "owl_closure": [x.to_dict() for x in owl],
            "falcon_closure": [x.to_dict() for x in falcon],
        }
        cdb.db.commit()

    (out / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the Equinox v0.45 numeric structural-closure demo.")
    ap.add_argument("--out", default="out/v0_45_structural_closure")
    args = ap.parse_args(argv)
    report = run_structural_closure_demo(args.out)
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
