from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
import argparse
import hashlib
import json
import sqlite3

from .compiler import CompilePolicy, aggregate_relations
from .ir import IRNode, Modality, NodeKind, RelationEvidence, RelationType, SourceRef

ID_SPACE_FORMAT = "equinox-canonical-id-space-v1"
ID_SPACE_VERSION = "0.44"
_ID_MASK = (1 << 63) - 1


@dataclass(frozen=True)
class CanonicalIdentity:
    numeric_id: int
    namespace: str
    symbolic_id: str
    identity_kind: str


@dataclass(frozen=True)
class CanonicalTriple:
    subject_id: int
    predicate_id: int
    object_id: int
    score: float
    confidence: float
    source_count: int
    evidence_count: int


@dataclass(frozen=True)
class SurfaceBinding:
    surface: str
    modality: str
    concept_id: int
    surface_node_id: str | None = None
    locale: str = ""


@dataclass
class CanonicalIDBuild:
    format: str
    version: str
    run_id: int
    identity_count: int
    node_identity_count: int
    relation_identity_count: int
    triple_count: int
    surface_binding_count: int
    digest: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def ensure_canonical_id_schema(db: sqlite3.Connection) -> None:
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS canonical_identities(
          numeric_id INTEGER PRIMARY KEY,
          namespace TEXT NOT NULL,
          symbolic_id TEXT NOT NULL,
          identity_kind TEXT NOT NULL DEFAULT '',
          metadata_json TEXT NOT NULL DEFAULT '{}',
          UNIQUE(namespace,symbolic_id)
        );
        CREATE INDEX IF NOT EXISTS idx_canonical_identity_symbolic
          ON canonical_identities(namespace,symbolic_id);

        CREATE TABLE IF NOT EXISTS canonical_surface_bindings(
          surface TEXT NOT NULL,
          modality TEXT NOT NULL,
          locale TEXT NOT NULL DEFAULT '',
          concept_numeric_id INTEGER NOT NULL,
          surface_node_id TEXT NOT NULL DEFAULT '',
          metadata_json TEXT NOT NULL DEFAULT '{}',
          PRIMARY KEY(surface,modality,locale,concept_numeric_id,surface_node_id),
          FOREIGN KEY(concept_numeric_id) REFERENCES canonical_identities(numeric_id)
        );
        CREATE INDEX IF NOT EXISTS idx_canonical_surface_lookup
          ON canonical_surface_bindings(surface,modality,locale);

        CREATE TABLE IF NOT EXISTS canonical_id_build_runs(
          canonical_id_run_id INTEGER PRIMARY KEY AUTOINCREMENT,
          format TEXT NOT NULL,
          version TEXT NOT NULL,
          policy_json TEXT NOT NULL,
          identity_count INTEGER NOT NULL,
          node_identity_count INTEGER NOT NULL,
          relation_identity_count INTEGER NOT NULL,
          triple_count INTEGER NOT NULL,
          surface_binding_count INTEGER NOT NULL,
          digest TEXT NOT NULL,
          metadata_json TEXT NOT NULL DEFAULT '{}',
          created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS canonical_relation_triples(
          canonical_id_run_id INTEGER NOT NULL,
          subject_id INTEGER NOT NULL,
          predicate_id INTEGER NOT NULL,
          object_id INTEGER NOT NULL,
          score REAL NOT NULL,
          confidence REAL NOT NULL,
          source_count INTEGER NOT NULL,
          evidence_count INTEGER NOT NULL,
          PRIMARY KEY(canonical_id_run_id,subject_id,predicate_id,object_id),
          FOREIGN KEY(canonical_id_run_id) REFERENCES canonical_id_build_runs(canonical_id_run_id),
          FOREIGN KEY(subject_id) REFERENCES canonical_identities(numeric_id),
          FOREIGN KEY(predicate_id) REFERENCES canonical_identities(numeric_id),
          FOREIGN KEY(object_id) REFERENCES canonical_identities(numeric_id)
        );
        CREATE INDEX IF NOT EXISTS idx_canonical_triple_subject
          ON canonical_relation_triples(canonical_id_run_id,subject_id,predicate_id);
        CREATE INDEX IF NOT EXISTS idx_canonical_triple_object
          ON canonical_relation_triples(canonical_id_run_id,object_id,predicate_id);
        """
    )


def _stable_numeric_id(namespace: str, symbolic_id: str) -> int:
    """Return a deterministic positive SQLite-safe 63-bit ID.

    Meaning is *not* encoded in the number.  The namespace and symbolic canonical
    identity merely make the handle stable across insertion order and rebuilds.
    A registry collision is treated as a hard error rather than silently merging
    two concepts.
    """
    payload = f"{namespace}\0{symbolic_id}".encode("utf-8")
    raw = hashlib.blake2b(payload, digest_size=8, person=b"EQNXID44").digest()
    value = int.from_bytes(raw, "big") & _ID_MASK
    return value or 1


def _register_identity(
    db: sqlite3.Connection,
    namespace: str,
    symbolic_id: str,
    identity_kind: str = "",
    metadata: dict[str, Any] | None = None,
) -> int:
    ensure_canonical_id_schema(db)
    numeric_id = _stable_numeric_id(namespace, symbolic_id)
    row = db.execute(
        "SELECT namespace,symbolic_id FROM canonical_identities WHERE numeric_id=?",
        (numeric_id,),
    ).fetchone()
    if row and tuple(row) != (namespace, symbolic_id):
        raise RuntimeError(
            "canonical numeric-ID collision: "
            f"{numeric_id} already belongs to {row[0]}:{row[1]}, "
            f"cannot assign {namespace}:{symbolic_id}"
        )
    db.execute(
        """INSERT INTO canonical_identities(numeric_id,namespace,symbolic_id,identity_kind,metadata_json)
           VALUES(?,?,?,?,?)
           ON CONFLICT(namespace,symbolic_id) DO UPDATE SET
             identity_kind=excluded.identity_kind,
             metadata_json=excluded.metadata_json""",
        (
            numeric_id,
            namespace,
            symbolic_id,
            identity_kind,
            json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True),
        ),
    )
    return numeric_id


def register_canonical_identity(
    db: sqlite3.Connection,
    *,
    namespace: str,
    symbolic_id: str,
    identity_kind: str = "",
    metadata: dict[str, Any] | None = None,
) -> int:
    """Public deterministic identity registration for higher-order numeric layers.

    v0.44 intentionally kept the stable-ID primitive private while proving the
    canonical node/relation space. Higher-order structures (macros in v0.46+)
    need the same collision-checked identity semantics without depending on
    insertion order or SQLite row IDs.
    """
    return _register_identity(db, namespace, symbolic_id, identity_kind, metadata)


def canonical_node_id(db: sqlite3.Connection, node_id: str) -> int:
    row = db.execute(
        "SELECT numeric_id FROM canonical_identities WHERE namespace='node' AND symbolic_id=?",
        (node_id,),
    ).fetchone()
    if row:
        return int(row[0])
    node = db.execute(
        "SELECT modality,kind,level FROM nodes WHERE node_id=?", (node_id,)
    ).fetchone()
    if not node:
        raise KeyError(f"unknown canonical node: {node_id}")
    return _register_identity(
        db, "node", node_id, str(node[1]),
        {"modality": str(node[0]), "level": int(node[2])},
    )


def canonical_relation_id(db: sqlite3.Connection, relation_type: str) -> int:
    return _register_identity(db, "relation", relation_type, "relation", {})


def bind_surface(
    db: sqlite3.Connection,
    *,
    surface: str,
    concept_node_id: str,
    modality: str = Modality.TEXT.value,
    locale: str = "",
    surface_node_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> int:
    """Bind an external surface label to an already canonical concept identity.

    Multiple surfaces/languages may point to one concept.  No string label is
    allowed to define concept identity by itself.
    """
    concept_id = canonical_node_id(db, concept_node_id)
    db.execute(
        """INSERT OR REPLACE INTO canonical_surface_bindings(
             surface,modality,locale,concept_numeric_id,surface_node_id,metadata_json
           ) VALUES(?,?,?,?,?,?)""",
        (
            surface,
            modality,
            locale,
            concept_id,
            surface_node_id or "",
            json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True),
        ),
    )
    return concept_id


def resolve_surface(
    db: sqlite3.Connection,
    surface: str,
    *,
    modality: str = Modality.TEXT.value,
    locale: str = "",
) -> list[int]:
    ensure_canonical_id_schema(db)
    rows = db.execute(
        """SELECT DISTINCT concept_numeric_id FROM canonical_surface_bindings
           WHERE surface=? AND modality=? AND locale=?
           ORDER BY concept_numeric_id""",
        (surface, modality, locale),
    ).fetchall()
    return [int(r[0]) for r in rows]


def materialize_canonical_id_space(
    db: sqlite3.Connection,
    *,
    policy: CompilePolicy | None = None,
    auto_bind_node_surfaces: bool = False,
    metadata: dict[str, Any] | None = None,
) -> CanonicalIDBuild:
    """Compile the current canonical DB into a language-agnostic numeric ID graph.

    Existing Forge symbolic IDs remain as provenance/debug identities. Runtime
    knowledge is represented by numeric (subject,predicate,object) triples.
    """
    ensure_canonical_id_schema(db)
    policy = policy or CompilePolicy()

    node_rows = db.execute(
        "SELECT node_id,modality,kind,level,surface FROM nodes ORDER BY node_id"
    ).fetchall()
    for node_id, modality, kind, level, surface in node_rows:
        nid = _register_identity(
            db, "node", str(node_id), str(kind),
            {"modality": str(modality), "level": int(level)},
        )
        if auto_bind_node_surfaces and surface:
            db.execute(
                """INSERT OR IGNORE INTO canonical_surface_bindings(
                     surface,modality,locale,concept_numeric_id,surface_node_id,metadata_json
                   ) VALUES(?,?,?,?,?,?)""",
                (str(surface), str(modality), "", nid, str(node_id), '{}'),
            )

    relations = aggregate_relations(db, policy)
    relation_types = sorted({r.relation_type for r in relations})
    for relation_type in relation_types:
        canonical_relation_id(db, relation_type)

    policy_json = json.dumps(asdict(policy), sort_keys=True, separators=(",", ":"))
    db.execute(
        """INSERT INTO canonical_id_build_runs(
             format,version,policy_json,identity_count,node_identity_count,
             relation_identity_count,triple_count,surface_binding_count,digest,metadata_json
           ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
        (ID_SPACE_FORMAT, ID_SPACE_VERSION, policy_json, 0, 0, 0, 0, 0, "pending",
         json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True)),
    )
    run_id = int(db.execute("SELECT last_insert_rowid()").fetchone()[0])

    triples: list[CanonicalTriple] = []
    for rel in relations:
        triple = CanonicalTriple(
            canonical_node_id(db, rel.from_id),
            canonical_relation_id(db, rel.relation_type),
            canonical_node_id(db, rel.to_id),
            float(rel.score), float(rel.confidence),
            int(rel.source_count), int(rel.evidence_count),
        )
        triples.append(triple)
        db.execute(
            """INSERT INTO canonical_relation_triples(
                 canonical_id_run_id,subject_id,predicate_id,object_id,score,confidence,source_count,evidence_count
               ) VALUES(?,?,?,?,?,?,?,?)""",
            (run_id, triple.subject_id, triple.predicate_id, triple.object_id,
             triple.score, triple.confidence, triple.source_count, triple.evidence_count),
        )

    counts = {
        ns: int(db.execute("SELECT COUNT(*) FROM canonical_identities WHERE namespace=?", (ns,)).fetchone()[0])
        for ns in ("node", "relation")
    }
    surface_count = int(db.execute("SELECT COUNT(*) FROM canonical_surface_bindings").fetchone()[0])
    identity_count = int(db.execute("SELECT COUNT(*) FROM canonical_identities").fetchone()[0])
    sig = [
        (t.subject_id, t.predicate_id, t.object_id, round(t.score, 12), round(t.confidence, 12), t.source_count, t.evidence_count)
        for t in sorted(triples, key=lambda x: (x.subject_id, x.predicate_id, x.object_id))
    ]
    digest = hashlib.sha256(
        json.dumps(sig, separators=(",", ":"), sort_keys=False).encode("utf-8")
    ).hexdigest()
    db.execute(
        """UPDATE canonical_id_build_runs SET
             identity_count=?,node_identity_count=?,relation_identity_count=?,
             triple_count=?,surface_binding_count=?,digest=?
           WHERE canonical_id_run_id=?""",
        (identity_count, counts["node"], counts["relation"], len(triples), surface_count, digest, run_id),
    )
    db.commit()
    return CanonicalIDBuild(
        ID_SPACE_FORMAT, ID_SPACE_VERSION, run_id,
        identity_count, counts["node"], counts["relation"], len(triples), surface_count, digest,
    )


def latest_run_id(db: sqlite3.Connection) -> int:
    ensure_canonical_id_schema(db)
    row = db.execute("SELECT MAX(canonical_id_run_id) FROM canonical_id_build_runs").fetchone()
    if not row or row[0] is None:
        raise ValueError("no canonical ID-space build exists")
    return int(row[0])


def load_numeric_triples(db: sqlite3.Connection, run_id: int | None = None) -> list[CanonicalTriple]:
    rid = int(run_id or latest_run_id(db))
    rows = db.execute(
        """SELECT subject_id,predicate_id,object_id,score,confidence,source_count,evidence_count
           FROM canonical_relation_triples WHERE canonical_id_run_id=?
           ORDER BY subject_id,predicate_id,object_id""",
        (rid,),
    ).fetchall()
    return [CanonicalTriple(int(a), int(p), int(b), float(s), float(c), int(sc), int(ec))
            for a,p,b,s,c,sc,ec in rows]


def numeric_closure(
    db: sqlite3.Connection,
    root_id: int,
    *,
    run_id: int | None = None,
    max_depth: int = 4,
    predicate_ids: set[int] | None = None,
) -> list[CanonicalTriple]:
    """Return the outgoing relational closure of a numeric concept handle.

    This performs no semantic inference or inheritance.  It only exposes the
    graph addressed by the ID so later releases can collapse/rout it explicitly.
    """
    if max_depth < 0:
        raise ValueError("max_depth must be >= 0")
    triples = load_numeric_triples(db, run_id)
    by_subject: dict[int, list[CanonicalTriple]] = {}
    for t in triples:
        by_subject.setdefault(t.subject_id, []).append(t)
    frontier = {int(root_id)}
    seen_nodes = {int(root_id)}
    out: dict[tuple[int,int,int], CanonicalTriple] = {}
    for _ in range(max_depth + 1):
        next_frontier: set[int] = set()
        for subject in frontier:
            for t in by_subject.get(subject, []):
                if predicate_ids is not None and t.predicate_id not in predicate_ids:
                    continue
                out[(t.subject_id, t.predicate_id, t.object_id)] = t
                if t.object_id not in seen_nodes:
                    seen_nodes.add(t.object_id)
                    next_frontier.add(t.object_id)
        if not next_frontier:
            break
        frontier = next_frontier
    return [out[k] for k in sorted(out)]


def identity_record(db: sqlite3.Connection, numeric_id: int) -> CanonicalIdentity:
    row = db.execute(
        "SELECT numeric_id,namespace,symbolic_id,identity_kind FROM canonical_identities WHERE numeric_id=?",
        (int(numeric_id),),
    ).fetchone()
    if not row:
        raise KeyError(numeric_id)
    return CanonicalIdentity(int(row[0]), str(row[1]), str(row[2]), str(row[3]))


def _demo_node(node_id: str, kind: str = NodeKind.CONCEPT.value, level: int = 3) -> IRNode:
    return IRNode(node_id, Modality.CROSSMODAL.value, kind, level, surface=None)


def run_canonical_id_demo(out_dir: str | Path) -> dict[str, Any]:
    """Small language-separated proof that semantic identity lives in numeric relations."""
    from .database import CanonicalDB

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    db_path = out / "canonical-id-demo.sqlite"
    if db_path.exists():
        db_path.unlink()

    nodes = {
        "owl": _demo_node("concept:owl"),
        "bird": _demo_node("concept:bird"),
        "animal": _demo_node("concept:animal"),
        "fly": _demo_node("concept:fly", level=2),
        "feathers": _demo_node("concept:feathers", level=2),
    }
    src = SourceRef("v044-demo", "1", "canonical-id-demo")
    evidence = [
        RelationEvidence(nodes["owl"].node_id, nodes["bird"].node_id, RelationType.IS_A.value, 1.0, 1.0, src),
        RelationEvidence(nodes["bird"].node_id, nodes["animal"].node_id, RelationType.IS_A.value, 1.0, 1.0, src),
        RelationEvidence(nodes["owl"].node_id, nodes["fly"].node_id, RelationType.ACTION.value, 1.0, 1.0, src),
        RelationEvidence(nodes["bird"].node_id, nodes["feathers"].node_id, RelationType.PROPERTY.value, 1.0, 1.0, src),
    ]
    with CanonicalDB(db_path) as cdb:
        for node in nodes.values():
            cdb.upsert_node(node)
        for ev in evidence:
            cdb.add_evidence(ev)
        cdb.db.commit()
        owl_numeric = canonical_node_id(cdb.db, nodes["owl"].node_id)
        bind_surface(cdb.db, surface="Eule", concept_node_id=nodes["owl"].node_id, locale="de")
        bind_surface(cdb.db, surface="owl", concept_node_id=nodes["owl"].node_id, locale="en")
        build = materialize_canonical_id_space(cdb.db)
        owl_numeric_after = canonical_node_id(cdb.db, nodes["owl"].node_id)
        triples = load_numeric_triples(cdb.db, build.run_id)
        closure = numeric_closure(cdb.db, owl_numeric, run_id=build.run_id, max_depth=4)
        relation_ids = {
            r: canonical_relation_id(cdb.db, r)
            for r in (RelationType.IS_A.value, RelationType.ACTION.value, RelationType.PROPERTY.value)
        }
        report = {
            "format": "equinox-forge-canonical-id-demo-v1",
            "version": ID_SPACE_VERSION,
            "build": build.to_dict(),
            "controls": {
                "numeric_id_is_stable_across_materialization": owl_numeric == owl_numeric_after,
                "german_and_english_surface_share_concept": (
                    resolve_surface(cdb.db, "Eule", locale="de") == [owl_numeric]
                    and resolve_surface(cdb.db, "owl", locale="en") == [owl_numeric]
                ),
                "triples_are_numeric": all(
                    isinstance(x, int) for t in triples for x in (t.subject_id, t.predicate_id, t.object_id)
                ),
                "surface_strings_absent_from_triples": all(
                    "Eule" not in str(asdict(t)) and "owl" not in str(asdict(t)) for t in triples
                ),
                "closure_reaches_animal": canonical_node_id(cdb.db, nodes["animal"].node_id)
                    in {t.object_id for t in closure},
            },
            "numeric_ids": {
                key: canonical_node_id(cdb.db, node.node_id) for key, node in nodes.items()
            },
            "relation_ids": relation_ids,
            "triples": [asdict(t) for t in triples],
            "owl_closure": [asdict(t) for t in closure],
            "surface_resolution": {
                "Eule@de": resolve_surface(cdb.db, "Eule", locale="de"),
                "owl@en": resolve_surface(cdb.db, "owl", locale="en"),
            },
        }
        cdb.db.commit()

    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the Equinox v0.44 canonical numeric-ID space demo.")
    ap.add_argument("--out", default="out/v0_44_canonical_id_space")
    args = ap.parse_args(argv)
    report = run_canonical_id_demo(args.out)
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
