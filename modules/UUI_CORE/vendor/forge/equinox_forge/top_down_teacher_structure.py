from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable
import argparse
import hashlib
import json
import math
import sqlite3

from .canonical import normalize_surface
from .canonical_id_space import (
    bind_surface,
    canonical_node_id,
    load_numeric_triples,
    materialize_canonical_id_space,
)
from .database import CanonicalDB
from .ir import IRNode, Modality, NodeKind, RelationEvidence, RelationType, SourceRef
from .recursive_macro_collapse import (
    RecursiveMacroPolicy,
    macro_members,
    materialize_recursive_macro_collapse,
)
from .structural_closure import StructuralClosurePolicy, materialize_structural_closures
from .teacher_behavior import CausalTeacherBackend, TeacherCandidate

TOP_DOWN_TEACHER_FORMAT = "equinox-top-down-teacher-structure-v1"
TOP_DOWN_TEACHER_VERSION = "0.47"


@dataclass(frozen=True)
class AttributeProbeSpec:
    relation_type: str
    prompt_template: str

    def validate(self) -> None:
        if "{surface}" not in self.prompt_template:
            raise ValueError("attribute prompt_template must contain {surface}")
        if self.relation_type not in {RelationType.PROPERTY.value, RelationType.ACTION.value}:
            raise ValueError("v0.47 attribute probes are restricted to property/action")


@dataclass(frozen=True)
class TopDownTeacherPolicy:
    parent_prompt_template: str = "{surface} is a"
    attribute_probes: tuple[AttributeProbeSpec, ...] = ()
    stop_surfaces: tuple[str, ...] = ()
    max_parent_depth: int = 8
    top_k: int = 4
    temperature: float = 1.0
    min_probability: float = 0.55
    evidence_confidence: float = 0.95
    extraction_namespace: str = "default"
    attributes_on_internal_concepts_only: bool = True

    def validate(self) -> None:
        if "{surface}" not in self.parent_prompt_template:
            raise ValueError("parent_prompt_template must contain {surface}")
        if self.max_parent_depth < 1:
            raise ValueError("max_parent_depth must be >= 1")
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and > 0")
        if not 0.0 <= self.min_probability <= 1.0:
            raise ValueError("min_probability must be in [0,1]")
        if not 0.0 <= self.evidence_confidence <= 1.0:
            raise ValueError("evidence_confidence must be in [0,1]")
        if not self.extraction_namespace:
            raise ValueError("extraction_namespace must not be empty")
        for spec in self.attribute_probes:
            spec.validate()


@dataclass(frozen=True)
class TeacherParentObservation:
    child_surface: str
    parent_surface: str
    probability: float
    prompt: str
    discovery_depth: int


@dataclass(frozen=True)
class TeacherAttributeObservation:
    subject_surface: str
    relation_type: str
    object_surface: str
    probability: float
    prompt: str


@dataclass
class TopDownTeacherBuild:
    format: str
    version: str
    run_id: int
    source_id: str
    source_version: str
    backend_id: str
    seed_count: int
    concept_count: int
    root_count: int
    parent_edge_count: int
    attribute_edge_count: int
    max_hierarchy_depth: int
    teacher_queries: int
    digest: str
    top_down_order: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def ensure_top_down_teacher_schema(db: sqlite3.Connection) -> None:
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS top_down_teacher_runs(
          top_down_teacher_run_id INTEGER PRIMARY KEY AUTOINCREMENT,
          format TEXT NOT NULL,
          version TEXT NOT NULL,
          source_id TEXT NOT NULL,
          source_version TEXT NOT NULL,
          backend_id TEXT NOT NULL,
          policy_json TEXT NOT NULL,
          seed_count INTEGER NOT NULL,
          concept_count INTEGER NOT NULL,
          root_count INTEGER NOT NULL,
          parent_edge_count INTEGER NOT NULL,
          attribute_edge_count INTEGER NOT NULL,
          max_hierarchy_depth INTEGER NOT NULL,
          teacher_queries INTEGER NOT NULL,
          digest TEXT NOT NULL,
          metadata_json TEXT NOT NULL DEFAULT '{}',
          created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS top_down_teacher_parent_edges(
          top_down_teacher_run_id INTEGER NOT NULL,
          child_surface TEXT NOT NULL,
          parent_surface TEXT NOT NULL,
          probability REAL NOT NULL,
          prompt TEXT NOT NULL,
          discovery_depth INTEGER NOT NULL,
          PRIMARY KEY(top_down_teacher_run_id,child_surface)
        );

        CREATE TABLE IF NOT EXISTS top_down_teacher_attribute_edges(
          top_down_teacher_run_id INTEGER NOT NULL,
          subject_surface TEXT NOT NULL,
          relation_type TEXT NOT NULL,
          object_surface TEXT NOT NULL,
          probability REAL NOT NULL,
          prompt TEXT NOT NULL,
          PRIMARY KEY(top_down_teacher_run_id,subject_surface,relation_type)
        );
        """
    )


def _clean_surface(text: str) -> str:
    return normalize_surface(str(text)).strip()


def _best_candidate(
    backend: CausalTeacherBackend,
    prompt: str,
    policy: TopDownTeacherPolicy,
    *,
    reject: set[str] | None = None,
) -> TeacherCandidate | None:
    reject = {_clean_surface(x) for x in (reject or set())}
    for cand in backend.next_token(prompt, policy.top_k, policy.temperature):
        if cand.special or float(cand.probability) + 1e-15 < policy.min_probability:
            continue
        surface = _clean_surface(cand.surface)
        if not surface or surface in reject:
            continue
        return TeacherCandidate(
            int(cand.token_id), str(cand.raw_token), surface,
            float(cand.probability), None if cand.logit is None else float(cand.logit), bool(cand.special)
        )
    return None


def _discover_parent_forest(
    backend: CausalTeacherBackend,
    seed_surfaces: Iterable[str],
    policy: TopDownTeacherPolicy,
) -> tuple[list[TeacherParentObservation], int]:
    stop = {_clean_surface(x) for x in policy.stop_surfaces}
    seeds = sorted({_clean_surface(x) for x in seed_surfaces if _clean_surface(x)})
    queue: list[tuple[str, int]] = [(x, 0) for x in seeds]
    queried: set[str] = set()
    parent_by_child: dict[str, TeacherParentObservation] = {}
    query_count = 0

    while queue:
        child, depth = queue.pop(0)
        if child in queried or child in stop or depth >= policy.max_parent_depth:
            continue
        queried.add(child)
        prompt = policy.parent_prompt_template.format(surface=child)
        query_count += 1
        cand = _best_candidate(backend, prompt, policy, reject={child})
        if cand is None:
            continue
        parent = _clean_surface(cand.surface)
        obs = TeacherParentObservation(child, parent, float(cand.probability), prompt, depth)
        parent_by_child[child] = obs
        if parent not in queried and parent not in stop:
            queue.append((parent, depth + 1))

    return [parent_by_child[k] for k in sorted(parent_by_child)], query_count


def _validate_parent_forest(parent_edges: Iterable[TeacherParentObservation]) -> dict[str, str]:
    parent = {e.child_surface: e.parent_surface for e in parent_edges}
    for start in sorted(parent):
        seen: set[str] = set()
        cur = start
        while cur in parent:
            if cur in seen:
                raise ValueError(f"teacher hierarchy cycle detected at surface {cur!r}")
            seen.add(cur)
            cur = parent[cur]
    return parent


def _hierarchy_depths(parent: dict[str, str]) -> tuple[dict[str, int], list[str]]:
    all_surfaces = set(parent) | set(parent.values())
    roots = sorted(x for x in all_surfaces if x not in parent)
    if not all_surfaces:
        return {}, []
    if not roots:
        raise ValueError("teacher hierarchy has no root")

    children: dict[str, list[str]] = {}
    for child, par in parent.items():
        children.setdefault(par, []).append(child)
    for values in children.values():
        values.sort()

    depth: dict[str, int] = {}
    frontier = [(root, 0) for root in roots]
    while frontier:
        node, d = frontier.pop(0)
        old = depth.get(node)
        if old is not None and old <= d:
            continue
        depth[node] = d
        frontier.extend((child, d + 1) for child in children.get(node, []))
    return depth, roots


def _discover_attributes(
    backend: CausalTeacherBackend,
    concept_surfaces: Iterable[str],
    policy: TopDownTeacherPolicy,
) -> tuple[list[TeacherAttributeObservation], int]:
    rows: list[TeacherAttributeObservation] = []
    queries = 0
    for surface in sorted(set(concept_surfaces)):
        for spec in policy.attribute_probes:
            prompt = spec.prompt_template.format(surface=surface)
            queries += 1
            cand = _best_candidate(backend, prompt, policy, reject={surface})
            if cand is None:
                continue
            rows.append(TeacherAttributeObservation(
                surface, spec.relation_type, _clean_surface(cand.surface),
                float(cand.probability), prompt,
            ))
    return rows, queries


def _concept_node_id(namespace: str, surface: str) -> str:
    raw = f"{namespace}\0concept\0{surface}".encode("utf-8")
    return "teacher_concept:" + hashlib.sha256(raw).hexdigest()[:24]


def _value_node_id(namespace: str, relation_type: str, surface: str) -> str:
    raw = f"{namespace}\0value\0{relation_type}\0{surface}".encode("utf-8")
    return "teacher_value:" + hashlib.sha256(raw).hexdigest()[:24]


def extract_top_down_teacher_structure(
    db: sqlite3.Connection,
    backend: CausalTeacherBackend,
    *,
    seed_surfaces: Iterable[str],
    source_id: str,
    source_version: str = "",
    policy: TopDownTeacherPolicy | None = None,
    metadata: dict[str, Any] | None = None,
) -> TopDownTeacherBuild:
    """Extract a teacher-owned hierarchy into Forge without domain ontology in code.

    Phase 1 observes generic parent/attribute probes. Phase 2 validates the resulting
    forest and materializes it root-to-leaf into canonical concept nodes. Concrete
    category names remain teacher evidence/surface bindings; Forge only knows the
    generic operations IS_A, PROPERTY and ACTION.
    """
    policy = policy or TopDownTeacherPolicy()
    policy.validate()
    ensure_top_down_teacher_schema(db)

    manifest = backend.manifest()
    manifest_json = json.dumps(manifest, ensure_ascii=False, sort_keys=True)
    adapter_hash = hashlib.sha256(manifest_json.encode("utf-8")).hexdigest()
    db.execute(
        """INSERT OR REPLACE INTO source_models(
             source_id,source_version,adapter_id,adapter_hash,model_type,manifest_json
           ) VALUES(?,?,?,?,?,?)""",
        (source_id, source_version, str(backend.backend_id), adapter_hash,
         "teacher_structure", manifest_json),
    )

    seeds = sorted({_clean_surface(x) for x in seed_surfaces if _clean_surface(x)})
    if not seeds:
        raise ValueError("seed_surfaces must not be empty")

    parent_rows, parent_queries = _discover_parent_forest(backend, seeds, policy)
    parent = _validate_parent_forest(parent_rows)
    depth, roots = _hierarchy_depths(parent)
    all_concepts = sorted(set(depth) | set(seeds))

    if policy.attributes_on_internal_concepts_only:
        # Internal concepts are both a child of something more general and a parent
        # of something more specific.  This keeps top-down attributes attached to
        # established abstraction levels instead of probing every leaf blindly.
        attribute_subjects = sorted(set(parent) & set(parent.values()))
    else:
        attribute_subjects = all_concepts
    attr_rows, attr_queries = _discover_attributes(backend, attribute_subjects, policy)
    total_queries = parent_queries + attr_queries
    top_down = sorted(all_concepts, key=lambda x: (depth.get(x, 10**9), x))

    payload = {
        "parents": [asdict(x) for x in parent_rows],
        "attributes": [asdict(x) for x in attr_rows],
        "roots": roots,
        "order": top_down,
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    db.execute(
        """INSERT INTO top_down_teacher_runs(
             format,version,source_id,source_version,backend_id,policy_json,
             seed_count,concept_count,root_count,parent_edge_count,attribute_edge_count,
             max_hierarchy_depth,teacher_queries,digest,metadata_json
           ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            TOP_DOWN_TEACHER_FORMAT, TOP_DOWN_TEACHER_VERSION, source_id, source_version,
            str(backend.backend_id), json.dumps(asdict(policy), ensure_ascii=False, sort_keys=True, default=asdict),
            len(seeds), len(all_concepts), len(roots), len(parent_rows), len(attr_rows),
            max(depth.values(), default=0), total_queries, digest,
            json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True),
        ),
    )
    run_id = int(db.execute("SELECT last_insert_rowid()").fetchone()[0])

    for row in parent_rows:
        db.execute(
            """INSERT INTO top_down_teacher_parent_edges(
                 top_down_teacher_run_id,child_surface,parent_surface,probability,prompt,discovery_depth
               ) VALUES(?,?,?,?,?,?)""",
            (run_id, row.child_surface, row.parent_surface, row.probability, row.prompt, row.discovery_depth),
        )
    for row in attr_rows:
        db.execute(
            """INSERT INTO top_down_teacher_attribute_edges(
                 top_down_teacher_run_id,subject_surface,relation_type,object_surface,probability,prompt
               ) VALUES(?,?,?,?,?,?)""",
            (run_id, row.subject_surface, row.relation_type, row.object_surface, row.probability, row.prompt),
        )

    src = SourceRef(source_id, source_version, "teacher_top_down_structure")
    concept_nodes: dict[str, IRNode] = {}
    for surface in top_down:
        concept_nodes[surface] = IRNode(
            _concept_node_id(policy.extraction_namespace, surface),
            Modality.CROSSMODAL.value,
            NodeKind.CONCEPT.value,
            3,
            surface=None,
            metadata={
                "teacher_surface": surface,
                "teacher_structure_depth": int(depth.get(surface, 0)),
                "top_down_teacher_run_id": run_id,
            },
        )
        node = concept_nodes[surface]
        db.execute(
            "INSERT OR IGNORE INTO nodes(node_id,modality,kind,level,surface,metadata_json) VALUES(?,?,?,?,?,?)",
            (node.node_id, node.modality, node.kind, node.level, node.surface,
             json.dumps(node.metadata, ensure_ascii=False, sort_keys=True)),
        )

    # Materialize generic hierarchy relations after parent concepts already exist.
    for row in parent_rows:
        child = concept_nodes[row.child_surface]
        par = concept_nodes[row.parent_surface]
        ev = RelationEvidence(
            child.node_id, par.node_id, RelationType.IS_A.value,
            row.probability, policy.evidence_confidence, src,
            {
                "evidence_kind": "teacher_structure_parent",
                "prompt": row.prompt,
                "teacher_probability": row.probability,
                "teacher_structure_run_id": run_id,
            },
        )
        # Duplicate CanonicalDB.add_evidence semantics locally because this API accepts sqlite3.
        metadata_json = json.dumps(ev.metadata, ensure_ascii=False, sort_keys=True)
        raw = json.dumps({
            "from": ev.from_id, "to": ev.to_id, "type": ev.relation_type,
            "score": float(ev.score), "confidence": float(ev.confidence),
            "source": ev.source.source_id, "version": ev.source.source_version,
            "extraction": ev.source.extraction, "metadata": ev.metadata,
        }, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
        fp = hashlib.sha256(raw).hexdigest()
        db.execute(
            """INSERT OR IGNORE INTO evidence(
                 evidence_fingerprint,from_id,to_id,relation_type,score,confidence,
                 source_id,source_version,extraction,metadata_json
               ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (fp, ev.from_id, ev.to_id, ev.relation_type, ev.score, ev.confidence,
             ev.source.source_id, ev.source.source_version, ev.source.extraction, metadata_json),
        )

    value_nodes: dict[tuple[str, str], IRNode] = {}
    for row in attr_rows:
        key = (row.relation_type, row.object_surface)
        if key not in value_nodes:
            n = IRNode(
                _value_node_id(policy.extraction_namespace, row.relation_type, row.object_surface),
                Modality.CROSSMODAL.value, NodeKind.CONCEPT.value, 2, surface=None,
                metadata={"teacher_surface": row.object_surface, "semantic_role": row.relation_type},
            )
            value_nodes[key] = n
            db.execute(
                "INSERT OR IGNORE INTO nodes(node_id,modality,kind,level,surface,metadata_json) VALUES(?,?,?,?,?,?)",
                (n.node_id, n.modality, n.kind, n.level, n.surface,
                 json.dumps(n.metadata, ensure_ascii=False, sort_keys=True)),
            )
        subj = concept_nodes[row.subject_surface]
        obj = value_nodes[key]
        metadata_row = {
            "evidence_kind": "teacher_structure_attribute",
            "prompt": row.prompt,
            "teacher_probability": row.probability,
            "teacher_structure_run_id": run_id,
        }
        raw = json.dumps({
            "from": subj.node_id, "to": obj.node_id, "type": row.relation_type,
            "score": float(row.probability), "confidence": float(policy.evidence_confidence),
            "source": source_id, "version": source_version,
            "extraction": "teacher_top_down_structure", "metadata": metadata_row,
        }, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
        fp = hashlib.sha256(raw).hexdigest()
        db.execute(
            """INSERT OR IGNORE INTO evidence(
                 evidence_fingerprint,from_id,to_id,relation_type,score,confidence,
                 source_id,source_version,extraction,metadata_json
               ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (fp, subj.node_id, obj.node_id, row.relation_type, row.probability,
             policy.evidence_confidence, source_id, source_version,
             "teacher_top_down_structure", json.dumps(metadata_row, ensure_ascii=False, sort_keys=True)),
        )

    # Surfaces are labels/routing entries only; concept identity is already separate.
    for surface in top_down:
        bind_surface(
            db, surface=surface, concept_node_id=concept_nodes[surface].node_id,
            modality=Modality.TEXT.value, locale="",
            metadata={"source": "teacher_top_down_structure", "run_id": run_id},
        )
    for (rel, surface), node in sorted(value_nodes.items()):
        bind_surface(
            db, surface=surface, concept_node_id=node.node_id,
            modality=Modality.TEXT.value, locale="",
            metadata={"source": "teacher_top_down_structure", "semantic_role": rel, "run_id": run_id},
        )

    db.commit()
    return TopDownTeacherBuild(
        TOP_DOWN_TEACHER_FORMAT, TOP_DOWN_TEACHER_VERSION, run_id,
        source_id, source_version, str(backend.backend_id), len(seeds), len(all_concepts),
        len(roots), len(parent_rows), len(attr_rows), max(depth.values(), default=0),
        total_queries, digest, top_down,
    )


def load_parent_edges(db: sqlite3.Connection, run_id: int) -> list[TeacherParentObservation]:
    rows = db.execute(
        """SELECT child_surface,parent_surface,probability,prompt,discovery_depth
           FROM top_down_teacher_parent_edges WHERE top_down_teacher_run_id=?
           ORDER BY child_surface""", (int(run_id),)
    ).fetchall()
    return [TeacherParentObservation(str(a), str(b), float(p), str(pr), int(d)) for a,b,p,pr,d in rows]


def hierarchy_chain(db: sqlite3.Connection, run_id: int, leaf_surface: str) -> list[str]:
    parent = {x.child_surface: x.parent_surface for x in load_parent_edges(db, run_id)}
    cur = _clean_surface(leaf_surface)
    chain = [cur]
    seen = {cur}
    while cur in parent:
        cur = parent[cur]
        if cur in seen:
            raise ValueError("cycle while reading hierarchy chain")
        seen.add(cur)
        chain.append(cur)
    return list(reversed(chain))


def _fixture_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "v0_47_top_down_teacher_fixture.json"


def run_top_down_teacher_demo(out_dir: str | Path, fixture_path: str | Path | None = None) -> dict[str, Any]:
    """Controlled neural-teacher proof for the generic v0.47 extractor.

    Concrete animal/category data is loaded from the benchmark fixture, not encoded
    in the extractor. The same extractor accepts any CausalTeacherBackend, including
    the local Hugging Face backend when transformers/model files are available.
    """
    from .real_teacher_domain_trial import _NeuralDomainTeacher

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fixture = json.loads(Path(fixture_path or _fixture_path()).read_text(encoding="utf-8"))
    teacher = _NeuralDomainTeacher(
        [(str(a), str(b)) for a,b in fixture["training_pairs"]],
        out / "teacher", seed=4701, max_epochs=700,
    )
    policy = TopDownTeacherPolicy(
        parent_prompt_template=str(fixture["parent_probe_template"]),
        attribute_probes=tuple(AttributeProbeSpec(str(x["relation_type"]), str(x["prompt_template"])) for x in fixture["attribute_probes"]),
        stop_surfaces=tuple(str(x) for x in fixture["stop_surfaces"]),
        max_parent_depth=8,
        top_k=4,
        temperature=1.0,
        min_probability=0.55,
        evidence_confidence=0.95,
        extraction_namespace="v047-demo",
        attributes_on_internal_concepts_only=True,
    )

    db_path = out / "teacher-structure.sqlite"
    if db_path.exists():
        db_path.unlink()
    with CanonicalDB(db_path) as cdb:
        build = extract_top_down_teacher_structure(
            cdb.db, teacher, seed_surfaces=fixture["seed_surfaces"],
            source_id="v047-neural-teacher", source_version="1", policy=policy,
            metadata={"fixture_format": fixture["format"]},
        )
        parent_edges = load_parent_edges(cdb.db, build.run_id)
        canonical = materialize_canonical_id_space(cdb.db)
        seed_node_ids = {
            s: _concept_node_id(policy.extraction_namespace, str(s)) for s in fixture["seed_surfaces"]
        }
        seed_numeric = {s: canonical_node_id(cdb.db, nid) for s,nid in seed_node_ids.items()}
        closure = materialize_structural_closures(
            cdb.db, canonical_run_id=canonical.run_id, root_ids=seed_numeric.values(),
            policy=StructuralClosurePolicy(max_parent_depth=8),
        )
        macro = materialize_recursive_macro_collapse(
            cdb.db, canonical_run_id=canonical.run_id, structural_closure_run_id=closure.run_id,
            root_ids=seed_numeric.values(),
            policy=RecursiveMacroPolicy(min_members=2, min_common_facts=2, max_levels=4),
        )
        chains = {s: hierarchy_chain(cdb.db, build.run_id, s) for s in fixture["seed_surfaces"]}
        actual_edges = sorted((x.child_surface, x.parent_surface) for x in parent_edges)
        expected_edges = sorted(tuple(x) for x in fixture["expected_parent_edges"])
        roots = sorted(set(build.top_down_order) - {x.child_surface for x in parent_edges})
        macro_rows = [
            {"macro_id": int(mid), "level": int(level), "members": macro_members(cdb.db, int(mid), run_id=macro.run_id)}
            for mid,level in cdb.db.execute(
                "SELECT macro_id,level FROM recursive_macros WHERE recursive_macro_run_id=? ORDER BY level,macro_id",
                (macro.run_id,),
            ).fetchall()
        ]
        report = {
            "format": "equinox-v047-top-down-teacher-demo-v1",
            "version": TOP_DOWN_TEACHER_VERSION,
            "teacher": teacher.manifest(),
            "teacher_structure_build": build.to_dict(),
            "canonical_build": canonical.to_dict(),
            "closure_build": closure.to_dict(),
            "macro_build": macro.to_dict(),
            "top_down_order": build.top_down_order,
            "parent_edges": [asdict(x) for x in parent_edges],
            "chains": chains,
            "numeric_seed_ids": seed_numeric,
            "macro_rows": macro_rows,
            "controls": {
                "teacher_is_neural": teacher.manifest().get("parameters", 0) > 0,
                "source_manifest_persisted": cdb.db.execute(
                    "SELECT COUNT(*) FROM source_models WHERE source_id=? AND source_version=?",
                    ("v047-neural-teacher", "1"),
                ).fetchone()[0] == 1,
                "extractor_matches_teacher_parent_structure": actual_edges == expected_edges,
                "single_expected_root": roots == sorted(fixture["expected_roots"]),
                "top_down_order_is_root_first": bool(build.top_down_order) and build.top_down_order[0] in fixture["expected_roots"],
                "leaf_chains_match_fixture": chains == fixture["expected_leaf_chains"],
                "canonical_graph_is_numeric": all(
                    isinstance(v, int) for t in load_numeric_triples(cdb.db, canonical.run_id)
                    for v in (t.subject_id, t.predicate_id, t.object_id)
                ),
                "closure_materialized_for_all_seeds": closure.root_count == len(fixture["seed_surfaces"]),
                "recursive_macro_collapse_reuses_extracted_structure": macro.levels_built >= 2 and macro.macro_count >= 3,
                "concrete_ontology_absent_from_extractor_policy": all(
                    str(term).lower() not in json.dumps(asdict(TopDownTeacherPolicy()), ensure_ascii=False).lower()
                    for term in sorted({
                        str(x) for edge in fixture["expected_parent_edges"] for x in edge
                    })
                ),
            },
        }
        cdb.db.commit()

    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Extract a top-down Equinox ID hierarchy from teacher evidence (v0.47).")
    ap.add_argument("--out", default="out/v0_47_top_down_teacher_structure")
    ap.add_argument("--fixture", default=None, help="optional controlled benchmark fixture JSON")
    args = ap.parse_args(argv)
    report = run_top_down_teacher_demo(args.out, args.fixture)
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
