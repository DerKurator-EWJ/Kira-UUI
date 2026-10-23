from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
import hashlib
import json
import sqlite3

from .ir import RelationType, NodeKind

DEPENDENCY_FORMAT = "equinox-forge-dependency-graph-v1"
DEPENDENCY_VERSION = "0.11"

AUTO_KINDS = {
    "composition_member",
    "evidence_from_source",
    "evidence_to_source",
    "sense_target_node",
    "sense_context",
    "sense_prediction_template",
    "sense_output",
    "derived_output",
}


@dataclass(frozen=True, order=True)
class ArtifactRef:
    artifact_type: str
    artifact_id: str

    @property
    def key(self) -> str:
        return f"{self.artifact_type}:{self.artifact_id}"

    def to_dict(self) -> dict[str, str]:
        return {"artifact_type": self.artifact_type, "artifact_id": self.artifact_id, "key": self.key}


@dataclass
class DependencyImpact:
    roots: list[dict[str, str]]
    affected: list[dict[str, str]]
    affected_build_targets: list[str]
    affected_nodes: int
    affected_manual_artifacts: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def ensure_dependency_schema(db: sqlite3.Connection) -> None:
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS dependency_edges(
          dependent_type TEXT NOT NULL,
          dependent_id TEXT NOT NULL,
          dependency_type TEXT NOT NULL,
          dependency_id TEXT NOT NULL,
          dependency_kind TEXT NOT NULL,
          metadata_json TEXT NOT NULL DEFAULT '{}',
          PRIMARY KEY(dependent_type,dependent_id,dependency_type,dependency_id,dependency_kind)
        );
        CREATE INDEX IF NOT EXISTS idx_dependency_lookup
          ON dependency_edges(dependency_type,dependency_id,dependent_type,dependent_id);
        CREATE INDEX IF NOT EXISTS idx_dependent_lookup
          ON dependency_edges(dependent_type,dependent_id,dependency_type,dependency_id);

        CREATE TABLE IF NOT EXISTS build_targets(
          target_id TEXT PRIMARY KEY,
          builder TEXT NOT NULL,
          policy_json TEXT NOT NULL DEFAULT '{}',
          active INTEGER NOT NULL DEFAULT 1,
          last_output_hash TEXT NOT NULL DEFAULT '',
          metadata_json TEXT NOT NULL DEFAULT '{}',
          updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_build_targets_builder ON build_targets(builder,active,target_id);

        CREATE TABLE IF NOT EXISTS incremental_rebuild_runs(
          incremental_run_id INTEGER PRIMARY KEY AUTOINCREMENT,
          change_set_hash TEXT NOT NULL,
          source_policy_hash TEXT NOT NULL,
          roots_json TEXT NOT NULL,
          affected_json TEXT NOT NULL,
          rebuilt_json TEXT NOT NULL,
          deferred_json TEXT NOT NULL,
          output_hash TEXT NOT NULL,
          created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
          metadata_json TEXT NOT NULL DEFAULT '{}'
        );
        """
    )


def add_dependency(
    db: sqlite3.Connection,
    dependent: ArtifactRef,
    dependency: ArtifactRef,
    dependency_kind: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    if dependent == dependency:
        return
    db.execute(
        "INSERT OR REPLACE INTO dependency_edges(dependent_type,dependent_id,dependency_type,dependency_id,dependency_kind,metadata_json) VALUES(?,?,?,?,?,?)",
        (
            dependent.artifact_type,
            dependent.artifact_id,
            dependency.artifact_type,
            dependency.artifact_id,
            dependency_kind,
            json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True),
        ),
    )


def register_manual_dependency(
    db: sqlite3.Connection,
    dependent_type: str,
    dependent_id: str,
    dependency_type: str,
    dependency_id: str,
    dependency_kind: str = "manual",
    metadata: dict[str, Any] | None = None,
) -> None:
    ensure_dependency_schema(db)
    add_dependency(
        db,
        ArtifactRef(dependent_type, dependent_id),
        ArtifactRef(dependency_type, dependency_id),
        dependency_kind,
        metadata,
    )
    db.commit()


def register_build_target(
    db: sqlite3.Connection,
    target_id: str,
    builder: str,
    policy: dict[str, Any] | None = None,
    *,
    active: bool = True,
    last_output_hash: str = "",
    metadata: dict[str, Any] | None = None,
) -> None:
    ensure_dependency_schema(db)
    if not target_id or not builder:
        raise ValueError("target_id and builder are required")
    meta = {"format": DEPENDENCY_FORMAT, "version": DEPENDENCY_VERSION, **(metadata or {})}
    db.execute(
        "INSERT INTO build_targets(target_id,builder,policy_json,active,last_output_hash,metadata_json,updated_at) "
        "VALUES(?,?,?,?,?,?,CURRENT_TIMESTAMP) "
        "ON CONFLICT(target_id) DO UPDATE SET builder=excluded.builder,policy_json=excluded.policy_json,active=excluded.active,last_output_hash=excluded.last_output_hash,metadata_json=excluded.metadata_json,updated_at=CURRENT_TIMESTAMP",
        (
            target_id, builder, json.dumps(policy or {}, ensure_ascii=False, sort_keys=True),
            int(active), last_output_hash, json.dumps(meta, ensure_ascii=False, sort_keys=True),
        ),
    )


def register_sense_build_target(
    db: sqlite3.Connection,
    target_id: str,
    sense_policy: Any,
    consensus_policy: Any,
    *,
    track_stability: bool = True,
    last_output_hash: str = "",
) -> None:
    payload = {
        "sense_policy": asdict(sense_policy),
        "consensus_policy": asdict(consensus_policy),
        "track_stability": bool(track_stability),
    }
    register_build_target(db, target_id, "sense", payload, last_output_hash=last_output_hash)




def register_macro_build_target(
    db: sqlite3.Connection,
    target_id: str,
    macro_policy: Any,
    *,
    active: bool = True,
    last_output_hash: str = "",
) -> None:
    """Register a reproducible macro recipe plus explicit dependency boundary."""
    payload = asdict(macro_policy) if hasattr(macro_policy, "__dataclass_fields__") else dict(macro_policy)
    register_build_target(db, target_id, "macro", payload, active=active, last_output_hash=last_output_hash)
    # Replace only this builder's explicit input edges; unrelated manual/plugin edges survive.
    db.execute(
        "DELETE FROM dependency_edges WHERE dependent_type='build_target' AND dependent_id=? AND dependency_kind IN ('macro_input_target','macro_input_node')",
        (target_id,),
    )
    for upstream in sorted(set(payload.get("input_build_targets") or [])):
        add_dependency(db, ArtifactRef("build_target", target_id), ArtifactRef("build_target", upstream), "macro_input_target")
    for node_id in sorted(set(payload.get("input_nodes") or [])):
        add_dependency(db, ArtifactRef("build_target", target_id), ArtifactRef("node", node_id), "macro_input_node")
    db.commit()

def _members(db: sqlite3.Connection, node_id: str) -> list[str]:
    return [r[0] for r in db.execute(
        "SELECT child_id FROM node_members WHERE parent_id=? ORDER BY position", (node_id,)
    ).fetchall()]


def _matched_templates(db: sqlite3.Connection, contexts: list[str]) -> set[str]:
    if not contexts:
        return set()
    context_members = {c: _members(db, c) for c in contexts}
    templates = [(r[0], _members(db, r[0])) for r in db.execute(
        "SELECT node_id FROM nodes WHERE kind=? ORDER BY node_id", (NodeKind.TEMPLATE.value,)
    ).fetchall()]
    matched: set[str] = set()
    for tid, tm in templates:
        if not tm:
            continue
        for cm in context_members.values():
            if len(tm) <= len(cm) and any(cm[i:i+len(tm)] == tm for i in range(len(cm)-len(tm)+1)):
                matched.add(tid)
                break
    return matched


def refresh_dependency_index(db: sqlite3.Connection) -> dict[str, int]:
    """Rebuild only automatically-derived dependency edges.

    Manual/plugin edges survive. The index is conservative: source evidence invalidates
    both endpoints, composition flows upward through node_members, and registered sense
    targets depend on their contextual-use nodes and matching prediction templates.
    """
    ensure_dependency_schema(db)
    placeholders = ",".join("?" for _ in AUTO_KINDS)
    db.execute(f"DELETE FROM dependency_edges WHERE dependency_kind IN ({placeholders})", tuple(sorted(AUTO_KINDS)))

    composition_edges = 0
    for parent, child in db.execute("SELECT parent_id,child_id FROM node_members ORDER BY parent_id,position").fetchall():
        add_dependency(db, ArtifactRef("node", parent), ArtifactRef("node", child), "composition_member")
        composition_edges += 1

    evidence_edges = 0
    for a, b, src, ver in db.execute(
        "SELECT from_id,to_id,source_id,source_version FROM main.evidence ORDER BY evidence_id"
    ).fetchall():
        source = ArtifactRef("source", f"{src}@{ver}")
        add_dependency(db, ArtifactRef("node", a), source, "evidence_from_source")
        add_dependency(db, ArtifactRef("node", b), source, "evidence_to_source")
        evidence_edges += 2

    sense_edges = 0
    targets = [r[0] for r in db.execute(
        "SELECT target_id FROM build_targets WHERE active=1 AND builder='sense' ORDER BY target_id"
    ).fetchall()]
    for target_id in targets:
        bt = ArtifactRef("build_target", target_id)
        add_dependency(db, bt, ArtifactRef("node", target_id), "sense_target_node")
        sense_edges += 1
        contexts = [r[0] for r in db.execute(
            "SELECT DISTINCT to_id FROM main.evidence WHERE from_id=? AND relation_type=? ORDER BY to_id",
            (target_id, RelationType.CONTEXTUAL_USE.value),
        ).fetchall()]
        for ctx in contexts:
            add_dependency(db, bt, ArtifactRef("node", ctx), "sense_context")
            sense_edges += 1
        for tid in sorted(_matched_templates(db, contexts)):
            add_dependency(db, bt, ArtifactRef("node", tid), "sense_prediction_template")
            sense_edges += 1

        # Persisted derived concepts / hypothesis IDs are outputs of this build target.
        derived_ids = {r[0] for r in db.execute(
            "SELECT hypothesis_id FROM hypothesis_states WHERE target_id=?", (target_id,)
        ).fetchall()}
        for nid, meta_json in db.execute(
            "SELECT node_id,metadata_json FROM nodes WHERE kind=?", (NodeKind.CONCEPT.value,)
        ).fetchall():
            try:
                meta = json.loads(meta_json or "{}")
            except Exception:
                meta = {}
            if meta.get("provisional") and meta.get("target_id") == target_id:
                derived_ids.add(nid)
        for nid in sorted(derived_ids):
            add_dependency(db, ArtifactRef("node", nid), bt, "sense_output")
            sense_edges += 1

    derived_edges = 0
    try:
        rows = db.execute(
            "SELECT build_target_id,node_id FROM derived_artifacts WHERE node_id IS NOT NULL AND status!='stale' ORDER BY build_target_id,artifact_id"
        ).fetchall()
    except sqlite3.OperationalError:
        rows = []
    for target_id, node_id in rows:
        add_dependency(db, ArtifactRef("node", node_id), ArtifactRef("build_target", target_id), "derived_output")
        derived_edges += 1

    db.commit()
    return {
        "composition_edges": composition_edges,
        "evidence_source_edges": evidence_edges,
        "sense_edges": sense_edges,
        "derived_output_edges": derived_edges,
        "total_edges": int(db.execute("SELECT COUNT(*) FROM dependency_edges").fetchone()[0]),
        "sense_build_targets": len(targets),
        "registered_build_targets": int(db.execute("SELECT COUNT(*) FROM build_targets WHERE active=1").fetchone()[0]),
    }


def transitive_dependents(db: sqlite3.Connection, roots: Iterable[ArtifactRef]) -> set[ArtifactRef]:
    ensure_dependency_schema(db)
    seen: set[ArtifactRef] = set(roots)
    queue = list(sorted(seen))
    while queue:
        cur = queue.pop(0)
        rows = db.execute(
            "SELECT dependent_type,dependent_id FROM dependency_edges WHERE dependency_type=? AND dependency_id=? ORDER BY dependent_type,dependent_id",
            (cur.artifact_type, cur.artifact_id),
        ).fetchall()
        for typ, aid in rows:
            ref = ArtifactRef(typ, aid)
            if ref not in seen:
                seen.add(ref)
                queue.append(ref)
    return seen


def impact_from_roots(db: sqlite3.Connection, roots: Iterable[ArtifactRef], *, refresh: bool = True) -> DependencyImpact:
    if refresh:
        refresh_dependency_index(db)
    roots = sorted(set(roots))
    affected = transitive_dependents(db, roots)
    build_targets = sorted(r.artifact_id for r in affected if r.artifact_type == "build_target")
    manual = sorted(r.key for r in affected if r.artifact_type not in {"source", "node", "build_target"})
    return DependencyImpact(
        [r.to_dict() for r in roots],
        [r.to_dict() for r in sorted(affected)],
        build_targets,
        sum(1 for r in affected if r.artifact_type == "node"),
        manual,
    )


def source_roots(source_keys: Iterable[str]) -> list[ArtifactRef]:
    out = []
    for key in source_keys:
        key = key.strip()
        if "@" not in key:
            raise ValueError(f"source root must be SOURCE@VERSION: {key!r}")
        src, ver = key.rsplit("@", 1)
        if not src or not ver:
            raise ValueError(f"source root must be SOURCE@VERSION: {key!r}")
        out.append(ArtifactRef("source", f"{src}@{ver}"))
    return out


def build_target_policy(db: sqlite3.Connection, target_id: str) -> dict[str, Any]:
    row = db.execute(
        "SELECT builder,policy_json,active,last_output_hash,metadata_json FROM build_targets WHERE target_id=?",
        (target_id,),
    ).fetchone()
    if not row:
        raise KeyError(target_id)
    return {
        "target_id": target_id,
        "builder": row[0],
        "policy": json.loads(row[1] or "{}"),
        "active": bool(row[2]),
        "last_output_hash": row[3],
        "metadata": json.loads(row[4] or "{}"),
    }


def change_set_hash(roots: Iterable[ArtifactRef]) -> str:
    raw = json.dumps([r.to_dict() for r in sorted(set(roots))], sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def topological_build_targets(db: sqlite3.Connection, target_ids: Iterable[str]) -> tuple[list[str], list[str]]:
    """Return dependency-first order plus cycle members for affected build targets."""
    ids = set(target_ids)
    if not ids:
        return [], []
    deps: dict[str, set[str]] = {x: set() for x in ids}
    rows = db.execute(
        "SELECT dependent_id,dependency_id FROM dependency_edges WHERE dependent_type='build_target' AND dependency_type='build_target'"
    ).fetchall()
    for dependent, dependency in rows:
        if dependent in ids and dependency in ids:
            deps[dependent].add(dependency)
    order: list[str] = []
    remaining = {k:set(v) for k,v in deps.items()}
    while remaining:
        ready = sorted(k for k,v in remaining.items() if not v)
        if not ready:
            return order, sorted(remaining)
        for k in ready:
            order.append(k)
            del remaining[k]
        for v in remaining.values():
            v.difference_update(ready)
    return order, []
