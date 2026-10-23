from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable
import hashlib
import json
import sqlite3

from .ir import ForgeDocument, IRNode

ARTIFACT_STORE_FORMAT = "equinox-forge-derived-artifact-store-v1"
ARTIFACT_STORE_VERSION = "0.11"


@dataclass
class DerivedArtifact:
    build_target_id: str
    artifact_id: str
    builder: str
    artifact_type: str
    node_id: str | None
    content_hash: str
    status: str
    generation: int
    source_policy_hash: str
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def ensure_artifact_schema(db: sqlite3.Connection) -> None:
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS derived_artifacts(
          build_target_id TEXT NOT NULL,
          artifact_id TEXT NOT NULL,
          builder TEXT NOT NULL,
          artifact_type TEXT NOT NULL,
          node_id TEXT,
          content_hash TEXT NOT NULL,
          status TEXT NOT NULL,
          generation INTEGER NOT NULL DEFAULT 1,
          source_policy_hash TEXT NOT NULL DEFAULT '',
          payload_json TEXT NOT NULL DEFAULT '{}',
          created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
          updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
          PRIMARY KEY(build_target_id, artifact_id)
        );
        CREATE INDEX IF NOT EXISTS idx_derived_artifacts_target
          ON derived_artifacts(build_target_id,status,artifact_type,artifact_id);
        CREATE INDEX IF NOT EXISTS idx_derived_artifacts_node
          ON derived_artifacts(node_id,status,build_target_id);

        CREATE TABLE IF NOT EXISTS derived_artifact_versions(
          artifact_version_id INTEGER PRIMARY KEY AUTOINCREMENT,
          build_target_id TEXT NOT NULL,
          artifact_id TEXT NOT NULL,
          builder TEXT NOT NULL,
          artifact_type TEXT NOT NULL,
          node_id TEXT,
          content_hash TEXT NOT NULL,
          status TEXT NOT NULL,
          generation INTEGER NOT NULL,
          source_policy_hash TEXT NOT NULL DEFAULT '',
          payload_json TEXT NOT NULL DEFAULT '{}',
          created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_derived_artifact_versions_lookup
          ON derived_artifact_versions(build_target_id,artifact_id,generation);
        """
    )


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _node_payload(node: IRNode) -> dict[str, Any]:
    return asdict(node)


def _content_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _write_version(db: sqlite3.Connection, artifact: DerivedArtifact) -> None:
    db.execute(
        "INSERT INTO derived_artifact_versions(build_target_id,artifact_id,builder,artifact_type,node_id,content_hash,status,generation,source_policy_hash,payload_json) "
        "VALUES(?,?,?,?,?,?,?,?,?,?)",
        (
            artifact.build_target_id, artifact.artifact_id, artifact.builder,
            artifact.artifact_type, artifact.node_id, artifact.content_hash,
            artifact.status, artifact.generation, artifact.source_policy_hash,
            _stable_json(artifact.payload),
        ),
    )


def record_document_outputs(
    db: sqlite3.Connection,
    build_target_id: str,
    builder: str,
    doc: ForgeDocument,
    output_node_ids: Iterable[str],
    *,
    status_by_node: dict[str, str] | None = None,
    source_policy_hash: str = "",
    artifact_type: str = "node",
) -> list[DerivedArtifact]:
    """Persist current derived outputs and append immutable version history.

    Nodes that were outputs of this target but disappear from the new build are retained
    in history and marked `stale` in the current store. Canonical nodes are never deleted.
    """
    ensure_artifact_schema(db)
    status_by_node = status_by_node or {}
    nodes = {n.node_id: n for n in doc.nodes}
    output_ids = sorted(set(output_node_ids))
    missing = [nid for nid in output_ids if nid not in nodes]
    if missing:
        raise KeyError(f"output nodes missing from document: {missing[:4]}")

    existing = {
        row[0]: row
        for row in db.execute(
            "SELECT artifact_id,builder,artifact_type,node_id,content_hash,status,generation,source_policy_hash,payload_json "
            "FROM derived_artifacts WHERE build_target_id=?",
            (build_target_id,),
        ).fetchall()
    }
    current: list[DerivedArtifact] = []

    for nid in output_ids:
        node = nodes[nid]
        payload = {
            "format": ARTIFACT_STORE_FORMAT,
            "version": ARTIFACT_STORE_VERSION,
            "node": _node_payload(node),
        }
        h = _content_hash(payload)
        status = status_by_node.get(nid, "active")
        old = existing.get(nid)
        generation = 1 if old is None else int(old[6])
        changed = old is None or old[4] != h or old[5] != status or old[7] != source_policy_hash
        if old is not None and changed:
            generation += 1
        artifact = DerivedArtifact(
            build_target_id, nid, builder, artifact_type, nid, h, status,
            generation, source_policy_hash, payload,
        )
        db.execute(
            "INSERT INTO derived_artifacts(build_target_id,artifact_id,builder,artifact_type,node_id,content_hash,status,generation,source_policy_hash,payload_json,updated_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP) "
            "ON CONFLICT(build_target_id,artifact_id) DO UPDATE SET builder=excluded.builder,artifact_type=excluded.artifact_type,node_id=excluded.node_id,content_hash=excluded.content_hash,status=excluded.status,generation=excluded.generation,source_policy_hash=excluded.source_policy_hash,payload_json=excluded.payload_json,updated_at=CURRENT_TIMESTAMP",
            (
                artifact.build_target_id, artifact.artifact_id, artifact.builder,
                artifact.artifact_type, artifact.node_id, artifact.content_hash,
                artifact.status, artifact.generation, artifact.source_policy_hash,
                _stable_json(artifact.payload),
            ),
        )
        if changed:
            _write_version(db, artifact)
        current.append(artifact)

    output_set = set(output_ids)
    for aid, old in sorted(existing.items()):
        if aid in output_set or old[5] == "stale":
            continue
        generation = int(old[6]) + 1
        payload = json.loads(old[8] or "{}")
        stale = DerivedArtifact(
            build_target_id, aid, old[1], old[2], old[3], old[4], "stale",
            generation, source_policy_hash or old[7], payload,
        )
        db.execute(
            "UPDATE derived_artifacts SET status='stale',generation=?,source_policy_hash=?,updated_at=CURRENT_TIMESTAMP WHERE build_target_id=? AND artifact_id=?",
            (generation, stale.source_policy_hash, build_target_id, aid),
        )
        _write_version(db, stale)

    db.commit()
    return current


def list_artifacts(
    db: sqlite3.Connection,
    *,
    build_target_id: str | None = None,
    statuses: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    ensure_artifact_schema(db)
    clauses: list[str] = []
    args: list[Any] = []
    if build_target_id is not None:
        clauses.append("build_target_id=?"); args.append(build_target_id)
    statuses = list(statuses or [])
    if statuses:
        clauses.append("status IN (%s)" % ",".join("?" for _ in statuses)); args.extend(statuses)
    sql = "SELECT build_target_id,artifact_id,builder,artifact_type,node_id,content_hash,status,generation,source_policy_hash,payload_json,created_at,updated_at FROM derived_artifacts"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY build_target_id,artifact_id"
    out = []
    for r in db.execute(sql, tuple(args)).fetchall():
        out.append({
            "build_target_id": r[0], "artifact_id": r[1], "builder": r[2],
            "artifact_type": r[3], "node_id": r[4], "content_hash": r[5],
            "status": r[6], "generation": int(r[7]), "source_policy_hash": r[8],
            "payload": json.loads(r[9] or "{}"), "created_at": r[10], "updated_at": r[11],
        })
    return out


def current_output_nodes(
    db: sqlite3.Connection,
    build_target_id: str,
    allowed_statuses: Iterable[str] = ("stable", "active"),
) -> list[tuple[str, str]]:
    ensure_artifact_schema(db)
    statuses = tuple(allowed_statuses)
    if not statuses:
        return []
    rows = db.execute(
        "SELECT node_id,status FROM derived_artifacts WHERE build_target_id=? AND node_id IS NOT NULL AND status IN (%s) ORDER BY artifact_id"
        % ",".join("?" for _ in statuses),
        (build_target_id, *statuses),
    ).fetchall()
    return [(r[0], r[1]) for r in rows]
