from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, asdict
from typing import Any, Iterator
import hashlib
import json
import sqlite3

from .ir import ForgeDocument
from .sense_compiler import SenseCompilePolicy, compile_provisional_senses
from .consensus import ConsensusPolicy, track_hypothesis_run, ConsensusReport

REBUILD_FORMAT = "equinox-forge-rebuild-v1"
REBUILD_VERSION = "0.11"


@dataclass(frozen=True)
class SourcePolicy:
    source_id: str
    source_version: str
    enabled: bool = True
    trust_weight: float = 1.0
    note: str = ""
    metadata: dict[str, Any] | None = None

    def validate(self) -> None:
        if not self.source_id:
            raise ValueError("source_id is required")
        if not self.source_version:
            raise ValueError("source_version is required")
        if not 0.0 <= float(self.trust_weight) <= 1.0:
            raise ValueError("trust_weight must be in [0,1]")


@dataclass
class RebuildReport:
    format: str
    version: str
    target_id: str
    source_policy_hash: str
    source_policies: list[dict[str, Any]]
    raw_evidence_count: int
    effective_evidence_count: int
    disabled_source_versions: int
    reweighted_source_versions: int
    sense_report: dict[str, Any]
    consensus_report: dict[str, Any] | None
    rebuild_run_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def set_source_policy(
    db: sqlite3.Connection,
    source_id: str,
    source_version: str,
    *,
    enabled: bool = True,
    trust_weight: float = 1.0,
    note: str = "",
    metadata: dict[str, Any] | None = None,
) -> SourcePolicy:
    policy = SourcePolicy(source_id, source_version, enabled, trust_weight, note, metadata or {})
    policy.validate()
    db.execute(
        "INSERT INTO source_policies(source_id,source_version,enabled,trust_weight,note,metadata_json,updated_at) "
        "VALUES(?,?,?,?,?,?,CURRENT_TIMESTAMP) "
        "ON CONFLICT(source_id,source_version) DO UPDATE SET enabled=excluded.enabled,trust_weight=excluded.trust_weight,note=excluded.note,metadata_json=excluded.metadata_json,updated_at=CURRENT_TIMESTAMP",
        (source_id, source_version, int(enabled), float(trust_weight), note, json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True)),
    )
    db.commit()
    return policy


def clear_source_policy(db: sqlite3.Connection, source_id: str, source_version: str) -> None:
    db.execute("DELETE FROM source_policies WHERE source_id=? AND source_version=?", (source_id, source_version))
    db.commit()


def replace_source_version(
    db: sqlite3.Connection,
    source_id: str,
    old_version: str,
    new_version: str,
    *,
    new_trust_weight: float = 1.0,
    note: str = "",
) -> tuple[SourcePolicy, SourcePolicy]:
    if old_version == new_version:
        raise ValueError("old_version and new_version must differ")
    old = SourcePolicy(source_id, old_version, False, 0.0, note or f"replaced by {new_version}", {"replacement": new_version})
    new = SourcePolicy(source_id, new_version, True, new_trust_weight, note or f"replaces {old_version}", {"replaces": old_version})
    old.validate(); new.validate()
    with db:
        for p in (old, new):
            db.execute(
                "INSERT INTO source_policies(source_id,source_version,enabled,trust_weight,note,metadata_json,updated_at) "
                "VALUES(?,?,?,?,?,?,CURRENT_TIMESTAMP) "
                "ON CONFLICT(source_id,source_version) DO UPDATE SET enabled=excluded.enabled,trust_weight=excluded.trust_weight,note=excluded.note,metadata_json=excluded.metadata_json,updated_at=CURRENT_TIMESTAMP",
                (p.source_id, p.source_version, int(p.enabled), float(p.trust_weight), p.note, json.dumps(p.metadata or {}, ensure_ascii=False, sort_keys=True)),
            )
    return old, new


def list_source_policies(db: sqlite3.Connection, include_defaults: bool = True) -> list[dict[str, Any]]:
    explicit = {
        (r[0], r[1]): {
            "source_id": r[0], "source_version": r[1], "enabled": bool(r[2]),
            "trust_weight": float(r[3]), "note": r[4], "metadata": json.loads(r[5] or "{}"),
            "updated_at": r[6], "explicit": True,
        }
        for r in db.execute(
            "SELECT source_id,source_version,enabled,trust_weight,note,metadata_json,updated_at FROM source_policies ORDER BY source_id,source_version"
        ).fetchall()
    }
    if not include_defaults:
        return list(explicit.values())
    source_keys = {
        (r[0], r[1]) for r in db.execute(
            "SELECT DISTINCT source_id,source_version FROM main.evidence ORDER BY source_id,source_version"
        ).fetchall()
    }
    out = []
    for key in sorted(source_keys | set(explicit)):
        if key in explicit:
            out.append(explicit[key])
        else:
            out.append({
                "source_id": key[0], "source_version": key[1], "enabled": True,
                "trust_weight": 1.0, "note": "", "metadata": {},
                "updated_at": None, "explicit": False,
            })
    return out


def source_policy_hash(db: sqlite3.Connection) -> str:
    # Hash only behaviourally relevant fields. Notes/metadata and the fact that a
    # default was made explicit must not invalidate an otherwise identical rebuild.
    snapshot = [
        {
            "source_id": row["source_id"],
            "source_version": row["source_version"],
            "enabled": bool(row["enabled"]),
            "trust_weight": float(row["trust_weight"]),
        }
        for row in list_source_policies(db, include_defaults=True)
    ]
    raw = json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@contextmanager
def effective_evidence_view(db: sqlite3.Connection) -> Iterator[dict[str, Any]]:
    """Temporarily shadows main.evidence with a weighted, filtered TEMP view.

    Raw evidence in main.evidence is never changed. Existing Forge compiler code can
    continue querying `evidence` and automatically sees the effective rebuild view.
    """
    db.execute("DROP VIEW IF EXISTS temp.evidence")
    db.execute(
        """
        CREATE TEMP VIEW evidence AS
        SELECT
          e.evidence_id,
          e.evidence_fingerprint,
          e.from_id,
          e.to_id,
          e.relation_type,
          e.score,
          MIN(1.0, MAX(0.0, e.confidence * COALESCE(sp.trust_weight,1.0))) AS confidence,
          e.source_id,
          e.source_version,
          e.extraction,
          e.metadata_json
        FROM main.evidence e
        LEFT JOIN main.source_policies sp
          ON sp.source_id=e.source_id AND sp.source_version=e.source_version
        WHERE COALESCE(sp.enabled,1)=1
          AND COALESCE(sp.trust_weight,1.0)>0.0
        """
    )
    raw_count = int(db.execute("SELECT COUNT(*) FROM main.evidence").fetchone()[0])
    effective_count = int(db.execute("SELECT COUNT(*) FROM evidence").fetchone()[0])
    policies = list_source_policies(db, include_defaults=True)
    meta = {
        "raw_evidence_count": raw_count,
        "effective_evidence_count": effective_count,
        "disabled_source_versions": sum(1 for p in policies if not p["enabled"]),
        "reweighted_source_versions": sum(1 for p in policies if p["enabled"] and abs(float(p["trust_weight"]) - 1.0) > 1e-12),
        "source_policy_hash": source_policy_hash(db),
        "source_policies": policies,
    }
    try:
        yield meta
    finally:
        db.execute("DROP VIEW IF EXISTS temp.evidence")


def rebuild_senses(
    db: sqlite3.Connection,
    target_id: str,
    sense_policy: SenseCompilePolicy,
    consensus_policy: ConsensusPolicy | None = None,
    *,
    track_stability: bool = True,
) -> tuple[ForgeDocument, RebuildReport]:
    with effective_evidence_view(db) as meta:
        doc, sense_report = compile_provisional_senses(db, target_id, sense_policy)
        consensus: ConsensusReport | None = None
        if track_stability:
            consensus = track_hypothesis_run(db, doc, target_id, sense_policy, consensus_policy or ConsensusPolicy())
        output_hash = hashlib.sha256(
            json.dumps(doc.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        cur = db.execute(
            "INSERT INTO rebuild_runs(target_id,source_policy_hash,output_hash,raw_evidence_count,effective_evidence_count,metadata_json) VALUES(?,?,?,?,?,?)",
            (
                target_id, meta["source_policy_hash"], output_hash,
                meta["raw_evidence_count"], meta["effective_evidence_count"],
                json.dumps({
                    "format": REBUILD_FORMAT,
                    "version": REBUILD_VERSION,
                    "disabled_source_versions": meta["disabled_source_versions"],
                    "reweighted_source_versions": meta["reweighted_source_versions"],
                }, sort_keys=True),
            ),
        )
        rebuild_run_id = int(cur.lastrowid)
        # v0.10 records the build recipe and dependency boundary. Local import avoids
        # making the dependency graph part of the lower-level v0.1-v0.9 modules.
        from .dependency import register_sense_build_target, refresh_dependency_index
        register_sense_build_target(
            db, target_id, sense_policy, consensus_policy or ConsensusPolicy(),
            track_stability=track_stability, last_output_hash=output_hash,
        )
        # v0.11 persists compiler outputs as derived artifacts. Only provisional
        # sense nodes are outputs; closure/input nodes remain canonical inputs.
        from .artifact_store import record_document_outputs
        output_ids = [n.node_id for n in doc.nodes if n.metadata.get("provisional") and n.metadata.get("target_id") == target_id]
        status_by_node = {}
        if consensus is not None:
            status_by_node = {h["hypothesis_id"]: h.get("status", "candidate") for h in consensus.hypotheses}
        else:
            status_by_node = {nid: "candidate" for nid in output_ids}
        record_document_outputs(
            db, target_id, "sense", doc, output_ids, status_by_node=status_by_node,
            source_policy_hash=meta["source_policy_hash"], artifact_type="sense_node",
        )
        refresh_dependency_index(db)
        db.commit()
        report = RebuildReport(
            REBUILD_FORMAT, REBUILD_VERSION, target_id,
            meta["source_policy_hash"], meta["source_policies"],
            meta["raw_evidence_count"], meta["effective_evidence_count"],
            meta["disabled_source_versions"], meta["reweighted_source_versions"],
            sense_report.to_dict(), None if consensus is None else consensus.to_dict(), rebuild_run_id,
        )
        return doc, report
