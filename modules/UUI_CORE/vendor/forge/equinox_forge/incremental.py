from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import hashlib
import json
import sqlite3

from .consensus import ConsensusPolicy
from .dependency import (
    ArtifactRef,
    build_target_policy,
    change_set_hash,
    impact_from_roots,
    refresh_dependency_index,
    source_roots,
    topological_build_targets,
)
from .rebuild import rebuild_senses, source_policy_hash
from .sense_compiler import FusionWeights, SenseCompilePolicy
from .macro_builder import MacroCompilePolicy, rebuild_macro

INCREMENTAL_FORMAT = "equinox-forge-incremental-rebuild-v1"
INCREMENTAL_VERSION = "0.11"


@dataclass
class IncrementalRebuildReport:
    format: str
    version: str
    change_set_hash: str
    source_policy_hash: str
    roots: list[dict[str, str]]
    affected_build_targets: list[str]
    affected_nodes: int
    affected_manual_artifacts: list[str]
    rebuilt_targets: list[dict[str, Any]]
    deferred_targets: list[dict[str, Any]]
    dry_run: bool
    dependency_stats: dict[str, int]
    incremental_run_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _ingest_document(db: sqlite3.Connection, doc: Any) -> None:
    """Non-destructive/idempotent CanonicalDB-equivalent ingest on an existing connection."""
    for n in sorted(doc.nodes, key=lambda x:(x.level,x.node_id)):
        meta = json.dumps(n.metadata,ensure_ascii=False,sort_keys=True)
        db.execute(
            "INSERT OR IGNORE INTO nodes(node_id,modality,kind,level,surface,metadata_json) VALUES(?,?,?,?,?,?)",
            (n.node_id,n.modality,n.kind,n.level,n.surface,meta),
        )
        row=db.execute("SELECT modality,kind,level,surface FROM nodes WHERE node_id=?",(n.node_id,)).fetchone()
        if tuple(row)!=(n.modality,n.kind,n.level,n.surface):
            raise ValueError(f"canonical collision for {n.node_id}")
        for pos,child in enumerate(n.members):
            existing=db.execute("SELECT child_id FROM node_members WHERE parent_id=? AND position=?",(n.node_id,pos)).fetchone()
            if existing is not None and existing[0]!=child:
                raise ValueError(f"member collision for {n.node_id} position {pos}")
            db.execute("INSERT OR IGNORE INTO node_members(parent_id,position,child_id) VALUES(?,?,?)",(n.node_id,pos,child))
    for e in doc.evidence:
        metadata_json=json.dumps(e.metadata,ensure_ascii=False,sort_keys=True)
        raw=json.dumps({
            "from":e.from_id,"to":e.to_id,"type":e.relation_type,
            "score":float(e.score),"confidence":float(e.confidence),
            "source":e.source.source_id,"version":e.source.source_version,
            "extraction":e.source.extraction,"metadata":e.metadata,
        },ensure_ascii=False,separators=(",",":"),sort_keys=True).encode("utf-8")
        fp=hashlib.sha256(raw).hexdigest()
        db.execute(
            "INSERT OR IGNORE INTO evidence(evidence_fingerprint,from_id,to_id,relation_type,score,confidence,source_id,source_version,extraction,metadata_json) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (fp,e.from_id,e.to_id,e.relation_type,float(e.score),float(e.confidence),e.source.source_id,e.source.source_version,e.source.extraction,metadata_json),
        )


def _sense_policy(payload: dict[str, Any]) -> SenseCompilePolicy:
    s = payload.get("sense_policy") or {}
    w = s.get("weights") or {}
    weights = FusionWeights(**{k: w[k] for k in asdict(FusionWeights()) if k in w})
    kwargs = {k: s[k] for k in (
        "threshold", "min_channels", "min_contexts", "include_singletons",
        "min_relation_score", "min_relation_confidence", "min_relation_sources",
    ) if k in s}
    kwargs["weights"] = weights
    return SenseCompilePolicy(**kwargs)


def _consensus_policy(payload: dict[str, Any]) -> ConsensusPolicy:
    c = payload.get("consensus_policy") or {}
    valid = asdict(ConsensusPolicy()).keys()
    return ConsensusPolicy(**{k: c[k] for k in valid if k in c})


def _macro_policy(payload: dict[str, Any]) -> MacroCompilePolicy:
    valid = {
        "input_build_targets", "input_nodes", "allowed_statuses", "min_members",
        "ordered_members", "modality", "kind", "surface",
    }
    return MacroCompilePolicy(**{k: payload[k] for k in valid if k in payload})


def incremental_rebuild(
    db: sqlite3.Connection,
    roots: list[ArtifactRef],
    *,
    output_dir: str | Path | None = None,
    dry_run: bool = False,
    ingest_derived: bool = False,
) -> IncrementalRebuildReport:
    dep_stats = refresh_dependency_index(db)
    impact = impact_from_roots(db, roots, refresh=False)
    rebuilt: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    outdir = Path(output_dir) if output_dir else None
    if outdir and not dry_run:
        outdir.mkdir(parents=True, exist_ok=True)

    # Current built-in executor is the sense compiler. The scheduler and dependency
    # graph are generic; future macro/crossmodal builders register here without
    # changing invalidation semantics. Build targets run dependency-first.
    target_order, cycle_targets = topological_build_targets(db, impact.affected_build_targets)
    for target_id in cycle_targets:
        deferred.append({"target_id": target_id, "builder": build_target_policy(db,target_id)["builder"], "reason": "dependency_cycle"})
    deferred_ids = set(cycle_targets)
    for target_id in target_order:
        info = build_target_policy(db, target_id)
        if not info["active"]:
            continue
        upstream = [r[0] for r in db.execute(
            "SELECT dependency_id FROM dependency_edges WHERE dependent_type='build_target' AND dependent_id=? AND dependency_type='build_target'",
            (target_id,),
        ).fetchall()]
        if any(x in deferred_ids for x in upstream):
            deferred.append({"target_id":target_id,"builder":info["builder"],"reason":"upstream_builder_deferred"})
            deferred_ids.add(target_id)
            continue
        if info["builder"] not in {"sense", "macro"}:
            deferred.append({"target_id": target_id, "builder": info["builder"], "reason": "builder_not_installed"})
            deferred_ids.add(target_id)
            continue
        if info["builder"] == "macro" and not ((info["policy"].get("input_build_targets") or []) or (info["policy"].get("input_nodes") or [])):
            # Preserve the v0.10 placeholder behaviour: merely naming a future
            # macro builder is not enough to execute it until a real recipe exists.
            deferred.append({"target_id": target_id, "builder": "macro", "reason": "builder_not_installed"})
            deferred_ids.add(target_id)
            continue
        if dry_run:
            rebuilt.append({"target_id": target_id, "builder": info["builder"], "status": "planned"})
            continue
        policy_payload = info["policy"]
        if info["builder"] == "sense":
            sense = _sense_policy(policy_payload)
            consensus = _consensus_policy(policy_payload)
            track = bool(policy_payload.get("track_stability", True))
            doc, report = rebuild_senses(db, target_id, sense, consensus, track_stability=track)
            extra = {
                "rebuild_run_id": report.rebuild_run_id,
                "clusters_created": report.sense_report.get("clusters_created"),
            }
        else:
            macro_policy = _macro_policy(policy_payload)
            doc, report = rebuild_macro(db, target_id, macro_policy, record_store=True)
            extra = {
                "built": report.built,
                "output_node_id": report.output_node_id,
                "members": len(report.members),
                "reason": report.reason,
            }
        if ingest_derived:
            _ingest_document(db, doc)
        output_path = None
        if outdir:
            safe = target_id.replace("/", "_").replace(":", "_")
            output_path = outdir / f"{safe}.eqir.json"
            output_path.write_text(json.dumps(doc.to_dict(),ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
        rebuilt.append({
            "target_id": target_id,
            "builder": info["builder"],
            "status": "rebuilt" if getattr(report, "built", True) else "not_built",
            "output": None if output_path is None else str(output_path),
            "output_hash": hashlib.sha256(json.dumps(doc.to_dict(),sort_keys=True,ensure_ascii=False,separators=(",",":")).encode()).hexdigest(),
            **extra,
        })

    if not dry_run:
        refresh_dependency_index(db)
    csh = change_set_hash(roots)
    sph = source_policy_hash(db)
    payload_for_hash = {
        "roots": [r.to_dict() for r in roots],
        "affected": impact.affected_build_targets,
        "rebuilt": rebuilt,
        "deferred": deferred,
        "dry_run": dry_run,
    }
    output_hash = hashlib.sha256(json.dumps(payload_for_hash,sort_keys=True,ensure_ascii=False,separators=(",",":")).encode()).hexdigest()
    run_id = None
    if not dry_run:
        cur = db.execute(
            "INSERT INTO incremental_rebuild_runs(change_set_hash,source_policy_hash,roots_json,affected_json,rebuilt_json,deferred_json,output_hash,metadata_json) VALUES(?,?,?,?,?,?,?,?)",
            (
                csh, sph,
                json.dumps([r.to_dict() for r in roots],sort_keys=True),
                json.dumps(impact.to_dict(),sort_keys=True),
                json.dumps(rebuilt,sort_keys=True),
                json.dumps(deferred,sort_keys=True),
                output_hash,
                json.dumps({"format":INCREMENTAL_FORMAT,"version":INCREMENTAL_VERSION},sort_keys=True),
            ),
        )
        run_id = int(cur.lastrowid)
        db.commit()

    return IncrementalRebuildReport(
        INCREMENTAL_FORMAT, INCREMENTAL_VERSION, csh, sph,
        [r.to_dict() for r in roots], impact.affected_build_targets,
        impact.affected_nodes, impact.affected_manual_artifacts,
        rebuilt, deferred, dry_run, dep_stats, run_id,
    )


def incremental_rebuild_from_sources(
    db: sqlite3.Connection,
    changed_sources: list[str],
    **kwargs: Any,
) -> IncrementalRebuildReport:
    return incremental_rebuild(db, source_roots(changed_sources), **kwargs)
