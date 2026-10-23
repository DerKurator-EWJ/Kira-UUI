from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
import hashlib
import json
import sqlite3

from .ir import ForgeDocument, RelationType
from .source_lineage import SourceLineageGraph

EXCEPTION_CONSENSUS_FORMAT = "equinox-forge-exception-consensus-v1"
EXCEPTION_CONSENSUS_VERSION = "0.38"


def ensure_exception_consensus_schema(db: sqlite3.Connection) -> None:
    db.executescript("""
    CREATE TABLE IF NOT EXISTS exception_hypothesis_runs(
      run_id INTEGER PRIMARY KEY AUTOINCREMENT,
      policy_hash TEXT NOT NULL,
      output_hash TEXT NOT NULL,
      created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
      metadata_json TEXT NOT NULL DEFAULT '{}'
    );
    CREATE TABLE IF NOT EXISTS exception_hypothesis_states(
      override_id TEXT NOT NULL,
      policy_hash TEXT NOT NULL,
      general_frame_id TEXT NOT NULL,
      subtype_prototype_id TEXT NOT NULL,
      status TEXT NOT NULL,
      first_seen_run INTEGER NOT NULL,
      last_seen_run INTEGER NOT NULL,
      seen_runs INTEGER NOT NULL DEFAULT 1,
      missed_runs INTEGER NOT NULL DEFAULT 0,
      consensus_score REAL NOT NULL DEFAULT 0.0,
      support_sources INTEGER NOT NULL DEFAULT 0,
      conflict_sources INTEGER NOT NULL DEFAULT 0,
      support_weight REAL NOT NULL DEFAULT 0.0,
      conflict_weight REAL NOT NULL DEFAULT 0.0,
      metadata_json TEXT NOT NULL DEFAULT '{}',
      PRIMARY KEY(override_id,policy_hash)
    );
    CREATE INDEX IF NOT EXISTS idx_exception_states_status
      ON exception_hypothesis_states(policy_hash,status,override_id);
    """)


@dataclass(frozen=True)
class ExceptionConsensusPolicy:
    source_min_context_fraction: float = 1.0
    source_vote_threshold: float = 0.55
    source_vote_margin: float = 0.15
    stable_min_sources: int = 2
    stable_min_runs: int = 2
    stable_consensus: float = 0.75
    stable_min_support_weight: float = 1.5
    quarantine_conflicts: int = 2
    quarantine_conflict_weight: float = 1.5
    quarantine_after_misses: int = 2

    def validate(self) -> None:
        for name in ("source_min_context_fraction","source_vote_threshold","source_vote_margin","stable_consensus"):
            v=float(getattr(self,name))
            if not 0.0 <= v <= 1.0: raise ValueError(f"{name} must be in [0,1]")
        for name in ("stable_min_sources","stable_min_runs","quarantine_conflicts","quarantine_after_misses"):
            if int(getattr(self,name)) < 1: raise ValueError(f"{name} must be >= 1")
        for name in ("stable_min_support_weight","quarantine_conflict_weight"):
            if float(getattr(self,name)) <= 0: raise ValueError(f"{name} must be > 0")


@dataclass
class ExceptionSourceVote:
    source_key: str
    source_weight: float
    vote: str
    observed_contexts: int
    required_contexts: int
    exception_score: float
    general_score: float
    margin: float
    raw_source_keys: list[str] = field(default_factory=list)
    lineage_roots: list[str] = field(default_factory=list)
    lineage_internal_conflict: bool = False


@dataclass
class ExceptionLifecycleAssessment:
    override_id: str
    status: str
    previous_status: str | None
    consensus_score: float
    support_sources: int
    conflict_sources: int
    abstain_sources: int
    support_weight: float
    conflict_weight: float
    support_streak: int
    missed_runs: int
    votes: list[dict[str,Any]]
    reason: str


@dataclass
class ExceptionConsensusReport:
    format: str
    version: str
    run_id: int
    policy_hash: str
    assessments: list[dict[str,Any]]
    missing_overrides: list[dict[str,Any]]
    policy: dict[str,Any]
    def to_dict(self) -> dict[str,Any]: return asdict(self)


def _policy_hash(policy: ExceptionConsensusPolicy) -> str:
    return hashlib.sha256(json.dumps(asdict(policy),sort_keys=True,separators=(",",":")).encode()).hexdigest()[:20]


def _source_weights(db: sqlite3.Connection) -> dict[str,float]:
    try:
        rows=db.execute("SELECT source_id,source_version,enabled,trust_weight FROM source_policies").fetchall()
    except sqlite3.OperationalError:
        return {}
    return {f"{s}@{v}": (max(0.0,min(1.0,float(w))) if bool(en) else 0.0) for s,v,en,w in rows}


def _context_source_scores(db: sqlite3.Connection, context_ids: list[str], target_id: str) -> dict[str,dict[str,float]]:
    """Return per-source per-context confidence-weighted prediction scores."""
    if not context_ids: return {}
    qmarks=",".join("?" for _ in context_ids)
    rows=db.execute(
        f"SELECT from_id,score,confidence,source_id,source_version FROM evidence WHERE relation_type=? AND to_id=? AND from_id IN ({qmarks}) ORDER BY evidence_id",
        (RelationType.PREDICTION.value,target_id,*context_ids),
    ).fetchall()
    buckets: dict[str,dict[str,list[tuple[float,float]]]]={}
    for ctx,score,conf,src,ver in rows:
        buckets.setdefault(f"{src}@{ver}",{}).setdefault(str(ctx),[]).append((float(score),float(conf)))
    out: dict[str,dict[str,float]]={}
    for source,contexts in buckets.items():
        out[source]={}
        for ctx,evs in contexts.items():
            den=sum(c for _,c in evs)
            if den>0: out[source][ctx]=max(0.0,min(1.0,sum(s*c for s,c in evs)/den))
    return out


def _votes_for_override(db: sqlite3.Connection, meta: dict[str,Any], policy: ExceptionConsensusPolicy) -> list[ExceptionSourceVote]:
    contexts=sorted(str(x) for x in (meta.get("support_context_ids") or []))
    exception_id=str(meta.get("exception_top_target_id") or "")
    general_id=str(meta.get("exception_general_top_target_id") or "")
    if not contexts or not exception_id or not general_id: return []
    exc=_context_source_scores(db,contexts,exception_id)
    gen=_context_source_scores(db,contexts,general_id)
    sources=sorted(set(exc)|set(gen))
    weights=_source_weights(db)
    required=max(1,int(len(contexts)*policy.source_min_context_fraction + 0.999999999))
    votes=[]
    for source in sources:
        weight=weights.get(source,1.0)
        if weight <= 0.0: continue
        observed=set(exc.get(source,{}))|set(gen.get(source,{}))
        es=sum(exc.get(source,{}).get(c,0.0) for c in contexts)/len(contexts)
        gs=sum(gen.get(source,{}).get(c,0.0) for c in contexts)/len(contexts)
        vote="abstain"
        if len(observed)>=required:
            if es>=policy.source_vote_threshold and es>=gs+policy.source_vote_margin: vote="support"
            elif gs>=policy.source_vote_threshold and gs>=es+policy.source_vote_margin: vote="conflict"
        votes.append(ExceptionSourceVote(source,weight,vote,len(observed),required,es,gs,es-gs))
    return votes




def _collapse_exception_votes_by_lineage(db: sqlite3.Connection, votes: list[ExceptionSourceVote]) -> list[ExceptionSourceVote]:
    if not votes:
        return []
    graph = SourceLineageGraph.from_db(db)
    keys = [v.source_key for v in votes]
    if not any(key in graph.parents for key in keys):
        return votes
    by_key = {v.source_key: v for v in votes}
    out: list[ExceptionSourceVote] = []
    for component in graph.components(keys):
        members = [by_key[k] for k in component.sources if k in by_key]
        if not members:
            continue
        decisive = {v.vote for v in members if v.vote in {"support", "conflict"}}
        internal_conflict = len(decisive) > 1
        if internal_conflict or not decisive:
            vote = "abstain"
            selected = members
        else:
            vote = next(iter(decisive))
            selected = [v for v in members if v.vote == vote]
        weight = max((v.source_weight for v in selected), default=0.0)
        den = sum(max(0.0, v.source_weight) for v in members)
        if den > 0.0:
            es = sum(v.exception_score * max(0.0, v.source_weight) for v in members) / den
            gs = sum(v.general_score * max(0.0, v.source_weight) for v in members) / den
        else:
            es = gs = 0.0
        out.append(ExceptionSourceVote(
            source_key="lineage:" + "|".join(component.roots),
            source_weight=weight,
            vote=vote,
            observed_contexts=max((v.observed_contexts for v in members), default=0),
            required_contexts=max((v.required_contexts for v in members), default=0),
            exception_score=es,
            general_score=gs,
            margin=es-gs,
            raw_source_keys=list(component.sources),
            lineage_roots=list(component.roots),
            lineage_internal_conflict=internal_conflict,
        ))
    return out

def _set_node_lifecycle(db: sqlite3.Connection, override_id: str, status: str, payload: dict[str,Any]) -> None:
    row=db.execute("SELECT metadata_json FROM nodes WHERE node_id=?",(override_id,)).fetchone()
    if not row: return
    try: meta=json.loads(row[0] or "{}")
    except Exception: meta={}
    meta.update({
        "exception_lifecycle_status":status,
        "exception_consensus_score":payload.get("consensus_score",0.0),
        "exception_support_sources":payload.get("support_sources",0),
        "exception_conflict_sources":payload.get("conflict_sources",0),
        "exception_support_weight":payload.get("support_weight",0.0),
        "exception_conflict_weight":payload.get("conflict_weight",0.0),
        "exception_support_streak":payload.get("support_streak",0),
        "exception_lifecycle_reason":payload.get("reason",""),
        "exception_consensus_version":EXCEPTION_CONSENSUS_VERSION,
    })
    db.execute("UPDATE nodes SET metadata_json=? WHERE node_id=?",(json.dumps(meta,ensure_ascii=False,sort_keys=True),override_id))


def track_exception_run(db: sqlite3.Connection, doc: ForgeDocument, policy: ExceptionConsensusPolicy|None=None) -> ExceptionConsensusReport:
    policy=policy or ExceptionConsensusPolicy(); policy.validate(); ensure_exception_consensus_schema(db)
    ph=_policy_hash(policy)
    output_hash=hashlib.sha256(json.dumps(doc.to_dict(),sort_keys=True,ensure_ascii=False,separators=(",",":")).encode()).hexdigest()
    cur=db.execute("INSERT INTO exception_hypothesis_runs(policy_hash,output_hash,metadata_json) VALUES(?,?,?)",(ph,output_hash,json.dumps({"version":EXCEPTION_CONSENSUS_VERSION},sort_keys=True)))
    run_id=int(cur.lastrowid)
    overrides=[n for n in doc.nodes if (n.metadata or {}).get("exception_override")]
    current={n.node_id for n in overrides}
    missing=[]
    for oid,status,missed,meta_json in db.execute("SELECT override_id,status,missed_runs,metadata_json FROM exception_hypothesis_states WHERE policy_hash=?",(ph,)).fetchall():
        if oid in current: continue
        nm=int(missed)+1
        ns="quarantined" if nm>=policy.quarantine_after_misses else "contested"
        meta=json.loads(meta_json or "{}"); meta.update({"reason":"missing_from_rebuild","support_streak":0})
        db.execute("UPDATE exception_hypothesis_states SET status=?,missed_runs=?,metadata_json=? WHERE override_id=? AND policy_hash=?",(ns,nm,json.dumps(meta,sort_keys=True),oid,ph))
        _set_node_lifecycle(db,oid,ns,{"reason":"missing_from_rebuild","support_streak":0})
        missing.append({"override_id":oid,"previous_status":status,"status":ns,"missed_runs":nm})

    assessments=[]
    for node in sorted(overrides,key=lambda n:n.node_id):
        meta=dict(node.metadata or {})
        raw_votes=_votes_for_override(db,meta,policy)
        votes=_collapse_exception_votes_by_lineage(db,raw_votes)
        support=[v for v in votes if v.vote=="support"]; conflict=[v for v in votes if v.vote=="conflict"]; abstain=[v for v in votes if v.vote=="abstain"]
        sw=sum(v.source_weight for v in support); cw=sum(v.source_weight for v in conflict); dw=sw+cw
        consensus=0.0 if dw<=0 else sw/dw
        row=db.execute("SELECT status,first_seen_run,seen_runs,metadata_json FROM exception_hypothesis_states WHERE override_id=? AND policy_hash=?",(node.node_id,ph)).fetchone()
        if row:
            prev,first,seen,oldmeta=row; oldmeta=json.loads(oldmeta or "{}")
            prev_streak=int(oldmeta.get("support_streak",0))
            first=int(first); seen=int(seen)+1
        else:
            prev=None; first=run_id; seen=1; prev_streak=0
        clean_support=(len(support)>=policy.stable_min_sources and sw>=policy.stable_min_support_weight and consensus>=policy.stable_consensus and not conflict)
        streak=prev_streak+1 if clean_support else 0
        if len(conflict)>=policy.quarantine_conflicts and cw>=policy.quarantine_conflict_weight:
            status="quarantined"; reason="weighted_multi_lineage_conflict"
        elif conflict:
            status="contested"; reason="lineage_conflict"
        elif clean_support and streak>=policy.stable_min_runs:
            status="stable"; reason="repeated_cross_lineage_consensus"
        else:
            status="candidate"; reason="insufficient_repeated_independent_lineage_confirmation"
        payload={
            "consensus_score":consensus,"support_sources":len(support),"conflict_sources":len(conflict),
            "support_weight":sw,"conflict_weight":cw,"support_streak":streak,"reason":reason,
            "votes":[asdict(v) for v in votes],"version":EXCEPTION_CONSENSUS_VERSION,
        }
        db.execute(
            "INSERT INTO exception_hypothesis_states(override_id,policy_hash,general_frame_id,subtype_prototype_id,status,first_seen_run,last_seen_run,seen_runs,missed_runs,consensus_score,support_sources,conflict_sources,support_weight,conflict_weight,metadata_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(override_id,policy_hash) DO UPDATE SET status=excluded.status,last_seen_run=excluded.last_seen_run,seen_runs=excluded.seen_runs,missed_runs=0,consensus_score=excluded.consensus_score,support_sources=excluded.support_sources,conflict_sources=excluded.conflict_sources,support_weight=excluded.support_weight,conflict_weight=excluded.conflict_weight,metadata_json=excluded.metadata_json",
            (node.node_id,ph,str(meta.get("overridden_frame_node_id") or ""),str(meta.get("exception_subtype_prototype_node_id") or ""),status,first,run_id,seen,0,consensus,len(support),len(conflict),sw,cw,json.dumps(payload,ensure_ascii=False,sort_keys=True)),
        )
        node.metadata.update({
            "exception_lifecycle_status":status,"exception_consensus_score":consensus,
            "exception_support_sources":len(support),"exception_conflict_sources":len(conflict),
            "exception_support_weight":sw,"exception_conflict_weight":cw,"exception_support_streak":streak,
            "exception_lifecycle_reason":reason,"exception_consensus_policy_hash":ph,
        })
        _set_node_lifecycle(db,node.node_id,status,payload)
        assessments.append(asdict(ExceptionLifecycleAssessment(node.node_id,status,prev,consensus,len(support),len(conflict),len(abstain),sw,cw,streak,0,[asdict(v) for v in votes],reason)))
    db.commit()
    return ExceptionConsensusReport(EXCEPTION_CONSENSUS_FORMAT,EXCEPTION_CONSENSUS_VERSION,run_id,ph,assessments,missing,asdict(policy))


def list_exception_states(db: sqlite3.Connection) -> list[dict[str,Any]]:
    ensure_exception_consensus_schema(db)
    rows=db.execute("SELECT override_id,policy_hash,general_frame_id,subtype_prototype_id,status,first_seen_run,last_seen_run,seen_runs,missed_runs,consensus_score,support_sources,conflict_sources,support_weight,conflict_weight,metadata_json FROM exception_hypothesis_states ORDER BY override_id,policy_hash").fetchall()
    keys=["override_id","policy_hash","general_frame_id","subtype_prototype_id","status","first_seen_run","last_seen_run","seen_runs","missed_runs","consensus_score","support_sources","conflict_sources","support_weight","conflict_weight"]
    out=[]
    for row in rows:
        d=dict(zip(keys,row[:-1])); d["metadata"]=json.loads(row[-1] or "{}"); out.append(d)
    return out
