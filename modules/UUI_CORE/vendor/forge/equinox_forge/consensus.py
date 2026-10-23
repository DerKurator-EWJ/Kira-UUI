from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any
import hashlib
import json
import math
import sqlite3

from .ir import ForgeDocument, IRNode, RelationType, NodeKind
from .source_lineage import SourceLineageGraph
from .sense_compiler import (
    SenseCompilePolicy,
    FusionWeights,
    _members,
    _node_from_db,
    _jaccard,
    _cross_affinity,
    _cosine_dict,
    _weighted_fuse,
)

CONSENSUS_FORMAT = "equinox-forge-consensus-v1"
CONSENSUS_VERSION = "0.38"


@dataclass(frozen=True)
class ConsensusPolicy:
    source_vote_threshold: float = 0.58
    source_vote_margin: float = 0.08
    min_observed_pairs: int = 1
    stable_min_sources: int = 2
    stable_min_runs: int = 2
    stable_consensus: float = 0.75
    stable_min_support_weight: float = 1.5
    quarantine_conflicts: int = 2
    quarantine_conflict_weight: float = 1.5
    quarantine_after_misses: int = 2

    def validate(self) -> None:
        for name in ("source_vote_threshold", "source_vote_margin", "stable_consensus"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        for name in ("min_observed_pairs", "stable_min_sources", "stable_min_runs", "quarantine_conflicts", "quarantine_after_misses"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be >= 1")
        for name in ("stable_min_support_weight", "quarantine_conflict_weight"):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be > 0")


@dataclass
class SourceVote:
    source_key: str
    source_weight: float
    vote: str
    internal_score: float | None
    competing_score: float | None
    internal_pairs: int
    competing_pairs: int
    channel_observations: dict[str, int]
    quality: float
    raw_source_keys: list[str] = field(default_factory=list)
    lineage_roots: list[str] = field(default_factory=list)
    lineage_internal_conflict: bool = False


@dataclass
class HypothesisAssessment:
    hypothesis_id: str
    membership_hash: str
    status: str
    consensus_score: float
    support_sources: int
    conflict_sources: int
    abstain_sources: int
    decisive_sources: int
    support_weight: float
    conflict_weight: float
    decisive_weight: float
    mean_support_quality: float
    votes: list[dict[str, Any]]
    previous_status: str | None = None
    seen_runs: int = 0
    consecutive_seen: int = 0
    missed_runs: int = 0


@dataclass
class ConsensusReport:
    format: str
    version: str
    target_id: str
    policy_hash: str
    run_id: int | None
    hypotheses: list[dict[str, Any]]
    missing_hypotheses: list[dict[str, Any]]
    policy: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _policy_hash(sense_policy: SenseCompilePolicy, consensus_policy: ConsensusPolicy) -> str:
    raw = {
        "sense": {
            "threshold": sense_policy.threshold,
            "min_channels": sense_policy.min_channels,
            "min_contexts": sense_policy.min_contexts,
            "include_singletons": sense_policy.include_singletons,
            "weights": asdict(sense_policy.weights),
        },
        "consensus": asdict(consensus_policy),
    }
    return hashlib.sha256(json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:20]


def _source_relation_maps(db: sqlite3.Connection) -> dict[str, dict[tuple[str, str, str], float]]:
    rows = db.execute(
        "SELECT from_id,to_id,relation_type,score,confidence,source_id,source_version FROM evidence ORDER BY evidence_id"
    ).fetchall()
    buckets: dict[str, dict[tuple[str, str, str], list[tuple[float, float]]]] = {}
    for a, b, typ, score, conf, src, ver in rows:
        key = f"{src}@{ver}"
        buckets.setdefault(key, {}).setdefault((a, b, typ), []).append((float(score), float(conf)))
    out: dict[str, dict[tuple[str, str, str], float]] = {}
    for src, rels in buckets.items():
        out[src] = {}
        for key, evs in rels.items():
            den = sum(c for _, c in evs)
            if den > 0:
                out[src][key] = max(0.0, min(1.0, sum(s * c for s, c in evs) / den))
    return out


def _source_weights(db: sqlite3.Connection) -> dict[str, float]:
    try:
        rows = db.execute("SELECT source_id,source_version,enabled,trust_weight FROM main.source_policies").fetchall()
    except sqlite3.OperationalError:
        return {}
    out: dict[str, float] = {}
    for src, ver, enabled, weight in rows:
        out[f"{src}@{ver}"] = 0.0 if not bool(enabled) else max(0.0, min(1.0, float(weight)))
    return out




def _weighted_optional_mean(values: list[tuple[float | None, float]]) -> float | None:
    rows = [(float(v), max(0.0, float(w))) for v, w in values if v is not None and float(w) > 0.0]
    den = sum(w for _, w in rows)
    if den <= 0.0:
        return None
    return sum(v * w for v, w in rows) / den


def _collapse_votes_by_lineage(db: sqlite3.Connection, votes: list[SourceVote]) -> list[SourceVote]:
    """Collapse correlated source votes into conservative epistemic components.

    If no participating source has lineage metadata, the original vote list is
    returned unchanged for backward compatibility.  Within a correlated family,
    unanimous decisive votes count once; support/conflict disagreement inside the
    family becomes abstention instead of a within-family majority vote.
    """
    if not votes:
        return []
    graph = SourceLineageGraph.from_db(db)
    keys = [v.source_key for v in votes]
    if not any(key in graph.parents for key in keys):
        return votes
    by_key = {v.source_key: v for v in votes}
    out: list[SourceVote] = []
    for component in graph.components(keys):
        members = [by_key[k] for k in component.sources if k in by_key]
        if not members:
            continue
        decisive = {v.vote for v in members if v.vote in {"support", "conflict"}}
        internal_conflict = len(decisive) > 1
        if internal_conflict or not decisive:
            vote = "abstain"
            selected = members
            quality = 0.0
            weight = max((v.source_weight for v in members), default=0.0)
        else:
            vote = next(iter(decisive))
            selected = [v for v in members if v.vote == vote]
            weight = max((v.source_weight for v in selected), default=0.0)
            den = sum(max(0.0, v.source_weight) for v in selected)
            quality = (
                sum(v.quality * max(0.0, v.source_weight) for v in selected) / den
                if den > 0.0 else 0.0
            )
        channel_observations: dict[str, int] = {}
        for v in members:
            for name, count in v.channel_observations.items():
                channel_observations[name] = channel_observations.get(name, 0) + int(count)
        weights = [(v.internal_score, v.source_weight) for v in members]
        comp = [(v.competing_score, v.source_weight) for v in members]
        out.append(SourceVote(
            source_key="lineage:" + "|".join(component.roots),
            source_weight=weight,
            vote=vote,
            internal_score=_weighted_optional_mean(weights),
            competing_score=_weighted_optional_mean(comp),
            internal_pairs=max((v.internal_pairs for v in members), default=0),
            competing_pairs=max((v.competing_pairs for v in members), default=0),
            channel_observations=channel_observations,
            quality=max(0.0, min(1.0, quality)),
            raw_source_keys=list(component.sources),
            lineage_roots=list(component.roots),
            lineage_internal_conflict=internal_conflict,
        ))
    return out

def _sym(relmap: dict[tuple[str, str, str], float], a: str, b: str, typ: str) -> float | None:
    vals = []
    if (a, b, typ) in relmap:
        vals.append(relmap[(a, b, typ)])
    if (b, a, typ) in relmap:
        vals.append(relmap[(b, a, typ)])
    return max(vals) if vals else None


def _prediction_signature_for_source(db: sqlite3.Connection, relmap: dict[tuple[str, str, str], float], members: list[str]) -> dict[str, float]:
    if not members:
        return {}
    sig: dict[str, float] = {}
    tids = [r[0] for r in db.execute("SELECT node_id FROM nodes WHERE kind=?", (NodeKind.TEMPLATE.value,)).fetchall()]
    for tid in tids:
        tm = _members(db, tid)
        if not tm or len(tm) > len(members):
            continue
        if not any(members[i:i+len(tm)] == tm for i in range(len(members)-len(tm)+1)):
            continue
        for (a, b, typ), value in relmap.items():
            if a == tid and typ == RelationType.PREDICTION.value:
                sig[b] = max(sig.get(b, 0.0), value)
    return sig


def _source_pair_score(
    db: sqlite3.Connection,
    relmap: dict[tuple[str, str, str], float],
    a: str,
    b: str,
    context_lex: dict[str, list[str]],
    context_members: dict[str, list[str]],
    weights: FusionWeights,
) -> tuple[float | None, int, dict[str, int]]:
    hidden = _sym(relmap, a, b, RelationType.CONTEXTUAL_SIMILARITY.value)

    def cross(typ: str) -> float | None:
        aa, bb = context_lex[a], context_lex[b]
        if not aa or not bb:
            return None
        vals: list[float] = []
        for x in aa:
            for y in bb:
                v = _sym(relmap, x, y, typ)
                if v is not None:
                    vals.append(v)
        return max(vals) if vals else None

    semantic = cross(RelationType.SEMANTIC_SIMILARITY.value)
    substitution = cross(RelationType.SUBSTITUTION.value)
    transition = cross(RelationType.TRANSITION.value)
    pa = _prediction_signature_for_source(db, relmap, context_members[a])
    pb = _prediction_signature_for_source(db, relmap, context_members[b])
    prediction = _cosine_dict(pa, pb)
    scores = {
        "hidden_similarity": hidden,
        # lexical overlap is structural, not an independent source vote.
        "lexical_overlap": None,
        "semantic_similarity": semantic,
        "substitution": substitution,
        "transition": transition,
        "prediction_signature": prediction,
    }
    fused, channels, _ = _weighted_fuse(scores, weights)
    observations = {k: int(v is not None) for k, v in scores.items() if k != "lexical_overlap"}
    if channels == 0:
        return None, 0, observations
    return fused, channels, observations


def _vote_for_cluster(
    db: sqlite3.Connection,
    relmap: dict[tuple[str, str, str], float],
    cluster: list[str],
    all_contexts: list[str],
    context_lex: dict[str, list[str]],
    context_members: dict[str, list[str]],
    weights: FusionWeights,
    policy: ConsensusPolicy,
    source_key: str,
    source_weight: float = 1.0,
) -> SourceVote:
    internal: list[float] = []
    competing: list[float] = []
    observations: dict[str, int] = {}
    for i, a in enumerate(cluster):
        for b in cluster[i+1:]:
            score, nch, obs = _source_pair_score(db, relmap, a, b, context_lex, context_members, weights)
            for k, v in obs.items(): observations[k] = observations.get(k, 0) + v
            if score is not None and nch > 0:
                internal.append(score)
    outside = [x for x in all_contexts if x not in set(cluster)]
    for a in cluster:
        for b in outside:
            score, nch, obs = _source_pair_score(db, relmap, a, b, context_lex, context_members, weights)
            for k, v in obs.items(): observations[k] = observations.get(k, 0) + v
            if score is not None and nch > 0:
                competing.append(score)
    internal_score = None if not internal else sum(internal) / len(internal)
    competing_score = None if not competing else max(competing)
    vote = "abstain"
    quality = 0.0
    if len(internal) >= policy.min_observed_pairs:
        comp = 0.0 if competing_score is None else competing_score
        if internal_score is not None and internal_score >= policy.source_vote_threshold and internal_score >= comp + policy.source_vote_margin:
            vote = "support"
            quality = max(0.0, min(1.0, internal_score - comp))
    if competing_score is not None and competing_score >= policy.source_vote_threshold:
        base = 0.0 if internal_score is None else internal_score
        if competing_score >= base + policy.source_vote_margin:
            vote = "conflict"
            quality = max(0.0, min(1.0, competing_score - base))
    return SourceVote(source_key, float(source_weight), vote, internal_score, competing_score, len(internal), len(competing), observations, quality)


def assess_hypotheses(
    db: sqlite3.Connection,
    doc: ForgeDocument,
    target_id: str,
    sense_policy: SenseCompilePolicy,
    consensus_policy: ConsensusPolicy | None = None,
) -> tuple[list[HypothesisAssessment], str]:
    cp = consensus_policy or ConsensusPolicy()
    cp.validate()
    policy_hash = _policy_hash(sense_policy, cp)
    source_maps = _source_relation_maps(db)
    source_weights = _source_weights(db)
    all_contexts = sorted({r[0] for r in db.execute(
        "SELECT DISTINCT to_id FROM evidence WHERE from_id=? AND relation_type=?",
        (target_id, RelationType.CONTEXTUAL_USE.value),
    ).fetchall()})
    context_members = {c: _members(db, c) for c in all_contexts}
    context_lex = {c: [m for m in context_members[c] if m != target_id] for c in all_contexts}
    assessments: list[HypothesisAssessment] = []
    for node in sorted((n for n in doc.nodes if n.metadata.get("provisional") and n.metadata.get("target_id") == target_id), key=lambda n: n.node_id):
        cluster = sorted(node.metadata.get("contexts") or [m for m in node.members if m != target_id])
        raw_votes = [
            _vote_for_cluster(db, relmap, cluster, all_contexts, context_lex, context_members, sense_policy.weights, cp, src, source_weights.get(src, 1.0))
            for src, relmap in sorted(source_maps.items())
        ]
        votes = _collapse_votes_by_lineage(db, raw_votes)
        support = [v for v in votes if v.vote == "support"]
        conflict = [v for v in votes if v.vote == "conflict"]
        abstain = [v for v in votes if v.vote == "abstain"]
        decisive = len(support) + len(conflict)
        support_weight = sum(v.source_weight for v in support)
        conflict_weight = sum(v.source_weight for v in conflict)
        decisive_weight = support_weight + conflict_weight
        consensus = 0.0 if decisive_weight <= 0.0 else support_weight / decisive_weight
        mean_quality = 0.0 if not support else sum(v.quality * v.source_weight for v in support) / max(1e-12, support_weight)
        membership_hash = hashlib.sha256("\n".join(cluster).encode()).hexdigest()[:20]
        assessments.append(HypothesisAssessment(
            node.node_id, membership_hash, "candidate", consensus,
            len(support), len(conflict), len(abstain), decisive,
            support_weight, conflict_weight, decisive_weight, mean_quality,
            [asdict(v) for v in votes],
        ))
    return assessments, policy_hash


def track_hypothesis_run(
    db: sqlite3.Connection,
    doc: ForgeDocument,
    target_id: str,
    sense_policy: SenseCompilePolicy,
    consensus_policy: ConsensusPolicy | None = None,
) -> ConsensusReport:
    cp = consensus_policy or ConsensusPolicy()
    assessments, policy_hash = assess_hypotheses(db, doc, target_id, sense_policy, cp)
    output_hash = hashlib.sha256(json.dumps(doc.to_dict(), sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    cur = db.execute(
        "INSERT INTO hypothesis_runs(compiler,target_id,policy_hash,output_hash,metadata_json) VALUES(?,?,?,?,?)",
        (CONSENSUS_FORMAT, target_id, policy_hash, output_hash, json.dumps({"version": CONSENSUS_VERSION}, sort_keys=True)),
    )
    run_id = int(cur.lastrowid)
    current_ids = {a.hypothesis_id for a in assessments}
    missing_reports: list[dict[str, Any]] = []

    # Mark previously known hypotheses that disappeared under the same policy.
    previous = db.execute(
        "SELECT hypothesis_id,status,seen_runs,consecutive_seen,missed_runs,metadata_json FROM hypothesis_states WHERE target_id=? AND policy_hash=?",
        (target_id, policy_hash),
    ).fetchall()
    for hid, status, seen_runs, consecutive, missed, meta_json in previous:
        if hid in current_ids:
            continue
        new_missed = int(missed) + 1
        new_status = "quarantined" if new_missed >= cp.quarantine_after_misses else "contested"
        meta = json.loads(meta_json or "{}")
        meta["last_transition_reason"] = "missing_from_rebuild"
        db.execute(
            "UPDATE hypothesis_states SET status=?,consecutive_seen=0,missed_runs=?,metadata_json=? WHERE hypothesis_id=? AND policy_hash=?",
            (new_status, new_missed, json.dumps(meta, sort_keys=True), hid, policy_hash),
        )
        missing_reports.append({"hypothesis_id": hid, "previous_status": status, "status": new_status, "missed_runs": new_missed})

    node_map = {n.node_id: n for n in doc.nodes}
    for a in assessments:
        row = db.execute(
            "SELECT status,first_seen_run,last_seen_run,seen_runs,consecutive_seen,missed_runs,membership_hash FROM hypothesis_states WHERE hypothesis_id=? AND policy_hash=?",
            (a.hypothesis_id, policy_hash),
        ).fetchone()
        if row:
            prev_status, first_run, last_run, seen_runs, consecutive, missed, old_membership = row
            consecutive_seen = int(consecutive) + 1 if int(last_run) == run_id - 1 and old_membership == a.membership_hash else 1
            seen_runs = int(seen_runs) + 1
            first_seen = int(first_run)
        else:
            prev_status = None
            consecutive_seen = 1
            seen_runs = 1
            first_seen = run_id
        status = "candidate"
        reason = "insufficient_independent_lineage_confirmation"
        if a.conflict_sources >= cp.quarantine_conflicts and a.conflict_weight >= cp.quarantine_conflict_weight:
            status = "quarantined"; reason = "weighted_multi_lineage_conflict"
        elif a.conflict_weight > 0.0 and a.consensus_score < cp.stable_consensus:
            status = "contested"; reason = "weighted_lineage_conflict"
        elif a.support_sources >= cp.stable_min_sources and a.support_weight >= cp.stable_min_support_weight and consecutive_seen >= cp.stable_min_runs and a.consensus_score >= cp.stable_consensus:
            status = "stable"; reason = "repeated_weighted_cross_lineage_consensus"
        a.previous_status = prev_status
        a.status = status
        a.seen_runs = seen_runs
        a.consecutive_seen = consecutive_seen
        a.missed_runs = 0
        meta = {
            "version": CONSENSUS_VERSION,
            "reason": reason,
            "votes": a.votes,
            "mean_support_quality": a.mean_support_quality,
            "support_weight": a.support_weight,
            "conflict_weight": a.conflict_weight,
            "decisive_weight": a.decisive_weight,
        }
        db.execute(
            "INSERT INTO hypothesis_states(hypothesis_id,target_id,policy_hash,membership_hash,status,first_seen_run,last_seen_run,seen_runs,consecutive_seen,missed_runs,consensus_score,support_sources,conflict_sources,metadata_json) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(hypothesis_id,policy_hash) DO UPDATE SET membership_hash=excluded.membership_hash,status=excluded.status,last_seen_run=excluded.last_seen_run,seen_runs=excluded.seen_runs,consecutive_seen=excluded.consecutive_seen,missed_runs=0,consensus_score=excluded.consensus_score,support_sources=excluded.support_sources,conflict_sources=excluded.conflict_sources,metadata_json=excluded.metadata_json",
            (a.hypothesis_id, target_id, policy_hash, a.membership_hash, status, first_seen, run_id, seen_runs, consecutive_seen, 0, a.consensus_score, a.support_sources, a.conflict_sources, json.dumps(meta, ensure_ascii=False, sort_keys=True)),
        )
        if a.hypothesis_id in node_map:
            node_map[a.hypothesis_id].metadata.update({
                "lifecycle_status": status,
                "consensus_score": a.consensus_score,
                "support_sources": a.support_sources,
                "conflict_sources": a.conflict_sources,
                "support_weight": a.support_weight,
                "conflict_weight": a.conflict_weight,
                "seen_runs": seen_runs,
                "consecutive_seen": consecutive_seen,
                "consensus_policy_hash": policy_hash,
            })
    db.commit()
    return ConsensusReport(
        CONSENSUS_FORMAT, CONSENSUS_VERSION, target_id, policy_hash, run_id,
        [asdict(a) for a in assessments], missing_reports, asdict(cp),
    )


def list_hypothesis_states(db: sqlite3.Connection, target_id: str | None = None) -> list[dict[str, Any]]:
    sql = "SELECT hypothesis_id,target_id,policy_hash,status,first_seen_run,last_seen_run,seen_runs,consecutive_seen,missed_runs,consensus_score,support_sources,conflict_sources,metadata_json FROM hypothesis_states"
    args: tuple[Any, ...] = ()
    if target_id:
        sql += " WHERE target_id=?"; args = (target_id,)
    sql += " ORDER BY target_id,policy_hash,hypothesis_id"
    out = []
    for row in db.execute(sql, args).fetchall():
        out.append({
            "hypothesis_id": row[0], "target_id": row[1], "policy_hash": row[2], "status": row[3],
            "first_seen_run": row[4], "last_seen_run": row[5], "seen_runs": row[6], "consecutive_seen": row[7],
            "missed_runs": row[8], "consensus_score": row[9], "support_sources": row[10], "conflict_sources": row[11],
            "metadata": json.loads(row[12] or "{}"),
        })
    return out
