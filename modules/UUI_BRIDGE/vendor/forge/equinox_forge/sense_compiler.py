from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any
import hashlib
import json
import math
import sqlite3

from .canonical import composite_id
from .compiler import CompilePolicy, aggregate_relations
from .ir import ForgeDocument, IRNode, RelationEvidence, RelationType, SourceRef, NodeKind, Modality

SENSE_COMPILER_FORMAT = "equinox-forge-sense-compiler-v1"
SENSE_COMPILER_VERSION = "0.7"


@dataclass(frozen=True)
class FusionWeights:
    hidden_similarity: float = 0.55
    lexical_overlap: float = 0.05
    semantic_similarity: float = 0.15
    substitution: float = 0.10
    transition: float = 0.05
    prediction_signature: float = 0.10

    def validate(self) -> None:
        vals = asdict(self)
        if any(v < 0 for v in vals.values()):
            raise ValueError("fusion weights must be >= 0")
        if sum(vals.values()) <= 0:
            raise ValueError("at least one fusion weight must be > 0")


@dataclass
class SenseCompilePolicy:
    threshold: float = 0.62
    min_channels: int = 2
    min_contexts: int = 2
    include_singletons: bool = False
    min_relation_score: float = 0.05
    min_relation_confidence: float = 0.05
    min_relation_sources: int = 1
    weights: FusionWeights = FusionWeights()

    def validate(self) -> None:
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError("threshold must be in [0,1]")
        if self.min_channels < 1:
            raise ValueError("min_channels must be >= 1")
        if self.min_contexts < 1:
            raise ValueError("min_contexts must be >= 1")
        self.weights.validate()


@dataclass
class PairFusion:
    a_context: str
    b_context: str
    fused_score: float
    available_channels: int
    channel_scores: dict[str, float | None]
    used_weights: dict[str, float]


@dataclass
class SenseCluster:
    sense_id: str
    target_id: str
    target_surface: str
    contexts: list[str]
    mean_internal_score: float
    min_internal_score: float
    evidence_channels: list[str]
    confidence: float


@dataclass
class SenseCompileReport:
    format: str
    target_id: str
    target_surface: str
    contexts_found: int
    pair_count: int
    accepted_pair_count: int
    clusters_created: int
    unresolved_contexts: int
    threshold: float
    min_channels: int
    weights: dict[str, float]
    clusters: list[dict[str, Any]]
    pairs: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _members(db: sqlite3.Connection, node_id: str) -> list[str]:
    return [r[0] for r in db.execute(
        "SELECT child_id FROM node_members WHERE parent_id=? ORDER BY position", (node_id,)
    ).fetchall()]


def _node_row(db: sqlite3.Connection, node_id: str):
    return db.execute(
        "SELECT node_id,modality,kind,level,surface,metadata_json FROM nodes WHERE node_id=?", (node_id,)
    ).fetchone()


def _node_from_db(db: sqlite3.Connection, node_id: str) -> IRNode:
    row = _node_row(db, node_id)
    if not row:
        raise KeyError(node_id)
    nid, modality, kind, level, surface, meta = row
    return IRNode(nid, modality, kind, int(level), surface, _members(db, nid), json.loads(meta or "{}"))


def _compiled_map(db: sqlite3.Connection, policy: SenseCompilePolicy):
    rels = aggregate_relations(db, CompilePolicy(
        min_score=policy.min_relation_score,
        min_confidence=policy.min_relation_confidence,
        min_sources=policy.min_relation_sources,
    ))
    out: dict[tuple[str,str,str], Any] = {}
    for r in rels:
        out[(r.from_id, r.to_id, r.relation_type)] = r
    return out


def _sym_relation(relmap, a: str, b: str, relation_type: str) -> float | None:
    vals = []
    r = relmap.get((a,b,relation_type))
    if r is not None: vals.append(float(r.score) * float(r.confidence))
    r = relmap.get((b,a,relation_type))
    if r is not None: vals.append(float(r.score) * float(r.confidence))
    return None if not vals else max(vals)


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b: return 1.0
    u = a | b
    return 0.0 if not u else len(a & b) / len(u)


def _cross_affinity(relmap, aa: list[str], bb: list[str], relation_type: str) -> float | None:
    if not aa or not bb:
        return None
    per_a: list[float] = []
    for a in aa:
        vals = []
        for b in bb:
            v = _sym_relation(relmap, a, b, relation_type)
            if v is not None:
                vals.append(v)
        if vals:
            per_a.append(max(vals))
    per_b: list[float] = []
    for b in bb:
        vals = []
        for a in aa:
            v = _sym_relation(relmap, a, b, relation_type)
            if v is not None:
                vals.append(v)
        if vals:
            per_b.append(max(vals))
    if not per_a and not per_b:
        return None
    sides = []
    if per_a: sides.append(sum(per_a)/len(per_a))
    if per_b: sides.append(sum(per_b)/len(per_b))
    return sum(sides)/len(sides)


def _prediction_signature(db: sqlite3.Connection, relmap, context_members: list[str]) -> dict[str, float]:
    # Any canonical TEMPLATE whose ordered members occur contiguously in this
    # context contributes its compiled prediction distribution. This lets corpus
    # n-grams and causal-teacher probes form one behavioural signature without
    # making their tokenizer/model identity part of the canonical node.
    if not context_members:
        return {}
    sig: dict[str,float] = {}
    template_ids = [r[0] for r in db.execute("SELECT node_id FROM nodes WHERE kind=?", (NodeKind.TEMPLATE.value,)).fetchall()]
    for tid in template_ids:
        tm = _members(db, tid)
        if not tm or len(tm) > len(context_members):
            continue
        matched = any(context_members[i:i+len(tm)] == tm for i in range(len(context_members)-len(tm)+1))
        if not matched:
            continue
        for (a,b,t), r in relmap.items():
            if a == tid and t == RelationType.PREDICTION.value:
                sig[b] = max(sig.get(b,0.0), float(r.score)*float(r.confidence))
    return sig


def _cosine_dict(a: dict[str,float], b: dict[str,float]) -> float | None:
    keys = set(a) | set(b)
    if not keys or not a or not b:
        return None
    dot = sum(a.get(k,0.0)*b.get(k,0.0) for k in keys)
    na = math.sqrt(sum(v*v for v in a.values())); nb = math.sqrt(sum(v*v for v in b.values()))
    if na <= 0 or nb <= 0:
        return None
    return max(0.0, min(1.0, dot/(na*nb)))


def _weighted_fuse(scores: dict[str,float|None], weights: FusionWeights) -> tuple[float,int,dict[str,float]]:
    wd = asdict(weights)
    used = {k: wd[k] for k,v in scores.items() if v is not None and wd.get(k,0.0) > 0}
    denom = sum(used.values())
    if denom <= 0:
        return 0.0, 0, {}
    score = sum(float(scores[k]) * w for k,w in used.items()) / denom
    supporting = sum(1 for k in used if float(scores[k]) > 0.0)
    return float(max(0.0,min(1.0,score))), supporting, used


def _complete_link_clusters(contexts: list[str], pair_scores: dict[tuple[str,str], PairFusion], threshold: float, min_channels: int) -> list[list[str]]:
    clusters = [[c] for c in sorted(contexts)]

    def pair(a,b):
        lo,hi=sorted((a,b)); return pair_scores.get((lo,hi))

    def can_merge(x: list[str], y: list[str]) -> tuple[bool,float]:
        vals=[]
        for a in x:
            for b in y:
                p=pair(a,b)
                if p is None or p.available_channels < min_channels or p.fused_score < threshold:
                    return False, 0.0
                vals.append(p.fused_score)
        return True, min(vals) if vals else 0.0

    while True:
        best=None
        for i in range(len(clusters)):
            for j in range(i+1,len(clusters)):
                ok,quality=can_merge(clusters[i],clusters[j])
                if not ok: continue
                key=(quality, -len(clusters[i])-len(clusters[j]), tuple(clusters[i]), tuple(clusters[j]))
                if best is None or key > best[0]: best=(key,i,j)
        if best is None: break
        _,i,j=best
        merged=sorted(clusters[i]+clusters[j])
        clusters=[c for k,c in enumerate(clusters) if k not in {i,j}] + [merged]
        clusters.sort(key=lambda c: tuple(c))
    return clusters


def compile_provisional_senses(
    db: sqlite3.Connection,
    target_id: str,
    policy: SenseCompilePolicy | None = None,
) -> tuple[ForgeDocument, SenseCompileReport]:
    policy = policy or SenseCompilePolicy(); policy.validate()
    target = _node_from_db(db, target_id)
    if target.modality != Modality.TEXT.value or target.kind != NodeKind.SURFACE.value:
        raise ValueError("sense compiler target must be a canonical text surface")

    relmap = _compiled_map(db, policy)
    # Contextual-use edges are target -> context. Aggregate endpoints from raw DB
    # so the compiler does not silently lose a context because of another channel's threshold.
    contexts = sorted({r[0] for r in db.execute(
        "SELECT DISTINCT to_id FROM evidence WHERE from_id=? AND relation_type=?",
        (target_id, RelationType.CONTEXTUAL_USE.value),
    ).fetchall() if _node_row(db, r[0])})

    context_members: dict[str,list[str]] = {}
    context_lex: dict[str,list[str]] = {}
    pred_sig: dict[str,dict[str,float]] = {}
    for c in contexts:
        mem=_members(db,c)
        context_members[c]=mem
        # Exclude the target itself for context-content similarity.
        context_lex[c]=[m for m in mem if m != target_id]
        pred_sig[c]=_prediction_signature(db, relmap, mem)

    pair_map: dict[tuple[str,str],PairFusion] = {}
    for i,a in enumerate(contexts):
        for b in contexts[i+1:]:
            hidden=_sym_relation(relmap,a,b,RelationType.CONTEXTUAL_SIMILARITY.value)
            lexical=_jaccard(set(context_lex[a]),set(context_lex[b]))
            semantic=_cross_affinity(relmap,context_lex[a],context_lex[b],RelationType.SEMANTIC_SIMILARITY.value)
            substitution=_cross_affinity(relmap,context_lex[a],context_lex[b],RelationType.SUBSTITUTION.value)
            transition=_cross_affinity(relmap,context_lex[a],context_lex[b],RelationType.TRANSITION.value)
            prediction=_cosine_dict(pred_sig[a],pred_sig[b])
            scores={
                "hidden_similarity":hidden,
                "lexical_overlap":lexical,
                "semantic_similarity":semantic,
                "substitution":substitution,
                "transition":transition,
                "prediction_signature":prediction,
            }
            fused,nch,used=_weighted_fuse(scores,policy.weights)
            lo,hi=sorted((a,b))
            pair_map[(lo,hi)]=PairFusion(lo,hi,fused,nch,scores,used)

    clusters_raw=_complete_link_clusters(contexts,pair_map,policy.threshold,policy.min_channels)
    selected=[c for c in clusters_raw if len(c)>=policy.min_contexts or policy.include_singletons]

    nodes: dict[str,IRNode] = {}
    def add_closure(nid: str) -> None:
        if nid in nodes:
            return
        node=_node_from_db(db,nid)
        nodes[nid]=node
        for child in node.members:
            add_closure(child)
    add_closure(target.node_id)
    for c in contexts:
        add_closure(c)
    evidence: list[RelationEvidence] = []
    cluster_reports: list[SenseCluster] = []
    src=SourceRef("equinox-forge-sense-compiler",SENSE_COMPILER_VERSION,"evidence_fusion_complete_link")

    for cluster in selected:
        internal=[]; channel_union=set()
        for i,a in enumerate(cluster):
            for b in cluster[i+1:]:
                p=pair_map[tuple(sorted((a,b)))]
                internal.append(p.fused_score)
                channel_union.update(k for k,v in p.channel_scores.items() if v is not None)
        mean_score=sum(internal)/len(internal) if internal else 1.0
        min_score=min(internal) if internal else 1.0
        members=[target_id]+sorted(cluster)
        label=f"sense:{target.surface or target_id}"
        sid=composite_id(Modality.TEXT.value,NodeKind.CONCEPT.value,members,label)
        level=max(_node_from_db(db,x).level for x in members)+1
        node=IRNode(
            sid,Modality.TEXT.value,NodeKind.CONCEPT.value,level,
            surface=label,members=members,
            metadata={
                "provisional":True,
                "compiler":SENSE_COMPILER_FORMAT,
                "compiler_version":SENSE_COMPILER_VERSION,
                "target_id":target_id,
                "target_surface":target.surface,
                "contexts":sorted(cluster),
                "threshold":policy.threshold,
                "min_channels":policy.min_channels,
                "fusion_weights":asdict(policy.weights),
            },
        )
        nodes[sid]=node
        confidence=float(max(0.0,min(1.0,mean_score)))
        evidence.append(RelationEvidence(
            target_id,sid,RelationType.ASSOCIATION.value,mean_score,confidence,src,
            {
                "derived":True,"role":"provisional_sense",
                "context_count":len(cluster),"min_internal_score":min_score,
                "channels":sorted(channel_union),
                "policy_hash":hashlib.sha256(json.dumps({"threshold":policy.threshold,"min_channels":policy.min_channels,"weights":asdict(policy.weights)},sort_keys=True).encode()).hexdigest()[:16],
            }
        ))
        cluster_reports.append(SenseCluster(sid,target_id,target.surface or "",sorted(cluster),mean_score,min_score,sorted(channel_union),confidence))

    selected_contexts={c for cl in selected for c in cl}
    accepted_pairs=sum(1 for p in pair_map.values() if p.available_channels>=policy.min_channels and p.fused_score>=policy.threshold)
    report=SenseCompileReport(
        format=SENSE_COMPILER_FORMAT,
        target_id=target_id,target_surface=target.surface or "",
        contexts_found=len(contexts),pair_count=len(pair_map),accepted_pair_count=accepted_pairs,
        clusters_created=len(cluster_reports),unresolved_contexts=len(set(contexts)-selected_contexts),
        threshold=policy.threshold,min_channels=policy.min_channels,weights=asdict(policy.weights),
        clusters=[asdict(c) for c in cluster_reports],pairs=[asdict(pair_map[k]) for k in sorted(pair_map)],
    )
    doc=ForgeDocument(nodes=sorted(nodes.values(),key=lambda n:(n.level,n.node_id)),evidence=evidence)
    return doc,report


def find_text_surface_id(db: sqlite3.Connection, surface: str) -> str:
    rows=db.execute("SELECT node_id FROM nodes WHERE modality=? AND kind=? AND surface=? ORDER BY node_id",
                    (Modality.TEXT.value,NodeKind.SURFACE.value,surface)).fetchall()
    if not rows:
        raise KeyError(f"no canonical text surface: {surface!r}")
    if len(rows)>1:
        raise ValueError(f"ambiguous canonical surface {surface!r}: {len(rows)} nodes")
    return rows[0][0]
