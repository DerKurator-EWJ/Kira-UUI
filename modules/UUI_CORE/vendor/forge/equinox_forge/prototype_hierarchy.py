from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import hashlib
import json
import sqlite3

from .canonical import composite_id
from .compiler import CompilePolicy, aggregate_relations
from .frame_induction import _load_node_recursive
from .ir import ForgeDocument, IRNode, Modality, NodeKind, RelationEvidence, RelationType, SourceRef

PROTOTYPE_HIERARCHY_FORMAT = "equinox-forge-prototype-hierarchy-v1"
PROTOTYPE_HIERARCHY_VERSION = "0.25"


@dataclass(frozen=True)
class PrototypeHierarchyPolicy:
    min_children: int = 2
    max_depth: int = 4
    min_cross_leaf_score: float = 0.80
    score_decay_per_depth: float = 0.10
    min_score_floor: float = 0.55
    min_cross_leaf_pair_fraction: float = 1.0
    relation_types: tuple[str, ...] = (
        RelationType.SEMANTIC_SIMILARITY.value,
        RelationType.SUBSTITUTION.value,
        RelationType.CONTEXTUAL_SIMILARITY.value,
    )
    membership_confidence: float = 0.92
    require_disjoint_leaf_sets: bool = True

    def validate(self) -> None:
        if self.min_children < 2:
            raise ValueError("min_children must be >= 2")
        if self.max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        for name, value in (
            ("min_cross_leaf_score", self.min_cross_leaf_score),
            ("score_decay_per_depth", self.score_decay_per_depth),
            ("min_score_floor", self.min_score_floor),
            ("min_cross_leaf_pair_fraction", self.min_cross_leaf_pair_fraction),
            ("membership_confidence", self.membership_confidence),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        if not self.relation_types:
            raise ValueError("relation_types must not be empty")


@dataclass
class HierarchyMergeReport:
    node_id: str
    hierarchy_depth: int
    threshold: float
    child_node_ids: list[str]
    child_surfaces: list[str]
    leaf_node_ids: list[str]
    leaf_surfaces: list[str]
    cross_pair_count: int
    supporting_pair_count: int
    supporting_pair_fraction: float
    mean_cross_score: float
    min_cross_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PrototypeHierarchyReport:
    format: str
    version: str
    base_prototypes: int
    generated_concepts: int
    generated_memberships: int
    levels_generated: int
    highest_hierarchy_depth: int
    frontier_remaining: int
    merges: list[HierarchyMergeReport]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["merges"] = [x.to_dict() for x in self.merges]
        return d


def _stable_hash(obj: Any) -> str:
    raw = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _metadata(row_json: str | None) -> dict[str, Any]:
    try:
        return json.loads(row_json or "{}")
    except Exception:
        return {}


def _base_prototype_ids(db: sqlite3.Connection) -> list[str]:
    out: list[str] = []
    for node_id, metadata_json in db.execute(
        "SELECT node_id,metadata_json FROM nodes WHERE modality=? AND kind=? ORDER BY node_id",
        (Modality.TEXT.value, NodeKind.CONCEPT.value),
    ).fetchall():
        if _metadata(metadata_json).get("role") == "latent_slot_prototype":
            out.append(str(node_id))
    return out


def _semantic_quality_map(db: sqlite3.Connection, relation_types: tuple[str, ...]) -> dict[tuple[str, str], float]:
    allowed = set(relation_types)
    out: dict[tuple[str, str], float] = {}
    for rel in aggregate_relations(db, CompilePolicy(min_score=0.0, min_confidence=0.0, min_sources=1)):
        if rel.relation_type not in allowed:
            continue
        q = max(0.0, min(1.0, float(rel.score) * float(rel.confidence)))
        key = (rel.from_id, rel.to_id)
        if q > out.get(key, 0.0):
            out[key] = q
    return out


def _leaf_ids(node: IRNode, node_map: dict[str, IRNode]) -> list[str]:
    meta = node.metadata or {}
    explicit = meta.get("leaf_node_ids") or meta.get("filler_node_ids")
    if isinstance(explicit, list) and explicit:
        return sorted({str(x) for x in explicit})
    if meta.get("role") == "latent_slot_prototype":
        return sorted({str(x) for x in node.members})
    leaves: set[str] = set()
    for child_id in node.members:
        child = node_map.get(child_id)
        if child is None:
            continue
        if child.kind == NodeKind.SURFACE.value:
            leaves.add(child.node_id)
        elif child.kind == NodeKind.CONCEPT.value:
            leaves.update(_leaf_ids(child, node_map))
    return sorted(leaves)


def _leaf_surfaces(leaf_ids: list[str], node_map: dict[str, IRNode]) -> list[str]:
    return sorted({str(node_map[x].surface) for x in leaf_ids if x in node_map and node_map[x].surface is not None})


def _pair_stats(
    a: IRNode,
    b: IRNode,
    node_map: dict[str, IRNode],
    quality: dict[tuple[str, str], float],
    threshold: float,
) -> dict[str, Any]:
    a_leaves = _leaf_ids(a, node_map)
    b_leaves = _leaf_ids(b, node_map)
    scores: list[float] = []
    supporting = 0
    for x in a_leaves:
        for y in b_leaves:
            if x == y:
                # Overlapping leaf support is not independent evidence for a higher class.
                continue
            score = max(quality.get((x, y), 0.0), quality.get((y, x), 0.0))
            scores.append(score)
            if score + 1e-12 >= threshold:
                supporting += 1
    count = len(scores)
    return {
        "a_leaf_ids": a_leaves,
        "b_leaf_ids": b_leaves,
        "pair_count": count,
        "supporting_pair_count": supporting,
        "supporting_pair_fraction": (supporting / count if count else 0.0),
        "mean_score": (sum(scores) / count if count else 0.0),
        "min_score": (min(scores) if scores else 0.0),
    }


def _compatible(
    a: IRNode,
    b: IRNode,
    node_map: dict[str, IRNode],
    quality: dict[tuple[str, str], float],
    threshold: float,
    policy: PrototypeHierarchyPolicy,
) -> tuple[bool, dict[str, Any]]:
    a_leaves = set(_leaf_ids(a, node_map))
    b_leaves = set(_leaf_ids(b, node_map))
    if policy.require_disjoint_leaf_sets and a_leaves & b_leaves:
        return False, {
            "pair_count": 0, "supporting_pair_count": 0, "supporting_pair_fraction": 0.0,
            "mean_score": 0.0, "min_score": 0.0, "reason": "overlapping_leaf_support",
        }
    stats = _pair_stats(a, b, node_map, quality, threshold)
    ok = (
        stats["pair_count"] > 0
        and stats["supporting_pair_fraction"] + 1e-12 >= policy.min_cross_leaf_pair_fraction
        and stats["mean_score"] + 1e-12 >= threshold
    )
    stats["reason"] = "compatible" if ok else "insufficient_cross_leaf_support"
    return ok, stats


def _cluster_frontier(
    frontier: list[IRNode],
    node_map: dict[str, IRNode],
    quality: dict[tuple[str, str], float],
    threshold: float,
    policy: PrototypeHierarchyPolicy,
) -> tuple[list[list[IRNode]], list[IRNode]]:
    # Deterministic complete-link agglomeration. A candidate can join a cluster only
    # when it is independently compatible with every current member.
    remaining = sorted(frontier, key=lambda n: n.node_id)
    clusters: list[list[IRNode]] = []
    used: set[str] = set()
    for seed in remaining:
        if seed.node_id in used:
            continue
        cluster = [seed]
        used.add(seed.node_id)
        changed = True
        while changed:
            changed = False
            best: tuple[float, str, IRNode] | None = None
            for cand in remaining:
                if cand.node_id in used:
                    continue
                pair_results = [_compatible(cand, member, node_map, quality, threshold, policy) for member in cluster]
                if not all(ok for ok, _ in pair_results):
                    continue
                score = min(float(stats["mean_score"]) for _, stats in pair_results)
                key = (score, cand.node_id, cand)
                if best is None or (key[0], key[1]) > (best[0], best[1]):
                    best = key
            if best is not None:
                cluster.append(best[2])
                used.add(best[2].node_id)
                changed = True
        clusters.append(sorted(cluster, key=lambda n: n.node_id))
    accepted = [c for c in clusters if len(c) >= policy.min_children]
    singles = [n for c in clusters if len(c) < policy.min_children for n in c]
    return accepted, singles


def _hierarchy_node(
    children: list[IRNode],
    depth: int,
    node_map: dict[str, IRNode],
    quality: dict[tuple[str, str], float],
    threshold: float,
    policy: PrototypeHierarchyPolicy,
) -> tuple[IRNode, HierarchyMergeReport]:
    child_ids = sorted(n.node_id for n in children)
    leaf_ids = sorted({x for child in children for x in _leaf_ids(child, node_map)})
    pair_scores: list[float] = []
    pair_min_scores: list[float] = []
    support_pairs = 0
    total_pairs = 0
    for i, a in enumerate(children):
        for b in children[i + 1:]:
            ok, stats = _compatible(a, b, node_map, quality, threshold, policy)
            if not ok:
                raise ValueError("attempted to build an incompatible recursive concept")
            total_pairs += int(stats["pair_count"])
            support_pairs += int(stats["supporting_pair_count"])
            # Weight pair means by the number of underlying cross-leaf pairs.
            pair_scores.extend([float(stats["mean_score"])] * max(1, int(stats["pair_count"])))
            pair_min_scores.append(float(stats["min_score"]))
    signature = {
        "role": "recursive_latent_concept",
        "depth": depth,
        "children": child_ids,
        "leaves": leaf_ids,
        "relation_types": list(policy.relation_types),
    }
    short = _stable_hash(signature)[:10]
    node_id = composite_id(Modality.TEXT.value, NodeKind.CONCEPT.value, child_ids, label="recursive-latent-concept:" + short)
    mean_score = sum(pair_scores) / len(pair_scores) if pair_scores else 0.0
    min_score = min(pair_min_scores) if pair_min_scores else 0.0
    node = IRNode(
        node_id=node_id,
        modality=Modality.TEXT.value,
        kind=NodeKind.CONCEPT.value,
        level=max(n.level for n in children) + 1,
        surface=f"Concept_{short}",
        members=child_ids,
        metadata={
            "role": "recursive_latent_concept",
            "provisional": True,
            "hierarchy_depth": depth,
            "child_concept_ids": child_ids,
            "child_concept_surfaces": sorted(str(n.surface) for n in children if n.surface is not None),
            "leaf_node_ids": leaf_ids,
            "leaf_surfaces": _leaf_surfaces(leaf_ids, node_map),
            "support_count": len(child_ids),
            "cross_leaf_threshold": threshold,
            "cross_leaf_pair_fraction": (support_pairs / total_pairs if total_pairs else 0.0),
            "cross_leaf_mean_score": mean_score,
            "cross_leaf_min_score": min_score,
            "semantic_relation_types": list(policy.relation_types),
            "forge_version": PROTOTYPE_HIERARCHY_VERSION,
        },
    )
    report = HierarchyMergeReport(
        node_id=node.node_id,
        hierarchy_depth=depth,
        threshold=threshold,
        child_node_ids=child_ids,
        child_surfaces=sorted(str(n.surface) for n in children if n.surface is not None),
        leaf_node_ids=leaf_ids,
        leaf_surfaces=_leaf_surfaces(leaf_ids, node_map),
        cross_pair_count=total_pairs,
        supporting_pair_count=support_pairs,
        supporting_pair_fraction=(support_pairs / total_pairs if total_pairs else 0.0),
        mean_cross_score=mean_score,
        min_cross_score=min_score,
    )
    return node, report


def induce_prototype_hierarchy(
    db: sqlite3.Connection,
    policy: PrototypeHierarchyPolicy | None = None,
    *,
    source_id: str = "derived:prototype-hierarchy",
    source_version: str = PROTOTYPE_HIERARCHY_VERSION,
) -> tuple[ForgeDocument, PrototypeHierarchyReport]:
    policy = policy or PrototypeHierarchyPolicy()
    policy.validate()
    base_ids = _base_prototype_ids(db)
    all_nodes: dict[str, IRNode] = {}
    base_nodes: list[IRNode] = []
    for node_id in base_ids:
        base_nodes.append(_load_node_recursive(db, node_id, all_nodes))
    quality = _semantic_quality_map(db, policy.relation_types)
    src = SourceRef(source_id, source_version, "recursive_prototype_hierarchy_v1")
    evidence: list[RelationEvidence] = []
    merges: list[HierarchyMergeReport] = []
    generated: list[IRNode] = []

    frontier = sorted(base_nodes, key=lambda n: n.node_id)
    highest = 0
    levels_generated = 0
    for depth in range(1, policy.max_depth + 1):
        if len(frontier) < policy.min_children:
            break
        threshold = max(policy.min_score_floor, policy.min_cross_leaf_score - (depth - 1) * policy.score_decay_per_depth)
        clusters, singles = _cluster_frontier(frontier, all_nodes, quality, threshold, policy)
        if not clusters:
            break
        next_frontier: list[IRNode] = []
        for cluster in clusters:
            parent, merge_report = _hierarchy_node(cluster, depth, all_nodes, quality, threshold, policy)
            if parent.node_id in all_nodes:
                parent = all_nodes[parent.node_id]
            else:
                all_nodes[parent.node_id] = parent
                generated.append(parent)
                merges.append(merge_report)
                for child in cluster:
                    evidence.append(RelationEvidence(
                        from_id=child.node_id,
                        to_id=parent.node_id,
                        relation_type=RelationType.IS_A.value,
                        score=max(0.0, min(1.0, merge_report.mean_cross_score)),
                        confidence=float(policy.membership_confidence),
                        source=src,
                        metadata={
                            "evidence_kind": "recursive_prototype_membership",
                            "hierarchy_depth": depth,
                            "derived_from_leaf_support": True,
                            "cross_leaf_threshold": threshold,
                            "cross_leaf_pair_fraction": merge_report.supporting_pair_fraction,
                            "provisional": True,
                        },
                    ))
            next_frontier.append(parent)
        # Unmerged concepts stay in the frontier so they may join a broader class
        # when the threshold relaxes at a higher recursion depth.
        next_frontier.extend(singles)
        next_frontier = sorted({n.node_id: n for n in next_frontier}.values(), key=lambda n: n.node_id)
        if {n.node_id for n in next_frontier} == {n.node_id for n in frontier}:
            break
        frontier = next_frontier
        highest = depth
        levels_generated += 1

    doc = ForgeDocument(
        nodes=sorted(all_nodes.values(), key=lambda n: (n.level, n.node_id)),
        evidence=sorted(evidence, key=lambda e: (e.from_id, e.to_id, e.relation_type)),
    )
    report = PrototypeHierarchyReport(
        format=PROTOTYPE_HIERARCHY_FORMAT,
        version=PROTOTYPE_HIERARCHY_VERSION,
        base_prototypes=len(base_nodes),
        generated_concepts=len(generated),
        generated_memberships=len(evidence),
        levels_generated=levels_generated,
        highest_hierarchy_depth=highest,
        frontier_remaining=len(frontier),
        merges=merges,
        notes=[
            "Recursive concepts are built only from previously induced concepts whose underlying leaf fillers have independent semantic support.",
            "Higher concepts contain lower concept nodes as members; raw surface/Q nodes are not flattened into the composition hierarchy.",
            "The cross-leaf threshold may relax by depth, but never below min_score_floor; every merge remains complete-link compatible at its own level.",
            "Child->parent IS_A edges are derived and provisional. Raw semantic evidence remains unchanged and can rebuild the hierarchy later.",
            "No ontology label such as ANIMAL is asserted automatically; Concept_* names remain latent until separately grounded.",
        ],
    )
    return doc, report
