from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import json
import sqlite3

from .canonical import composite_id
from .compiler import CompilePolicy, aggregate_relations
from .frame_induction import _load_node_recursive
from .ir import ForgeDocument, IRNode, Modality, NodeKind, RelationEvidence, RelationType, SourceRef

HIERARCHICAL_FRAME_TRANSFER_FORMAT = "equinox-forge-hierarchical-frame-transfer-v1"
HIERARCHICAL_FRAME_TRANSFER_VERSION = "0.26"


@dataclass(frozen=True)
class HierarchicalFrameTransferPolicy:
    min_ancestor_hops: int = 1
    max_ancestor_hops: int = 4
    min_path_edge_quality: float = 0.0
    prefer_highest_ancestor: bool = True
    require_recursive_concept: bool = True
    prediction_confidence_scale: float = 1.0
    hierarchical_admission_min_path_score: float = 0.55

    def validate(self) -> None:
        if self.min_ancestor_hops < 1:
            raise ValueError("min_ancestor_hops must be >= 1")
        if self.max_ancestor_hops < self.min_ancestor_hops:
            raise ValueError("max_ancestor_hops must be >= min_ancestor_hops")
        for name, value in (
            ("min_path_edge_quality", self.min_path_edge_quality),
            ("prediction_confidence_scale", self.prediction_confidence_scale),
            ("hierarchical_admission_min_path_score", self.hierarchical_admission_min_path_score),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")


@dataclass
class HierarchicalFrameBinding:
    original_frame_node_id: str
    transferred_frame_node_id: str
    original_slot_node_id: str
    transferred_slot_node_id: str
    base_prototype_node_id: str
    higher_concept_node_id: str
    higher_concept_surface: str | None
    ancestor_hops: int
    path_min_quality: float
    target_hierarchy_depth: int
    prediction_targets: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HierarchicalFrameTransferReport:
    format: str
    version: str
    frames_considered: int
    frames_with_prototype: int
    frames_promoted: int
    frames_without_eligible_ancestor: int
    generated_nodes: int
    generated_prediction_evidence: int
    bindings: list[HierarchicalFrameBinding]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["bindings"] = [x.to_dict() for x in self.bindings]
        return d


def _metadata(raw: str | None) -> dict[str, Any]:
    try:
        return json.loads(raw or "{}")
    except Exception:
        return {}


def _node_metadata_map(db: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    return {str(node_id): _metadata(metadata_json) for node_id, metadata_json in db.execute(
        "SELECT node_id,metadata_json FROM nodes"
    ).fetchall()}


def _is_a_adjacency(db: sqlite3.Connection, min_edge_quality: float) -> dict[str, list[tuple[str, float]]]:
    out: dict[str, list[tuple[str, float]]] = {}
    policy = CompilePolicy(min_score=0.0, min_confidence=0.0, min_sources=1)
    for rel in aggregate_relations(db, policy):
        if rel.relation_type != RelationType.IS_A.value:
            continue
        q = max(0.0, min(1.0, float(rel.score) * float(rel.confidence)))
        if q + 1e-12 < min_edge_quality:
            continue
        out.setdefault(rel.from_id, []).append((rel.to_id, q))
    for key in out:
        out[key].sort(key=lambda x: x[0])
    return out


def _eligible_ancestors(
    start_id: str,
    adjacency: dict[str, list[tuple[str, float]]],
    metadata: dict[str, dict[str, Any]],
    policy: HierarchicalFrameTransferPolicy,
) -> list[dict[str, Any]]:
    # Keep the best (highest minimum-edge quality) path seen for each node at a
    # given/fewer hop count. IS_A is expected to be acyclic, but the seen guard also
    # makes malformed input bounded and deterministic.
    queue: list[tuple[str, int, float, list[str]]] = [(start_id, 0, 1.0, [start_id])]
    best: dict[str, tuple[int, float, list[str]]] = {start_id: (0, 1.0, [start_id])}
    pos = 0
    while pos < len(queue):
        node_id, hops, path_quality, path = queue[pos]
        pos += 1
        if hops >= policy.max_ancestor_hops:
            continue
        for nxt, edge_q in adjacency.get(node_id, []):
            if nxt in path:
                continue
            nh = hops + 1
            nq = min(path_quality, edge_q)
            old = best.get(nxt)
            if old is not None and (old[0] < nh or (old[0] == nh and old[1] >= nq - 1e-12)):
                continue
            npath = path + [nxt]
            best[nxt] = (nh, nq, npath)
            queue.append((nxt, nh, nq, npath))

    rows: list[dict[str, Any]] = []
    for node_id, (hops, quality, path) in best.items():
        if node_id == start_id or hops < policy.min_ancestor_hops or hops > policy.max_ancestor_hops:
            continue
        meta = metadata.get(node_id, {})
        if policy.require_recursive_concept and meta.get("role") != "recursive_latent_concept":
            continue
        rows.append({
            "node_id": node_id,
            "hops": hops,
            "path_min_quality": quality,
            "path": path,
            "hierarchy_depth": int(meta.get("hierarchy_depth", 0) or 0),
        })
    rows.sort(key=lambda r: (
        -int(r["hops"]) if policy.prefer_highest_ancestor else int(r["hops"]),
        -int(r["hierarchy_depth"]),
        -float(r["path_min_quality"]),
        str(r["node_id"]),
    ))
    return rows


def _frame_rows(db: sqlite3.Connection) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    for node_id, metadata_json in db.execute(
        "SELECT node_id,metadata_json FROM nodes WHERE modality=? AND kind=? ORDER BY node_id",
        (Modality.TEXT.value, NodeKind.TEMPLATE.value),
    ).fetchall():
        meta = _metadata(metadata_json)
        if meta.get("role") == "induced_frame_schema" and not meta.get("hierarchical_transfer"):
            rows.append((str(node_id), meta))
    return rows


def promote_frames_to_higher_concepts(
    db: sqlite3.Connection,
    policy: HierarchicalFrameTransferPolicy | None = None,
    *,
    source_id: str = "derived:hierarchical-frame-transfer",
    source_version: str = HIERARCHICAL_FRAME_TRANSFER_VERSION,
) -> tuple[ForgeDocument, HierarchicalFrameTransferReport]:
    """Promote prototype-typed frame slots to a reachable higher latent concept.

    The operation copies the frame into a new derived frame. It does not rewrite or
    delete the original frame and it does not add any new target knowledge. Only the
    slot admission type is broadened through an already compiled IS_A hierarchy.
    """
    policy = policy or HierarchicalFrameTransferPolicy()
    policy.validate()
    metadata = _node_metadata_map(db)
    adjacency = _is_a_adjacency(db, policy.min_path_edge_quality)
    compiled = aggregate_relations(db, CompilePolicy(min_score=0.0, min_confidence=0.0, min_sources=1))
    prediction_from: dict[str, list[Any]] = {}
    for rel in compiled:
        if rel.relation_type == RelationType.PREDICTION.value:
            prediction_from.setdefault(rel.from_id, []).append(rel)

    all_nodes: dict[str, IRNode] = {}
    evidence: list[RelationEvidence] = []
    bindings: list[HierarchicalFrameBinding] = []
    considered = with_proto = no_ancestor = 0
    src = SourceRef(source_id, source_version, "hierarchical_frame_transfer_v1")

    for frame_id, frame_meta in _frame_rows(db):
        considered += 1
        slot_ids = [str(x) for x in (frame_meta.get("slot_node_ids") or [])]
        if len(slot_ids) != 1:
            continue
        original_slot = _load_node_recursive(db, slot_ids[0], all_nodes)
        slot_meta = dict(original_slot.metadata or {})
        profile = dict(slot_meta.get("semantic_admission") or {})
        base_prototype_id = str(profile.get("prototype_node_id") or slot_meta.get("latent_prototype_node_id") or "")
        if not base_prototype_id:
            continue
        with_proto += 1
        ancestors = _eligible_ancestors(base_prototype_id, adjacency, metadata, policy)
        if not ancestors:
            no_ancestor += 1
            continue
        selected = ancestors[0]
        higher_id = str(selected["node_id"])
        higher = _load_node_recursive(db, higher_id, all_nodes)
        original_frame = _load_node_recursive(db, frame_id, all_nodes)

        new_profile = {
            **profile,
            "mode": "hierarchical_prototype_admission_v1",
            "hierarchical_prototype_node_id": higher_id,
            "hierarchical_prototype_surface": higher.surface,
            "hierarchical_membership_relation_type": RelationType.IS_A.value,
            "hierarchical_max_hops": int(policy.max_ancestor_hops + 1),
            "hierarchical_min_path_score": float(policy.hierarchical_admission_min_path_score),
            "hierarchical_base_prototype_node_id": base_prototype_id,
            "hierarchical_compile_ancestor_hops": int(selected["hops"]),
            "hierarchical_compile_path_min_quality": float(selected["path_min_quality"]),
            "hierarchical_compile_path": list(selected["path"]),
            "provisional": True,
        }
        slot_label = f"[SLOT_H:{higher.surface or higher_id[:8]}]"
        new_slot_id = composite_id(
            Modality.TEXT.value,
            NodeKind.CONCEPT.value,
            [higher_id],
            label=f"hierarchical-frame-slot:{original_slot.node_id}:{higher_id}",
        )
        transferred_slot = IRNode(
            node_id=new_slot_id,
            modality=Modality.TEXT.value,
            kind=NodeKind.CONCEPT.value,
            level=max(original_slot.level, higher.level) + 1,
            surface=slot_label,
            members=[higher_id],
            metadata={
                **slot_meta,
                "role": "hierarchical_frame_slot",
                "provisional": True,
                "semantic_admission": new_profile,
                "semantic_typed": True,
                "original_slot_node_id": original_slot.node_id,
                "base_prototype_node_id": base_prototype_id,
                "higher_concept_node_id": higher_id,
                "higher_concept_surface": higher.surface,
                "hierarchical_transfer": True,
                "forge_version": HIERARCHICAL_FRAME_TRANSFER_VERSION,
            },
        )
        all_nodes[transferred_slot.node_id] = transferred_slot

        new_members = [transferred_slot.node_id if x == original_slot.node_id else x for x in original_frame.members]
        new_frame_id = composite_id(
            Modality.TEXT.value,
            NodeKind.TEMPLATE.value,
            new_members,
            label=f"hierarchical-frame:{original_frame.node_id}:{higher_id}",
        )
        transferred_frame = IRNode(
            node_id=new_frame_id,
            modality=Modality.TEXT.value,
            kind=NodeKind.TEMPLATE.value,
            level=max((all_nodes.get(x) or _load_node_recursive(db, x, all_nodes)).level for x in new_members) + 1,
            surface=original_frame.surface,
            members=new_members,
            metadata={
                **dict(original_frame.metadata or {}),
                "role": "induced_frame_schema",
                "provisional": True,
                "slot_node_ids": [transferred_slot.node_id],
                "hierarchical_transfer": True,
                "original_frame_node_id": original_frame.node_id,
                "base_prototype_node_id": base_prototype_id,
                "higher_concept_node_id": higher_id,
                "higher_concept_surface": higher.surface,
                "ancestor_hops": int(selected["hops"]),
                "ancestor_path_min_quality": float(selected["path_min_quality"]),
                "forge_version": HIERARCHICAL_FRAME_TRANSFER_VERSION,
            },
        )
        all_nodes[transferred_frame.node_id] = transferred_frame

        targets: list[str] = []
        for rel in sorted(prediction_from.get(original_frame.node_id, []), key=lambda r: (r.to_id, -r.score)):
            _load_node_recursive(db, rel.to_id, all_nodes)
            evidence.append(RelationEvidence(
                transferred_frame.node_id,
                rel.to_id,
                RelationType.PREDICTION.value,
                float(rel.score),
                max(0.0, min(1.0, float(rel.confidence) * float(policy.prediction_confidence_scale))),
                src,
                {
                    "evidence_kind": "hierarchical_frame_prediction_transfer",
                    "original_frame_node_id": original_frame.node_id,
                    "base_prototype_node_id": base_prototype_id,
                    "higher_concept_node_id": higher_id,
                    "no_new_target_knowledge": True,
                    "provisional": True,
                },
            ))
            targets.append(rel.to_id)

        bindings.append(HierarchicalFrameBinding(
            original_frame_node_id=original_frame.node_id,
            transferred_frame_node_id=transferred_frame.node_id,
            original_slot_node_id=original_slot.node_id,
            transferred_slot_node_id=transferred_slot.node_id,
            base_prototype_node_id=base_prototype_id,
            higher_concept_node_id=higher_id,
            higher_concept_surface=higher.surface,
            ancestor_hops=int(selected["hops"]),
            path_min_quality=float(selected["path_min_quality"]),
            target_hierarchy_depth=int(selected["hierarchy_depth"]),
            prediction_targets=sorted(set(targets)),
        ))

    doc = ForgeDocument(
        nodes=sorted(all_nodes.values(), key=lambda n: (n.level, n.node_id)),
        evidence=sorted(evidence, key=lambda e: (e.from_id, e.to_id, e.relation_type)),
    )
    report = HierarchicalFrameTransferReport(
        format=HIERARCHICAL_FRAME_TRANSFER_FORMAT,
        version=HIERARCHICAL_FRAME_TRANSFER_VERSION,
        frames_considered=considered,
        frames_with_prototype=with_proto,
        frames_promoted=len(bindings),
        frames_without_eligible_ancestor=no_ancestor,
        generated_nodes=sum(2 for _ in bindings),
        generated_prediction_evidence=len(evidence),
        bindings=bindings,
        notes=[
            "Promotion broadens only slot admission through an already compiled IS_A hierarchy; it does not invent new prediction targets.",
            "Original frames remain intact. The promoted frame is a separate provisional derived artifact and can be removed/rebuilt independently.",
            "A runtime candidate must reach the promoted higher concept through compiled IS_A membership edges within the configured hop and path-quality limits.",
            "Higher-concept labels remain latent Concept_* identifiers; no human ontology name is asserted by this step.",
        ],
    )
    return doc, report
