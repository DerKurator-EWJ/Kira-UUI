from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import json
import sqlite3

from .canonical import composite_id
from .compiler import CompilePolicy, aggregate_relations
from .frame_induction import _load_node_recursive, _template_contexts
from .ir import ForgeDocument, IRNode, Modality, NodeKind, RelationEvidence, RelationType, SourceRef

EXCEPTION_INDUCTION_FORMAT = "equinox-forge-exception-induction-v1"
EXCEPTION_INDUCTION_VERSION = "0.28"


@dataclass(frozen=True)
class ExceptionInductionPolicy:
    min_contexts: int = 2
    min_unique_fillers: int = 2
    min_target_context_fraction: float = 1.0
    min_exception_target_score: float = 0.50
    min_disagreement_margin: float = 0.20
    min_membership_quality: float = 0.50
    min_prototype_path_quality: float = 0.50
    max_prototype_hops: int = 4
    evidence_confidence: float = 0.92
    require_different_top1: bool = True

    def validate(self) -> None:
        if self.min_contexts < 2:
            raise ValueError("min_contexts must be >= 2")
        if self.min_unique_fillers < 2:
            raise ValueError("min_unique_fillers must be >= 2")
        if self.max_prototype_hops < 1:
            raise ValueError("max_prototype_hops must be >= 1")
        for name, value in (
            ("min_target_context_fraction", self.min_target_context_fraction),
            ("min_exception_target_score", self.min_exception_target_score),
            ("min_disagreement_margin", self.min_disagreement_margin),
            ("min_membership_quality", self.min_membership_quality),
            ("min_prototype_path_quality", self.min_prototype_path_quality),
            ("evidence_confidence", self.evidence_confidence),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")


@dataclass
class ExceptionOverrideBinding:
    general_frame_node_id: str
    override_frame_node_id: str
    subtype_prototype_node_id: str
    subtype_prototype_surface: str | None
    slot_node_id: str
    higher_concept_node_id: str
    pattern: list[str]
    support_context_ids: list[str]
    support_fillers: list[str]
    general_top_target_id: str
    general_top_target_surface: str | None
    general_top_score: float
    exception_top_target_id: str
    exception_top_target_surface: str | None
    exception_top_score: float
    subtype_general_target_score: float
    disagreement_margin: float
    prototype_path_hops: int
    prototype_path_quality: float
    prediction_targets: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExceptionInductionReport:
    format: str
    version: str
    general_frames_considered: int
    subtype_candidates_considered: int
    overrides_induced: int
    rejected_insufficient_contexts: int
    rejected_insufficient_fillers: int
    rejected_no_systematic_deviation: int
    rejected_existing_specific_frame: int
    generated_nodes: int
    generated_prediction_evidence: int
    bindings: list[ExceptionOverrideBinding]
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
    return {str(n): _metadata(m) for n, m in db.execute("SELECT node_id,metadata_json FROM nodes").fetchall()}


def _node_surface_map(db: sqlite3.Connection) -> dict[str, str | None]:
    return {str(n): (None if s is None else str(s)) for n, s in db.execute("SELECT node_id,surface FROM nodes").fetchall()}


def _compiled_relations(db: sqlite3.Connection):
    return aggregate_relations(db, CompilePolicy(min_score=0.0, min_confidence=0.0, min_sources=1))


def _is_a_graph(db: sqlite3.Connection, min_quality: float) -> tuple[dict[str, list[tuple[str, float]]], dict[tuple[str, str], float]]:
    adjacency: dict[str, list[tuple[str, float]]] = {}
    direct: dict[tuple[str, str], float] = {}
    for rel in _compiled_relations(db):
        if rel.relation_type != RelationType.IS_A.value:
            continue
        q = max(0.0, min(1.0, float(rel.score) * float(rel.confidence)))
        if q + 1e-12 < min_quality:
            continue
        adjacency.setdefault(rel.from_id, []).append((rel.to_id, q))
        direct[(rel.from_id, rel.to_id)] = max(direct.get((rel.from_id, rel.to_id), 0.0), q)
    for key in adjacency:
        adjacency[key].sort(key=lambda x: (x[0], -x[1]))
    return adjacency, direct


def _best_path_to(start_id: str, target_id: str, adjacency: dict[str, list[tuple[str, float]]], max_hops: int) -> tuple[int, float, list[str]] | None:
    queue: list[tuple[str, int, float, list[str]]] = [(start_id, 0, 1.0, [start_id])]
    best: dict[str, tuple[int, float]] = {start_id: (0, 1.0)}
    pos = 0
    answer = None
    while pos < len(queue):
        node, hops, quality, path = queue[pos]
        pos += 1
        if node == target_id and hops > 0:
            candidate = (hops, quality, path)
            if answer is None or quality > answer[1] + 1e-12 or (abs(quality - answer[1]) <= 1e-12 and hops < answer[0]):
                answer = candidate
            continue
        if hops >= max_hops:
            continue
        for nxt, edge_q in adjacency.get(node, []):
            if nxt in path:
                continue
            nh = hops + 1
            nq = min(quality, edge_q)
            old = best.get(nxt)
            if old is not None and old[0] <= nh and old[1] >= nq - 1e-12:
                continue
            best[nxt] = (nh, nq)
            queue.append((nxt, nh, nq, path + [nxt]))
    return answer


def _prototype_nodes(db: sqlite3.Connection) -> list[tuple[str, str | None, dict[str, Any]]]:
    rows = []
    for node_id, surface, metadata_json in db.execute(
        "SELECT node_id,surface,metadata_json FROM nodes WHERE modality=? AND kind=? ORDER BY node_id",
        (Modality.TEXT.value, NodeKind.CONCEPT.value),
    ).fetchall():
        meta = _metadata(metadata_json)
        if meta.get("role") == "latent_slot_prototype":
            rows.append((str(node_id), None if surface is None else str(surface), meta))
    return rows


def _general_frames(db: sqlite3.Connection) -> list[tuple[str, dict[str, Any]]]:
    out = []
    for node_id, metadata_json in db.execute(
        "SELECT node_id,metadata_json FROM nodes WHERE modality=? AND kind=? ORDER BY node_id",
        (Modality.TEXT.value, NodeKind.TEMPLATE.value),
    ).fetchall():
        meta = _metadata(metadata_json)
        if meta.get("role") != "induced_frame_schema" or not meta.get("hierarchical_transfer"):
            continue
        pattern = meta.get("pattern_tokens")
        slot_positions = meta.get("slot_positions") or []
        if not isinstance(pattern, list) or pattern.count("*") != 1 or len(slot_positions) != 1:
            continue
        out.append((str(node_id), meta))
    return out


def _prediction_map(db: sqlite3.Connection) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {}
    for rel in _compiled_relations(db):
        if rel.relation_type == RelationType.PREDICTION.value:
            out.setdefault(rel.from_id, []).append(rel)
    for key in out:
        out[key].sort(key=lambda r: (-float(r.score), r.to_id))
    return out


def _existing_specific_patterns(db: sqlite3.Connection) -> set[tuple[tuple[str, ...], str]]:
    out: set[tuple[tuple[str, ...], str]] = set()
    for _, metadata_json in db.execute(
        "SELECT node_id,metadata_json FROM nodes WHERE modality=? AND kind=? ORDER BY node_id",
        (Modality.TEXT.value, NodeKind.TEMPLATE.value),
    ).fetchall():
        meta = _metadata(metadata_json)
        if meta.get("role") != "induced_frame_schema" or meta.get("hierarchical_transfer"):
            continue
        pattern = meta.get("pattern_tokens")
        slot_ids = [str(x) for x in (meta.get("slot_node_ids") or [])]
        if not isinstance(pattern, list) or len(slot_ids) != 1:
            continue
        slot_meta_raw = db.execute("SELECT metadata_json FROM nodes WHERE node_id=?", (slot_ids[0],)).fetchone()
        if not slot_meta_raw:
            continue
        smeta = _metadata(slot_meta_raw[0])
        profile = dict(smeta.get("semantic_admission") or {})
        proto = str(profile.get("prototype_node_id") or smeta.get("latent_prototype_node_id") or "")
        if proto:
            out.add((tuple(str(x) for x in pattern), proto))
    return out


def _slot_for_prototype(db: sqlite3.Connection, prototype_id: str) -> str | None:
    candidates: list[str] = []
    for node_id, metadata_json in db.execute(
        "SELECT node_id,metadata_json FROM nodes WHERE modality=? AND kind=? ORDER BY node_id",
        (Modality.TEXT.value, NodeKind.CONCEPT.value),
    ).fetchall():
        meta = _metadata(metadata_json)
        if meta.get("role") not in {"induced_frame_slot", "hierarchical_frame_slot"}:
            continue
        profile = dict(meta.get("semantic_admission") or {})
        proto = str(profile.get("prototype_node_id") or meta.get("latent_prototype_node_id") or "")
        if proto == prototype_id and not meta.get("hierarchical_transfer"):
            candidates.append(str(node_id))
    return sorted(candidates)[0] if candidates else None


def _generic_prototype_slot(db: sqlite3.Connection, prototype_id: str, position: int, all_nodes: dict[str, IRNode]) -> IRNode:
    prototype = _load_node_recursive(db, prototype_id, all_nodes)
    meta = dict(prototype.metadata or {})
    filler_ids = [str(x) for x in (meta.get("filler_node_ids") or prototype.members or [])]
    filler_surfaces = [str(x) for x in (meta.get("filler_surfaces") or [])]
    slot_id = composite_id(
        Modality.TEXT.value, NodeKind.CONCEPT.value, [prototype_id],
        label=f"exception-slot:{position}:{prototype_id}",
    )
    return IRNode(
        node_id=slot_id,
        modality=Modality.TEXT.value,
        kind=NodeKind.CONCEPT.value,
        level=prototype.level + 1,
        surface=f"[SLOT_{position}]",
        members=[prototype_id],
        metadata={
            "role": "induced_frame_slot",
            "provisional": True,
            "slot_position": position,
            "open_text_slot": True,
            "filler_surfaces": filler_surfaces,
            "support_count": len(filler_ids),
            "semantic_typed": True,
            "latent_prototype_node_id": prototype_id,
            "semantic_admission": {
                "mode": "latent_prototype_admission_v1",
                "enabled": True,
                "profile_filler_node_ids": filler_ids,
                "profile_filler_surfaces": filler_surfaces,
                "relation_types": list(meta.get("semantic_relation_types") or []),
                "min_candidate_score": 0.0,
                "min_support_fillers": min(2, max(1, len(filler_ids))),
                "prototype_node_id": prototype_id,
                "prototype_surface": prototype.surface,
                "prototype_membership_relation_type": RelationType.IS_A.value,
                "prototype_min_candidate_score": 0.0,
                "provisional": True,
            },
            "exception_induction_generated": True,
            "forge_version": EXCEPTION_INDUCTION_VERSION,
        },
    )


def _target_distribution(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    targets: dict[str, list[Any]] = {}
    for row in rows:
        seen: set[str] = set()
        for rel in row["predictions"]:
            if rel.to_id in seen:
                continue
            seen.add(rel.to_id)
            targets.setdefault(rel.to_id, []).append(rel)
    out: dict[str, dict[str, float]] = {}
    total = len(rows)
    for target_id, rels in targets.items():
        support = len(rels)
        support_fraction = support / total if total else 0.0
        mean_present_score = sum(float(r.score) for r in rels) / support if support else 0.0
        mean_all_score = sum(float(r.score) for r in rels) / total if total else 0.0
        mean_conf = sum(float(r.confidence) for r in rels) / support if support else 0.0
        out[target_id] = {
            "support_count": float(support),
            "support_fraction": support_fraction,
            "mean_present_score": mean_present_score,
            "mean_all_score": mean_all_score,
            "mean_confidence": mean_conf,
        }
    return out


def induce_exception_overrides(
    db: sqlite3.Connection,
    policy: ExceptionInductionPolicy | None = None,
    *,
    source_id: str = "derived:exception-induction",
    source_version: str = EXCEPTION_INDUCTION_VERSION,
) -> tuple[ForgeDocument, ExceptionInductionReport]:
    """Induce subtype-specific frame overrides from systematic deviations.

    v0.28 deliberately does *not* invent a subtype. It requires an already compiled
    latent prototype/hierarchy from independent semantic evidence. It then asks
    whether multiple concrete predictive contexts from that subtype consistently
    disagree with a broader hierarchical frame. Only that residual behavior is
    compiled as an override.
    """
    policy = policy or ExceptionInductionPolicy()
    policy.validate()
    metadata = _node_metadata_map(db)
    surfaces = _node_surface_map(db)
    preds = _prediction_map(db)
    contexts = _template_contexts(db)
    adjacency, direct_is_a = _is_a_graph(db, min(policy.min_membership_quality, policy.min_prototype_path_quality))
    prototypes = _prototype_nodes(db)
    existing_specific = _existing_specific_patterns(db)

    all_nodes: dict[str, IRNode] = {}
    evidence: list[RelationEvidence] = []
    bindings: list[ExceptionOverrideBinding] = []
    src = SourceRef(source_id, source_version, "automatic_exception_override_induction_v1")

    subtype_candidates = 0
    rej_contexts = rej_fillers = rej_deviation = rej_existing = 0
    general_frames = _general_frames(db)

    for general_id, gmeta in general_frames:
        pattern = [str(x) for x in (gmeta.get("pattern_tokens") or [])]
        slot_position = int((gmeta.get("slot_positions") or [pattern.index("*")])[0])
        higher_id = str(gmeta.get("higher_concept_node_id") or "")
        if not higher_id:
            continue
        general_preds = preds.get(general_id, [])
        if not general_preds:
            continue
        general_top = general_preds[0]

        for prototype_id, prototype_surface, _ in prototypes:
            path = _best_path_to(prototype_id, higher_id, adjacency, policy.max_prototype_hops)
            if path is None or path[1] + 1e-12 < policy.min_prototype_path_quality:
                continue
            subtype_candidates += 1
            if (tuple(pattern), prototype_id) in existing_specific:
                rej_existing += 1
                continue

            matched: list[dict[str, Any]] = []
            fillers: set[str] = set()
            for row in contexts:
                tokens = [str(x) for x in row["tokens"]]
                if len(tokens) != len(pattern):
                    continue
                ok = True
                for i, (p, tok) in enumerate(zip(pattern, tokens)):
                    if p == "*":
                        continue
                    if p != tok:
                        ok = False
                        break
                if not ok:
                    continue
                filler_node_id = str(row["members"][slot_position])
                membership_q = direct_is_a.get((filler_node_id, prototype_id), 0.0)
                if membership_q + 1e-12 < policy.min_membership_quality:
                    continue
                matched.append(row)
                fillers.add(tokens[slot_position])

            if len(matched) < policy.min_contexts:
                rej_contexts += 1
                continue
            if len(fillers) < policy.min_unique_fillers:
                rej_fillers += 1
                continue

            dist = _target_distribution(matched)
            accepted = [
                (target_id, vals) for target_id, vals in dist.items()
                if vals["support_fraction"] + 1e-12 >= policy.min_target_context_fraction
                and vals["mean_all_score"] + 1e-12 >= policy.min_exception_target_score
            ]
            accepted.sort(key=lambda x: (-x[1]["mean_all_score"], x[0]))
            if not accepted:
                rej_deviation += 1
                continue
            exception_top_id, exception_top_vals = accepted[0]
            general_subtype_score = float(dist.get(general_top.to_id, {}).get("mean_all_score", 0.0))
            margin = float(exception_top_vals["mean_all_score"] - general_subtype_score)
            if policy.require_different_top1 and exception_top_id == general_top.to_id:
                rej_deviation += 1
                continue
            if margin + 1e-12 < policy.min_disagreement_margin:
                rej_deviation += 1
                continue

            specific_slot_id = _slot_for_prototype(db, prototype_id)
            if specific_slot_id:
                specific_slot = _load_node_recursive(db, specific_slot_id, all_nodes)
            else:
                specific_slot = _generic_prototype_slot(db, prototype_id, slot_position, all_nodes)
                all_nodes[specific_slot.node_id] = specific_slot

            general_frame = _load_node_recursive(db, general_id, all_nodes)
            new_members = list(general_frame.members)
            if not (0 <= slot_position < len(new_members)):
                continue
            new_members[slot_position] = specific_slot.node_id
            for member_id in new_members:
                if member_id not in all_nodes:
                    _load_node_recursive(db, member_id, all_nodes)
            override_id = composite_id(
                Modality.TEXT.value,
                NodeKind.TEMPLATE.value,
                new_members,
                label=f"exception-override:{general_id}:{prototype_id}",
            )
            override = IRNode(
                node_id=override_id,
                modality=Modality.TEXT.value,
                kind=NodeKind.TEMPLATE.value,
                level=max(all_nodes[m].level for m in new_members) + 1,
                surface=general_frame.surface,
                members=new_members,
                metadata={
                    **dict(general_frame.metadata or {}),
                    "role": "induced_frame_schema",
                    "provisional": True,
                    "slot_node_ids": [specific_slot.node_id],
                    "hierarchical_transfer": False,
                    "ancestor_hops": 0,
                    "exception_override": True,
                    "automatic_exception_induction": True,
                    "overridden_frame_node_id": general_id,
                    "exception_subtype_prototype_node_id": prototype_id,
                    "base_prototype_node_id": prototype_id,
                    "higher_concept_node_id": higher_id,
                    "support_context_ids": sorted(str(r["node_id"]) for r in matched),
                    "support_count": len(matched),
                    "unique_fillers": len(fillers),
                    "exception_general_top_target_id": general_top.to_id,
                    "exception_general_top_score": float(general_top.score),
                    "exception_subtype_general_target_score": general_subtype_score,
                    "exception_top_target_id": exception_top_id,
                    "exception_top_score": float(exception_top_vals["mean_all_score"]),
                    "exception_disagreement_margin": margin,
                    "exception_prototype_path": list(path[2]),
                    "exception_prototype_path_quality": float(path[1]),
                    "forge_version": EXCEPTION_INDUCTION_VERSION,
                },
            )
            all_nodes[override.node_id] = override

            prediction_targets: list[str] = []
            for target_id, vals in accepted:
                _load_node_recursive(db, target_id, all_nodes)
                confidence = max(0.0, min(1.0, float(policy.evidence_confidence) * float(vals["mean_confidence"]) * float(vals["support_fraction"])))
                evidence.append(RelationEvidence(
                    override.node_id,
                    target_id,
                    RelationType.PREDICTION.value,
                    max(0.0, min(1.0, float(vals["mean_all_score"]))),
                    confidence,
                    src,
                    {
                        "evidence_kind": "automatic_exception_override_prediction",
                        "overridden_frame_node_id": general_id,
                        "subtype_prototype_node_id": prototype_id,
                        "support_context_ids": sorted(str(r["node_id"]) for r in matched),
                        "support_fillers": sorted(fillers),
                        "target_support_fraction": float(vals["support_fraction"]),
                        "general_top_target_id": general_top.to_id,
                        "subtype_general_target_score": general_subtype_score,
                        "disagreement_margin": margin,
                        "provisional": True,
                    },
                ))
                prediction_targets.append(target_id)

            bindings.append(ExceptionOverrideBinding(
                general_frame_node_id=general_id,
                override_frame_node_id=override.node_id,
                subtype_prototype_node_id=prototype_id,
                subtype_prototype_surface=prototype_surface,
                slot_node_id=specific_slot.node_id,
                higher_concept_node_id=higher_id,
                pattern=pattern,
                support_context_ids=sorted(str(r["node_id"]) for r in matched),
                support_fillers=sorted(fillers),
                general_top_target_id=general_top.to_id,
                general_top_target_surface=surfaces.get(general_top.to_id),
                general_top_score=float(general_top.score),
                exception_top_target_id=exception_top_id,
                exception_top_target_surface=surfaces.get(exception_top_id),
                exception_top_score=float(exception_top_vals["mean_all_score"]),
                subtype_general_target_score=general_subtype_score,
                disagreement_margin=margin,
                prototype_path_hops=int(path[0]),
                prototype_path_quality=float(path[1]),
                prediction_targets=sorted(set(prediction_targets)),
            ))

    doc = ForgeDocument(
        nodes=sorted(all_nodes.values(), key=lambda n: (n.level, n.node_id)),
        evidence=sorted(evidence, key=lambda e: (e.from_id, e.to_id, e.relation_type, -e.score)),
    )
    report = ExceptionInductionReport(
        format=EXCEPTION_INDUCTION_FORMAT,
        version=EXCEPTION_INDUCTION_VERSION,
        general_frames_considered=len(general_frames),
        subtype_candidates_considered=subtype_candidates,
        overrides_induced=len(bindings),
        rejected_insufficient_contexts=rej_contexts,
        rejected_insufficient_fillers=rej_fillers,
        rejected_no_systematic_deviation=rej_deviation,
        rejected_existing_specific_frame=rej_existing,
        generated_nodes=len(doc.nodes),
        generated_prediction_evidence=len(doc.evidence),
        bindings=bindings,
        notes=[
            "v0.28 induces an override only from repeated concrete predictive contexts that belong to an already compiled subtype prototype and systematically disagree with a broader hierarchical frame.",
            "Subtype identity is not inferred from the exception target. It must already exist from independent prototype/hierarchy evidence, preventing target-label leakage into the class structure.",
            "A single anomalous context cannot create an override under the default policy; at least two contexts and two distinct fillers are required.",
            "The general hierarchical frame is never rewritten or deleted. The override is a separate provisional direct-prototype frame and v0.27 specificity resolution decides precedence at runtime.",
        ],
    )
    return doc, report
