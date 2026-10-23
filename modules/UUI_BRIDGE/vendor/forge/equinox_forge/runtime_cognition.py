from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable
import heapq
import json
import math

from .ir import CompiledRelation, NodeKind, RelationType
from .runtime_pack import LoadedRuntimePack, RuntimeNode, load_base_pack
from .runtime_query import RuntimeIndex

RUNTIME_COGNITION_FORMAT = "equinox-runtime-cognition-v2"
RUNTIME_COGNITION_VERSION = "0.40"


@dataclass
class RuntimeCognitionPolicy:
    max_recall_nodes: int = 96
    max_recall_depth: int = 3
    max_relation_neighbors: int = 24
    max_structural_parents: int = 24
    max_predictions: int = 8
    min_activation: float = 0.06
    min_prediction_score: float = 0.01
    depth_decay: float = 0.82
    parent_weight: float = 0.72
    transition_weight: float = 0.72
    recalled_prediction_weight: float = 0.72
    exact_context_prediction_weight: float = 1.0
    # v0.40: strong direct evidence from an exact or suffix context forms an
    # authority tier over recall-only alternatives. This is deliberately not an
    # unconditional "exact always wins" rule: the tier activates only when at
    # least one direct PREDICTION candidate clears the configured support floor.
    # Competing strong direct candidates remain mutually comparable, and weak
    # direct evidence falls back to ordinary aggregation.
    enable_exact_context_authority: bool = True
    exact_context_authority_min_direct_score: float = 0.50
    exact_context_authority_include_suffix: bool = True
    seed_recent_word_weight: float = 0.82
    seed_older_word_weight: float = 0.48
    # Runtime mechanism switches. Defaults preserve v0.16-v0.20 behavior.
    enable_exact_context: bool = True
    enable_suffix_context: bool = True
    enable_lexical_seeds: bool = True
    enable_parent_routing: bool = True
    enable_relation_recall: bool = True
    enable_direct_context_prediction: bool = True
    enable_transition_prediction: bool = True
    enable_recalled_context_prediction: bool = True
    # v0.32: recalled predictive contexts are candidates for memory/association,
    # not automatically sufficient evidence for output. Require live-context
    # overlap before a recalled PREDICTION relation can execute. Unknown live
    # tokens must be admitted by an explicit compiled context/frame rather than
    # guessed through lexical recall alone.
    enable_support_gated_recalled_prediction: bool = True
    recalled_support_min_shared_tokens: int = 2
    recalled_support_min_overlap_fraction: float = 0.66
    recalled_support_allow_unknown_tokens_without_context: bool = False
    # v0.33: an entity may be new to predictive training while already known to
    # the BASE through independent semantic evidence. Such a token can substitute
    # for a trained entity only when compiled semantic relations support the
    # substitution. Mere presence of the surface in the BASE is not sufficient.
    enable_evidence_backed_novel_entity_admission: bool = True
    novel_entity_semantic_relation_types: tuple[str, ...] = (
        RelationType.SEMANTIC_SIMILARITY.value,
        RelationType.CONTEXTUAL_SIMILARITY.value,
    )
    novel_entity_min_semantic_quality: float = 0.65
    # v0.34: a relation family that has never been compiled as direct predictive
    # evidence may execute only through an explicit relation-composition schema.
    # The schema binds one live entity slot and traverses a fixed, audited sequence
    # of non-predictive semantic relations. Truly opaque relation families remain
    # unsupported and must abstain.
    enable_relation_composition_prediction: bool = True
    relation_composition_weight: float = 0.86
    relation_composition_min_path_quality: float = 0.50
    relation_composition_max_hops: int = 4
    relation_composition_max_matches: int = 8
    relation_composition_allowed_premise_types: tuple[str, ...] = (
        RelationType.IS_A.value,
        RelationType.PROPERTY.value,
        RelationType.ACTION.value,
    )
    relation_composition_allowed_statuses: tuple[str, ...] = ("stable",)
    relation_composition_min_schema_sources: int = 1
    relation_composition_min_schema_support: int = 1
    # v0.22: explicit induced frame schemas such as [SLOT_0] ist ein.
    enable_frame_schema_matching: bool = True
    # v0.23: if an induced slot carries a compiled semantic admission profile,
    # enforce it before the frame can become a predictive context.
    enable_semantic_slot_admission: bool = True
    # v0.24: prefer a compiled latent class membership edge (candidate -> prototype)
    # over re-evaluating pairwise candidate-to-filler similarity at runtime.
    enable_slot_prototype_admission: bool = True
    # v0.26: allow a slot to admit a candidate through a compiled multi-hop IS_A
    # chain into a higher recursive latent concept.
    enable_hierarchical_slot_admission: bool = True
    # v0.27: when multiple frame schemas of identical structural width admit the
    # same live context, retain only the most specific scope. Exact contexts are
    # already resolved before frame matching; among frames the order is direct
    # lower/prototype scope > hierarchical transferred scope > untyped open scope.
    enable_frame_specificity_override: bool = True
    # v0.28: allow compile-time automatically induced exception frames. Kept as a
    # separate switch so the exact same BASE can be ablated without recompilation.
    enable_exception_override_frames: bool = True
    # v0.29: lifecycle-aware exception gating. Exception frames without lifecycle
    # metadata remain allowed for backward compatibility with v0.28 packs. Once a
    # lifecycle status is present, only explicitly allowed statuses may execute.
    enable_exception_lifecycle_gating: bool = True
    allowed_exception_lifecycle_statuses: tuple[str, ...] = ("stable",)
    max_hierarchical_slot_hops: int = 8
    frame_match_weight: float = 0.82
    max_frame_matches: int = 8
    # Relation weights are recall routing priors, not learned semantics.
    relation_weights: dict[str, float] = field(default_factory=lambda: {
        RelationType.ASSOCIATION.value: 0.55,
        RelationType.SEMANTIC_SIMILARITY.value: 0.50,
        RelationType.SUBSTITUTION.value: 0.42,
        RelationType.IS_A.value: 0.50,
        RelationType.PROPERTY.value: 0.46,
        RelationType.ACTION.value: 0.46,
        RelationType.CROSSMODAL.value: 0.44,
        RelationType.CONTEXTUAL_USE.value: 0.45,
        RelationType.CONTEXTUAL_SIMILARITY.value: 0.48,
    })

    def validate(self) -> None:
        if self.max_recall_nodes < 1: raise ValueError("max_recall_nodes must be >= 1")
        if self.max_recall_depth < 0: raise ValueError("max_recall_depth must be >= 0")
        if self.max_relation_neighbors < 1: raise ValueError("max_relation_neighbors must be >= 1")
        if self.max_structural_parents < 1: raise ValueError("max_structural_parents must be >= 1")
        if self.max_predictions < 1: raise ValueError("max_predictions must be >= 1")
        if self.max_frame_matches < 1: raise ValueError("max_frame_matches must be >= 1")
        if self.max_hierarchical_slot_hops < 1: raise ValueError("max_hierarchical_slot_hops must be >= 1")
        if self.recalled_support_min_shared_tokens < 1: raise ValueError("recalled_support_min_shared_tokens must be >= 1")
        if not 0 <= self.recalled_support_min_overlap_fraction <= 1: raise ValueError("recalled_support_min_overlap_fraction must be in [0,1]")
        if not 0 <= self.novel_entity_min_semantic_quality <= 1: raise ValueError("novel_entity_min_semantic_quality must be in [0,1]")
        if not self.novel_entity_semantic_relation_types: raise ValueError("novel_entity_semantic_relation_types must not be empty")
        if not 0 < self.relation_composition_weight <= 1: raise ValueError("relation_composition_weight must be in (0,1]")
        if not 0 <= self.relation_composition_min_path_quality <= 1: raise ValueError("relation_composition_min_path_quality must be in [0,1]")
        if self.relation_composition_max_hops < 1: raise ValueError("relation_composition_max_hops must be >= 1")
        if self.relation_composition_max_matches < 1: raise ValueError("relation_composition_max_matches must be >= 1")
        if not self.relation_composition_allowed_premise_types: raise ValueError("relation_composition_allowed_premise_types must not be empty")
        if not self.relation_composition_allowed_statuses: raise ValueError("relation_composition_allowed_statuses must not be empty")
        if self.relation_composition_min_schema_sources < 1: raise ValueError("relation_composition_min_schema_sources must be >= 1")
        if self.relation_composition_min_schema_support < 1: raise ValueError("relation_composition_min_schema_support must be >= 1")
        if not 0 <= self.exact_context_authority_min_direct_score <= 1: raise ValueError("exact_context_authority_min_direct_score must be in [0,1]")
        if not 0 <= self.min_activation <= 1: raise ValueError("min_activation must be in [0,1]")
        if not 0 <= self.min_prediction_score <= 1: raise ValueError("min_prediction_score must be in [0,1]")
        if not 0 < self.depth_decay <= 1: raise ValueError("depth_decay must be in (0,1]")
        if not 0 < self.frame_match_weight <= 1: raise ValueError("frame_match_weight must be in (0,1]")


@dataclass
class RecallTrace:
    node_id: str
    surface: str | None
    kind: str
    level: int
    activation: float
    depth: int
    via: str
    from_node_id: str | None

    def to_dict(self) -> dict[str, Any]: return asdict(self)


@dataclass
class PredictionContribution:
    from_node_id: str
    from_surface: str | None
    relation_type: str
    relation_score: float
    relation_confidence: float
    source_count: int
    context_activation: float
    channel_weight: float
    contribution: float
    direct_context: bool

    def to_dict(self) -> dict[str, Any]: return asdict(self)


@dataclass
class PredictionCandidate:
    node_id: str
    surface: str | None
    score: float
    direct_score: float
    recalled_score: float
    transition_score: float
    contribution_count: int
    contributions: list[PredictionContribution]
    composition_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["contributions"] = [c.to_dict() for c in self.contributions]
        return d


@dataclass
class RuntimePredictionReport:
    format: str
    version: str
    context: str
    resolved_context_ids: list[str]
    lexical_seed_ids: list[str]
    recall: list[RecallTrace]
    predictions: list[PredictionCandidate]
    work: dict[str, int]
    policy: dict[str, Any]
    frame_admission: list[dict[str, Any]] = field(default_factory=list)
    relation_composition: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "version": self.version,
            "context": self.context,
            "resolved_context_ids": self.resolved_context_ids,
            "lexical_seed_ids": self.lexical_seed_ids,
            "recall": [x.to_dict() for x in self.recall],
            "predictions": [x.to_dict() for x in self.predictions],
            "work": dict(self.work),
            "policy": self.policy,
            "frame_admission": list(self.frame_admission),
            "relation_composition": list(self.relation_composition),
        }


@dataclass
class PredictionBenchmarkCase:
    context: str
    expected: str


@dataclass
class PredictionBenchmarkReport:
    format: str
    version: str
    cases: int
    hit_at_1: float
    hit_at_k: float
    mean_reciprocal_rank: float
    unresolved_contexts: int
    expected_not_in_pack: int
    average_recall_nodes: float
    average_relation_checks: float
    top_k: int
    details: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]: return asdict(self)


def _surface_tokens(text: str) -> list[str]:
    # Runtime context matching intentionally stays tokenizer-independent. This first
    # text consumer uses whitespace-delimited surfaces already present in the pack.
    return [x for x in text.strip().split() if x]


def _node_preference(node: RuntimeNode) -> tuple[int, int, str]:
    # Predictive templates first, then composites/concepts, then lexical surfaces.
    kind_rank = {
        NodeKind.TEMPLATE.value: 0,
        NodeKind.COMPOSITE.value: 1,
        NodeKind.CONCEPT.value: 2,
        NodeKind.SURFACE.value: 3,
        NodeKind.QUANT.value: 9,
    }.get(node.kind, 5)
    return (kind_rank, -node.level, node.node_id)


def _noisy_or(values: Iterable[float]) -> float:
    product = 1.0
    seen = False
    for v in values:
        seen = True
        product *= 1.0 - max(0.0, min(1.0, float(v)))
    return 0.0 if not seen else 1.0 - product


class RuntimeCognition:
    """Bounded read-only recall and prediction over a compiled .eqbase graph.

    Recall may use structural and associative relations to expose relevant contexts.
    Prediction authority comes from compiled directed PREDICTION/TRANSITION evidence
    or, since v0.34, from an explicit stable relation-composition schema whose full
    semantic premise path exists in the BASE. Semantic similarity alone never becomes
    a next-word claim by itself.
    """

    def __init__(self, pack: LoadedRuntimePack, policy: RuntimeCognitionPolicy | None = None):
        self.pack = pack
        self.index = RuntimeIndex(pack)
        self.policy = policy or RuntimeCognitionPolicy()
        self.policy.validate()
        # Frame schemas are an explicit compiled abstraction, not inferred ad hoc at
        # query time. Build a tiny width index once when the pack is opened.
        self._frames_by_width: dict[int, list[dict[str, Any]]] = {}
        for node in pack.nodes:
            meta = node.metadata or {}
            if meta.get("role") != "induced_frame_schema":
                continue
            pattern = meta.get("pattern_tokens")
            if not isinstance(pattern, list) or not pattern or not all(isinstance(x, str) for x in pattern):
                continue
            support = int(meta.get("support_count", 0) or 0)
            open_slot = bool(meta.get("open_text_slots", True))
            allowed_fillers: set[str] = set()
            slot_ids = [str(x) for x in (meta.get("slot_node_ids", []) or [])]
            slot_profiles: list[dict[str, Any]] = []
            for slot_id in slot_ids:
                slot_node = self.index.node_by_id.get(slot_id)
                if slot_node is not None:
                    smeta = slot_node.metadata or {}
                    allowed_fillers.update(str(x) for x in (smeta.get("filler_surfaces", []) or []))
                    slot_profiles.append(dict(smeta.get("semantic_admission") or {}))

            hierarchical = bool(meta.get("hierarchical_transfer"))
            exception_override = bool(meta.get("exception_override"))
            ancestor_hops = int(meta.get("ancestor_hops", 0) or 0)
            typed = any(bool(x.get("enabled")) for x in slot_profiles)
            prototype_typed = any(bool(x.get("prototype_node_id")) for x in slot_profiles)
            # The absolute numbers are categorical ranks, not probabilities. They
            # exist solely to make scope precedence deterministic. Lower ancestor
            # hop count means a narrower hierarchical promotion.
            if hierarchical:
                scope_class = "hierarchical_concept"
                scope_rank = 3000 - max(0, ancestor_hops)
            elif prototype_typed:
                scope_class = "direct_prototype"
                scope_rank = 5000
            elif typed or not open_slot:
                scope_class = "direct_typed"
                scope_rank = 4500
            else:
                scope_class = "open_untyped"
                scope_rank = 1000
            self._frames_by_width.setdefault(len(pattern), []).append({
                "node_id": node.node_id,
                "pattern": list(pattern),
                "support": support,
                "open_slot": open_slot,
                "allowed_fillers": allowed_fillers,
                "slot_ids": slot_ids,
                "scope_class": scope_class,
                "scope_rank": scope_rank,
                "ancestor_hops": ancestor_hops,
                "hierarchical_transfer": hierarchical,
                "exception_override": exception_override,
                "exception_lifecycle_status": str(meta.get("exception_lifecycle_status") or ""),
            })
        for width in self._frames_by_width:
            self._frames_by_width[width].sort(key=lambda x: (-int(x["scope_rank"]), -int(x["support"]), str(x["node_id"])))

        # v0.34 relation-composition schemas are explicit compiled rules. They are
        # indexed by live token width but are not prediction relations themselves.
        # A schema may therefore expose a held-out *predictive* relation family only
        # when a fixed semantic path already exists in the BASE.
        self._relation_composition_by_width: dict[int, list[dict[str, Any]]] = {}
        allowed_premises = set(self.policy.relation_composition_allowed_premise_types)
        for node in pack.nodes:
            meta = node.metadata or {}
            if meta.get("role") != "relation_composition_schema":
                continue
            pattern = meta.get("pattern_tokens")
            path = meta.get("path")
            status = str(meta.get("status") or "candidate")
            source_count = int(meta.get("source_count", 0) or 0)
            support_count = int(meta.get("support_count", 0) or 0)
            if status not in set(self.policy.relation_composition_allowed_statuses):
                continue
            if source_count < self.policy.relation_composition_min_schema_sources:
                continue
            if support_count < self.policy.relation_composition_min_schema_support:
                continue
            if not isinstance(pattern, list) or not pattern or not all(isinstance(x, str) for x in pattern):
                continue
            if not isinstance(path, list) or not path or len(path) > self.policy.relation_composition_max_hops:
                continue
            slot_positions = [i for i, tok in enumerate(pattern) if tok.startswith("[SLOT_") and tok.endswith("]")]
            if len(slot_positions) != 1:
                continue
            clean_path: list[dict[str, str]] = []
            valid = True
            for step in path:
                if not isinstance(step, dict):
                    valid = False; break
                rtype = str(step.get("relation_type") or "")
                direction = str(step.get("direction") or "out")
                if rtype not in allowed_premises or direction not in {"out", "in"}:
                    valid = False; break
                clean_path.append({"relation_type": rtype, "direction": direction})
            if not valid:
                continue
            self._relation_composition_by_width.setdefault(len(pattern), []).append({
                "node_id": node.node_id,
                "surface": node.surface,
                "pattern": list(pattern),
                "slot_position": slot_positions[0],
                "path": clean_path,
                "status": status,
                "support_count": support_count,
                "source_count": source_count,
                "min_path_quality": float(meta.get("min_path_quality", self.policy.relation_composition_min_path_quality) or self.policy.relation_composition_min_path_quality),
            })
        for width in self._relation_composition_by_width:
            self._relation_composition_by_width[width].sort(
                key=lambda x: (-int(x["source_count"]), -int(x["support_count"]), str(x["node_id"]))
            )

        # Tokens that have actually participated in a compiled predictive context.
        # v0.33 uses this vocabulary to distinguish "known somewhere in the BASE"
        # from "already trained for this predictive mechanism".
        self._predictive_context_token_surfaces: set[str] = set()
        for from_id, rels in self.index.relations_from.items():
            if not any(r.relation_type == RelationType.PREDICTION.value for r in rels):
                continue
            node = self.index.node_by_id.get(from_id)
            if node is None or not node.surface:
                continue
            self._predictive_context_token_surfaces.update(_surface_tokens(node.surface))

    @classmethod
    def from_path(cls, path: str | Path, policy: RuntimeCognitionPolicy | None = None) -> "RuntimeCognition":
        return cls(load_base_pack(path), policy)

    def _semantic_slot_admission(self, token: str, slot_node_id: str) -> dict[str, Any]:
        slot = self.index.node_by_id.get(slot_node_id)
        profile = dict((slot.metadata or {}).get("semantic_admission") or {}) if slot else {}
        result: dict[str, Any] = {
            "token": token, "slot_node_id": slot_node_id, "typed": bool(profile.get("enabled")),
            "admitted": True, "reason": "untyped_open_slot", "candidate_node_ids": [],
            "supporting_fillers": [], "best_scores": {},
        }
        if not profile.get("enabled") or not self.policy.enable_semantic_slot_admission:
            if profile.get("enabled") and not self.policy.enable_semantic_slot_admission:
                result["reason"] = "semantic_admission_disabled"
            return result
        filler_ids = {str(x) for x in profile.get("profile_filler_node_ids", []) or []}
        allowed_types = {str(x) for x in profile.get("relation_types", []) or []}
        min_score = float(profile.get("min_candidate_score", 0.0) or 0.0)
        min_support = int(profile.get("min_support_fillers", 1) or 1)
        candidate_ids = [nid for nid in self.index.surface_to_nodes.get(token, [])
                         if self.index.node_by_id[nid].kind == NodeKind.SURFACE.value]
        result["candidate_node_ids"] = sorted(candidate_ids)
        if not candidate_ids:
            result.update({"admitted": False, "reason": "candidate_surface_unknown", "required_support": min_support, "min_score": min_score})
            return result
        if filler_ids.intersection(candidate_ids):
            result.update({"admitted": True, "reason": "observed_filler", "supporting_fillers": sorted(filler_ids.intersection(candidate_ids)),
                           "required_support": min_support, "min_score": min_score})
            return result

        hierarchical_id = str(profile.get("hierarchical_prototype_node_id") or "")
        if hierarchical_id and self.policy.enable_hierarchical_slot_admission:
            relation_type = str(profile.get("hierarchical_membership_relation_type") or RelationType.IS_A.value)
            max_hops = min(
                int(profile.get("hierarchical_max_hops", self.policy.max_hierarchical_slot_hops) or self.policy.max_hierarchical_slot_hops),
                int(self.policy.max_hierarchical_slot_hops),
            )
            min_path_score = float(profile.get("hierarchical_min_path_score", 0.0) or 0.0)
            best_path = None
            total_index_lookups = 0
            total_relation_checks = 0
            for cid in candidate_ids:
                queue = [(cid, 0, 1.0, [cid])]
                seen_best = {cid: (0, 1.0)}
                pos = 0
                while pos < len(queue):
                    current, hops, quality, path = queue[pos]; pos += 1
                    if current == hierarchical_id and hops > 0:
                        candidate = (quality, -hops, cid, list(path))
                        if best_path is None or candidate[:3] > best_path[:3]:
                            best_path = candidate
                        continue
                    if hops >= max_hops:
                        continue
                    total_index_lookups += 1
                    bucket = self.index.relations_from.get(current, [])
                    total_relation_checks += len(bucket)
                    for rel in bucket:
                        if rel.relation_type != relation_type:
                            continue
                        nxt = rel.to_id
                        if nxt in path:
                            continue
                        edge_q = max(0.0, min(1.0, float(rel.score) * float(rel.confidence)))
                        nq = min(quality, edge_q)
                        nh = hops + 1
                        old = seen_best.get(nxt)
                        if old is not None and old[0] <= nh and old[1] >= nq - 1e-12:
                            continue
                        seen_best[nxt] = (nh, nq)
                        queue.append((nxt, nh, nq, path + [nxt]))
            if best_path is not None and best_path[0] + 1e-12 >= min_path_score:
                result.update({
                    "hierarchical_prototype_node_id": hierarchical_id,
                    "hierarchical_relation_type": relation_type,
                    "hierarchical_path_score": float(best_path[0]),
                    "hierarchical_hops": int(-best_path[1]),
                    "hierarchical_candidate_node_id": best_path[2],
                    "hierarchical_path": best_path[3],
                    "hierarchical_min_path_score": min_path_score,
                    "hierarchical_index_lookups": total_index_lookups,
                    "hierarchical_relation_checks": total_relation_checks,
                    "admitted": True,
                    "reason": "hierarchical_prototype_membership",
                })
                return result
            # Keep evaluating the direct base-prototype path below. This makes the
            # transferred frame backward-compatible for its original subtype.
            result.update({
                "hierarchical_prototype_node_id": hierarchical_id,
                "hierarchical_relation_type": relation_type,
                "hierarchical_path_score": float(best_path[0]) if best_path is not None else 0.0,
                "hierarchical_min_path_score": min_path_score,
                "hierarchical_index_lookups": total_index_lookups,
                "hierarchical_relation_checks": total_relation_checks,
            })

        prototype_id = str(profile.get("prototype_node_id") or "")
        if prototype_id and self.policy.enable_slot_prototype_admission:
            relation_type = str(profile.get("prototype_membership_relation_type") or RelationType.IS_A.value)
            prototype_min = float(profile.get("prototype_min_candidate_score", min_score) or min_score)
            best = 0.0
            matched_candidate = None
            prototype_index_lookups = 0
            for cid in candidate_ids:
                prototype_index_lookups += 1
                rel = self.index.relation_by_key.get((cid, prototype_id, relation_type))
                if rel is None:
                    continue
                q = max(0.0, min(1.0, float(rel.score) * float(rel.confidence)))
                if q > best:
                    best = q; matched_candidate = cid
            admitted = best + 1e-12 >= prototype_min
            result.update({
                "prototype_node_id": prototype_id,
                "prototype_relation_type": relation_type,
                "prototype_score": best,
                "prototype_min_score": prototype_min,
                "prototype_candidate_node_id": matched_candidate,
                "prototype_index_lookups": prototype_index_lookups,
                "admitted": admitted,
                "reason": "prototype_membership" if admitted else "insufficient_prototype_membership",
            })
            return result

        best_by_filler: dict[str, float] = {}
        for cid in candidate_ids:
            for rel in self.index.relations_from.get(cid, []):
                if rel.relation_type in allowed_types and rel.to_id in filler_ids:
                    q = max(0.0, min(1.0, float(rel.score) * float(rel.confidence)))
                    best_by_filler[rel.to_id] = max(best_by_filler.get(rel.to_id, 0.0), q)
            for rel in self.index.relations_to.get(cid, []):
                if rel.relation_type in allowed_types and rel.from_id in filler_ids:
                    q = max(0.0, min(1.0, float(rel.score) * float(rel.confidence)))
                    best_by_filler[rel.from_id] = max(best_by_filler.get(rel.from_id, 0.0), q)
        supporters = sorted(fid for fid, score in best_by_filler.items() if score + 1e-12 >= min_score)
        result.update({
            "best_scores": {k: best_by_filler[k] for k in sorted(best_by_filler)},
            "supporting_fillers": supporters, "required_support": min_support, "min_score": min_score,
            "admitted": len(supporters) >= min_support,
            "reason": "semantic_support" if len(supporters) >= min_support else "insufficient_semantic_support",
        })
        return result

    def _exception_metadata_allowed(self, meta: dict[str, Any] | None) -> bool:
        meta = meta or {}
        if not bool(meta.get("exception_override")):
            return True
        if not self.policy.enable_exception_override_frames:
            return False
        if not self.policy.enable_exception_lifecycle_gating:
            return True
        status = str(meta.get("exception_lifecycle_status") or "")
        # v0.28 compatibility: packs compiled before lifecycle tracking have no status.
        if not status:
            return True
        return status in set(self.policy.allowed_exception_lifecycle_statuses)

    def _match_frame_schemas(self, tokens: list[str]) -> tuple[list[str], int, list[dict[str, Any]]]:
        if not self.policy.enable_frame_schema_matching or not tokens:
            return [], 0, []
        checks = 0
        diagnostics: list[dict[str, Any]] = []
        # Prefer the longest structural frame first. Within the same width v0.27
        # applies *scope specificity* before support/score aggregation: a direct
        # subtype frame suppresses a broader hierarchical promotion when both admit
        # the same live filler. This is override semantics, not a score bonus.
        for width in sorted((w for w in self._frames_by_width if w <= len(tokens)), reverse=True):
            live = tokens[-width:]
            matches: list[dict[str, Any]] = []
            width_had_structural_match = False
            for spec in self._frames_by_width[width]:
                if bool(spec.get("exception_override")):
                    node_obj = self.index.node_by_id.get(str(spec.get("node_id")))
                    node_meta = dict((node_obj.metadata or {}) if node_obj else {})
                    if not self._exception_metadata_allowed(node_meta):
                        # Preserve an audit trace when a lifecycle-gated exception
                        # structurally matches the live context, while allowing a
                        # broader stable frame of the same width to take over.
                        pattern0=list(spec.get("pattern") or [])
                        structural = len(pattern0)==len(live) and all(p=="*" or p==t for p,t in zip(pattern0,live))
                        if structural:
                            diagnostics.append({
                                "frame_node_id":str(spec.get("node_id")),"pattern":pattern0,"live":list(live),
                                "slot_checks":[],"structural_match":True,"admitted":False,
                                "reason":"exception_lifecycle_blocked",
                                "exception_lifecycle_status":str(node_meta.get("exception_lifecycle_status") or ""),
                                "specificity_scope_class":spec.get("scope_class"),
                                "specificity_scope_rank":int(spec.get("scope_rank",0)),
                                "specificity_ancestor_hops":int(spec.get("ancestor_hops",0)),
                                "suppressed_by_specificity":False,
                            })
                        continue
                node_id = str(spec["node_id"]); pattern = list(spec["pattern"]); support = int(spec["support"])
                open_slot = bool(spec["open_slot"]); allowed_fillers = set(spec["allowed_fillers"]); slot_ids = list(spec["slot_ids"])
                checks += 1
                ok = True
                slot_cursor = 0
                frame_diag = {
                    "frame_node_id": node_id, "pattern": pattern, "live": list(live), "slot_checks": [],
                    "structural_match": True, "admitted": False,
                    "specificity_scope_class": spec["scope_class"],
                    "specificity_scope_rank": int(spec["scope_rank"]),
                    "specificity_ancestor_hops": int(spec["ancestor_hops"]),
                    "suppressed_by_specificity": False,
                }
                for p, t in zip(pattern, live):
                    if p == "*":
                        sid = slot_ids[slot_cursor] if slot_cursor < len(slot_ids) else ""
                        slot_cursor += 1
                        if not open_slot and t not in allowed_fillers:
                            ok = False
                            frame_diag["slot_checks"].append({"token": t, "admitted": False, "reason": "closed_slot"})
                            break
                        if sid:
                            admission = self._semantic_slot_admission(t, sid)
                            frame_diag["slot_checks"].append(admission)
                            if not admission.get("admitted", False):
                                ok = False
                                break
                    elif p != t:
                        ok = False
                        frame_diag["structural_match"] = False
                        break
                if frame_diag["structural_match"]:
                    width_had_structural_match = True
                    frame_diag["admitted"] = ok
                    diagnostics.append(frame_diag)
                if ok:
                    matches.append({**spec, "diag": frame_diag})
            if matches:
                selected = list(matches)
                if self.policy.enable_frame_specificity_override:
                    best_rank = max(int(x["scope_rank"]) for x in matches)
                    selected = [x for x in matches if int(x["scope_rank"]) == best_rank]
                    selected_ids = {str(x["node_id"]) for x in selected}
                    winner_ids = sorted(selected_ids)
                    for x in matches:
                        diag = x["diag"]
                        diag["specificity_winner_rank"] = best_rank
                        diag["specificity_winner_frame_ids"] = winner_ids
                        if str(x["node_id"]) not in selected_ids:
                            diag["suppressed_by_specificity"] = True
                            diag["specificity_suppressed_by"] = winner_ids
                selected.sort(key=lambda x: (-int(x["support"]), str(x["node_id"])))
                return [str(x["node_id"]) for x in selected[:self.policy.max_frame_matches]], checks, diagnostics
            # If a structurally matching longest frame was rejected by typed admission,
            # do not silently fall through to a shorter, broader schema.
            if width_had_structural_match:
                return [], checks, diagnostics
        return [], checks, diagnostics

    def _resolve_context(self, context: str) -> tuple[list[str], list[str], list[str], int, list[dict[str, Any]], str]:
        text = " ".join(_surface_tokens(context))
        exact = list(self.index.surface_to_nodes.get(text, [])) if text else []
        exact_nodes = sorted((self.index.node_by_id[x] for x in exact), key=_node_preference)
        exact_context = [n.node_id for n in exact_nodes if n.kind in {
            NodeKind.TEMPLATE.value, NodeKind.COMPOSITE.value, NodeKind.CONCEPT.value
        } and self._exception_metadata_allowed(n.metadata or {})
        and any(r.relation_type == RelationType.PREDICTION.value for r in self.index.relations_from.get(n.node_id, []))]

        tokens = _surface_tokens(context)
        lexical: list[str] = []
        for tok in tokens:
            ids = [x for x in self.index.surface_to_nodes.get(tok, [])
                   if self.index.node_by_id[x].kind == NodeKind.SURFACE.value]
            if ids:
                lexical.append(sorted(ids)[0])

        if not self.policy.enable_exact_context:
            exact_context = []

        # If an exact context label does not exist, use the longest known suffix
        # context. This makes n-gram BASE packs useful with longer live prompts.
        fallback: list[str] = []
        if not exact_context and self.policy.enable_suffix_context:
            for width in range(min(len(tokens), 8), 1, -1):
                label = " ".join(tokens[-width:])
                ids = [x for x in self.index.surface_to_nodes.get(label, [])
                       if self.index.node_by_id[x].kind in {NodeKind.TEMPLATE.value, NodeKind.COMPOSITE.value}
                       and any(r.relation_type == RelationType.PREDICTION.value for r in self.index.relations_from.get(x, []))]
                if ids:
                    fallback = sorted(ids, key=lambda x: _node_preference(self.index.node_by_id[x]))
                    break

        frames: list[str] = []
        frame_checks = 0
        frame_admission: list[dict[str, Any]] = []
        if not exact_context and not fallback:
            frames, frame_checks, frame_admission = self._match_frame_schemas(tokens)

        if not self.policy.enable_lexical_seeds:
            lexical = []
        resolution_kind = "exact" if exact_context else ("suffix" if fallback else ("frame" if frames else "none"))
        return exact_context or fallback or frames, lexical, tokens, frame_checks, frame_admission, resolution_kind

    def _relation_quality(self, rel: CompiledRelation, weight: float) -> float:
        # source_count is only a mild confidence bonus; it never substitutes for
        # independent-source tracking upstream in Forge.
        source_bonus = 0.90 + 0.10 * min(1.0, rel.source_count / 2.0)
        return max(0.0, min(1.0, rel.score * rel.confidence * weight * source_bonus))

    def recall(self, context: str) -> tuple[list[RecallTrace], dict[str, Any]]:
        context_ids, lexical_ids, tokens, frame_checks, frame_admission, resolution_kind = self._resolve_context(context)
        seeds: list[tuple[str, float, str]] = []
        for nid in context_ids:
            node = self.index.node_by_id.get(nid)
            is_frame = bool(node and (node.metadata or {}).get("role") == "induced_frame_schema")
            seeds.append((nid, self.policy.frame_match_weight if is_frame else 1.0, "frame_schema" if is_frame else "exact_or_suffix_context"))
        for i, nid in enumerate(lexical_ids):
            if i == len(lexical_ids) - 1:
                act = self.policy.seed_recent_word_weight
            else:
                distance = (len(lexical_ids) - 1) - i
                act = self.policy.seed_older_word_weight * (self.policy.depth_decay ** max(0, distance - 1))
            seeds.append((nid, act, "lexical_seed"))

        best: dict[str, float] = {}
        best_meta: dict[str, tuple[int, str, str | None]] = {}
        heap: list[tuple[float, int, str, str, str | None]] = []
        serial = 0
        for nid, act, via in seeds:
            if act < self.policy.min_activation: continue
            if act > best.get(nid, -1.0):
                best[nid] = act
                best_meta[nid] = (0, via, None)
                heapq.heappush(heap, (-act, serial, nid, via, None)); serial += 1

        relation_checks = parent_checks = popped = 0
        while heap and len(best_meta) <= self.policy.max_recall_nodes:
            neg_act, _, nid, _, _ = heapq.heappop(heap)
            act = -neg_act
            if act + 1e-12 < best.get(nid, -1.0):
                continue
            depth, _, _ = best_meta[nid]
            popped += 1
            if depth >= self.policy.max_recall_depth:
                continue

            # Structural upward routing is central to Equinox: known lower nodes can
            # expose a bounded set of contexts/macros without scanning all parents.
            parents = (self.index.parents.get(nid, [])[:self.policy.max_structural_parents]
                       if self.policy.enable_parent_routing else [])
            parent_checks += 1 + len(parents)
            for pid in parents:
                pnode = self.index.node_by_id[pid]
                if pnode.kind == NodeKind.QUANT.value:
                    continue
                if not self._exception_metadata_allowed(pnode.metadata or {}):
                    continue
                nxt = act * self.policy.parent_weight * self.policy.depth_decay
                if nxt < self.policy.min_activation:
                    continue
                if nxt > best.get(pid, -1.0) + 1e-12:
                    best[pid] = nxt
                    best_meta[pid] = (depth + 1, "parent", nid)
                    heapq.heappush(heap, (-nxt, serial, pid, "parent", nid)); serial += 1

            # Associative/semantic recall relations. PREDICTION and TRANSITION are
            # deliberately excluded here; they are output channels, not recall links.
            rels = self.index.relations_from.get(nid, []) if self.policy.enable_relation_recall else []
            relation_checks += 1 + len(rels)
            ranked_rels: list[tuple[float, CompiledRelation]] = []
            for rel in rels:
                weight = self.policy.relation_weights.get(rel.relation_type)
                if weight is None:
                    continue
                q = self._relation_quality(rel, weight)
                if q > 0:
                    ranked_rels.append((q, rel))
            ranked_rels.sort(key=lambda x: (-x[0], x[1].to_id, x[1].relation_type))
            for q, rel in ranked_rels[:self.policy.max_relation_neighbors]:
                target = self.index.node_by_id.get(rel.to_id)
                if target is None or target.kind == NodeKind.QUANT.value:
                    continue
                if not self._exception_metadata_allowed(target.metadata or {}):
                    continue
                nxt = act * q * self.policy.depth_decay
                if nxt < self.policy.min_activation:
                    continue
                if nxt > best.get(rel.to_id, -1.0) + 1e-12:
                    best[rel.to_id] = nxt
                    best_meta[rel.to_id] = (depth + 1, f"relation:{rel.relation_type}", nid)
                    heapq.heappush(heap, (-nxt, serial, rel.to_id, f"relation:{rel.relation_type}", nid)); serial += 1

            if len(best_meta) >= self.policy.max_recall_nodes:
                break

        ordered = sorted(best, key=lambda n: (-best[n], best_meta[n][0], n))[:self.policy.max_recall_nodes]
        traces = [RecallTrace(
            n, self.index.node_by_id[n].surface, self.index.node_by_id[n].kind,
            self.index.node_by_id[n].level, best[n], best_meta[n][0], best_meta[n][1], best_meta[n][2]
        ) for n in ordered]
        meta = {
            "resolved_context_ids": context_ids,
            "lexical_seed_ids": lexical_ids,
            "tokens": tokens,
            "context_resolution_kind": resolution_kind,
            "frame_admission": frame_admission,
            "work": {
                "recall_nodes": len(traces),
                "queue_pops": popped,
                "relation_checks": relation_checks,
                "parent_checks": parent_checks,
                "frame_match_checks": frame_checks,
                "frame_admission_checks": sum(len(x.get("slot_checks", [])) for x in frame_admission),
                "frame_admission_rejections": sum(1 for x in frame_admission if x.get("structural_match") and not x.get("admitted")),
                "frame_specificity_suppressed": sum(1 for x in frame_admission if x.get("suppressed_by_specificity")),
            },
        }
        return traces, meta

    def _match_relation_composition_schema(self, live_tokens: list[str], schema: dict[str, Any]) -> tuple[bool, str | None]:
        pattern = list(schema.get("pattern") or [])
        if len(pattern) != len(live_tokens):
            return False, None
        slot_position = int(schema.get("slot_position", -1))
        if slot_position < 0 or slot_position >= len(pattern):
            return False, None
        for i, (want, got) in enumerate(zip(pattern, live_tokens)):
            if i == slot_position:
                continue
            if want != got:
                return False, None
        return True, live_tokens[slot_position]

    def _execute_relation_composition(self, live_tokens: list[str]) -> tuple[list[dict[str, Any]], dict[str, int]]:
        """Execute stable compiled semantic path schemas for the live query.

        This mechanism deliberately does not invent a missing relation. The query
        wording must match an explicit schema and each hop must be backed by a
        compiled non-predictive relation. The terminal node becomes a prediction
        candidate only if the complete path clears the configured quality floor.
        """
        diagnostics: list[dict[str, Any]] = []
        work = {
            "relation_composition_schema_checks": 0,
            "relation_composition_path_relation_checks": 0,
            "relation_composition_matches": 0,
            "relation_composition_admissions": 0,
            "relation_composition_rejections": 0,
        }
        if not self.policy.enable_relation_composition_prediction:
            return diagnostics, work
        schemas = self._relation_composition_by_width.get(len(live_tokens), [])
        for schema in schemas[: self.policy.relation_composition_max_matches]:
            work["relation_composition_schema_checks"] += 1
            matched, slot_surface = self._match_relation_composition_schema(live_tokens, schema)
            diag: dict[str, Any] = {
                "schema_node_id": schema.get("node_id"),
                "schema_surface": schema.get("surface"),
                "pattern_tokens": list(schema.get("pattern") or []),
                "path": list(schema.get("path") or []),
                "status": schema.get("status"),
                "source_count": int(schema.get("source_count", 0) or 0),
                "support_count": int(schema.get("support_count", 0) or 0),
                "matched": matched,
                "admitted": False,
                "reason": "pattern_mismatch" if not matched else "",
                "slot_surface": slot_surface,
                "terminals": [],
            }
            if not matched or not slot_surface:
                diagnostics.append(diag)
                continue
            work["relation_composition_matches"] += 1
            starts = [
                nid for nid in self.index.surface_to_nodes.get(slot_surface, [])
                if self.index.node_by_id.get(nid) is not None
                and self.index.node_by_id[nid].kind != NodeKind.QUANT.value
            ]
            if not starts:
                diag["reason"] = "slot_surface_unknown"
                work["relation_composition_rejections"] += 1
                diagnostics.append(diag)
                continue

            # state = (node_id, quality, path_node_ids, path_edges)
            states: list[tuple[str, float, list[str], list[dict[str, Any]]]] = [
                (nid, 1.0, [nid], []) for nid in starts
            ]
            for step in list(schema.get("path") or []):
                next_states: dict[str, tuple[float, list[str], list[dict[str, Any]]]] = {}
                rtype = str(step["relation_type"]); direction = str(step.get("direction") or "out")
                table = self.index.relations_from if direction == "out" else self.index.relations_to
                for current, quality, path_nodes, path_edges in states:
                    bucket = table.get(current, [])
                    work["relation_composition_path_relation_checks"] += len(bucket)
                    for rel in bucket:
                        if rel.relation_type != rtype:
                            continue
                        nxt = rel.to_id if direction == "out" else rel.from_id
                        if nxt in path_nodes:
                            continue
                        target = self.index.node_by_id.get(nxt)
                        if target is None or target.kind == NodeKind.QUANT.value:
                            continue
                        edge_q = max(0.0, min(1.0, float(rel.score) * float(rel.confidence)))
                        nq = quality * edge_q
                        edge = {
                            "from_id": rel.from_id, "to_id": rel.to_id,
                            "relation_type": rel.relation_type, "direction": direction,
                            "score": float(rel.score), "confidence": float(rel.confidence),
                            "source_count": int(rel.source_count), "edge_quality": edge_q,
                        }
                        old = next_states.get(nxt)
                        candidate = (nq, path_nodes + [nxt], path_edges + [edge])
                        if old is None or nq > old[0] + 1e-12:
                            next_states[nxt] = candidate
                states = [(nid, row[0], row[1], row[2]) for nid, row in next_states.items()]
                if not states:
                    break

            threshold = max(0.0, min(1.0, float(schema.get("min_path_quality", self.policy.relation_composition_min_path_quality))))
            terminals = []
            for nid, quality, path_nodes, path_edges in states:
                node = self.index.node_by_id.get(nid)
                if node is None or node.kind == NodeKind.QUANT.value or not node.surface:
                    continue
                if quality + 1e-12 < threshold:
                    continue
                terminals.append({
                    "target_id": nid, "target_surface": node.surface, "path_quality": quality,
                    "path_node_ids": path_nodes, "path_edges": path_edges,
                })
            terminals.sort(key=lambda x: (-float(x["path_quality"]), str(x["target_surface"]), str(x["target_id"])))
            diag["terminals"] = terminals[: self.policy.max_predictions]
            diag["min_path_quality"] = threshold
            if not terminals:
                diag["reason"] = "no_supported_composition_path"
                work["relation_composition_rejections"] += 1
            else:
                diag["admitted"] = True
                diag["reason"] = "compiled_relation_composition_path"
                work["relation_composition_admissions"] += 1
            diagnostics.append(diag)
        return diagnostics, work

    def _novel_entity_semantic_substitution_support(
        self,
        live_tokens: list[str],
        source_tokens: list[str],
    ) -> dict[str, Any]:
        """Require independent semantic evidence for tokens new to prediction.

        The comparison is deliberately conservative: v0.33 only admits a single
        aligned substitution in an otherwise matching context. This avoids turning
        generic graph proximity into prediction authority and gives later versions a
        clean baseline for richer prototype/hierarchy admission.
        """
        diag: dict[str, Any] = {
            "required": False, "admitted": True, "reason": "no_novel_predictive_token",
            "novel_tokens": [], "substitutions": [], "relation_checks": 0,
        }
        novel = sorted({t for t in live_tokens if t not in self._predictive_context_token_surfaces})
        diag["novel_tokens"] = novel
        if not novel or not self.policy.enable_evidence_backed_novel_entity_admission:
            if novel and not self.policy.enable_evidence_backed_novel_entity_admission:
                diag.update({"required": True, "admitted": True, "reason": "novel_entity_admission_disabled"})
            return diag
        diag["required"] = True
        if len(live_tokens) != len(source_tokens):
            diag.update({"admitted": False, "reason": "novel_entity_unaligned_context"})
            return diag
        diffs = [(i, a, b) for i, (a, b) in enumerate(zip(live_tokens, source_tokens)) if a != b]
        if len(diffs) != 1:
            diag.update({"admitted": False, "reason": "novel_entity_requires_single_substitution"})
            return diag
        pos, live_tok, source_tok = diffs[0]
        if live_tok not in novel:
            diag.update({"admitted": False, "reason": "novel_token_not_aligned_substitution"})
            return diag
        live_ids = [nid for nid in self.index.surface_to_nodes.get(live_tok, [])
                    if self.index.node_by_id[nid].kind == NodeKind.SURFACE.value]
        source_ids = [nid for nid in self.index.surface_to_nodes.get(source_tok, [])
                      if self.index.node_by_id[nid].kind == NodeKind.SURFACE.value]
        if not live_ids or not source_ids:
            diag.update({"admitted": False, "reason": "novel_entity_missing_compiled_semantics"})
            return diag
        allowed = set(self.policy.novel_entity_semantic_relation_types)
        best = 0.0
        best_rel: dict[str, Any] | None = None
        checks = 0
        for a in live_ids:
            for b in source_ids:
                for rtype in allowed:
                    for left, right in ((a, b), (b, a)):
                        checks += 1
                        rel = self.index.relation_by_key.get((left, right, rtype))
                        if rel is None:
                            continue
                        quality = max(0.0, min(1.0, float(rel.score) * float(rel.confidence)))
                        if quality > best:
                            best = quality
                            best_rel = {
                                "from_id": rel.from_id, "to_id": rel.to_id,
                                "relation_type": rel.relation_type, "score": rel.score,
                                "confidence": rel.confidence, "quality": quality,
                            }
        diag["relation_checks"] = checks
        substitution = {
            "position": pos, "live_token": live_tok, "source_token": source_tok,
            "best_quality": best, "threshold": self.policy.novel_entity_min_semantic_quality,
            "best_relation": best_rel,
        }
        diag["substitutions"] = [substitution]
        if best + 1e-12 < self.policy.novel_entity_min_semantic_quality:
            diag.update({"admitted": False, "reason": "insufficient_novel_entity_semantic_support"})
            return diag
        diag.update({"admitted": True, "reason": "evidence_backed_novel_entity_substitution"})
        return diag

    def _recalled_prediction_support(
        self,
        source: RuntimeNode | None,
        live_tokens: list[str],
        lexical_ids: list[str],
        *,
        has_resolved_context: bool,
    ) -> tuple[bool, dict[str, Any]]:
        """Decide whether associative recall is sufficient to execute a prediction.

        v0.31 showed that a remembered predictive parent could fire merely because
        one known token was present. v0.32 separates *memory access* from
        *prediction authority*. A recalled context needs substantial lexical
        overlap with the live context. If the live context contains an unseen token,
        lexical recall alone may not generalize over it; a compiled exact/suffix or
        admitted frame context must provide that authority.
        """
        diag: dict[str, Any] = {
            "source_node_id": source.node_id if source else None,
            "source_surface": source.surface if source else None,
            "admitted": False,
            "reason": "",
            "shared_tokens": [],
            "shared_token_count": 0,
            "overlap_fraction": 0.0,
            "unknown_live_tokens": [],
        }
        if not self.policy.enable_support_gated_recalled_prediction:
            diag["admitted"] = True
            diag["reason"] = "support_gate_disabled"
            return True, diag
        if source is None or not source.surface:
            diag["reason"] = "source_has_no_surface_support"
            return False, diag

        known_live_surfaces = {
            self.index.node_by_id[nid].surface
            for nid in lexical_ids
            if nid in self.index.node_by_id and self.index.node_by_id[nid].surface
        }
        unknown = [tok for tok in live_tokens if tok not in known_live_surfaces]
        diag["unknown_live_tokens"] = unknown
        if (unknown and not has_resolved_context
                and not self.policy.recalled_support_allow_unknown_tokens_without_context):
            diag["reason"] = "unknown_live_token_without_compiled_context"
            return False, diag

        live_set = set(live_tokens)
        source_set = set(_surface_tokens(source.surface))
        shared = sorted(live_set & source_set)
        denom = min(len(live_set), len(source_set))
        fraction = (len(shared) / denom) if denom else 0.0
        diag["shared_tokens"] = shared
        diag["shared_token_count"] = len(shared)
        diag["overlap_fraction"] = fraction
        if len(shared) < self.policy.recalled_support_min_shared_tokens:
            diag["reason"] = "insufficient_shared_tokens"
            return False, diag
        if fraction + 1e-12 < self.policy.recalled_support_min_overlap_fraction:
            diag["reason"] = "insufficient_overlap_fraction"
            return False, diag

        # If no exact/suffix/frame context already carries the live query, tokens
        # that are new to predictive training require independent semantic support
        # for the aligned substitution. This is the v0.33 L4 recovery path.
        if not has_resolved_context:
            semantic = self._novel_entity_semantic_substitution_support(live_tokens, _surface_tokens(source.surface))
            diag["novel_entity_admission"] = semantic
            if semantic.get("required") and not semantic.get("admitted"):
                diag["reason"] = str(semantic.get("reason") or "novel_entity_rejected")
                return False, diag
            if semantic.get("required") and semantic.get("admitted"):
                diag["admitted"] = True
                diag["reason"] = str(semantic.get("reason") or "evidence_backed_novel_entity_substitution")
                return True, diag

        diag["admitted"] = True
        diag["reason"] = "context_overlap_supported"
        return True, diag

    def predict(self, context: str) -> RuntimePredictionReport:
        recall, meta = self.recall(context)
        context_ids = set(meta["resolved_context_ids"])
        lexical_ids = list(meta["lexical_seed_ids"])
        recalled = {x.node_id: x for x in recall}
        contributions: dict[str, list[PredictionContribution]] = {}
        prediction_relations_checked = 0
        support_gate_checks = 0
        support_gate_rejections = 0
        support_gate_reasons: dict[str, int] = {}

        def add(rel: CompiledRelation, activation: float, weight: float, direct: bool) -> None:
            nonlocal prediction_relations_checked
            prediction_relations_checked += 1
            target = self.index.node_by_id.get(rel.to_id)
            if target is None or target.kind == NodeKind.QUANT.value:
                return
            c = max(0.0, min(1.0, activation * rel.score * rel.confidence * weight))
            if c < self.policy.min_prediction_score:
                return
            source = self.index.node_by_id.get(rel.from_id)
            contributions.setdefault(rel.to_id, []).append(PredictionContribution(
                rel.from_id, source.surface if source else None, rel.relation_type,
                rel.score, rel.confidence, rel.source_count, activation, weight, c, direct,
            ))

        # Exact/suffix predictive contexts get full-weight direct predictions.
        # Induced frame schemas are explicit compiled hypotheses and therefore use
        # the lower frame_match_weight even when they resolve the live context.
        if self.policy.enable_direct_context_prediction:
            for cid in meta["resolved_context_ids"]:
                node = self.index.node_by_id.get(cid)
                is_frame = bool(node and (node.metadata or {}).get("role") == "induced_frame_schema")
                activation = self.policy.frame_match_weight if is_frame else 1.0
                for rel in self.index.relations_from.get(cid, []):
                    if rel.relation_type == RelationType.PREDICTION.value:
                        add(rel, activation, self.policy.exact_context_prediction_weight, True)

        # Latest lexical item supplies bigram transition predictions.
        if lexical_ids and self.policy.enable_transition_prediction:
            last = lexical_ids[-1]
            act = recalled[last].activation if last in recalled else self.policy.seed_recent_word_weight
            for rel in self.index.relations_from.get(last, []):
                if rel.relation_type == RelationType.TRANSITION.value:
                    add(rel, act, self.policy.transition_weight, True)

        # v0.34 explicit relation-composition schemas can derive a target without
        # any direct PREDICTION evidence for the live relation family. The schema
        # and every semantic path hop are compiled in the BASE; opaque families
        # therefore still abstain.
        composition_diagnostics, composition_work = self._execute_relation_composition(list(meta.get("tokens", [])))
        for diag in composition_diagnostics:
            if not diag.get("admitted"):
                continue
            schema_id = str(diag.get("schema_node_id") or "")
            schema_node = self.index.node_by_id.get(schema_id)
            for terminal in diag.get("terminals", []):
                target_id = str(terminal.get("target_id") or "")
                target = self.index.node_by_id.get(target_id)
                if target is None:
                    continue
                quality = max(0.0, min(1.0, float(terminal.get("path_quality", 0.0))))
                contribution = quality * self.policy.relation_composition_weight
                if contribution < self.policy.min_prediction_score:
                    continue
                contributions.setdefault(target_id, []).append(PredictionContribution(
                    schema_id, schema_node.surface if schema_node else None, RelationType.COMPOSITION.value,
                    quality, 1.0, int(diag.get("source_count", 1) or 1), 1.0,
                    self.policy.relation_composition_weight, contribution, True,
                ))

        # Recalled contexts can contribute prediction evidence at a lower weight.
        if self.policy.enable_recalled_context_prediction:
            live_tokens = list(meta.get("tokens", []))
            for trace in recall:
                if trace.node_id in context_ids:
                    continue
                trace_node = self.index.node_by_id.get(trace.node_id)
                if trace_node is not None and not self._exception_metadata_allowed(trace_node.metadata or {}):
                    continue
                predictive = [
                    rel for rel in self.index.relations_from.get(trace.node_id, [])
                    if rel.relation_type == RelationType.PREDICTION.value
                ]
                if not predictive:
                    continue
                admitted, gate = self._recalled_prediction_support(
                    trace_node, live_tokens, lexical_ids, has_resolved_context=bool(context_ids)
                )
                support_gate_checks += 1
                reason = str(gate.get("reason", "unknown"))
                support_gate_reasons[reason] = support_gate_reasons.get(reason, 0) + 1
                if not admitted:
                    support_gate_rejections += 1
                    continue
                for rel in predictive:
                    add(rel, trace.activation, self.policy.recalled_prediction_weight, False)

        candidates: list[PredictionCandidate] = []
        for target_id, cs in contributions.items():
            direct = _noisy_or(c.contribution for c in cs if c.relation_type == RelationType.PREDICTION.value and c.direct_context)
            recalled_score = _noisy_or(c.contribution for c in cs if c.relation_type == RelationType.PREDICTION.value and not c.direct_context)
            transition = _noisy_or(c.contribution for c in cs if c.relation_type == RelationType.TRANSITION.value)
            composition = _noisy_or(c.contribution for c in cs if c.relation_type == RelationType.COMPOSITION.value)
            score = _noisy_or(c.contribution for c in cs)
            node = self.index.node_by_id[target_id]
            cs.sort(key=lambda c: (-c.contribution, c.relation_type, c.from_node_id))
            candidates.append(PredictionCandidate(
                target_id, node.surface, score, direct, recalled_score, transition, len(cs), cs, composition,
            ))

        # v0.40 evidence ordering. Direct exact/suffix prediction evidence is a
        # stronger *kind* of support than associative recall once it clears a
        # minimum quality floor. This prevents many weak recalled contexts from
        # winning merely by noisy-OR accumulation. If no direct candidate clears
        # the floor, ordinary score ordering is preserved. Frames and composition
        # are deliberately excluded from this authority tier.
        resolution_kind = str(meta.get("context_resolution_kind") or "none")
        authority_resolution = resolution_kind == "exact" or (
            resolution_kind == "suffix" and self.policy.exact_context_authority_include_suffix
        )
        authority_active = False
        authority_ids: set[str] = set()
        if self.policy.enable_exact_context_authority and authority_resolution:
            authority_ids = {
                c.node_id for c in candidates
                if c.direct_score + 1e-12 >= self.policy.exact_context_authority_min_direct_score
            }
            authority_active = bool(authority_ids)
        if authority_active:
            candidates.sort(key=lambda x: (
                0 if x.node_id in authority_ids else 1,
                -x.direct_score if x.node_id in authority_ids else 0.0,
                -x.score, -x.transition_score, x.surface or "", x.node_id,
            ))
        else:
            candidates.sort(key=lambda x: (-x.score, -x.direct_score, -x.transition_score, x.surface or "", x.node_id))
        candidates = candidates[:self.policy.max_predictions]

        work = dict(meta["work"])
        work["exact_context_authority_active"] = int(authority_active)
        work["exact_context_authority_candidates"] = len(authority_ids)
        work["context_resolution_kind:exact"] = int(resolution_kind == "exact")
        work["context_resolution_kind:suffix"] = int(resolution_kind == "suffix")
        work["context_resolution_kind:frame"] = int(resolution_kind == "frame")
        work.update(composition_work)
        work["prediction_relation_checks"] = prediction_relations_checked
        work["support_gate_checks"] = support_gate_checks
        work["support_gate_rejections"] = support_gate_rejections
        work["support_gate_admissions"] = support_gate_checks - support_gate_rejections
        for reason, count in sorted(support_gate_reasons.items()):
            work[f"support_gate_reason:{reason}"] = count
        work["prediction_candidates_before_limit"] = len(contributions)
        work["prediction_candidates_returned"] = len(candidates)
        return RuntimePredictionReport(
            RUNTIME_COGNITION_FORMAT, RUNTIME_COGNITION_VERSION, context,
            list(meta["resolved_context_ids"]), lexical_ids, recall, candidates, work,
            asdict(self.policy), list(meta.get("frame_admission", [])), composition_diagnostics,
        )


def load_prediction_cases(paths: Iterable[str | Path]) -> list[PredictionBenchmarkCase]:
    out: list[PredictionBenchmarkCase] = []
    for p in paths:
        path = Path(p)
        if path.suffix.lower() == ".jsonl":
            for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip() or line.lstrip().startswith("#"): continue
                obj = json.loads(line)
                if not isinstance(obj, dict) or not obj.get("context") or not obj.get("expected"):
                    raise ValueError(f"{path}:{line_no}: expected JSON object with context + expected")
                out.append(PredictionBenchmarkCase(str(obj["context"]), str(obj["expected"])))
        else:
            for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip() or line.lstrip().startswith("#"): continue
                if "\t" not in line:
                    raise ValueError(f"{path}:{line_no}: expected TAB-separated context and expected surface")
                context, expected = line.split("\t", 1)
                out.append(PredictionBenchmarkCase(context.strip(), expected.strip()))
    return out


def benchmark_predictions(
    pack: LoadedRuntimePack | str | Path,
    cases: list[PredictionBenchmarkCase],
    *,
    policy: RuntimeCognitionPolicy | None = None,
    top_k: int = 5,
) -> PredictionBenchmarkReport:
    if top_k < 1: raise ValueError("top_k must be >= 1")
    cog = RuntimeCognition.from_path(pack, policy) if isinstance(pack, (str, Path)) else RuntimeCognition(pack, policy)
    hit1 = hitk = unresolved = expected_missing = 0
    rr_total = 0.0
    recall_total = relation_total = 0
    details: list[dict[str, Any]] = []
    for case in cases:
        report = cog.predict(case.context)
        surfaces = [x.surface for x in report.predictions]
        rank = None
        for i, surface in enumerate(surfaces, 1):
            if surface == case.expected:
                rank = i; break
        if rank == 1: hit1 += 1
        if rank is not None and rank <= top_k: hitk += 1
        if rank is not None: rr_total += 1.0 / rank
        if not report.resolved_context_ids: unresolved += 1
        if not cog.index.surface_to_nodes.get(case.expected): expected_missing += 1
        recall_total += report.work.get("recall_nodes", 0)
        relation_total += report.work.get("relation_checks", 0) + report.work.get("prediction_relation_checks", 0)
        details.append({
            "context": case.context,
            "expected": case.expected,
            "rank": rank,
            "top_surfaces": surfaces[:top_k],
            "resolved_context_ids": report.resolved_context_ids,
            "recall_nodes": report.work.get("recall_nodes", 0),
            "relation_checks": report.work.get("relation_checks", 0) + report.work.get("prediction_relation_checks", 0),
        })
    n = len(cases)
    return PredictionBenchmarkReport(
        RUNTIME_COGNITION_FORMAT, RUNTIME_COGNITION_VERSION, n,
        (hit1 / n if n else 0.0), (hitk / n if n else 0.0), (rr_total / n if n else 0.0),
        unresolved, expected_missing,
        (recall_total / n if n else 0.0), (relation_total / n if n else 0.0), top_k, details,
    )
